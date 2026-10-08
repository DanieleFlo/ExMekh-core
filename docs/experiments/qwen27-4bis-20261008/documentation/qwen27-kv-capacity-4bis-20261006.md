# 4 BIS — Modalità di contesto esteso per Qwen3.8 27B

Implementazione esplicita per Qwen3.8 27B, 6 ottobre 2026. L'utente ha
esteso il lavoro della sessione includendo il 4 BIS e una prova con almeno
65536 token effettivi. Accetta un rallentamento in cambio di più contesto e
vuole scegliere il compromesso dai benchmark; non è fissato un limite
universale del 10% o del 20%. Il percorso ordinario resta il default.

## Vincoli verificati nel checkout

`qwen35`: 64 layer target, attenzione completa ogni quattro layer, 4 KV head,
dimensione 256. `qwen_graph_alloc` alloca K e V FP32 per 16 layer. Il target
richiede quindi 131072 byte di KV per token, esclusi MTP, ricorrenza, vision,
workspace e pesi. Non dispone della selezione QSA sparsa di Flash.

Il PCIe locale è Gen4 x2; il report del 4 ottobre misura H2D pinned circa
2,74 GiB/s e la fotografia del 6 ottobre conferma ancora x2. Una migrazione
di sistema/BIOS non è un prerequisito software né una modifica inclusa qui.

La rimisura del 6 ottobre, con due warm-up e cinque campioni su buffer pinned
da 64 MiB, dà una mediana H2D di 3,025 GiB/s e D2H di 3,067 GiB/s.
Con questa H2D, i 12 GiB remoti a 128K richiederebbero comunque circa
3,97 s/token nel disegno ingenuo. Le due misure usano buffer diversi e non
dimostrano da sole un cambiamento della banda del sistema.

| Token effettivi | KV target FP32 | KV remoto con 32K residenti | Solo H2D a 2,74 GiB/s, se riletto per token |
| ---: | ---: | ---: | ---: |
| 28000 | 3,42 GiB | 0 | 0 |
| 32768 | 4 GiB | 0 | 0 |
| 65536 | 8 GiB | 4 GiB | 1,46 s/token |
| 131072 | 16 GiB | 12 GiB | 4,38 s/token |
| 262144 | 32 GiB | 28 GiB | 10,22 s/token |

Sono stime del disegno che trasferisce tutto il KV remoto a ogni token,
non benchmark. Overlap/tile possono nascondere una parte delle attese;
non eliminano il volume di trasferimenti. Questo disegno non soddisfa da
solo la richiesta di un rallentamento leggero sul link attuale.

## Percorsi implementati

1. **Crescita progressiva in VRAM (`grow`).** Blocchi da 4096 celle allocati sulle celle realmente
   utilizzate, mantenendo indici logici, causalità e precisione FP32. Non
   alloca in anticipo il KV della massima capacità configurata. Può evitare
   OOM iniziali e sprechi, ma il limite fisico del contesto pieno resta.
2. **Offload FP32 con attention GPU per tile (`hybrid-attention`).** Host backing ordinario,
   staging pinned bounded, stream/eventi e doppio buffer. Le statistiche
   softmax `(max, normalizzatore, output)` includono l'intero passato.
   Il decode e la verifica MTP mantengono le 32 partizioni e il merge
   originali, raggruppando partizioni consecutive in tile fino a 8192 celle.
   Copie e calcolo usano due buffer con eventi di fine copia e consumo.
   Il prefill usa un workspace di un layer intero e la stessa attention
   CUDA originale, preservandone precisione e ordine della riduzione.
   Offre capacità reale; sulla macchina locale il costo
   PCIe stimato ne limita fortemente la velocità.
3. **Ricerca successiva: attention del passato remoto sulla CPU.** Trasferire
   query e risultati parziali, preservando FP32, invece di tutto il KV.
   Riutilizzare K/V tra i sei query head di ogni KV group. Unire le statistiche
   softmax di entrambe le parti sulla GPU. Questa è la ricerca più pertinente
   per ridurre il costo PCIe, ma richiede una misura preliminare del throughput
   AVX2 e della banda RAM visibili a WSL; non è un guadagno già dimostrato.

Nessun prototipo elimina token o introduce una sliding window semantica.
Quantizzazione KV e cambiamenti dei pesi non appartengono a questa scheda.

## Parametri e lifetime

- `--qwen-kv-capacity-mode off|grow|hybrid-attention`: default `off`.
- `--qwen-kv-resident N`: capacità fisica del prefisso residente per il percorso ibrido, default 32768, intervallo 64..262144.
- `--ctx N`: continua a indicare la capacità logica richiesta; non viene
  ridotta silenziosamente. Il runtime espone memoria fisica, RAM, byte per
  token e frontiera effettiva separatamente.

Questi parametri sono implementati in CLI, server e benchmark. Le variabili
equivalenti sono `DS4_CUDA_QWEN_KV_CAPACITY_MODE` e
`DS4_CUDA_QWEN_KV_RESIDENT`. Il percorso richiede CUDA, singola GPU e
capacità logica fino a 262144 celle. Flag Flash ereditati non lo attivano.

K/V e frontiere appartengono alla sessione; stage e tile sono workspace
serializzati dell'engine. Le letture e scritture del checkpoint usano il KV
logico FP32 completo. Reset, restore e distruzione attendono tutti i
consumatori e le copie. Le nuove righe oltre una frontiera ripristinata
sono sovrascritte dall'append; i kernel conservano la maschera causale.
I checkpoint esistenti e gli snapshot in memoria conservano il formato
logico originale. Le views e copie fisiche incompatibili vengono rifiutate.

Esempio esplicito per la capacità estesa:

```sh
./ds4 --cuda -m gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  --ctx 131072 --qwen-kv-capacity-mode hybrid-attention \
  --qwen-kv-resident 32768 -p '...'
```

L'esempio configura la capacità: non equivale a una qualifica con 128K
token effettivi. A 64K il target passa da 8 a 4 GiB di KV residente; bisogna
aggiungere workspace, MTP, pesi e ricorrenza. Il report runtime misura
separatamente massimo KV residente, RAM, workspace e traffico DMA.

## Esperimenti e decisione

`tests/test_qwen35_kv_capacity` confronta off, grow e hybrid-attention
con le esatte geometrie locali e un budget residente forzato a 64 celle.
Le nove primitive v24 attraversano residenza e blocchi di crescita, query
ordinarie e cinque righe MTP, mRoPE, prefill GEMM da 512 righe, più tile
a 32768/65536 celle e restore in una cache con budget residente diverso:
output e byte logici K/V sono bit-identici all'oracle CUDA.
Il callback MTP sul GGUF reale supera cancellazione, snapshot completo
e RNG esatto con spill forzato (prompt da 2066 token); anche lo snapshot
denso in memoria supera i controlli esistenti.

Il nuovo driver `tools/qualify_qwen27_capacity.py` usa frontiere effettive,
stessi prompt/chunk e otto token teacher-forced, confronta tutti i logits e
campiona la memoria GPU ogni due secondi. La prova v24 su
`Qwen3.8-27B-UD-Q4_K_S.gguf` ha completato 65536 token effettivi e otto token
teacher-forced, con logits finiti sull'intero vocabolario:

| Modo | Prefill TPS | Decode TPS | Tempo totale s | Picco GPU campionato MiB |
| --- | ---: | ---: | ---: | ---: |
| hybrid-attention, 32768 residenti, chunk 1024 | 396,78 | 0,68 | 186,286 | 22073 |

Un solo campione diagnostico, con log di avanzamento per chunk e senza
profiler. Il runtime misura 8,002 GiB di KV host, 4,109 GiB di KV residente
di picco (compresa la vecchia allocazione durante la crescita), 0,625 GiB di
workspace, 99,990 GiB H2D e 8,001 GiB D2H. Il log iniziale dei context
buffers resta una stima della capacità logica ordinaria; i numeri fisici
sono quelli del runtime e del campionamento GPU. Dati:
`qwen27-capacity-v24-64-hybrid/report.json`.

La successiva v27 completa anche il confronto numerico del modello intero
a 64K effettivi. Il riferimento diagnostico usa
`DS4_CUDA_QWEN_KV_REFERENCE_ATTENTION=1`: esegue i kernel originali su K/V
contigui dell'intero layer, copiati dal mirror logico. Usa un solo workspace
di layer e non è una misura del modello con tutti i layer KV in VRAM.
Il percorso a tile e il riferimento hanno logits completi bit-identici
alla frontiera e dopo otto token teacher-forced, KL zero. I nove oracle
CUDA indipendenti con KV residente a 64K restano una verifica distinta.

| v27, 65536 token effettivi, 32768 residenti | Prefill TPS | Decode TPS | Tempo totale s | Picco GPU campionato MiB |
| --- | ---: | ---: | ---: | ---: |
| Attention originale su un layer contiguo | 416,34 | 0,69 | 177,314 | 21612 |
| Attention per tile | 414,10 | 0,69 | 177,527 | 21684 |

Stato `PASS` per capacità e numerica, un campione diagnostico per modalità.
Dati: `qwen27-capacity-v27-reference64/report.json`. Il decode di circa
1,45 s/token è coerente con il costo
del trasferimento di circa 4 GiB/token sul link locale: offre capacità con
un costo di decode significativo, che l'utente può valutare dai benchmark.
Il primo tentativo residente con chunk 1024 è stato interrotto per dare
precedenza alla prova di offload: ancora in prefill dopo circa 58 minuti,
senza frontiera o TPS completi. È conservato in
`qwen27-capacity-v23-64/report.json`; non prova un OOM o un limite di capacità.

Poi confronti a token effettivi 8K, 28K, 32K, 64K e, se il costo misurato
lo consente, 128K/262K. Stessi pesi, prefissi, sampling, chunk e precisione.
Registrare prefill, TTFT, decode p50/p90, RAM/VRAM, trasferimenti e capacità
massima realmente completata. Testare mRoPE/vision, MTP/exact sampling,
rewind, restore in una nuova sessione e checkpoint con resident budget diverso.

Confrontare logits, loss teacher-forced e task-quality oltre alla memoria;
un seed uguale o una risposta plausibile non certificano la correttezza.
Consegnare una curva contesto/velocità e classificare ogni profilo come
`guadagno prestazionale`, `maggiore capacità con rallentamento misurato` o
`non conveniente`. L'utente decide il compromesso dai risultati; la modalità
rimane esplicita e disattivata per default.

## Speculazione nel contesto esteso — aggiornamento 7 ottobre

Richiesta: scegliere il drafter affidabile più veloce sul GGUF/3090 locale e
valutare 14–15 proposte, riutilizzando DS4. NextN già presente e DSpark v2 sono
modelli distinti; la sola estensione del verifier non implementa DSpark.

La v35 qualifica verifier fino a 16 righe (15 proposte più seed). Q4_K, Q5_K
e Q6_K coincidono bit per bit con le proiezioni decode singole; ogni frontiera
di rollback riproduce stato recurrent, logits completi e continuazione. Gli
snapshot intermedi esatti di 15 righe occupano 2382888960 byte in RAM, con
49643520 byte di scratch GPU condiviso tra i 48 layer recurrent. Le primitive
KV esercitano 64K effettivi e batch di 8/15/16 righe, inclusa mRoPE. Nel modo
`hybrid-attention` la limitazione automatica NextN a una proposta oltre 2K è
rimossa; gli override espliciti restano rispettati. Il default ordinario resta
invariato. Benchmark reali AR/NextN sono separati dai test del verifier.

Checkpoint candidato: [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/tree/b9a5dbdf03bc999c6c73c426b19c2d9041cea393),
revision `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`, SHA-256 safetensors
`2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7`.
Scaricato e verificato. Il servizio ufficiale usa 7 proposte, mentre il blocco
di training è 16: 14–15 va confrontato, non dedotto dai TPS H200. Il modello ha
5 layer GQA 32Q/8KV, head 128, tap target 5/19/33/47/61, projector 25600→5120,
RoPE YaRN, Markov rank 256 e confidence head. Il vecchio DSpark DS4 usa MLA
DeepSeek: richiede un nuovo backbone Qwen e cattura dei tap target.

Il converter `gguf-tools/qwen_dspark_convert.py` valida schema, configurazione
e hash del checkpoint, usa il quantizer DS4 e mantiene norme/confidence F32.
La quantizzazione opzionale riguarda soltanto il drafter. La conversione da
sola non certifica runtime, acceptance o velocità: questi controlli sono
separati e descritti sotto.

### Backbone DSpark e qualificazione numerica (7 ottobre)

Il backbone DSpark è ora implementato, riusando embedding, GEMM, RMSNorm,
SwiGLU, head target e verifier DS4. La cattura dei cinque tap è successiva
al layer; ogni sessione conserva il KV del drafter in RAM FP16, con copie
esplicite attraverso lo staging bounded condiviso. L'attenzione GQA include
tutto il contesto e il blocco noise bidirezionale. Il target conserva FP32.
Il Markov rank 256 e il confidence head appartengono al checkpoint ufficiale.

`tests/test_qwen_dspark_primitives.c` verifica embedding F16/F32, concat dei
tap, norme learned e YaRN con oracle CPU float64, più attention sulle celle
effettive a 64K. Esercita contesti vuoti, tile non allineati e width 7/15;
max_abs YaRN circa 5,3e-7 e attention circa 6,2e-7. La fase YaRN è calcolata
in float64 per eliminare l'errore fast-math nei token distanti, senza cambiare
la tolleranza di normalizzazione. Log `dspark-primitives-v39-tight.log`.

Il verifier target ora conserva la policy decode anche per tail T2..5 e
per i formati IQ/Q3 del GGUF locale; le piccole proiezioni F32 usano la
stessa riduzione scalar del decode. Il test wide v40 passa per ogni formato
effettivamente trovato e per T2/3/4/5/8/15/16. Il test DSpark v40 forza
ogni accettazione da 1 a 16 righe: l'intero payload logico del target coincide
bit per bit con la valutazione sequenziale degli stessi token. Passano
cancellazione dei campioni/RNG, identity mismatch, checkpoint troncato,
restore, isolamento e ricreazione della sessione. Log `qwen-wide-v40.log`
e `dspark-state-v40.log`.

Il payload DSpark aggiunge un flag opzionale al formato logico versione 1,
fingerprint del GGUF completo e KV FP16 del drafter alla frontiera viva.
Non salva indirizzi o slot fisici. Un checkpoint di modalità diversa
richiede il rebuild del prompt; i vecchi checkpoint AR/NextN conservano il
percorso precedente. La nuova estensione non promuove da sola velocità o
qualità applicativa; vision/API e misure a cinque coppie alternate restano
gate separati.

### NextN e DSpark a 64K effettivi (7 ottobre)

I confronti v41–v43 generano 32 token dopo 65.536 token reali, mantenendo
32.768 celle residenti per layer target. AR, NextN7/15 e DSpark v2 Q8/Q4
riproducono gli stessi token e i logits completi prima e dopo la continuazione
bit per bit (KL zero). Sono diagnostiche singole, non cinque coppie A/B.

| Percorso | Prefill token/s | Decode complessivo token/s |
| --- | ---: | ---: |
| AR v43 | 402,30 | 0,67 |
| NextN7 v43 | 371,13 | 1,43 |
| NextN15 v43 | 370,13 | 1,07 |
| DSpark v2 Q4, 7 proposte v43 | 396,36 | 0,71 |

NextN7 e NextN15 accettano entrambi 23 proposte nei medesimi otto cicli
del campione: 50 proposte generate con width 7, 93 con width 15. La media
è 3,875 token utili per verifica, incluso il seed. Il costo medio della
proposta cresce da 579 a 1075 ms e quello della verifica da 1982 a 2406 ms;
il rollback resta circa 56 ms. La profondità configurata non è il numero
di token accettati. Report `qwen27-nextn-v43-64/report.json`.

Il default sperimentale di `hybrid-attention` seleziona NextN; l'override
`--qwen-extended-draft off|nextn|dspark` e `--mtp-draft N` consente confronti
espliciti. La profondità automatica NextN è 5, scelta dopo lo sweep
3/5/7/9/11/13/15 e confermata dall'utente. Nella scelta automatica, la parte di contesto ancora
interamente residente usa una sola proposta; la profondità esplicita viene
rispettata. La modalità ordinaria conserva le impostazioni precedenti.

I pesi DSpark sono residenti in VRAM (Q8 circa 1,89 GiB; Q4 circa 1,06 GiB).
Il suo KV storico è FP16 in RAM, circa 1,25 GiB a 64K per i cinque layer,
e viene letto per tile durante ogni proposta. Questo distingue la residenza
dei pesi da quella dell'intera inferenza. Il target mantiene il KV FP32.

La variante successiva separa il KV NextN dal budget target:
`DS4_CUDA_QWEN_MTP_KV_MODE=shared|auto|resident|staged`. `shared` riproduce il
limite del target; `resident` richiede il budget per l'intero KV del singolo
layer NextN, comprese le celle target promesse, una crescita transitoria e
512 MiB di margine. A 64K il KV NextN completo occupa circa 512 MiB, contro
256 MiB della configurazione condivisa. Non cambia formato o precisione.

`auto` usa `staged`: riusa il workspace KV del prefill target già allocato,
mentre il target in decode consuma buffer per tile separati. Il KV NextN
completo resta in quel workspace tra le proposte e i cicli; append GPU,
tag di ownership e frontiere evitano letture RAM ripetute. Prefill di altri
layer/sessioni, restore logico, reset e resize invalidano la mappa prima di
riusare il workspace. Non aumenta il budget VRAM della configurazione
condivisa. La cache logica RAM FP32 rimane autorevole per checkpoint e
cambio sessione. Il nuovo default dell'allocator in modalità capacità è
`auto`, da qualificare sul binario finale prima della promozione.

I contesti offloaded usano snapshot intermedi recurrent in RAM anche sotto
width 5, per limitare la pressione VRAM nel prefill WSL. Il ciclo lean V(2)
riusa lo snapshot GPU prebatch per la sola riga di rollback: quel ciclo non
salva lo stato prebatch e invalida la sessione se la verifica fallisce.
Nessuna nuova allocazione per token. Il percorso con snapshot sicuro mantiene
il salvataggio precedente. Primitive, ogni frontiera di rollback e callback
di cancellazione devono passare prima del confronto prestazionale.

La v46 passa le primitive KV a 64K, tutte le frontiere recurrent/logits a
3/5/7/9/11/13/14/15 e i tre gate di cancellazione: percorso ordinario con
variabili Flash ereditate, KV condiviso in capacità, KV NextN residente.
Lo sweep v46 completo a 65.536 token effettivi genera gli stessi 32 token
con tutte le profondità; i logits completi prima e dopo il decode coincidono
bit per bit con AR (KL zero). Un warm-up di 16 token viene ripristinato alla
frontiera originale prima della misura. Sono campioni diagnostici singoli,
con trace dei cicli; non costituiscono cinque coppie di promozione.

| Proposte NextN | Prefill token/s | Decode complessivo token/s |
| --- | ---: | ---: |
| Nessuna (AR) | 392,97 | 0,67 |
| 3 | 361,54 | 1,68 |
| 5 | 359,99 | 1,90 |
| 7 | 361,24 | 1,78 |
| 9 | 360,15 | 1,70 |
| 11 | 361,21 | 1,61 |
| 13 | 361,32 | 1,53 |
| 15 | 361,13 | 1,49 |

Le profondità 5..15 accettano le stesse 23 proposte e committano in media
3,875 token per verifica. Aumentare la profondità aumenta quindi il lavoro
scartato nel campione, senza ridurre le otto verifiche target. La proposta
NextN5 costa in media 18,73 ms, la verifica target 1841,09 ms e il rollback
56,76 ms. La cache NextN nel workspace evita i circa 90 ms/proposta del
precedente KV condiviso. Report `qwen27-nextn-v46-depth-sweep-64/report.json`.

### Perché NextN può aggiungere tempo al prefill

La profondità di proposta appartiene al decode. Il prefill DS4 prepara però
anche il KV NextN, usando ogni coppia `(token[p], target_h[p-1])`. Non esegue
il FFN o il classificatore del drafter durante questa preparazione: normalizza
la coppia, proietta EH, K/V e applica il rotary alle chiavi. I pesi e gli
snapshot di verifica occupano già memoria prima che il prefill inizi.

Il percorso storico usa `qwen_graph_mtp_catchup_rows` con una sincronizzazione
GPU e due scritture KV per ogni token, anche dentro un chunk da 1024. Il
timer di prefill applicativo include questo lavoro; non coincide con il solo
tempo target. `DS4_QWEN_PREFILL_PHASES=1` misura separatamente target e
preparazione NextN, aggiungendo fence soltanto nella diagnostica.

La variante `DS4_CUDA_QWEN_MTP_BATCH_CATCHUP=1` conserva i
kernel scalar e la riduzione di ogni proiezione. Accumula le K/V in viste
persistenti del workspace esistente, poi effettua store, rotary e copie RAM
per chunk. Evita una sincronizzazione per token e le allocazioni di viste
temporanee. Il carry shift, mRoPE e l'intero KV sono confrontati con il
percorso seriale originale tramite `tests/qwen_mtp_catchup_probe.c`, anche
attraversando 32K e 64K celle e con embedding vision dentro il chunk.
I sei gate v47 passano: normale e capacità, ogni rollback, cancellazione,
checkpoint e restore. Log `nextn-gates-v47/qualification.json`.

La diagnostica v48 a 65.536 token effettivi separa 167,64 s target e
13,18 s di preparazione seriale NextN da 168,10 s target e 7,95 s di
preparazione per chunk. Riduce la sola preparazione di circa il 40%;
il prefill totale passa da 361,56 a 371,32 token/s (+2,70%). Il decode
NextN5 resta circa 1,90–1,91 token/s, contro 0,67 di AR. Sono campioni
singoli con fence diagnostiche, non una promozione a cinque coppie.

Le continuazioni NextN5 seriale/per chunk coincidono bit per bit prima e
dopo 32 token; anche il controllo AR ripetuto coincide. Il secondo AR
originario aveva differenze massime 0,00717 nei logits di frontiera: quel
report rimane `all_pass=false`, conservato accanto alla ripetizione esatta
`qwen27-nextn-v48-phases-64-on-control2/report.json`. Non sono cambiate
tolleranze o golden. La variante successiva abilita la preparazione per
chunk soltanto nella modalità CUDA `hybrid-attention` con capacità oltre
le celle residenti; `DS4_CUDA_QWEN_MTP_BATCH_CATCHUP=0` mantiene l'override
seriale. Il percorso ordinario conserva il default precedente.

## Riesame Strata del 7 ottobre e prova 128K

Il checkout Strata esaminato è `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`.
Il confronto v49 usa 131072 token effettivamente elaborati, residenti 32768,
NextN5, preparazione per chunk e riuso dello staging target per KV NextN.
Dopo warm-up di 16 token, ripristina la stessa frontiera e genera 32 token.
Token e logits completi di frontiera/fine coincidono bit per bit con AR:
`qwen27-nextn-v49-128/report.json`, `all_pass=true`.

AR misura 224,68 token/s prefill, circa 0,233 token/s decode. NextN5 misura
179,05 prefill, 0,348 token/s complessivi e 0,36 nello steady decode.
Nel campione l'acceptance è 16,7%, con 1,824 token utili per verifica,
circa 5000 ms di verifica e 55,4 ms di rollback. È una diagnostica singola:
non confronta altre profondità e non dimostra una promozione a cinque
coppie. Il contesto 128K passa la correttezza, ma rimane costoso sul PCIe x2.

La variante sperimentale `DS4_CUDA_QWEN_MTP_DECODE_ROWS=1` conserva i
rollback FP32 in un'arena GPU durante il solo decode. Il budget separato
`DS4_CUDA_QWEN_MTP_DECODE_ROWS_MB=0..8192` è 1024 MiB inizialmente; zero
forza il fallback RAM. Prima del tentativo viene restituito lo scratch
grande del prefill. Crescita KV e workspace di verifica restano riservati.
Il tentativo avviene una volta per fase, i cicli riutilizzano arena e view,
un nuovo prefill restituisce l'arena. Recurrent checkpoint di annullamento,
KV logico e formato dei checkpoint non cambiano. I flag Flash non attivano
questo percorso; la variante richiede NextN CUDA e contesto offloaded.

Se entra soltanto una parte delle righe, la v53 conserva un prefisso
nell'arena e la coda nel backing RAM originale. Il kernel mantiene FP32,
ordine e snapshot; il trasferimento D2H riguarda soltanto la coda. Il restore
sceglie D2D o H2D secondo la posizione logica. Il test a 512 MiB forza tre
righe su cinque, con confronto indipendente di ogni rollback e continuazione;
passano anche il budget zero e la cancellazione con payload/RNG completi.
Tutti i 12 gate preliminari v53 passano.

La diagnostica v51 a 64K effettivi ammette tutte le cinque righe, conserva
token e logits completi bit per bit e misura circa 2,246 token/s complessivi
contro 1,916 del controllo NextN5. Il rollback medio passa da 56,07 a
1,26 ms; la verifica da 1824,49 a 1574,58 ms. A 128K la v51 ricade su RAM
per insufficiente spazio per l'arena intera e passa gli stessi controlli
numerici (circa 0,380 token/s). Sono misure diagnostiche singole, non una
promozione della variante. Il test reale 128K del prefisso parziale v53
interrotto durante il decode è conservato ed escluso; viene ripetuto in una
cartella distinta dopo la ripresa, senza cambiare NextN5.

La ripetizione `decode-rows-v53-128-resume1/report.json` passa token e logits
completi bit per bit: 131072 token effettivi più 32, due snapshot su cinque
ammesse in VRAM. Misura 0,39160 token/s complessivi nel decode. Il prefill
è invece 81,15 token/s, sensibilmente peggiore di v51 e del v53 interrotto
sullo stesso binario. La causa rimane da diagnosticare; la modalità continua
a essere sperimentale e la misura singola non promuove l'arena.
