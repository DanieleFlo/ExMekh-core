# Seconda ispezione delle pipeline Strata

## Decode: nuova ispezione del 6 ottobre

Su richiesta dell'utente il prefill qualificato diventa il punto di partenza;
il prossimo confronto si concentra sul decode e sulla verifica MTP. Nel
checkout Strata `1735d6471df29b42c26170efaac1f1446a58640f` sono stati riletti
`session.cpp`, `verify.cpp` e `pool.cpp`, non soltanto i README.

| Passaggio effettivo in Strata | Riscontro e intervento in DS4 |
| --- | --- |
| `session_loop`: due grafi per layer, GPU residenti avviati prima del pool CPU | L'ordine GPU/CPU esisteva già. Mancava invece l'uso dei grafi sul percorso Flash. Aggiunte due isole opzionali con `DS4_CUDA_QWEN4_DECODE_GRAPHS=1`: mixer e GDN/proiezioni dell'attenzione; combine e mixer FFN fino al router. |
| `session_capture_token`: grafo del token intero, pianificazione residenti sul device, doorbell in RAM mappata e copia con kernel | Il nuovo port cattura soltanto operazioni indipendenti dalle posizioni. Append KV, mRoPE, QSA, lettura del router e placement restano espliciti. Il grafo completo con doorbell non è implementato né dichiarato equivalente sotto WSL. |
| `verify.cpp`: residenti pianificati sul device e bypass dei doorbell quando tutto è residente | DS4 continua a leggere gli ID per alimentare la policy recente, il profilo e il placement dinamico. Il bypass sul device non è ancora implementato. Strata stessa lascia disattivato per default il bypass per layer, dichiarato neutro su alcune macchine. |
| `verify.cpp`: gruppi A/B della verifica, mixer del gruppo B sovrapposto agli esperti A | Non ancora portato. La deduplicazione dei pesi tra righe MTP esiste; non equivale a questo scheduler tra gruppi. |
| `verify.cpp`: stream separato dell'esperto condiviso, fork dopo il router; fusioni gate/up e combine | DS4 fonde già gate/up e la riduzione del gate condiviso. Lo stream separato dell'esperto condiviso non è ancora portato. |
| `pool.cpp`/`session.cpp`: worker su core distinti, core riservato al coordinatore, affinity mantenuta nella sessione | WSL espone 8 core/16 thread. Il pool DS4 persiste ma non imposta ancora questa affinity. |
| I grafi riducono chiamate al driver sul cammino critico | Oltre ai grafi, il sorgente successivo alla v25 raggruppa K/V e ID/conteggi QSA: un evento di pubblicazione per coppia, staging bounded condiviso, drain anche in caso di errore. |

I grafi usano l'infrastruttura DS4 esistente, conservano kernel e ordine
aritmetico e distinguono sessione, buffer residui, numero di righe, snapshot
MTP e modalità di verifica nella chiave. Il replay ripete anche gli scambi
dei buffer effettuati dal percorso eager. Prima di liberare una sessione si
attendono i consumatori e si invalidano i grafi. Una cattura fallita ripristina
gli scambi host e riesegue il segmento eager senza perdere il forward.
Il flag è inizialmente spento e non attiva il percorso Flash sul 27B.

La prova v25 a 256 token effettivi più 64 righe di decode ha catturato tutti
i 96 grafi, con warm-up e replay completati. Cinque A/B alternati rispetto
alla v24 ordinaria danno 35,09 → 37,83 TPS di decode steady (+7,81%), con
CV 2,52%/1,20% e tutti i logits completi bit-identici. Prefill 545,86 →
545,37 TPS (−0,09%). I dati sono `decode-graphs-v25-paired/graphs-paired.json`.
Nella v26 passano stato, checkpoint, rewind, MTP e KV con grafi; 68 righe
teacher-forced sono bit-identiche alla baseline iniziale, KL e delta loss
zero. Il caso con budget KV 1024 è stato rifiutato prima dell'inferenza dal
controllo minimo di 2052 celle; ripetuto con 4096, passa tutti i controlli.
La v26 a 65536 token effettivi, chunk 1024 e otto righe teacher-forced
completa il confronto diagnostico con la baseline iniziale compatibile:
prefill 201,78 → 235,76 TPS (+16,84%), decode steady 12,62 → 12,11 TPS
(−4,04%). I due logits completi sono bit-identici. La baseline è una misura
precedente riutilizzata, non una nuova serie di cinque A/B; gli otto token
comprendono anche la preparazione iniziale dei grafi. Il divario di decode
a 64K resta da risolvere. Dati: `kv-decode-v26-64/report.json`.
Non sono state introdotte la quantizzazione Q8
delle attivazioni native di Strata né letture GPU del KV in RAM mappata.

Il prefill misto CPU della v24 falliva 133 assert di stato. La v26 rende
coerente l'ordine FP32 di hit GPU e miss CPU, e supera i medesimi assert.
La successiva verifica di qualità rileva però KL massimo codice/prosa/tool
0,00748 / 0,02350 / 0,20535 e un argmax tool differente su 18 righe.
Questa variante non supera lo screen numerico dell'autotuner e resta
esclusa dalle preferenze automatiche. Il percorso GPU già qualificato
conserva i GEMM originali. I risultati migliori di loss su quelle tre
fixture non bastano a dichiarare equivalente una distribuzione differente.

### Selezione QSA e pubblicazione del KV nel decode

Riletto anche `src/kernels/cuda/qsa_select.cu` di Strata: il kernel
`block_topk_reg_kernel` conserva le chiavi nei registri e divide gli
istogrammi radix per warp. DS4 aveva già quattro passaggi radix esatti,
ma rileggeva i punteggi a ogni passaggio e usava un unico istogramma.
Il nuovo kernel DS4 conserva l'ordine originale «maggiore della soglia,
poi pari alla soglia», compresi i pareggi a zero e gli indici futuri;
non cambia score, selezione o arrotondamenti dell'attenzione.
SHA256 del file Strata:
`997023ff9c367cc5d43936371c363700313dbbeddb4a17ae0c52391bf7e5146f`.

Il probe CUDA con 21 campioni da 100 lanci, 20 warm-up, 24 casi e confronto
di tutti gli ID ordinati misura, per una query con score non uniformi:

| Blocchi | Contesto corrispondente | Radix originale µs | Chiavi nei registri, 1024 thread µs |
| ---: | ---: | ---: | ---: |
| 2048 | 8K | 22,323 | 29,191 |
| 16384 | 64K | 68,966 | 32,778 |
| 32768 | 128K | 139,725 | 47,114 |

Il probe è una misura di primitiva, non del token intero. Il dispatch
sperimentale è per una GPU, 1..8 righe e 8192..33792 blocchi; usa il kernel
precedente fuori intervallo. I prefills restano sul percorso esistente.
Flag: `DS4_CUDA_QWEN4_QSA_REGISTER_KEYS=1`. Dati e sorgente del probe:
`qsa-probe-v28/measurements.json` e `tools/qwen_qsa_probe.py`.

`DS4_CUDA_QWEN4_KV_DECODE_BATCH=1` tiene inoltre aperto il batch D2H delle
nuove righe K/V fino alla pubblicazione degli indici QSA dello stesso layer,
riducendo due attese host a una dopo lo spill. Usa lo stesso staging pinned
bounded, senza RAM mappata. Prima di letture logiche, rollback, nuova append,
prefetch e release si drenano i trasferimenti; i test coprono anche append
senza un'attenzione successiva. La v28 supera primitive default e batching,
stato CPU, stato CPU/MTP e stato con grafi/KV/batching. Il confronto end-to-end
delle ultime due ottimizzazioni resta da completare.

L'A/B v29 che isola soltanto i grafi, a 256 token effettivi più 64 righe di
decode e con policy/cache uguali sui due lati, misura 39,03 → 38,68 TPS
(−0,90%, CV 0,60%/1,05%). Prefill −0,39%; tutti i logits completi sono
bit-identici. Questo risultato distingue l'effetto dei soli grafi dal
precedente incremento combinato grafi/scheduler. Dati:
`decode-islands-v29-isolated/graphs-paired.json`.

Il successivo confronto v29 con la baseline iniziale misura decode
34,94 → 37,58 TPS (+7,56%), CV 2,75%/2,32%, e prefill caldo
546,86 → 474,73 TPS (−13,19%) per quel prompt breve con policy `stable`.
Il percorso resta sperimentale per quel workload: il prefill lungo migliore
non elimina questa regressione a cache calda. Cinque A/B alternati, due
warm-up per processo, logits completi bit-identici:
`decode-v29-initial-paired/final-paired.json`.

La v30 separa stream di cattura e di esecuzione. Le isole
Flash possono essere lanciate sulla coda ordinaria del forward, conservando
le dipendenze DMA esplicite, invece di cambiare stream a ogni replay. Il
flag `GRAPH_DEFAULT_STREAM=0` permette il confronto con la v29. CUDA usa lo
stream di lancio per l'ordinamento del grafo:
[documentazione NVIDIA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
La selezione sperimentale evita inoltre il kernel tile-max, il cui output
non viene letto dai selettori CUDA; l'API delle primitive rimane invariata.
Il mirror RAM del KV ancora residente registra il proprio evento subito
dopo l'append, poi viene pubblicato dopo l'avvio dell'attenzione: la sua
pubblicazione non deve attendere i kernel MoE successivi.

Il nuovo A/B isolato v30 usa policy recente e identica configurazione
KV/scheduler/QSA nei due percorsi; cambia soltanto l'abilitazione dei grafi.
Cinque coppie alternate, due warm-up, frontiera 256 e 64 righe teacher-forced:
decode 38,52 → 39,97 TPS (+3,76%), CV 2,86%/1,17%, prefill 538,68 →
533,51 TPS (−0,96%). Tutti i cinque campioni candidati superano il massimo
baseline; dieci confronti di tutti i logits sono bit-identici. Dati:
`decode-islands-v30-corrected/graphs-paired.json`. Il precedente tentativo
`decode-islands-v30-isolated` è stato interrotto perché due nomi di override
erano errati e ignorati: non è una qualifica. Il driver ora rifiuta variabili
Flash non riconosciute dai sorgenti prima di avviare il modello.

SHA256 dei file riletti: `session.cpp`
`b1479da93e913a0c944b1c61bb1fd623935cf68d7d02ae9ee1e32d7bb567625f`,
`verify.cpp` `938579652a136f14d4a01dc66da0035fca93863c66a89f01348b9bbf4a30c364`,
`pool.cpp` `c91ba11d4ee91adffe07387506e44825ef89839a8112c7802aba2959aedbef54`,
licenza MIT `949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0`.

Il 6 ottobre 2026 la repository è stata clonata in
`/tmp/ds4-strata-audit-20261006`. È stato scaricato e verificato il commit
`1735d6471df29b42c26170efaac1f1446a58640f`, licenza MIT. La testa remota
`82f46a8c8f475f001ad76d92f58f4a4f8ffb0253` non modifica i file di pipeline
esaminati rispetto a quel commit. Le implementazioni DS4 sono originali;
Strata serve da riferimento architetturale.

La lettura diretta usa
[`src/prefill/prefill.cpp`](https://github.com/Niko1221/Strata/blob/1735d6471df29b42c26170efaac1f1446a58640f/src/prefill/prefill.cpp)
(SHA256 `bd0dd999467168a9850d2d3fe5df7e4fd43a8b7e7815774bf60eb14a74b6b504`)
e
[`include/strata/core/dma_batch.hpp`](https://github.com/Niko1221/Strata/blob/1735d6471df29b42c26170efaac1f1446a58640f/include/strata/core/dma_batch.hpp)
(SHA256 `9b485734a866bee6ef0c927b838905ab7770cfdc1f7a7b19b97c4dea4fbbb07a`).
Il prefetch KV è opzionale in Strata e riusa un solo pool, protetto dagli
eventi `kv_released` e `kv_ready`; DS4 riproduce il riuso con DMA esplicito.

| Ottimizzazione osservata nel codice Strata | Situazione nel port DS4 |
| --- | --- |
| `session.cpp`: avvio dei residenti GPU prima del pool CPU, combine dello stesso layer | Scheduler presente; CPU con attivazioni FP32 e pesi originali. I gate stretti CPU e MTP sono superati nella v21; SiLU originale GPU tra le proiezioni CPU. |
| `verify.cpp`: quantizzazione e deduplicazione delle righe, separazione residenti/PCIe/CPU | Placement completo e deduplicazione dei pesi presenti. Quantizzazione delle attivazioni esclusa dai vincoli del port. |
| `expert_cache.cpp`: quote per layer, esperti interi, nessuna eviction durante il riempimento | Mancava: aggiunta policy sperimentale `EXPERT_POLICY=stable`, triplette scelte insieme e quote per byte nel prefill. Decode/MTP conservano l'adattamento recente. |
| `expert_source.cpp`: arena del complemento della VRAM, exchange esplicito del complemento | Il vecchio ranking RAM favoriva il complemento, ma la residenza GPU variabile causava ricircolo. La policy stabile mantiene l'ownership; la RAM resta bounded e il disco gestisce la capacità insufficiente. |
| `prefill.cpp`: residenti più ring di streaming, consumo protetto da eventi | Gruppi GPU doppi ed eventi presenti. Individuato anche il limite arbitrario di 256 righe nello scheduler completo: esteso a 8192 nel sorgente seguente alla v23, con gate e scelta automatica distinti per prefill e decode. |
| `prefill.cpp`: `Stager` copia RAM ordinaria in pinned prima dei lanci; ring tarato in byte | Aggiunto stager persistente bounded per il prefill, con worker solo CPU e drain prima del riuso delle sorgenti. |
| `prefill.cpp`: PLE del chunk seguente letta durante il corrente; primo gather accanto al layer 0 | Aggiunte `PLE_OVERLAP` e `PLE_PREFETCH`: gather dopo il layer 0 e lettura del chunk seguente con worker e buffer persistenti. La seconda modifica deve ancora terminare la qualificazione. |
| `prefill.cpp`: MMQ/fused native, chunk da 1024 a 8192 e prestito temporaneo di VRAM | Non estratto integralmente: MMQ/fused quantizzano le attivazioni, mentre DS4 conserva il percorso locale. Un confronto con chunk diversi richiede una baseline alla stessa dimensione. |
| `prefill.cpp`: staging KV di un layer intero e prefetch del successivo su stream separato | Aggiunti staging e prefetch del prefisso storico del layer successivo, bounded a 262144 celle. Un solo workspace come Strata, secondo buffer opzionale. Decode conserva la cache QSA; la nuova sovrapposizione deve ancora terminare la qualificazione. |
| `dma_batch.hpp`: batch di copie indipendenti su CUDA 13+, fallback a copie individuali | CUDA locale 12.4: aggiunto accorpamento delle destinazioni contigue, mantenendo l'ordine H2D/D2H e il pool pinned condiviso. |
| `kv_stream.cu`: mapped RAM letta da kernel, slot protetti, eviction | Protezione e mappe presenti; mapped RAM sostituita da DMA esplicito per WSL. |

## Diagnosi a 64K

La v12 conserva logits bit-identici al percorso iniziale, ma scende da
126,85 a 82,00 token/s di prefill e da 19,39 a 11,62 token/s di decode.
La cache riporta 1.380.207 demozioni GPU→RAM e 1.384.614 eviction RAM.
Il traffico KV misurato non basta a spiegare l'intero delta temporale.
Il confronto con Strata indica un'omissione concreta: una preferenza RAM
per il complemento non equivale a un placement GPU stabile.

La v13 introduce residenza stabile con lo stesso budget GPU/RAM, senza
cambiare GGUF, PLE BF16, attivazioni, contesto, chunk o kernel. Nei gate di
checkpoint/restore/rewind/MTP produce zero eviction GPU e zero demozioni,
con circa 22,1 GiB di arena RAM. Il confronto end-to-end a 64K è in corso;
128K verrà ripetuto solo dopo aver corretto e verificato 64K.
Il primo confronto v13 termina in 554,9 s: prefill 119,17 token/s e decode
10,42 token/s, con logits di frontiera e dopo otto token ancora bit-identici.
Il calo da 804,3 s conferma l'effetto delle demozioni, ma il decode perde
troppo con il ranking fisso del primo chunk. La v14 limita quindi il
placement stabile al prefill, conserva l'adattamento recente durante il
decode e aggiunge staging KV per layer e sovrapposizione PLE.
La v14 termina in 560,5 s (118,09 token/s di prefill; 7,50 di decode), ancora
bit-identica. La transizione alla policy recente avviene troppo tardi: i miss
vengono promossi durante gli otto token misurati, causando 1218 demozioni.
La correzione seguente prepara il ranking dall'ultimo window del prompt al
termine del prefill, attendendo prima tutti i consumatori. Il tempo e i byte
di questa preparazione appartengono al prefill/TTFT.

Le quote sono inoltre limitate dal budget fisico dell'allocatore, includendo
pesi densi ancora da caricare, riserva di 2 GiB e 256 MiB per scratch lazy.
Usare soltanto il limite logico da 20 GiB produceva residenza parziale negli
ultimi layer. L'accorpamento DMA rispetta il confine della singola
allocazione CUDA: indirizzi adiacenti di due slab non possono essere trattati
come un'unica destinazione. Il gate di stato sul modello reale ha rilevato
questo caso; il test sintetico iniziale non attraversava slab distinti.

## Controlli della nuova policy

La nuova ispezione ha inoltre rilevato un errore di integrazione: il riordino
`prefill_finish`, previsto una volta a fine prompt, era chiamato a ogni
forward di chunk con logits di frontiera. La v18 a 64K ha perciò eseguito
989381 eviction GPU e 361811 demozioni, impiegando 650,4 s. La v20/v21
sposta la preparazione del ranking fuori dal forward e al termine del prompt
nelle due entry point CLI e sessione. Il costo resta incluso nel prefill.
Il confronto v21 a chunk 256 termina in 589,1 s, 112,16 token/s di prefill
e 12,71 di decode: logits completi bit-identici, ma ancora regressione
rispetto alla baseline a chunk 256. La baseline con lo stesso 64K e chunk
1024 termina in 331,8 s, 201,78 token/s di prefill e 12,62 di decode.
Il candidato v21 con lo stesso chunk 1024 termina in 286,0 s, 233,15 token/s
di prefill e 11,85 di decode: +15,55% nel prefill e −6,10% nel decode, con
logits completi bit-identici. È un singolo campione diagnostico; non
costituisce promozione e il peggioramento nel decode va risolto o mantenuto
esplicitamente sperimentale. Una differenza di chunk non viene conteggiata
come guadagno del port. Dati: `kv-stable-v21-chunk1024/report.json`.

La v23 aggiunge il prefetch PLE del chunk seguente e KV del layer seguente.
Otto gate selezionati superati, inclusi KV con due workspace, stato Flash
stabile, CPU ordinaria/MTP e capacità 27B. Le prove lunghe devono ancora
qualificare queste sovrapposizioni. Nel sorgente successivo è corretta anche
la transizione spill → reset: uno storico vuoto torna agli slot identità,
evitando staging dell'intero prefisso nel prompt nuovo. Un test numerico
aggiunge reset, nuovi append e secondo spill dopo il rewind precedente.

`DS4_CUDA_QWEN4_EXPERT_POLICY=stable` seleziona una quota uguale di esperti
per layer, ponderata dai byte delle tre proiezioni. Il primo chunk completo
del router determina il ranking; un profilo compatibile può inizializzarlo.
Le proiezioni ammesse sono coerenti; durante il prefill i miss restano nello
staging e non espellono residenti in flight. Decode e verifica fino a otto
righe applicano la policy recente. Il piano appartiene all'engine; checkpoint e
sessioni mantengono lo stato logico esistente. La policy recente e LRU
rimangono disponibili per confronti e cambi di dominio.

`DS4_CUDA_QWEN4_PLE_OVERLAP=1` anticipa l'avvio GPU rispetto alla lettura
PLE del chunk. Il default resta spento sul percorso ordinario ed è acceso
con la policy stabile; `=0` consente un'ablation. Il codice mantiene gli
stessi hash EOS, snapshot MTP e conversioni BF16→FP32.

`DS4_CUDA_QWEN4_KV_PREFILL_STAGE=1` copia un layer KV in un workspace
condiviso e riusa l'attention originale sul batch completo. Il workspace è
contabilizzato separatamente, cresce alla creazione delle sessioni, è
bounded a 262144 celle e conserva il fallback per pagine se non disponibile.
`=0` consente di confrontare il percorso a selezioni bounded. Non cambia
QSA, precisione, causalità o formato dei checkpoint.

`DS4_CUDA_QWEN4_STAGING_WORKERS=0..4` controlla le copie RAM→pinned nel
prefill (default due worker con policy stabile). Le proiezioni con
destinazioni contigue diventano un solo upload DMA; una demozione successiva
forza prima la pubblicazione degli upload precedenti. I worker non chiamano
CUDA e tutti i task terminano prima del rilascio di RAM, staging o stream.

### Diagnosi del decode dopo 64K

L'ispezione aggiuntiva di `ExpertCache::admit` conferma che Strata conserva
gli slot residenti senza eviction per token, anche con quote per layer.
DS4 conserva questa scelta durante il prefill stabile e applica la policy
recente in decode, come richiesto dal piano. Questa differenza può aggiungere
trasferimenti e lavoro host; non è una differenza dei pesi o dell'aritmetica.

Tre prove sullo stesso binario v30, 65536 token effettivi e 64 righe decode,
ispezionano le API host solo nella generazione. Con 32768 celle KV residenti
il decode misura 10,23 TPS; con 69632, sufficienti anche per le nuove righe,
10,25 TPS. La scansione lineare delle vittime misura 9,63 TPS. Sono campioni
diagnostici con interposizione, non misure di promozione: non giustificano
né aumentare la residenza KV né sostituire gli heap. Nessuna lettura `pread`
avviene nel tratto decode misurato; il costo resta in memoria e nel forward.
Dati: `decode-host-v30-64-res32768`, `decode-host-v30-64-res69632`,
`decode-host-v30-64-res32768-linear`, ciascuno con `host-probe.json`.

La diagnosi individua un errore di integrazione: lo staging usava `route_T`
del router GEMM nativo. Un prefill da 1024 righe lasciava questo valore vivo
anche quando il decode ibrido non ricostruiva quelle liste. Copie piccole
venivano quindi mandate ai worker CPU. La v31 usa esplicitamente le righe
dell'operazione corrente per MoE e KV; il test prepara liste da 37 righe
prima di ciascuna quota decode/MTP e verifica che i job worker restino zero.
I nuovi contatori separano le attese dello staging e delle vittime per
richiesta e nella sola finestra decode. La qualifica della correzione resta
distinta dalle misure v30 qui riportate.

La prova v31 del 7 ottobre conferma la correttezza ma non recupera il decode:
225,53 TPS di prefill e 9,92 TPS steady a 65536 token effettivi più 64 righe.
Nella sola generazione, `victim_select_ms=2722,619` su 6,441 s di wall time;
lo staging attende 0,437 ms e non avvia job CPU. La sonda registra zero
`pread` e soltanto nove major fault: la ricostruzione dei ranking in RAM,
non una lettura SSD ordinaria, domina questo campione. L'assenza di
`pread` da sola non avrebbe escluso i fault di pagine mappate.

La v32 conserva la policy e sostituisce la ricostruzione degli heap con
alberi persistenti per capacità. Aggiorna soltanto frequenze modificate,
slot toccati e cambi di residenza; riabilita i pin della precedente epoca.
La RAM mantiene due minimi, preferendo esattamente gli offset presenti in
VRAM. Il conteggio della memoria include gli indici e il tempo delle
vittime comprende anche gli aggiornamenti degli alberi. Il prefill usa
ancora gli heap precedenti. Un oracle indipendente confronta 4096 epoche
con cambi di finestra, pin, crescita degli slot e residenza parziale;
il gate CUDA sotto pressione confronta inoltre ogni vittima con la scansione
lineare. Entrambi passano, insieme ai 14 gate di stato/numerica della v32.
Le misure di prestazioni v32 sono separate dalle prove v31.
