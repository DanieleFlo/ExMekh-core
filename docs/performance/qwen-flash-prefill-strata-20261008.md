# Qwen Flash: nuova verifica del prefill Strata, 8 ottobre 2026

Aggiornamento: la [riproduzione attraverso l'agente e il launcher reale](qwen-flash-agent-launcher-20261008.md)
spiega il prefill da circa 30 TPS osservato dall'utente e confronta blocchi
256/2048/4096 con MTP, vision e capacità 128K. Sono prove distinte dai
benchmark diretti qui sotto.

La v74 corregge la geometria delle proiezioni dense e promuove il profilo
standard del launcher Flash: chunk 2048, ring esperti, KV streaming esplicito,
ranking/grafi CUDA e GPU per prefill/MTP. 272 righe full-vocabulary sono
bit-identiche, passano stato e vision, e cinque coppie reali dall'agente
misurano prefill completo 37,83→140,31 TPS, decode codice 5,06→7,67 e prosa
5,33→8,92. Non servono variabili sperimentali per l'avvio standard; le sezioni
v64–v71 restano la cronologia delle prove precedenti e dei percorsi non promossi.

Il lavoro attivo riguarda Flash. Il 27B 4 BIS è stato ritirato su richiesta
dell'utente; sorgenti, integrazione recuperabile, motivazioni e risultati sono
in [docs/experiments](../experiments/qwen27-4bis-20261008/README.md).
Il normale 27B resta un controllo di regressione.

## Fonte ricontrollata

La testa `main` verificata durante questa sessione è
`6674a0065fb96bacde33e3eb10f91a1df86f95f2`, versione 0.1.40.4.
La copia separata è `performance-results/flash-strata-20261006/strata-source-20261008`;
quella del 7 ottobre rimane immutata. Licenza MIT; implementazione DS4 originale,
ispirata all'ordine delle operazioni, senza importazione di kernel.

Il confronto con `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a` non modifica
`src/prefill/prefill.cpp` o `src/core/expert_cache.cpp`. Le nuove modifiche
al dimensionamento Windows riguardano HIP; quelle ai kernel ristabiliscono
il decode Pascal sm_61. Non sono un nuovo guadagno di prefill per la 3090 sm_86.
Il [prefill fissato](https://github.com/Niko1221/Strata/blob/6674a0065fb96bacde33e3eb10f91a1df86f95f2/src/prefill/prefill.cpp)
resta la fonte per streaming anticipato, ring a byte e scheduling dello staging.

## Liste compatte, v64

L'ibrido può riservare `esperti × token` interi anziché `esperti × token × slot`.
Il router non ripete un esperto nel token; gli input diretti con duplicati
conservano tutti i contributi grazie alla crescita verificata sui conteggi.
Il risparmio a 8192 righe, 512 esperti e dieci slot è 144 MiB.
`DS4_CUDA_QWEN4_COMPACT_LISTS=1` resta sperimentale e disattivato inizialmente.

Cinque coppie A/B alternate, ciascuna con processo nuovo e un warm-up,
4096 token effettivi, capacità 8192, chunk 4096, 16 righe di decode
teacher-forced: tutti i logits di frontiera e decode sono bit-identici.
Riferimento v63 post-archiviazione; variante v64 congelata. Stessi pesi,
quota PCIe 1, percorso ibrido GPU e KV streaming, budget pesi 20 GiB.

| Metrica calda | Riferimento mediano | Variante mediana | Delta |
| --- | ---: | ---: | ---: |
| Prefill token/s | 430,36 | 440,04 | +2,25% |
| Decode totale token/s | 20,68 | 20,75 | +0,34% |

CV del prefill: 5,85% riferimento e 2,53% variante. Il +2,25% supera la
nuova soglia dell'1% richiesta dall'utente: la variante viene conservata
tra gli incrementi utili. La confidenza prestazionale è inferiore a quella
del ring anticipato: tre coppie su cinque hanno direzione positiva e il
riferimento è più rumoroso. Il primo screen singolo +5% non viene usato
come conclusione.
Artefatto: `compact-lists-v64-paired/compact-v64-paired.json` nella directory
dei risultati Flash. Nessuna modifica a quantizzazione o tolleranze.

## Anticipare le copie rispetto al router

Strata, dai chunk di 1024 righe, trasferisce tutti gli esperti non residenti
in ordine fisso prima che il router ne richieda il consumo. Il suo pack
utilizza quasi tutti gli esperti a queste dimensioni. Il ring DS4 già
accoppia gate/up/down, ma iniziava solo dopo il readback del router.

La v65 anticipa le prime due porzioni del ring dopo aver accodato il calcolo
dell'attenzione, prima del router. Riutilizza la stessa allocazione VRAM,
lo staging pinned bounded, il copy stream e gli eventi di consumo; non
aggiunge un secondo ring o una copia completa del modello. Gli eventi
proteggono i buffer precedenti e l'epoch della cache resta pinned fino
al termine dei consumatori. I contributi e il combine conservano il percorso
aritmetico originale. Sono disponibili contatori dei byte anticipati e
di quelli non utilizzati dal router.

Primitive con cache GPU zero/parziale, RAM bounded/zero, IQ2_XXS, Q2_K con
padding, sentinel, esperti non selezionati e upload abbandonato: PASS.
Checkpoint, restore/reuse, snapshot, rewind e MTP reali: PASS, incluse
anticipazioni forzate su chunk piccoli. Screen completo 4K: logits bit-identici.

Il singolo screen v65 aumenta il prefill caldo 509,41→541,26 token/s
(+6,25%), ma peggiora il primo prefill 303,02→261,67 (−13,64%). Dei
10,647 GB anticipati nel processo, 3,050 GB non sono utilizzati. La pipeline
non viene promossa: trasferire indiscriminatamente gli stessi esperti
di Strata non riproduce il working set del GGUF locale.

La v66 usa quindi le scelte osservate nell'ultimo chunk completo come
semplici hint dell'engine. Sul primo chunk freddo non anticipa esperti;
sui successivi anticipa solo quelli precedentemente selezionati, entro
le stesse due porzioni. Un cambio di dominio carica gli esperti mancanti
dopo il router, senza cambiare la scelta o perdere contributi. Gli hint
non entrano nei checkpoint logici e vengono cancellati col modello.

Controlli sperimentali: `DS4_CUDA_QWEN4_PREFILL_AHEAD=1`, policy `stable`
e `DS4_CUDA_QWEN4_PREFILL_RING=1`; soglia iniziale 1024 righe. La soglia
può essere forzata da `DS4_CUDA_QWEN4_PREFILL_AHEAD_MIN` per i gate.
`DS4_CUDA_QWEN4_PREFILL_AHEAD_ALL=1` riproduce l'anticipazione indiscriminata
per confronti. Il percorso ibrido prefill va disattivato quando si misura
il ring; il default di ogni nuova modalità resta disattivato.

### Confronto ripetuto degli hint v66

Cinque coppie alternate a 4096 token effettivi, capacità 8192, chunk 4096,
16 righe teacher-forced e un warm-up per processo. Stesso binario v66
congelato per riferimento e variante; cambia solo `PREFILL_AHEAD=0/1`.
Policy stabile, ring 128 MiB, KV streaming e budget pesi 20 GiB uguali.

| Metrica | Riferimento mediano | Hint mediani | Delta |
| --- | ---: | ---: | ---: |
| Prefill caldo token/s | 499,51 | 549,17 | +9,94% |
| Decode totale token/s | 32,20 | 31,74 | −1,43% |
| Primo prefill token/s | 304,54 | 301,61 | −0,96% |

Tutte e cinque le coppie migliorano il prefill caldo. CV 1,41% e 0,92%.
Dieci confronti di logits completi, prima e dopo decode: bit-identici,
KL zero. Il decode ha CV 3,76% e 3,57%; la variazione mediana negativa
viene registrata, senza presentare un guadagno su quella fase.
Artefatto: `prefill-ahead-v66-paired/ahead-v66-paired.json`.

La v67 mantiene questo percorso e ripristina la compatibilità del tuner
quando le nuove modalità sono disattivate. La chiave effettiva misurata
`6e7349718c8fc081` coincide con quella v57 a contesto 2048/chunk 256;
le preferenze decode, MTP e prefill precedenti rimangono caricabili.

I sette gate v67 passano: primitive ring con RAM bounded/zero, lettura
anticipata su file troncato, checkpoint/restore/MTP con hint, variante
ibrida a liste compatte, regressione statica e normale NextN 27B con
variabili Flash ereditate. Codice, prosa e tool teacher-forced totalizzano
68 righe da 248320 logits ciascuna: bit-identiche al riferimento v63,
KL zero e loss invariata. Include il cambio di working set degli hint;
nessuna soglia numerica viene allentata.

Qualifica vision v67: otto gate complessivi PASS, inclusi payload e
frontiere complete da 248320 logits sia target sia NextN, checkpoint
nidificati, restore e sessioni ricreate. Le due suite HTTP totalizzano
20 casi PASS: immagini singole/multiple, continuation e cache, thinking,
JSON, tool, SSE e API Anthropic. Le copie anticipate sono forzate anche
su chunk piccoli. Artefatto `prefill-ahead-v67-vision/report.json`.

### Soglia prestazionale dell'1%

Su richiesta dell'utente, il tuner usa `--min-gain 0.01` come default,
anziché richiedere oltre il 3%. Il parametro è esplicito e riproducibile.
I report A/B mostrano anche guadagni per coppia, numero di coppie positive
e soglia applicata. Restano i requisiti di numerica e qualità, warm-up,
almeno cinque campioni e registrazione delle regressioni delle altre fasi.
Un guadagno piccolo non viene eliminato automaticamente soltanto perché
la CV del riferimento è più alta; la variabilità è comunque documentata.

Gli incrementi sono misurati anche nel percorso completo: non si sommano
percentuali di contesti, binari o scheduler diversi. In particolare le
liste compatte servono il prefill ibrido, mentre questi hint servono il
ring del prefill GPU. I due risultati non dimostrano un +12,19% cumulativo.

## Contesto effettivo 64K: pipeline completa v67

Due confronti diagnostici separano l'anticipo dal percorso decode qualificato.
Ogni confronto usa lo stesso binario, 65536 token effettivi, capacità 65568,
chunk 4096 e 16 righe teacher-forced. Cambia soltanto `PREFILL_AHEAD=0/1`.
Sono coppie singole a cache fredda, non cinque campioni di promozione.

| Percorso | Prefill riferimento → hint | Delta prefill | Decode totale riferimento → hint | Delta decode |
| --- | ---: | ---: | ---: | ---: |
| GPU, ring stabile | 416,61 → 440,06 | +5,63% | 7,76 → 7,56 | −2,58% |
| Integrato, decode già qualificato | 415,86 → 440,34 | +5,89% | 14,43 → 14,71 | +1,94% |

Il percorso integrato conserva CPU 2 worker, quota PCIe 0,75 per decode,
prefill GPU (`HYBRID_PREFILL=0`), decode graphs, ranking di eviction,
KV decode batch e QSA register keys. Questi controlli sono uguali nei due
lati del confronto: il delta misura l'anticipo del ring, non la loro somma.
Il costo di scelta delle vittime su 16 decode è 48,033 ms nell'integrato,
contro 760,540 ms nel percorso GPU senza ranking: omettere ottimizzazioni
già qualificate falserebbe la valutazione dell'intera pipeline.

I logits completi alle frontiere prima e dopo decode sono bit-identici
in entrambe le coppie. Gli artefatti sono `prefill-ahead-v67-long` e
`prefill-ahead-v67-integrated`. La memoria GPU del confronto GPU lascia
circa 2,4 GiB liberi; nessun secondo ring viene aggiunto.

**Questi valori decode non includono DSpark né NextN/MTP.** Il benchmark
esegue il target su righe teacher-forced per confrontare tutti i logits.
Il numero di token accettati da un drafter e la velocità di generazione
speculativa vanno misurati in un benchmark distinto.

### Contesto effettivo 128K

Stesso confronto integrato, 131072 token effettivi, capacità 131104,
chunk 4096 e 16 righe teacher-forced, senza warm-up. Una coppia diagnostica
a cache fredda, con tutte le preferenze decode uguali e variando soltanto
`PREFILL_AHEAD`:

| Metrica | Riferimento | Hint | Delta |
| --- | ---: | ---: | ---: |
| Prefill token/s | 383,09 | 405,53 | +5,86% |
| Decode totale token/s | 12,57 | 11,64 | −7,40% |

Logits completi prima e dopo decode: bit-identici, KL zero. Memoria GPU
circa 21986 MiB usati / 2341 MiB liberi durante il prefill; PCIe Gen4 x2.
Artefatto `prefill-ahead-v67-128k`, riportato insieme alle altre misure
in `prefill-increments-v67.json`.

Il peggioramento decode supera il 5% in questa coppia. Non basta a stimare
la variabilità su 16 righe, ma va indagato e impedisce di promuovere gli
hint come default universale a 128K. La soglia di guadagno dell'1% non
elimina il controllo delle altre fasi. Non è un risultato con drafter.

## Richiesta DSpark v2 predefinito per Flash

Verifica del catalogo pubblico del 2026-10-08: 389 modelli trovati dalla
ricerca DSpark, nessun checkpoint nominato Qwen3.8 Flash; 824 risultati
dalla ricerca Qwen3.8-Flash. Il checkpoint ufficiale
[RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/blob/main/config.json)
è DSpark v2, ma usa hidden size 5120 e target layer 5, 19, 33, 47, 61
su 64 layer. Flash usa hidden size 2560 e 48 layer principali con quattro
stream HC. Le proiezioni apprese e i layer catturati sono incompatibili;
un flag non rende utilizzabile il drafter del 27B sul target Flash.
Il [catalogo RadixArk DSpark](https://huggingface.co/collections/RadixArk/dspark)
non elenca Flash. La ricerca non dimostra l'assenza di checkpoint privati.

Il codice `src/core/mtp.cpp` del checkout Strata fissato usa il NextN
embedded. DS4 ha già questo percorso per Flash; non viene rinominato DSpark.
Il codice specifico del 27B resta nell'archivio ritirato.

Il catalogo contiene anche
[PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash):
un DFlash da circa 498 milioni di parametri, senza Markov/confidence head,
addestrato sul target NVIDIA NVFP4. La card dichiara vantaggi nel dominio
matematico e regressioni in chat; non è DSpark v2 e non è qualificato sul
GGUF locale. La stessa card menziona una variante DSpark addestrata ma
non pubblicata, con rallentamenti nei test dell'autore. Questi risultati
esterni non sostituiscono benchmark locali e non rendono disponibile
il checkpoint. `tcclaviger/Qwen3.8-Flash-Next-Dflash2` contiene soltanto
una card «Soon» senza configurazione. Nessuno dei due giustifica attivare
DSpark v2 predefinito senza port e qualificazione specifici.

Metadata e ricerche, senza download dei pesi o esecuzione di codice remoto:
`dspark-flash-audit-20261008.json`, `dflash-flash-alternatives-20261008.json`.
L'utente ha scelto il NextN disponibile, chiarendo che la separazione
richiesta è tra prefill e decode. La ricerca della lunghezza migliore si svolgerà dopo gli
incrementi Flash, con baseline target, acceptance, latenza complessiva,
VRAM e distribuzione di sampling coerenti.

Il NextN Flash attuale usa `DS4_QWEN4_MTP_DEPTH=2/3` per finestre di
verifica comprendenti il token target iniziale: sono rispettivamente
uno o due token proposti. La policy automatica alterna queste lunghezze.
`--mtp-draft N` non amplia oggi questa finestra Flash. Per valutare
lunghezze maggiori serve estendere il ciclo Flash, le frontiere ricorrenti,
il catch-up del drafter e i gate di rollback; aumentare soltanto un numero
nel parser darebbe un benchmark ingannevole. L'eventuale default deve
anche preservare il sampling del target tramite la verifica esatta.

### v68: NextN predefinito per Flash CUDA

CLI e server attivano NextN embedded dopo la validazione del target Flash
su CUDA, senza sidecar e senza chiamarlo DSpark. `--no-mtp` disattiva il
default e le altre attivazioni speculative precedenti nella riga di comando;
un `--mtp` successivo resta un override esplicito. Il default abilita la
verifica esatta del sampling. Mantiene la policy adattiva di uno/due draft
greedy; a temperatura positiva il percorso esatto conserva il limite già
qualificato di un draft. La profondità migliore e l'eventuale estensione
della finestra rimangono misure successive, non un valore arbitrario fissato.

L'opzione interna `qwen4_mtp_auto` è impostata da CLI/server; opzioni library
azzerate, bench ed eval mantengono il target ordinario per i gate. Non attiva
NextN per il 27B, modelli senza head, ispezione o backend CPU. Il prefill
conserva il proprio scheduler e i controlli sperimentali separati dal
workload MTP/decode: questa modifica non promuove automaticamente gli hint
a 128K o il percorso CPU prefill non qualificato.

Build CPU e CUDA sm_86 v68 completata. Cinque gate iniziali PASS: suite
server, compatibilità statica, stato Flash tramite il default automatico
(senza flag di attivazione embedded/esatta), checkpoint/restore/snapshot,
rewind e resample a temperatura positiva, limiti MTP con scratch di una/due
righe e fusions disattivate, regressione NextN 27B con variabili Flash
ereditate. I confronti ordinari CLI specificano ora `--no-mtp` per evitare
confronti vacui tra due percorsi speculativi.
Artefatto `default-nextn-v68-gates/qualification.json`.

La successiva qualifica vision v68 passa dieci gate e venti casi API,
inclusi frontiere a vocabolario completo bit-identiche, payload multimodali,
restore/ricreazione e selezione automatica del server senza `--mtp`.
Artefatto `default-nextn-v68-vision/report.json`.

### Confronto per fase e costo del NextN v68

Cinque coppie CLI alternate per dominio, dopo una coppia di warm-up,
64 token generati greedy e stesso scheduler GPU. Il contesto **allocato**
è 8192; i prompt **effettivi** sono brevi (38 token codice, 47 prosa):
questi numeri non sono decode a 8K effettivi. Ogni processo riparte con
engine/cache esperti nuovi, mantenendo calda la cache dei file; nessun profiler.

| Dominio | Decode target TPS | Decode NextN TPS | Delta | Delta prefill |
| --- | ---: | ---: | ---: | ---: |
| Codice | 8,05 | 8,44 | +4,84% | +0,67% |
| Prosa | 11,11 | 9,93 | −10,62% | −0,30% |

Le cinque coppie codice sono tutte positive, le cinque prosa tutte negative;
gli output greedy sono identici. Il prefill non usa il drafter al posto del
target. Il CV decode è 1,22%/0,64% per codice e 2,55%/0,71% per prosa
(target/NextN). Artefatto `default-nextn-v68-gpu-only/report.json`.
La regressione sulla prosa impedisce di qualificare questa variante come
default universale: la richiesta «usa NextN se funziona meglio» si valuta
per la fase di decode, senza dedurre che il beneficio prefill la compensi.

Una quota PCIe unica 0,75 e due worker peggiorava entrambi i domini.
Separare decode (0,75/2), verifica MTP (1/2) e prefill (GPU ordinaria)
ha corretto parte del costo, ma non quello della prosa. Quel confronto è
diagnostico perché parte delle misure si è sovrapposta a una lettura Git
per Windows del checkout. Il confronto GPU riportato sopra è successivo,
senza tale attività, e conferma la regressione. Artefatti
`default-nextn-v68-paired` e `default-nextn-v68-split-phase`.

Due prove diagnostiche per dominio con profondità fissa 3 (due proposte)
o testa draft limitata a 65536 righe non eliminano il costo sulla prosa.
Non sono cinque coppie di qualifica e non promuovono un default.
Il target/verificatore mantiene sempre il vocabolario completo.
Artefatto `scan-nextn-v68/report.json`.

### v69: residenza degli esperti NextN entro la quota esistente

L'ispezione di `src/core/mtp.cpp` al commit Strata
`6674a0065fb96bacde33e3eb10f91a1df86f95f2` conferma il caricamento di tutti
gli esperti del drafter in VRAM. DS4 li trattava invece come quelli del
target. Il controllo sperimentale `DS4_CUDA_QWEN4_MTP_RESIDENT=1` aggiunge
questa riserva per il NextN effettivamente attivo, sottraendola dalla quota
già esistente. Nessun nuovo budget o cambiamento dei pesi. In questo GGUF
la riserva completa è 1,29 GiB; con policy stabile le quote target usano
il resto. Le altre policy conservano i loro criteri. Tutti i selettori di
vittime rispettano i pin del drafter, anche dopo cambio di epoch/restore.
Se la riserva non entra, resta il percorso bounded senza fallire l'inferenza.
La chiave tuner passa a 33 solo quando questa modalità è richiesta sul NextN;
la chiave già qualificata resta invariata con la modalità disattivata.

La prima qualifica ha trovato un errore di restore del checkpoint durante
il prefill: i suggerimenti di residenza GPU/RAM includevano pesi del drafter
precaricati ma senza cronologia del router. La correzione esporta soltanto
suggerimenti che appartengono a proiezioni con una cronologia reale. Conserva
il formato logico legacy e non inventa scelte del router. I pin permanenti
rimangono dell'engine e non vengono serializzati. Il fallimento è conservato
in `resident-nextn-v69-gates`, non cancellato o dichiarato PASS.

Build CPU/CUDA sm_86 corretta e cinque gate PASS in
`resident-nextn-v69-gates-r2/qualification.json`: oracle di replacement su
4096 epoch con pin permanenti, unit server, compatibilità statica, stato
Flash con checkpoint durante il prefill/restore/snapshot/rewind/resample,
27B ordinario con controlli Flash ereditati. La prova con quota esperti/RAM
256 MiB ha esercitato il fallback e passato i gate di stato; è distinta
dalla prova con pin residenti attivi. I/O troncato continua a essere un errore,
non un modo di ammettere triplet incompleti.

Cinque coppie alternate con lo stesso protocollo breve v68, dopo una coppia
di warm-up, target ordinario contro NextN con riserva attiva:

| Dominio | Decode target TPS | NextN residente TPS | Delta | Delta prefill |
| --- | ---: | ---: | ---: | ---: |
| Codice | 8,02 | 8,06 | +0,50% | +0,82% |
| Prosa | 11,05 | 10,08 | −8,78% | +0,15% |

Output greedy identici per tutte le coppie. Il CV decode è 1,45%/2,37%
per codice e 2,06%/2,72% per prosa. Nessun guadagno qualificato dell'1%:
la cache target più piccola può compensare i trasferimenti risparmiati
del drafter. La riserva resta sperimentale, disattivata inizialmente.
Non si deduce un delta diretto v68/v69 da benchmark eseguiti in momenti
diversi. Il caricamento iniziale della riserva ha richiesto circa 0,9 s
nelle prove di stato: è costo di avvio, non incluso nei TPS del prefill.
Il report conserva anche il tempo complessivo di processo, che comprende
caricamento, prompt e generazione. Artefatto
`default-nextn-v69-resident-gpu-only/report.json`.

### v70: decisione corrente per prefill e decode

La richiesta finale è «usa NextN se funziona meglio», separando prefill
e decode. Il prefill mantiene il target e il proprio scheduler; le
preferenze PCIe/worker possono essere diverse per prefill, decode singolo
e verifica MTP. Il default universale NextN v68 non viene promosso dopo
la regressione prosa ripetibile. CLI/server tornano al default ordinario;
`DS4_QWEN4_MTP_AUTO=1` permette di provare l'attivazione automatica Flash
con verifica esatta, senza `--mtp`; `--no-mtp` la disattiva. Il controllo
vale solo per Flash CUDA dopo la validazione del modello. `--mtp` resta
il percorso esplicito esistente. Si valuteranno la lunghezza e la policy
finali dopo gli incrementi, misurando anche avvio/TTFT e sampling positivo,
senza nascondere regressioni del decode dietro guadagni del prefill.

Il MTP embedded fornito da Qwen Flash è la testa NextN: non esistono due
algoritmi separati tra cui scegliere con questi nomi. La richiesta successiva
di conservare il MTP nativo mantiene la sua policy adattiva, senza imporre
la nuova residenza degli esperti. Questa policy adatta uno/due token proposti
in base all'acceptance; non spegne MTP automaticamente in base ai tempi.
La modalità automatica sperimentale e `--mtp` restano disponibili; non si
dichiara un default universalmente più veloce dai risultati codice/prosa.

La qualifica vision v70 passa dieci gate e venti casi API, compresi i logits
completi delle frontiere, bit-identici, checkpoint multimodali, ricreazione
della sessione, attivazione automatica sperimentale e override `--no-mtp`.
Artefatto `resident-nextn-v70-vision/report.json`. Il fallimento di compatibilità
in `resident-nextn-v70-static` deriva dal lock globale del processo DS4: il
gate avviava ispezioni mentre il server vision era attivo. Il risultato
rimane conservato; la ripetizione viene eseguita in serie con i gate modello.

### v71: fase esplicita, compresi i prefill brevi

Il numero di righe non identifica da solo il workload. I prompt e i loro
ultimi chunk da 1–8 token venivano classificati come decode/MTP, applicando
preferenze PCIe/CPU e ammissione della cache non pertinenti. Le entrate di
prefill ora impostano una fase esplicita; NextN e verifica impostano MTP.
Ogni wrapper ripristina la fase precedente anche quando il forward fallisce.
I chiamanti diretti delle primitive conservano l'inferenza precedente dalle
dimensioni. Anche il termine di un prefill breve prepara la cache decode.
Le famiglie del profilo tuner e la sua chiave ordinaria restano invariate;
le scelte del router, i pesi e la profondità nativa MTP non cambiano.

Build CPU/CUDA sm_86 e sette gate PASS in
`workload-phase-v71-gates/qualification.json`: ranking, selezione delle tre
fasi, unit server, compatibilità, stato Flash ordinario, NextN con riserva
e controlli Flash ereditati sul 27B. Checkpoint durante il prefill, restore,
rewind e resampling mantengono le tolleranze precedenti. La compatibilità
è stata ripetuta senza processi modello concorrenti e passa.

Il confronto mirato `workload-phase-v71-quality/report.json` esercita tutti
i prefill da 1–8 righe e tutte le code da 1–8 dopo un chunk da 128: sedici
casi, 68 righe, vocabolario completo 248320, bit-identici al binario congelato
v69; KL e differenza di loss zero. Le fixture separate codice/prosa/tool
in `workload-phase-v71-domains/report.json` aggiungono 68 righe complete,
anch'esse bit-identiche con KL e differenza di loss zero. Questi sono gate
numerici, non una nuova misura di guadagno prestazionale. Gli indici docs
comprendono 58 documenti e i quattro test dei collegamenti passano.
Sorgenti, diff e identità dei binari sono fissati in `candidate-v71`.

## Altri punti Strata da verificare

Strata prepara anche lo storico KV del NextN durante il prefill del target:
`MtpDrafter::prefill` in `src/core/mtp.cpp` e il percorso E-9
`Prefill::draft_kv` in `src/prefill/prefill.cpp`, al commit fissato
`6674a0065fb96bacde33e3eb10f91a1df86f95f2`. E-9 usa gli hidden del target e
il token successivo del prompt, riutilizza scratch bounded e calcola le
proiezioni/KV senza eseguire il MoE o la testa logits del drafter. DS4 avvia
attualmente lo storico NextN dalla frontiera del prompt. Questo pezzo non
è ancora portato: la sola riserva VRAM v69 non completa la pipeline MTP
Strata. L'effetto sull'acceptance e sulla velocità è da misurare, senza
dedurlo dai TPS esterni. Un port deve usare le primitive numeriche DS4,
coprire chunk, ultima riga, vision/mRoPE, cancellazione e checkpoint legacy.
È preparazione ausiliaria per le predizioni successive, quindi può aggiungere
costo al prefill; il target continua a elaborare il prompt.

Il prestito della cache e la scelta congiunta chunk/ring restano distinti
dal ring anticipato bounded. Serve misurare i byte reali dei workspace e
il traffico aggiuntivo causato dalla quota di esperti sacrificata; scegliere
soltanto il chunk più grande che entra in VRAM può peggiorare il risultato.
Non sono dichiarati implementati dal presente incremento.

Gli stager Strata usano attese sleeping su Linux per evitare starvation
dei worker. La coda di lettura/staging DS4 già attende tramite condition
variable e usa un pool bounded. I suoi eventi CUDA possono ancora essere
valutati con blocking sync: il guadagno documentato per 32 spinner Strata
non si applica automaticamente al nostro pool di due worker.

I kernel fusi Strata quantizzano le attivazioni in int8 e non coprono
direttamente il down Q2_K locale. PDL richiede sm_90+, assente sulla 3090.
Questi percorsi non vengono attivati aggirando i vincoli numerici del piano.
Le cinque coppie lunghe 64K/128K non sono sostituite dal test breve 4K.
