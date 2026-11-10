# Flash Metal: campagna prestazionale del 2 novembre 2026

Stato: **`KEEP`, qualifica locale del subset misurato**.
Checkout `1544bb91eafaefefe32862ab408c412f0beed354`
più il piano non committato. M5 Pro 24 GiB, GGUF Q2 originale da
147207127040 byte. Artefatti grezzi: `build/metal-perf-flash-20261102/`.

## Perimetro aggiornato dall'utente

L'utente richiede una campagna rapida soltanto per Flash e soltanto per la
parte Metal ereditata da DS4. Budget iniziale: al massimo due direzioni di
tuning per i kernel ereditati, solo se il profilo le giustifica. Audit e gate
numerici restano obbligatori. Lo streaming bounded aggiunto dal fork segue
il ciclo completo. Nessuna campagna Qwen27/Gemma/image avviata.

`metal/qwen4.metal` è identico a DS4
`0aaea5a238fb41a35106a551e73c8409dfb751ac` (`git diff` vuoto).
`ds4_qwen4_metal_stream.inc` viene invece dai commit del fork
`2b47e9c0b894fb709b8085a27f9fb0a22df53462` e
`fb1fdb3ecfe733bffb3e42749d7bcc5ebf10a833`.

## Baseline B0 e profilo, prima di cambiare codice

Il tentativo 8192/64/cache 2 GiB è rifiutato all'ammissione: necessari
2,89 GiB, ammessi 2,75 GiB dopo riserva host 2 GiB. Nessun guard aggirato.
La baseline separata usa cache **1 GiB**, sempre contesto 8192/chunk 64,
target-only e pesi originali. Il primo run 1 GiB termina dopo il prefill
perché mancava la directory di dump logits; non produce una misura valida.
Il rerun con directory preparata passa e conserva entrambi i tentativi.

Workload B0: prompt raw `tests/test-vectors/flash-0731/prompts/long_code_audit.txt`,
frontiere 64/128, otto token teacher-forced per frontiera, due sweep.
Prefill 9,06–10,43 token/s; decode 0,97–1,08 token/s. Sono misure
direzionali, non una qualifica statistica. A/B dovrà usare identico budget.

Trace separata: `DS4_METAL_ENCODER_TIMELINE`, `DS4_METAL_CB_TIMES` e
`DS4_BENCH_DECODE_TIMES`; 64 input e quattro token teacher-forced.
Prefill wall 6396,3 ms, somma span command buffer GPU 870,9 ms. I quattro
decode hanno wall 953,4–1192,2 ms e span GPU 110,1–114,6 ms.
La strumentazione altera lo scheduling: questi dati servono alla diagnosi,
non alla promozione. Gran parte del tempo è fuori dall'esecuzione GPU;
la trace non attribuisce da sola tutta la quota residua a letture SSD fisiche.

## F-P01: confronto delle fonti prima dell'implementazione

Ipotesi: serializzare tutte le letture degli esperti selezionati sul thread
d'inferenza limita prefill e decode. Una coda C bounded può riempire regioni
disgiunte dello stesso buffer Metal shared in parallelo, preservando byte,
ordine del layout compatto, router e riduzioni. Misurare throughput finale,
non il numero di worker. Nessun nuovo buffer di staging né prefetch predittivo.

Fonti acquisite/rilette il 2 novembre; URL e SHA-256 nel manifest `sources.json`:

| Fonte e revisione | File/simboli letti | Trasferibile / incompatibile |
| --- | --- | --- |
| Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097` | `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`: `Qwen4ExpTextTopKRouter`, `Qwen4ExpTextExperts`, `Qwen4ExpTextSparseMoeBlock`; configurazione e test del modello | Softmax FP32, top-k e pesi di routing, expert gate/up/down e shared expert restano invariati. Non eseguire sul Mac il modello HF pesante; non adottare il suo ordine di riduzione come nuova numerica Metal. |
| Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83` | `metal/src/engine/stream.{cpp,h}`: `io_main`, `submit`, `read_part`, `at_layer`; inventario test stream/arena | Worker e pubblicazione dopo completamento, slot protetti e letture dirette nei buffer. Arena grande, copia contigua `.experts`, predizione, `std::exit` sugli errori e C++ non si trasferiscono. |
| llama.cpp `dde879bc4f954b809892fd0d660515f42536394a` | `src/models/qwen4exp.cpp`: `build_moe_ffn`; `conversion/qwen4exp.py`; `ggml/src/ggml-metal/ggml-metal-ops.cpp`: `ggml_metal_op_mul_mat_id`; `tests/test-backend-ops.cpp`: `test_mul_mat_id` | Copertura effettiva del modello e dispatch indicizzato, ID I32/layout e selezione GEMM per capability. Non è un'implementazione della nostra coda SSD; la parity aritmetica ggml non prova lifetime/errore I/O. |
| Fork, HEAD sopra | `ds4_qwen4_metal_stream.inc`, `ds4_read_queue.h`, `ds4_qwen4_cuda.cuh`, `tests/test_read_queue.c`, `tests/test_qwen4_kernels.c`, `tests/test_flash_metal_stream_graph.c` | Riutilizzare la coda C esistente: payload e buffer del caller, wait/drain e risultati d'errore. Contatori/cache/Metal restano sul thread proprietario; nessun evento CUDA né copia H2D. |

Ricerche interne rilette: piano Flash del 4 ottobre, guida offload prefill,
report `qwen38-flash-prefill-async-20261004.md` e contratto prestazionale.
Il precedente speedup CUDA non è una previsione Metal. Il drift già osservato
aumentando chunk impedisce di cambiare chunk in questo esperimento.

Scelta F-P01: quattro worker persistenti creati solo al primo accesso routed,
al massimo 16 richieste vive, attesa di **tutti** i task prima di pubblicare o
restituire errore. I worker possiedono soltanto descriptor/offset/destinazione
e risultato; aggiornare i byte letti sul thread principale dopo il wait.
Letture dense, Qwen27, backend resident/CUDA e kernel restano sul percorso
esistente. Se non parte la coda, usare letture sincrone corrette.

Alternative rinviate: cache persistente e riuso dei buffer (cambiano residenza
e accounting), prefetch inter-layer (richiede lifetime e dipendenze aggiuntivi),
tuning kernel ereditati (circa 11–14% del wall nella trace; costo host dominante).

Gate previsti: coda host; letture routed scalari/mixed quant e unione token con
cache piccola; errore routed concorrente senza output parziale; parity grafo
resident/streamed; gate indipendente sul Flash completo con logits e replay;
poi A/B fuori dal profiler con warm-up e cinque coppie per promuovere.

Nessuna modifica aritmetica prevista, nessun allentamento delle soglie.

## Implementazione e riesami S2–S4

La modifica è confinata al trasporto routed in
`ds4_qwen4_metal_stream.inc`, con dipendenza Makefile dalla coda C già
esistente. Quattro worker vengono creati al primo caricamento esperti;
nessun worker per un modello che esegue soltanto letture dense. Al massimo
16 job vivono insieme e scrivono nelle posizioni compatte già determinate
dal router. Non sono introdotti buffer di staging o copie dei pesi.

Il caller attende anche tutti i job di un batch fallito, aggiorna i contatori
e pubblica `ready` soltanto al completamento. Il close drena/unisce i worker
prima del rilascio dei buffer e del descriptor. Fallimento di avvio della
coda: fallback sincrono. Kernel, dispatch aritmetico e chunk restano identici.

- **S2:** rilette pubblicazione, pinning e wait di Strata e della coda C a
  confronto col codice scritto. Le fixture verificano oracoli scalari,
  IQ2_XXS/Q2_K e altre coppie quant, 512 esperti, compattazione, split
  dell'unione dei token, eviction, cap ed errori. Il nuovo caso routed con
  quattro letture fallite verifica che l'output sentinella rimanga intatto.
- **S3:** confrontati nuovamente formato e ordinamento del trasporto con le
  fonti fissate. Una sola candidate, quattro worker fissi: nessuno sweep
  arbitrario di kernel/chunk/thread. Due sweep direzionali, poi warm-up e
  cinque coppie alternate fuori dal profiler. Il primo run direzionale della
  candidate coincide in parte con un link della build: è conservato e non
  usato per promuovere. Le cinque coppie finali non hanno build concorrenti.
- **S4:** riesaminati errore, cache/owner e close contro i contratti di
  `stream.cpp`, `ds4_read_queue.h` e i wrapper ggml. Ripetuti grafo
  resident/streamed, modello reale, park/release/reload, replay e MTP.
  Nessun codice esterno importato e nessun nuovo mismatch numerico.

Il probe esistente accetta ora `DS4_PROBE_SSD_CACHE_BYTES` per dichiarare il
budget del singolo run; il default resta 2 GiB. Questo consente il confronto
baseline/candidate a 1 GiB senza cambiare contesto o criteri numerici. Il
baseline probe viene collegato allo stesso harness e all'oggetto Metal B0
conservato prima della modifica.

## Misure promozionali

Mac17,9, Apple M5 Pro, 24 GiB; macOS 26.6.2 (25G83), SDK 26.5,
Apple clang 21.0.0. Nessuna misura contro un altro hardware/runtime.

Una coppia di warm-up scartata come previsto, poi cinque coppie A/B con
ordine alternato. Ogni processo nuovo esegue due frontiere, 64 e 128;
la seconda elabora **64 token nuovi**, non un prompt nuovo di 128 token.
Otto token teacher-forced per frontiera; capacità 8192, chunk 64 e cache
1 GiB identici. La cache applicativa inizia vuota; lo stato freddo dell'SSD
non è garantito e non è stata svuotata la page cache del Mac.

| Fase / occupazione | B0, mediana token/s | Candidate, mediana token/s | Incremento |
| --- | ---: | ---: | ---: |
| Prefill, primi 64 token | 10,19 | 19,28 | **+89,2%** |
| Prefill, 64 nuovi token fino a 128 | 9,17 | 18,16 | **+98,0%** |
| Decode, frontiera 64 | 0,95 | 1,10 | **+15,8%** |
| Decode, frontiera 128 | 0,94 | 1,11 | **+18,1%** |

CV delle metriche sopra: baseline 1,38–3,61%, candidate 0,40–2,57%.
Gli intervalli appaiati stimati restano positivi. Il CSV nativo arrotonda
throughput a due decimali: intervalli e percentuali derivati hanno questa
precisione, insufficiente da sola per futuri incrementi vicini all'1%.
Campioni, minimo/p10/mediana/media/p90/massimo/SD/CV e intervalli stimati sono
in [measurements.json](measurements.json).

La mediana dello steady decode passa 0,98 → 1,14 e 0,98 → 1,15 token/s.
La metrica del primo passo è separata e ha baseline CV >5%: nessuna
promozione autonoma del primo-token e nessuna misura HTTP TTFT rivendicata.

Controllo aggiuntivo: prompt effettivo di **512 token, otto chunk**, zero
output, stesso 8192/64/1 GiB. B0 9,14 token/s, candidate 16,92 token/s,
con tutti i logits finali identici. È una sola coppia: conferma numerica e
indicazione prestazionale, distinta dalle cinque coppie promozionali.

Tutti i dodici processi A/B terminano normalmente, pressione host normale,
nessuna crescita dello swap e riserva host conservata. Picchi di footprint
campionati nelle cinque misure: B0 1,06–1,41 GiB, candidate 1,34–1,53 GiB;
non si deduce memoria uguale dalla sola uguaglianza dei cap. Lo storage
pesi posseduto nel gate target-only è identico: 688578560 byte.

## Correttezza e regressioni

| Gate | Risultato |
| --- | --- |
| Coda host: concorrenza, limite, error payload, drain e restart | PASS |
| Primitive streamed, selezione scalare/mixed quant/cap/union e nuovo errore concorrente | PASS |
| Grafo ridotto resident/SSD, logits, stato GDN/KV/PLE e guasto n-gram | PASS bit-exact |
| Modello completo, input indipendente dal benchmark: prefill + otto posizioni | **9 vettori full-vocabulary bit-exact** con B0 |
| Stato prefill serializzato, prefix no-op, checkpoint, park/release/reload e continuazione | PASS, stato e logits bit-exact |
| Byte logici letti nel gate target-only completo | Identici, 48919904256 byte; non sono una misura di I/O SSD fisico |
| Profilo standard target-only 8192/64/cache 2 GiB | PASS; stessi nove vettori del gate 1 GiB |
| MTP reale, cache 1 GiB, due replay da otto token | PASS; stessi token, delta logits massimo **0** |
| Fixture fusioni decode del percorso residente e lifecycle Flash | PASS |
| Frontiere performance 64/128 e controllo 512 | Full-vocabulary identico alla baseline |
| Test host del qualificatore | 25 PASS; un primo run sandbox aveva 24 PASS e un errore di bind localhost, ripetuto fuori sandbox |
| Build CLI/server/bench/eval/suite e fixture pertinenti | PASS; due warning preesistenti di parametri inutilizzati nel core incluso |

Il primo build del probe ha rilevato l'include `errno.h` mancante nella nuova
lettura del budget: corretto e ricostruito prima dei gate. Nessun test fallito
è cancellato o trasformato in un PASS. Il gate full-model è indipendente dal
workload prestazionale; non è una nuova inferenza HF completa del 177B.

## Esito e lavoro residuo

**KEEP per Flash Next Q2 streamed sul subset M5 Pro verificato.** Le letture
parallele migliorano entrambe le fasi misurate; non serve un dispatch
prefill-only. Una sola candidate di trasporto, **zero tentativi di tuning
dei kernel ereditati**: il profilo non ne giustificava il costo in questa
campagna rapida. Non sono stati cambiati i default del launcher/cache.

La trace finale resta diagnostica: prefill 3320,2 ms wall, span GPU 597,4 ms;
decode 939,2–1050,7 ms wall, span GPU 117,7–137,1 ms. Lo scheduling e le
condizioni delle pagine influiscono sui tempi GPU: non si attribuisce uno
speedup dei kernel al confronto fra queste trace. Il tempo fuori GPU resta
dominante anche dopo la modifica.

Backlog residuo: riuso dei buffer/residenza dei pesi densi e della head,
costo delle barriere/readback, cache e possibile overlap inter-layer. Sono
nuovi interventi: prima di implementarli riaprire l'audit specifico e misurare
attese/allocazioni/byte per fase; questa prova non certifica quelle ipotesi.
Un cambio di budget cache va confrontato esplicitamente, senza attribuire
al kernel un guadagno dovuto a più memoria.

Non misurati qui: throughput vicino a 8k realmente occupati, cache grandi,
MTP performance, altre quant/hardware, HTTP/ExMekh e suite live estesa.
CUDA/ROCm/distribuito e gli altri modelli non vengono qualificati da questo
intervento; i loro sorgenti di produzione non sono cambiati. T07/F08 e le
qualifiche estese della roadmap originale mantengono i rispettivi stati.

Gli [artefatti compatti](experiment.json) conservano sorgenti, comandi,
hash, gate e limiti. Trace, binari B0/candidate, script riproducibili, dump e
report completi rimangono nella directory grezza ignorata sopra. Nessun commit
è stato eseguito durante la campagna di misura; il riesame successivo separa
implementazione, refactoring e documentazione in commit distinti.

## Riesame Clean Code successivo

Il [riesame del lettore e della cache](../../../architecture/metal-flash-stream-clean-code-review-20261102.md)
separa la lettura host in `ds4_file_reader.h`: il lettore possiede descriptor,
pool lazy, completamento, contatori e risultati d'errore; la cache Metal
possiede binding, routing e fence GPU. EOF prematuro viene riportato come EIO,
con errori pubblicati sul thread chiamante. I worker non stampano diagnostica.
I nuovi test host coprono anche il fallimento parziale dell'avvio dei worker.

I [risultati del riesame](clean-code-review.json) conservano hash e gate
successivi. Le misure delle cinque coppie sopra appartengono al codice
pre-refactoring e non vengono riscritte come nuove misure. Il riesame non
cambia kernel, layout o criteri numerici; i suoi esiti restano separati.

Il 2 novembre 2026 l'utente conferma il miglioramento nell'uso e segnala che
durante le misure post-refactoring stava svolgendo altre attività sul PC.
Accetta quindi il percorso ottimizzato come attivo: le letture parallele sono
già il percorso predefinito dello streaming Flash Metal, senza flag aggiuntivi.
`make -j2 all test-file-reader test-read-queue` conferma binari aggiornati e
test host PASS. Questa accettazione operativa non cambia i campioni o il
verdetto statistico del confronto post-refactoring; non si ripetono benchmark
durante l'uso concorrente del PC.
