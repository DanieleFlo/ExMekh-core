# Q-P03: letture dense suddivise nella coda bounded

S1 prima del codice, 2 novembre. B0/Bi invariata `089255a1d3b6162e59c61a6bede9ece5e672330f`:
Q-P02 è stato ritirato. Nessuna modifica numerica o aumento della cache.

## Attribuzione nuova e ipotesi

Sampling CPU separato, `sample` 10 s/1 ms dopo sette secondi di un benchmark
64+4: sul thread di inferenza **4844/6447 campioni (~75,1%) sono in pread**,
circa 1448/6447 nella fence di layer; allocazioni marginali nel campione.
Non sommare i thread di servizio al wall né interpretare i campioni come
byte SSD fisici. Trace precedente: decode GPU 16–17% del wall.
La diagnosi giustifica intervenire sulle letture prima di buffer pool o kernel.

Ipotesi: una lettura grande sincrona in parti da 8 MiB sottoutilizza le
richieste concorrenti. Suddividere la stessa regione contigua in un pezzo per
worker nella coda esistente, attenderli tutti e pubblicare gli stessi byte
può ridurre la latenza. Il throughput dello storage è un limite: se non migliora
il tempo richiesta, ritirare la modifica. Nessun nuovo buffer o prefetch.

## Riesame specifico delle fonti

Pin completi/URL/hash in [sources.json](sources.json), rilette il 2 novembre.
Le comparazioni esterne sono statiche, nessun confronto prestazionale eseguito.

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `models/qwen3_5/modeling_qwen3_5.py`: proiezioni, residui, GDN FP32, cast
  norm e softmax; confermati contro il config Qwen3.8 fissato nel Q-P02.
  Trasporto dei byte fuori dal grafo, nessuna operazione cambia ordine o dtype.
  Anche il Flash condiviso conserva aritmetica e routing dell’audit precedente.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp`: `read_part` suddivide una slice con
  `STRATA_METAL_READ_SPLIT`; `submit`/`io_main` contano i pezzi e pubblicano
  solo dopo l’ultimo. `load_dense` usa quattro lettori, ma per residenza intera.
  `stream.h` e `arena_budget_test.cpp` fissano slot e budget, non certificano
  il nostro errore I/O. Arena grande, file esperti contiguo, F_NOCACHE,
  predizione e abort su errore incompatibili; non adottati. Il percorso
  Qwen Strata al pin è CUDA qwen35moe, non copertura completa del 27B Metal.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/llama-mmap.cpp:read_raw_unsafe,read_raw,llama_mmap::impl`: retry EINTR,
  short read, mapping e caricamento; `ggml-metal.cpp` possiede buffer shared e
  view no-copy; `ggml-metal-ops.cpp`/device controllano shape/capability e
  `src/models/qwen35.cpp` esegue la famiglia densa. `test_mul_mat` verifica
  aritmetica backend, non drain/errore di una coda SSD equivalente. Non è un
  oracle per la concorrenza host; nessun nuovo percorso completo SSD qui provato.
- Ricerca locale riaperta: lettore/cache Clean Code, Flash F-P01, coda CUDA,
  `qwen27-weight-streaming-20261008.md`, report Metal T00–T07 e indice research.
  La RAM unificata consente letture dirette nei buffer borrowed; non trasferire
  H2D, pinned e eventi PCIe. Il guadagno Flash routed non prova il caso denso.
  Riletti reader/indexed queue, cache ready/pinning, close e test EOF/fallback.
- Ricerca web: RFC llama.cpp #31452 e discussione #27149 sono proposte,
  non prova di kernel/streaming merged. Il limite dense bytes/NVMe bandwidth
  è pertinente; non si adottano dichiarazioni prestazionali dei commenti.

Scelta minima: `ds4_file_reader_read` usa `read_indexed` esistente per regioni
almeno 32 MiB, un pezzo per worker (quattro nell’owner Metal), residuo massimo
workers−1 byte sincrono. Il percorso indexed drena tutti i job e pubblica
errori/contatori sul chiamante; fallback sincrono dopo startup fallito.
Letture piccole non avviano worker. API e lifetime restano invariati.

Alternative rinviate: overlap layer (lifetime doppio), cache maggiore
(cambia budget), riuso allocazioni (non dominante nel campione), MPP/tile
(nuova numerica o scarso beneficio Q-P02). Nessun codice esterno importato.

## Piano congelato

Test host: soglia ±1, offset e lunghezza dispari, 1/2/4/8 worker,
EOF parziale e drain, contatori, recovery, startup fallito anche parzialmente;
ASan/UBSan. Gate modello completo Qwen27 su prompt indipendente 128 token,
nove full-vocabulary e payload esatti vs B0; MTP e regressione Flash in sequenza,
poiché il reader è condiviso. Kernel e quant originali conservati.

Direzione: due sweep 64/128+ quattro teacher-forced, 8192/64/cache 2 GiB.
Se utili, una coppia warm-up e cinque alternate fuori dal profiler, stesso
workload e CSV locale a sei decimali già descritto nel Q-P02. Timeout120s/processo,
riserva host2GiB e guard invariati. Profilo finale separato. Piccoli guadagni
ammessi se riproducibili; nessuna soglia automatica2%.


## S2: lettore isolato

Implementazione minima: il metodo completo riusa i job indicizzati già
esistenti, con indici contigui e lo stesso owner. Non cambia la cache Metal.
Test host e ASan/UBSan PASS per soglia, code dispari, 1/2/4/8 worker,
EOF/drain, byte effettivi e fallback dopo startup parziale. Il primo test
aveva una fixture errata per un solo worker: il truncation point superava
la dimensione richiesta, quindi non produceva EOF. Corretta la fixture a
metà buffer, senza cambiare il lettore; tentativo fallito conservato nel log
della sessione di build. Nessun mismatch aritmetico.

Rilette per S2 `read_part`/pubblicazione finale di Strata, il loop EINTR/EOF
ggml e il contratto borrowed della coda locale: ogni task finisce prima del
ritorno, i contatori restano sul chiamante, nessun puntatore GPU passa ai
worker. Al momento il cap reader resta quattro nell’owner Metal, senza un
nuovo parametro utente.


## S3: gate completo e misura finale

Il gate indipendente Qwen27 passa su 128 input: nove full-vocabulary da
248320 logits e payload prefill bit-exact con B0; checkpoint, prefix no-op,
park/release/reload e continuazione esatti. Contatore finale del reader dopo reload e storage identici a
B0: 119969660928 byte logici e 1302325248 byte posseduti. Il counter si
azzera al reload: non è il totale I/O dell’intero processo. Fixture grafi Qwen/Flash PASS. La candidate
Q-P02 non è inclusa: shader e wrapper UD verificati identici agli hash B0.

Microbenchmark lettore: malloc + lettura completa + wait + free su regioni
GGUF reali, 32/64/128 MiB, F_NOCACHE e F_RDAHEAD=0 come il loader; un warm-up
poi dieci campioni per taglia. Incrementi mediani locali +10,83/+12,49/+21,37%.
È screening sequenziale B0 poi candidate, non una promozione statistica.
Un primo build del solo probe locale ometteva `_DARWIN_C_SOURCE`, nascondendo
le macro fcntl; nessuna inferenza avviata, errore e report NOT_VERIFIED conservati.

Trace finale separata: prefill 5475,1 ms wall, GPU 3674,0; quattro decode
2093,0–2185,5 ms wall, GPU 376,3–421,7. Kernel identici; non attribuire loro
la variazione GPU fra run. Fuori GPU rimane circa l’81–82% del decode.
Le coppie promozionali usano il workload 64/128+ quattro teacher-forced,
CSV a sei decimali, senza profiler e senza build concorrenti. Hash completo
del modello ricalcolato prima del warm-up e identico all’artefatto congelato.

Riesame S3: stessi `read_part`/`submit`/`io_main` Strata, bounds del reader e
wrapper buffer ggml. Una sola candidate con quattro worker/32 MiB; nessuno
sweep arbitrario di thread o cache. Il confronto non importa prefetch,
repack o F_NOCACHE da Strata; il loader locale aveva già la propria policy.


## Misure promozionali: cinque coppie

Una coppia di warm-up esclusa, poi cinque A/B alternate, nessun campione
selezionato/scartato dopo la raccolta. Processo nuovo, cache applicativa vuota,
8192/64/cache 2 GiB, pesi originali e identico input teacher-forced. La frontiera
128 riusa il prefisso: misura 64 nuovi token, non 128 token nuovi.

| Fase / occupazione | B0 mediana | Candidate mediana | Incremento / riduzione latenza |
| --- | ---: | ---: | ---: |
| Prefill primi64, TPS | 9,783366 | 10,979948 | +12,23% |
| Prefill altri64 fino128, TPS | 10,102997 | 11,058369 | +9,46% |
| Decode da64, TPS | 0,341370 | 0,433582 | +27,01% |
| Decode da128, TPS | 0,326383 | 0,439029 | +34,51% |
| Richiesta64 +4 eval, secondi | 18,206714 | 14,901786 | −18,15% |
| Richiesta128 (64 nuovi) +4 eval, secondi | 18,590294 | 14,899860 | −19,85% |

Il tempo richiesta sopra somma le finestre prefill/decode sincronizzate;
inizializzazione, dump, snapshot/ripristino e teardown restano fuori. Nessuna
misura HTTP TTFT o ExMekh rivendicata. Il primo passo decode e lo steady state
sono separati nei dati. Incremento locale e cumulativo coincidono: Q-P02 è
ritirato e Bi iniziale coincide con B0. Non si sommano percentuali di componenti.

Min/p10/mediana/media/p90/max/SD/CV, dieci run, warm-up, differenze e intervalli
sono in [measurements.json](measurements.json). Stime95% t appaiate, n=5:
prefill gain +7,96…+21,55% e +6,71…+11,30%; decode +16,19…+45,51% e
+20,74…+43,89%. Questi intervalli stimano la media dei cambiamenti appaiati,
non l’incertezza delle mediane riportate in tabella.

CV baseline prefill4,75/2,85%; candidate2,13/2,54%. Decode baseline6,98/6,40%
è **instabile** secondo il criterio congelato; candidate1,69/3,90%. La baseline
request64 ha CV6,62%, la request1284,36%; candidate1,38/2,07%.
Il segnale misurato supera questa dispersione, ogni confronto appaiato è
favorevole e gli intervalli restano positivi. La qualifica rimane locale;
non si aggiungono run selettivi per ridurre il CV. Nessuna soglia automatica2%.

Tutti i dodici processi (incluso warm-up) PASS. Nelle dieci misure: pressione1,
zero crescita swap, minimo host disponibile 5,12 GiB (riserva 2 GiB conservata).
Footprint di picco campionato2,20–3,15GiB: non confondere cap pesi con memoria
totale uguale fra processi. La cache/quantizzazione non cambia; nessun nuovo
buffer di staging e nessun aumento nascosto del budget.


## S4, integrazione ed esito

**KEEP nel subset M5 Pro misurato.** Il reader completo parallelo è attivo per
default per le regioni grandi nello streaming Metal. CLI, server, bench ed eval
ricostruiti con `make all`; host reader/queue e test finali scalari/feature PASS.
I primi due test finali lanciati nella sandbox non vedevano Metal: esiti
NOT_VERIFIED conservati, poi ripetizione verificata fuori sandbox. Due warning
preesistenti di parametri inutilizzati nei test che includono il core.

Riesame conclusivo: `read_part`/contatore pezzi e pubblicazione Strata,
loop ggml EINTR/EOF, `read_indexed`/wait/close locali, pinning/ready della cache
Metal e accounting controllati di nuovo. Nessun job sopravvive al ritorno o al
rilascio dei buffer; fallback dopo avvio fallito e contatori restano coerenti.
I 21 blob upstream iniziali sono verificati contro i Git object ID del pin;
un newline aggiunto dal trasporto locale dei file acquisiti è stato rimosso
solo quando ciò riproduceva il blob esatto. Hash prima/dopo documentati nel
manifest; nessuna differenza di codice interpretata come sorgente upstream.

Per il codice condiviso è stato riletto anche il Flash HF
`Qwen4ExpTextMLP/Experts/TopKRouter/SparseMoeBlock` e il grafo ggml
`qwen4exp.cpp:build_layer_ffn`: shared expert denso, softmax FP32, top-k,
ordine e pesi dei routed invariati. Il test Flash è regressione in sequenza,
non una nuova campagna o un suo benchmark prestazionale.

| Gate finale | Esito |
| --- | --- |
| Host reader, code, EOF, fallback e ASan/UBSan | PASS |
| Grafi ridotti Qwen residente/streamed e Flash | PASS |
| Qwen27 target, nove logits completi e payload prefill | Bit-exact B0/candidate |
| Qwen27 checkpoint/prefix no-op/park/reload/continuazione | PASS |
| Qwen27 MTP sidecar, due replay di otto token | Token esatti, delta logits max 1,1920929e-5 entro soglia congelata |
| Controllo MTP sul B0, stesso input/replay | Stesso delta 1,1920929e-5; nove vettori salvati B0/candidate bit-exact |
| Flash target-only, nove vettori completi, checkpoint e restore | Bit-exact con B0 Flash precedente |
| Oracle scalari UD/GDN e fixture feature finali | PASS sul codice di produzione |
| Warm-up + cinque coppie, guard memoria/cleanup | 12/12 PASS |

[gates.json](gates.json) conserva comandi, env, tempi, binari/log hash e criteri;
[inputs.json](inputs.json) fissa prompt/hash e token teacher-forced effettivi.
I dump full-vocabulary, payload, sampler CPU, trace GPU, script e binari sono
locali in `build/metal-perf-qwen27-20261102/`, ora ignorata esplicitamente da Git.
Nessun grande dump o peso versionato. Nessun commit durante queste misure.

## Hotspot residui e limiti

Il decode resta soprattutto fuori GPU; dopo questo intervento il prossimo
candidato è residenza/riuso bounded dei pesi e possibile overlap layer. Cambiano
budget/lifetime e richiedono una nuova campagna di fonti, contatori fisici I/O
e gate propri. Non si sostiene che il solo NVMe sia già al limite misurato.
Il tile IQ4_XS down con piccolo segnale positivo resta in backlog, da rivedere
per forma con nuova misura; il tile generalizzato Q-P02 non è incluso.

Prima campagna Qwen27 conclusa nel subset dichiarato: contesto **allocato8192**,
occupato64/128 nelle misure, quattro output teacher-forced e target-only.
MTP verificato per correttezza, non velocità. Non qualificati qui occupazione
quasi 8k, prompt 2k/4k, altri budget/hardware/quant, HTTP/ExMekh, full-resident27B,
CUDA/ROCm/distribuito o suite live. Il limite Q8 esteso di Q-P02 resta aperto
con le sue evidenze; nessuna soglia numerica è stata cambiata.
La prossima ottimizzazione non viene avviata senza un nuovo audit specifico.


Il [riesame successivo](../clean-code-review.json) separa il refactoring e i
suoi gate da questa qualifica storica. I commit del riesame includono codice,
test e provenienza; il controllo prestazionale successivo rimane distinto.
