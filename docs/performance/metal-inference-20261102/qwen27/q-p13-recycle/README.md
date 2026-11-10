# Q-P13: riciclo denso alla pressione del cap

3 novembre2026, S1 prima del codice. Bi Q-P11/Q-P12 corretti, nessun commit.
ProfiloP11 dopo gate:16 output,53542419632 byte logici dal reader,
55544856576 byte OS diskio_read (proc_pid_rusage V2),8.887964s decode.
Il secondo contatore include tutto il processo, non il solo reader/SSD
isolato. SDK26.5 `sys/resource.h:rusage_info_v2` e libproc, chiamate nel
solo harness diagnostico fuori dalle finestre; nessun claim dal timing trace.
GPU IQ4 verifier5 scende831.7->725.6ms aggregati rispetto aBi4; screening
wall1.758TPS variabile, non una regressione kernel già attribuita.

## Nuovo audit specifico lifetime/scheduling

- HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`, Qwen3_5 decoder/MLP/GDN:
  pesi immutabili, dipendenze e stato invariati; non richiede liberare un
  buffer dopo ciascun layer. Nessuna nuova precisione/quantizzazione.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp:pin/stamp/victim/read_part/io_main`, stream.h,
  `metal/src/mtl_rt.mm:Context::free` e window.cpp segmenti/stream:
  evizione solo di storage non protetto e dopo consumo, ready dopo ultima
  parte. Testarena/gemv letti. Routed/router e wrapper senza retained
  references non sono equivalenti al dense27B, ma il limite lifetime è
  trasferibile. Non importare prefetch predetto né abort su I/O.
- ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  ggml-metal.cpp/ops/device, llama-mmap.cpp e test_mul_mat: buffer shared
  immutabili validi fino al completamento; nessun oracle per la nostra coda.
- Locale: cache `load/layer/clear`, pin per argomento, `flush_commands`,
  pending CB/retire, registry/reconcile, reader drain/errori, CUDA ring e
  precedente report streaming. L'evizione esistente già sincronizza prima
  di sovrascrivere/liberare e protegge gli argomenti acquisiti. Quella fence
  può governare il riciclo denso, invece di un clear incondizionato per layer.

Scelta: solo cache con piano residente denso, senza router/compact, conservare
entry ready tra layer finché entra tutto nello stesso cap. Refresh delle
misure risorse comunque a ogni boundary, senza aspettare GPU inutilmente.
Pin pre-encode della fase abbandonata terminano al boundary; ogni eviction
continua a sincronizzare. Se una entry non è ready dopo un errore, usare
fence+clear esistenti prima del retry: non conservare una entry fallita che
impedisca recovery. Nessun aumento cache/staging/worker e nessun cambio Flash.

Alternative: ring aggiuntivo/prefetch più ampio spendono RAM extra; semplice
assenza di fence senza protezione/recupero è scorretta. Non adottata.

Gate: grafi ridotti byte-exact/cap, guasto EOF su riga fresca e recovery
anche con retention, lifecycle e full128/MTP4 indipendente8192, criteri frozen.
Screening/profilo, poi matrice appaiata con/senzaROWS5 e nuovo riciclo per
scegliere il bundle, conservando piccoli segnali e variabilità.

S2 PASS: graph/cap/fresh-row EOF e recovery, feature e lifecycle; modello
completo8192/128/MTP4 a1536MiB,11 ordinary/spec vettori bit-exact conBi.
Pin e readiness restano proprietà della cache; erroreI/O prende il percorso
di drain+clear prima del retry, senza entry fallite persistenti. Nuove misure
fuori profiler e traceI/O separate precedono l'intervento sulla soglia reader.

S3 screening1536MiB: prefill13.535961/decode1.934453; trace separata
13.665158/1.992031. Non promozione. Kernel5 ha tempoGPU minore, ma il wall
resta condizionato dall'I/O: serve matrice appaiata, non somme dei microgain.
