# Q-P14: letture parallele anche nella fascia8–32MiB

3 novembre2026, S1 prima del codice. Dopo profiloI/O P11, prova separata daP13.
Il file reader è incluso solo dall'owner Metal e dal suo test: nessuna modifica
al reader CUDA/ROCm o al runtime distribuito. Regressione Flash comunque richiesta.

## Audit specifico della soglia

Profilo misurato:53.54GB logici e55.54GB OS diskio per16 token MTP4,
8.89s decode. Inventario effettivo dei tensori di layer del GGUF:
174 regioni/3911270400byte tra8e32MiB,200/9678848000byte >=32MiB,
489/167829504byte sotto8MiB. Esclude embedding/head e non è hit-rate misurato.
La fascia~3.9GB/layer pass resta sincrona con la soglia32MiB diQ-P03.

Riletti prima dell'intervento:
- HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`, Qwen3_5 denso/GDN/residui:
  solo trasporto degli stessi byte, nessun ordine/dtype/mask alterato.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp:read_part/submit/io_main`, stream.h,
  testarena e wrapper/runtime: split delle slice anche più piccole e ready
  dopo tutte le parti. Trasferibili concorrenza bounded e pubblicazione finale;
  file esperti contiguo, allineamento4K, abort e default M5Max non trasferiti.
- ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/llama-mmap.cpp:read_raw_unsafe`, Metal wrapper/device/ops e
  test_mul_mat: EINTR/short-read e byte bounds; non offre un oracle equivalente
  della concorrenzaSSD. Nessuna deduzione di speedup da questi sorgenti.
- Locale: Q-P03/S1–S4 e micro32/64/128, Clean Code e test reader/coda,
  byte inventory dal GGUF corrente. La soglia32 fu provata solo sulle forme
  grandi; non è prova che8/16/24 siano già ottimali. Four worker esistenti,
  chunk8MiB, drain/counters/errori invariati.

Ipotesi: soglia8MiB, quattro pezzi su buffer originale, nessun thread/staging
aggiuntivo. Sotto8MiB resta sincrono/lazy. Alternative sweep worker e pipeline
nuova non sono il primo cambiamento necessario; confrontare prima8vs32.

Piano: snapshot reader32; micro host con due owner uguali e4worker, uno per
variante, stesso fd F_NOCACHE/F_RDAHEAD0, regioni8/16/24/32/64/128MiB.
Warmup escluso poi10 coppie alternate per taglia, read+wait e confronto byte
fuori dal timer. Nessuna inferenza concorrente. Testhost soglie8/32±1,
1/2/4/8worker, EOF/counters/recovery e startup parziale; sanitizer. Poi modello
Qwen/MTP4 e regressione Flash in sequenza, misure/pre-profile/post e bundle.
Conservare piccoli incrementi/rumore; nessuna soglia automatica2%.

Prima del codice: profiloP13 corretto appena completato, decode~1.99TPS sotto
profiler, residuoI/O ancora dominante; policy reader32 invariata rispetto al
profiloP11. Le medie dimensioni restano un obiettivo applicabile anche dopo
il riciclo a capacità. I nuovi test8/32 sono preparati e compatibili anche con
la policy32: output/errori/contatori, senza dipendere dalla formula dei pezzi.

S2 micro10 coppie alternate: soglia8 peggiora la fascia8/16/24MiB di
18.5/13.5/12.2% (CI stimati interamente negativi).32/64/128, la cui policy
non cambia, sono compatibili col rumore. Byte esatti, tutti66 pair inclusi
warmup PASS. **REJECT prima di integrare**: nessuna modifica al reader32 di
produzione. La lettura sincrona8MiB isolata è già~0.80ms: il solo numero
maggiore di richieste concorrenti non garantisce throughput.

Nuova ipotesi/per-intervento da auditare separatamente: destinazione riusata
rispetto a nuova allocazione può spiegare perché i vecchi microQ-P03 (malloc
inclusa) e l'attuale read+wait hanno costi diversi. Testare anche soglia64 e
buffer appena mappati, prima di attribuire aSSD/thread un nuovo cambiamento.
Nessun criterio numerico o limite di memoria modificato.
