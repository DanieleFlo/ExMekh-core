# Q-P06: residenza densa entro il cap, 3 novembre 2026

S1, piano prima del codice; implementazione dopo profilo Q-P05.
B0 continuazione `fc8fb63`, Bi Q-P05 con shader e numerica originali.

## Audit specifico della policy di residenza

Riletti per questo intervento, ai pin esatti e ai percorsi del manifest Q-P03:

- HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py`: tutti i layer densi
  sono usati a ogni token; separati embedding/head, ricorrenza e KV. Nessuna
  equivalenza semantica richiede espellere i pesi immutabili dopo un layer.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp:load_dense/make_arena/layer`, `stream.h`,
  `metal/src/mtl_rt.mm:Context::wrap/free`, `metal/tests/arena_budget_test.cpp`,
  `gemv_parity.cpp` e `metal/kernels/gemv.metal`:
  densi residenti separati da arena routed, slot protetti fino al consumo.
  Riserva minima prima di spendere il resto del budget è trasferibile;
  residenza totale densa, abort su errore, working set M5 Max e file esperti
  compatto non sono trasferibili al 27B/24GiB. Nessuna copertura Qwen27 Metal
  completa al pin. Gli errori/lifetime restano verificati dai test DS4.
- ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/models/qwen35.cpp`, `src/llama-mmap.cpp:impl/read_raw_unsafe/mlock`,
  `ggml/src/ggml-metal/{ggml-metal.cpp,ggml-metal-device.cpp,ggml-metal-ops.cpp}`,
  `tests/test-backend-ops.cpp:test_mul_mat`: ownership shared/no-copy,
  capability e algebra senza mutare i quant. Mmap/LRU del sistema non prova
  una cache applicativa bounded; non richiedere mlock né assumere pagine SSD fredde.
- Locale riletto: owner Metal `clear/load/range/close`, registry e retire,
  reader/coda, admission e park/reload, spans dei layer; report Q-P03 e Clean
  Code; `qwen27-weight-streaming-20261008.md` (nel relativo esperimento),
  `ds4_qwen4_transfer.cuh`, ricerca problem-to-source/piattaforme.
  Il ring CUDA conserva pesi separati e budget; non importare pinned/H2D.
- Ricerca nuova `quantumnic/ssd-llm@3e8c8b616e16552d1b840e2741a82f152155ef47`,
  `src/model/cache.rs:LayerCache/AdaptiveLayerPinner` letto via GitHub:
  pinning evita evizione di alcuni pesi; LRU puro in scansioni dense cicliche
  può produrre zero hit. Non adottare frequenza come segnale utile (tutti i
  layer sono eseguiti), sforamento se tutti pinned o prefetch fuori budget.
  Apple LLM in a Flash offre raggruppamento e riuso, ma sparsità approssimata
  e modelli con ReLU non equivalgono alla nostra SwiGLU e ai criteri congelati.

Scelta: il core fornisce regioni immutabili note tramite API, head prima e
poi tensori dei layer; il backend seleziona una quota persistente e conserva
lo spazio necessario per un intero layer non residente. I buffer persistenti
sono caricati lazy e sempre contabilizzati nello stesso cap, liberati solo
dopo fence in close/park. Nessuna introspezione della cache dal core.
Il resto usa il percorso transiente esistente; Flash e residente invariati.
Se head+riserva non entrano, conservare il percorso precedente.

Alternative: LRU ciclico e semplice aumento cap senza riuso non riducono
riletture; prefetch/overlap richiedono un intervento successivo con altro audit.
Nessun auto-aumento del cap e nessun nuovo flag. Provare2GiB prima, poi4GiB
solo se admission/guard lo consentono, mantenendo riserva2GiB e swap invariato.
Un cap maggiore è una variabile esplicita, non un guadagno a budget identico.

Gate: input ridotto identico resident/streamed, hit/riletture/cap, errori e
recovery/close senza leak; poi gate Qwen128 con payload e logits bit-exact,
MTP/lifecycle e regressione Flash sequenziali. Screening breve e profilo;
KEEP eventuale su warmup+5 coppie alternate del bundle, stessi criteri.

Profilo Bi appena acquisito prima del codice: quattro decode circa2s/token,
span GPU molto più breve, grandi attese host ancora presenti; Q-P05 non elimina
le scansioni dense. Sufficiente motivazione per verificare il riuso, senza
assumere che il solo SSD esaurisca tutto il residuo. Si mantengono le fence
esistenti in questo intervento per attribuire il solo beneficio della residenza.

S2: fixture ridotta PASS (cap12MiB riserva un layer completo; a4MiB la
policy deve fare fallback, non fingere hit). Gate128 bit-exact. Il contatore
ha rivelato che park/reload ricreava l'owner precedente senza la nuova policy:
nessun mismatch numerico, ma residenza persa. Riaperti
`ds4_lifecycle.inc:ds4_engine_weights_load`, release/retire e gli stessi
contratti Strata/ggml: trasporto/lifetime indipendenti dal payload. Centralizzare
l'apertura in un helper core comune, usato da open e reload, prima di misurare.
Fixture lifecycle aggiornata con un vero span scalare di4byte già presente
nel suo backing (prima il placeholder aveva bytes0). Gate da ripetere dopo.

Tentativo diagnostico `p06-resume-full` interrotto (FAIL/SIGTERM, non qualifica):
`DS4_QWEN_TRACE_DIR` attiva deliberatamente il replay token-wise dell'intero
prompt, anche filtrando posizione/stadio. Non usare questa trace per confrontare
il percorso layer-major. Nessun criterio cambiato; ripetizione completa senza
trace dopo gli isolati/lifecycle. L'acquisizione Q8 dovrà iniziare dopo prefill.

## S3 screening, non promozione

Dopo il fix reload: nove logits/payload128 bit-exact con B0. Cap2GiB: live
prefill1860560768byte, letture27298737712, finali104390989888 dal reload
(confronto row-gather senza retain:115599246528). Non sono byte SSD fisici.

Due screening: cap2GiB prefill11.42/11.06 decode0.46/0.44; cap4GiB
prefill10.23/11.36 decode0.47/0.46. Nessun grande guadagno dimostrato.
Trace cap4: decode0.49, span GPU successivi665.8/559.7/535.7ms contro
357.1/375.0/360.0 B0; non attribuire il peggioramento al solo I/O.
Nuovo sample separato (anche trace GPU, non timing promozionale):3435 sample
main,802 nella fence di layer, il resto principalmente wait coda/pread.
Guard normale, min host4.39GiB e picco footprint5.16GiB nella prima trace;
nessuna crescita swap. Riaperti Strata budget/lifetime e allocator/registry
locali: residenza cambia le pagine e il footprint, ma shader/algebra restano
identici. Nessuna spiegazione causale del diverso tempo GPU è già provata.
**PENDING_BUNDLE**: non promuovere la sola cache dal contatore dei byte.

Precisazione di provenienza nel riesame: il precedente streaming denso27B
CUDA citato sopra è il prototipo storico in
`docs/experiments/qwen27-4bis-20261008/shared-integration.patch`, non un
percorso production attuale. I file ring/transfer correnti sono Flash Next;
le loro assunzioni non qualificano automaticamente la cache27B Metal.
