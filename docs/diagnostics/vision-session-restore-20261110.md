# Ripristino della sessione dopo vision — 10 novembre 2026

## Osservazione iniziale

Il server Metal Qwen3.8-27B, contesto 8192, prefill 16, SSD streaming,
senza MTP e con `DS4_METAL_HOST_RESERVE_BYTES=0`, risponde 400 su
`/v1/responses` con `unable to restore session; checkpoint retained`.
Il catalogo continua a rispondere 200. Il checkpoint conservato contiene
2888 token testuali, misura 538400240 byte e supera il checksum SHA-256
dell'envelope. I log precedenti non conservano lo status del restore.
Questo non dimostra ancora quale risorsa o fase abbia fallito.

## Riesame prima dell'intervento

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py`, embedding delle
  immagini prima del decoder e stato `past_key_values`. L'ordine semantico
  resta applicabile, il suo lifetime non sostituisce il restore transazionale C.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp:ExpertStream::make_arena`, budget e minimi
  della cache prima della residenza. Non implementa questo checkpoint Qwen.
- llama.cpp/ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/ggml-metal.cpp:ggml_backend_metal_free`, fence prima
  del rilascio. Non autorizza a distruggere il grafo valido prima di validare
  il candidato del restore.
- Riletti `ds4_lifecycle.inc:lifecycle_checkpoint_restore`,
  `ds4.c:engine_vision_resume_sessions`, budget dinamico Qwen Metal,
  `ds4_qwen4_metal_stream.inc:ds4_gpu_qwen4_stream_resize` e i report
  `docs/performance/metal-inference-20261102/qwen27/adaptive-stream-20261103/README.md`
  (Q-P20/Q-P23), `docs/architecture/qwen27-vision-checkpoint-20261001.md`
  e `docs/performance/metal-apple-silicon-20261031/t08-20261106/README.md`.
  Il picco del candidato e la telemetria XNU erano già stati distinti dal
  normale live state. Non assumere automaticamente la stessa causa.

Prima fase scelta: conservare l'artefatto e aggiungere log della fase/status
falliti, quindi riprodurre su un input vision isolato. Nessun kernel,
dtype, maschera, tolleranza o margine di memoria viene modificato.
Diffusers non interviene nel restore del modello linguistico in esame.

## Causa isolata e scelta prima della correzione

Il replay del checkpoint originale riesce. Una successiva codifica di
`tests/vision-fixtures/qwen38/maple.png` riproduce lo stesso errore con
`DS4_STATUS_NOMEM` durante l'allocazione del candidato. Dopo park il ledger
device passa da 1239584832 a zero byte, ma Metal mantiene 1234370560 byte
allocati; dopo l'encoder sono 2177335296. La cache SSD possiede zero byte
in questo esperimento. Quindi non basta ridimensionare la cache.

Un input isolato di 64 MiB con `tensor_write` e `tensor_read` dimostra il
lifetime: `currentAllocatedSize` passa da 1769472 a 68878336 byte e rimane
a 68878336 dopo `tensor_free`; solo il drain del pool Objective-C lo
riporta a 1769472. I getter Objective-C dei trasferimenti host lasciano
riferimenti autoreleased oltre il ritorno delle API C. Il ledger ha già
rilasciato correttamente l'owner, mentre il buffer è ancora trattenuto.

Riaperto il confronto per questa discrepanza: Strata allo stesso pin,
`metal/src/mtl_rt.mm:Context::run_graph`, delimita gli oggetti temporanei
con `@autoreleasepool`; usa command buffer senza retained references,
un'assunzione che non trasferiamo ai nostri comandi. ggml allo stesso pin,
`ggml-metal.cpp:ggml_backend_metal_free`, richiede ancora il completamento
dei comandi prima del rilascio. Transformers allo stesso pin non governa
ARC e non giustifica modifiche a KV, maschere, dtypes o ordine dei calcoli.
Localmente `ds4_gpu_tensor_copy` ha già un pool per i chiamanti C; mancano
in `tensor_contents`, `tensor_read` e `tensor_write`. L'SDK macOS locale
espone il lifetime dei pool in `Foundation.framework/Headers/NSAutoreleasePool.h`.

Scelta: delimitare i temporanei nelle tre API host Metal, mantenendo handle,
alias, fence e batch attivi. Verificare prima il rilascio su input isolati,
poi ripetere il replay originale, confrontare gli embedding congelati e
verificare HTTP con il modello completo. Non introdurre purge dei KV,
stima artificiale di RAM libera o riduzione delle riserve del candidato.
Le implementazioni CUDA/ROCm non usano ARC e restano invariate.

## Verifiche dopo la correzione

Base sorgenti `a272633cc53d0b0b45ba48b5f8274c08d1b2c1a8`, modifiche locali
non committate. Artefatti e log in
`build/qwen27-vision-server/restore-diagnostic-20261110/`.

- Il nuovo test da 32 MiB fallisce prima della correzione e passa dopo su
  tutti e tre gli accessi host. Passa l'intera suite
  `test_qwen35_metal_features`, inclusi batch attivi, MTP, RNG, mRoPE e
  snapshot multimodali.
- `test_qwen_metal_lifecycle` passa per Qwen27 residente, Qwen27 streaming
  e Flash, inclusi checkpoint bit-exact, OOM, corruzione, truncation,
  file mancante e rollback atomico.
- Primitive Metal, grafo Qwen ridotto e grafo Flash residente/SSD passano
  con i criteri numerici congelati.
- `test-suite-http`, quattro test di testo prodotto e sedici contratti/
  manifest/artifact passano. Anche la build CPU passa.
- Il piano di build Linux/CUDA `sm_86` include CUDA ed esclude il file
  Objective-C modificato. GPU NVIDIA non disponibile per esecuzione.

Nel replay completo, dopo park `currentAllocatedSize` ora scende da
1241382912 a 1769472 byte; prima rimaneva a 1234370560. Dopo l'encoder
sono 929366016, invece di 2177335296 byte. Questo misura il rilascio,
non uno speedup di prefill/decode.

Gli embedding dell'immagine originale sono bit-exact prima/dopo, 6144000
byte, SHA-256 `2dfd817346fccc75e307979644d6e65c795829a8bb66944c6fa4ad8e152f17b2`.
I primi due replay successivi rifiutano ancora il candidato per capacità
host effettiva insufficiente (cache ammessa 0,64/0,44 GiB, inferiore al
binding minimo); non sono PASS end-to-end. La seconda prova è isolata
dalle compilazioni. Il difetto del lifetime risulta corretto, ma non
autorizza a ignorare il controllo delle risorse. L'utente libera RAM per
la ripetizione con le stesse impostazioni e riserva zero.

### Ripetizione con RAM disponibile

`replay-after-freed-ram.log`: PASS sul checkpoint originale, 2888 token,
encoder reale, resume con `parked=0` e payload completo con SHA-256 identico
all'originale. Gli embedding rimangono bit-exact. Questa prova non usa il
ramo diagnostico che esclude la pianificazione della cache. Il motore
conserva le normali riserve del grafo candidato, con riserva host zero.

Il primo avvio della suite completa dopo questo PASS si ferma invece alla
creazione della sessione: dopo apertura degli asset nativi, cache ammessa
0,27 GiB. È distinto dal test del solo motore e non viene presentato come
avvio riuscito. L'utente libera ulteriore RAM per riprovare la suite con
Qwen Image ed embeddings oltre a Qwen27 vision.

### Suite HTTP e riavvii

Il successivo avvio `native-suite-restore-fixed-20261110-retry` riesce,
con i tre modelli nel catalogo, contesto 8192, SSD streaming, senza MTP e
riserva host zero. HTTP 200 per testo, immagine dopo testo, nuova immagine
dopo stato multimodale anche in SSE; nessun evento di errore. Le richieste
sono limitate intenzionalmente a un token e terminano `incomplete` per
`max_tokens`: verificano il percorso e il restore, non la qualità della risposta.
Durante queste richieste tutti i 64 poll del catalogo rispondono 200,
massimo 1,270 ms. Log e JSON nella directory di quel server.

Su richiesta dell'utente si verificano anche tre arresti/riavvii completi,
con inferenza vision a ogni giro. Il primo tentativo dell'osservatore non
raccoglieva il processo figlio (`wait`) e scambiava lo zombie senza RSS/porta
per un arresto ancora in corso. Verificata l'assenza del server e dei suoi
file temporanei; corretto il solo script di prova e conservato il tentativo
in `restarts-initial-observer-error.json`. La ripetizione usa `Popen.wait`,
exit code, inventario processi, listener, directory agentic, checkpoint
temporanei e segnalazioni di handle Metal ancora vivi al cleanup.

La ripetizione `restarts.json` passa tutti e tre i cicli. Ogni processo
termina con exit 0, senza server/listener residui, directory agentic o
checkpoint temporanei e senza avvisi `Metal cleanup discarded`.

| Ciclo | Arresto | Aumento RAM libera + speculative | Test prima dell'arresto |
| --- | ---: | ---: | --- |
| 1 | 0,229 s | 7,90 GiB | Testo e vision HTTP 200 |
| 2 | 0,237 s | 8,09 GiB | Testo e vision HTTP 200 |
| 3 | 0,232 s | 8,83 GiB | Testo e vision HTTP 200 |

La RAM è campionata con contatori XNU globali e pagine da 16384 byte;
non è una misura esclusiva del singolo PID. La cache SSD è adattiva,
quindi RSS e incremento di memoria libera possono variare fra i giri.
Non si osserva accumulo di processi o risorse del server dopo gli arresti.
I test di lifetime e lifecycle verificano anche il rilascio nel processo,
prima della sua uscita.

Il server finale, PID 63989, rimane avviato in
`restore-restart-verified-3-20261110`, con i tre modelli, contesto 8192,
prefill 16, SSD streaming, senza MTP e riserva host zero. Anche le ultime
richieste testo/vision dopo il terzo riavvio rispondono 200. Nessun commit.
