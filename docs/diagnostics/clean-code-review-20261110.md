# Riesame di discovery, lifecycle e trasferimenti Metal

Il riesame di responsabilità, proprietà della memoria, sincronizzazione e
contratti HTTP parte da `a272633cc53d0b0b45ba48b5f8274c08d1b2c1a8` e dalle
modifiche descritte in `vision-session-restore-20261110.md`.

## Riscontri prima delle modifiche

- Lo snapshot della discovery è sincronizzato correttamente, ma apertura,
  sostituzione, lettura e distruzione accedono direttamente a mutex e JSON.
  Centralizzare queste operazioni nel proprietario `suite_catalog`.
- La discovery della lista evita il lock d'inferenza, mentre quella di un
  singolo modello nativo entra ancora nella coda del worker. I due endpoint
  devono leggere lo stesso snapshot senza attendere compute.
- La validazione di `image_timeout_seconds` avviene dopo l'allocazione
  della suite e richiede un rollback dedicato. Validare tutta la
  configurazione prima di creare risorse.
- Le diagnostiche del restore devono conservare la causa primaria anche
  se la chiusura del file fallisce, e identificare anche le fasi di
  apertura/identità oltre a validazione, candidato e payload.
- `tensor_read` e `tensor_write` ordinano il trasferimento rispetto ai
  produttori GPU. `tensor_fill_f32` scrive invece da CPU senza lo stesso
  confine: un blit già codificato può sovrascrivere il fill successivo.
  Il test esistente riempie prima di accodare il produttore, quindi non
  copre questa sequenza.

## Riesame upstream per l'ordine del fill, prima dell'intervento

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py`, aggiornamenti
  ricorrenti/KV nel forward: mantenere l'ordine delle mutazioni. Non
  trasferire l'allocator o lo scheduler PyTorch al backend C.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/mtl_rt.mm:Context::run_graph`, submit e completamento prima
  dell'osservazione host, con pool Objective-C delimitato. I suoi command
  buffer unretained non sono adottati qui.
- llama.cpp/ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/ggml-metal.cpp:ggml_backend_metal_synchronize`,
  confine esplicito di completamento. Non cambia i nostri layout di stato.
- `ds4_cuda.cu:ds4_gpu_tensor_fill_f32` accoda il kernel nello stream del
  device. `ds4_metal.m:ds4_gpu_host_transfer_ready` conserva già il batch
  quando drena i produttori per read/write e fornisce lo stesso confine al fill.
- Riletti il rapporto del restore, i test di copia dentro/fuori batch e
  Q-P23/Q-P25 in
  `docs/performance/metal-inference-20261102/qwen27/adaptive-stream-20261103/README.md`.
  Il problema di ordine è distinto dal precedente lifetime ARC.

La prova riproduce copia GPU → fill host sullo stesso input e verifica il
risultato tramite il confine host esistente. I gate sintetici e il gate reale
del checkpoint/encoder verificano la correttezza con gli stessi criteri numerici.

La regressione copia GPU → fill host fallisce prima della correzione
(`fill-order-before.log`, assert del valore finale `-7.0f`). Il fill deve
seguire il produttore già codificato. Il cambiamento riusa
`ds4_gpu_host_transfer_ready`, conserva il batch e lascia le operazioni
vuote o fuori bounds senza scritture. Le soglie numeriche restano immutate.

## Verifica e limiti conservati

- Nella suite HTTP la lista, il dettaglio dei due modelli e il 404 completano dodici
  richieste mentre il lock d'inferenza è occupato; refresh a owner libero.
- Fondamenta Metal, feature, lifecycle e fallback senza Metal 4 passano.
- Passano anche i test di testo prodotto (4), contratti (16) e release (8).
  La verifica statica del piano di build Linux include CUDA `sm_86`.
- I gate indipendenti Flash e Qwen27 target/MTP2 sono bit-exact sui
  10/10/12 artefatti congelati in
  `build/metal-perf-qwen27-20261102/review-20261103/`.
- Il primo MTP4 automatico si ferma al resume con NOMEM nella fase candidato.
  La cache ammessa è 0,63 GiB, inferiore al binding denso minimo. Il retry
  con tetto SSD a 1536 MiB si ferma già all'allocazione della sessione con
  0,76 GiB effettivamente ammessi. Entrambi gli esiti sono conservati.
  Q-P20/Q-P23/Q-P25 descrivono la distinzione fra memoria già rilasciata,
  disponibilità fisica osservata e picco del candidato.
- `make test` con accesso esclusivo completa le fixture host e fallisce
  sui test del modello Flash residente. Quattordici aperture sono rifiutate
  dall'admission (15,91 GiB richiesti, capacità osservata inferiore). Gli
  otto controlli host successivi sono stati eseguiti a parte e passano.
  I primi due tentativi erano bloccati rispettivamente dal bind di localhost
  nel sandbox e dal lock d'istanza del server attivo.
- Il replay finale del checkpoint reale si ferma prima alla creazione
  della sessione (cache ammessa 0,86 GiB) e al retry durante il resume
  (0 GiB). Il replay precedente al riesame aveva verificato il payload
  originale e gli embedding bit-exact con più RAM disponibile.
- Il primo riavvio della suite con la configurazione 8K, SSD streaming, vision,
  Qwen Image ed embeddings, senza MTP e riserva host zero si ferma alla
  creazione della sessione. La cache ammessa è 0,76 GiB. Il secondo avvio,
  con la stessa configurazione, riesce e rimane attivo sulla porta 8080.
  Le richieste testo e vision rispondono HTTP 200 con il limite di un token.
  Tutti i 60 poll della lista e dei dettagli nativi durante la richiesta
  vision rispondono HTTP 200, con un massimo osservato di 1,42 ms.

I log di verifica sono in `build/clean-code-review-20261110/` e i log dei
riavvii e la prova HTTP in `build/qwen27-vision-server/clean-code-review-20261110*/`.
