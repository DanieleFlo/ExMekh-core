# Qwen3.8 Flash Next con Qwen Image ed EmbeddingGemma 2

## Diagnosi prima della modifica

DS4 `a420a1398ae67dda8386c3475365cf2314f591a2`, RTX 3090 SM 8.6 su WSL.
Il catalogo Core del profilo attivo Flash Q2, contesto 65.536, MTP e visione,
espone soltanto `vision` in modalità `tiered`. `tools/managed_catalog.py`
ammette Image e Gemma soltanto in `vram` su CUDA SM 8.6. Flash ha soltanto
`tiered` su CUDA e `ssd` su Metal. Il worker in esecuzione è `ds4-server` e
`/v1/models` espone soltanto gli ID della chat.

`ds4_server_suite.inc:suite_collect_victims` può usare il checkpoint del
modello chat. `ds4_lifecycle.inc:lifecycle_supported` lo consente già per
Flash SSD su Metal e Qwen27 residente su CUDA. Su Flash CUDA i buffer del
grafo appartengono alla sessione, mentre gli esperti, i pesi densi e le viste
appartengono al runtime CUDA. `ds4_qwen4_kv.cuh:kv_io` salva le righe KV logiche
anche quando i cache slot sono su livelli diversi. Il payload Flash in
`ds4.c:qwen4_session_save_payload` include lo stato GDN, PLE, MTP, i logits e
lo stato della policy degli esperti. Il codice già verifica l'identità del
GGUF e scambia il grafo dopo la convalida del checkpoint.

`ds4_qwen_image_pipeline.c:configure_plan` dispone già di varianti residenti
e varianti che caricano i pesi dai file del bundle entro il budget della fase.
`ds4_server_suite_operations.inc:suite_plan_embedding` prova Gemma residente
e poi la residenza limitata. Entrambi possono usare una vittima di sessione
quando la richiesta eccede il margine disponibile. Il cambio completo del
worker per ogni tool avrebbe un costo di ricaricamento del modello chat e
richiederebbe il trasferimento dello stato fuori dal percorso di ripristino
esistente. Liberare anche i pesi Flash richiederebbe ripristinare i binding e
il budget della cache CUDA a livelli. La prima scelta è riusare il checkpoint
della sessione e le varianti native dei due modelli ausiliari, lasciando il
proprietario dei pesi Flash attivo durante il passaggio.

## Confronti upstream

- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
  `src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py` e
  `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`. Gemma sceglie la
  maschera bidirezionale per tipo di layer e assembla le modalità prima del
  backbone. Il text encoder Image usa l'ordine dei token e le posizioni del
  proprio modello. Si mantengono le implementazioni native qualificate.
- Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
  `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`,
  `src/diffusers/models/transformers/transformer_qwenimage21.py`,
  `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py` e
  `src/diffusers/hooks/group_offloading.py`. L'offload segue le fasi encoder,
  DiT e VAE. Il cast all'ingresso del VAE e i cast dopo le normalizzazioni
  del DiT sono parte del percorso numerico. La suite DS4 riusa il proprio
  trasporto per fasi e il budget del registro.
- Strata `6674a0065fb96bacde33e3eb10f91a1df86f95f2`,
  `src/prefill/prefill.cpp`, `src/prefill/moe_mmq.cu` e
  `tests/cuda/prefill_mmq_kquant_test.cpp`. Il prefill conta separatamente
  spazio per esperti residenti, staging e scratch. I test controllano MMQ
  degli esperti quantizzati del modello Flash. Questo confronto riguarda
  il confine tra cache e stato della chat, mentre Image e Gemma usano i
  backend numerici DS4.
- llama.cpp `b38960f52079e41f91863dd0905ec13432966b39`,
  `src/models/gemma-embedding2.cpp`, separa attenzione e pooling Gemma.
  Nel checkout `1a064ab0921238c1daa397d6f4a900ef33884de2`,
  `src/llama-memory-hybrid.cpp` salva memoria ricorrente e di attenzione.
  `ggml/src/ggml-cuda/ggml-cuda.cu` verifica le operazioni supportate dal
  device e `tests/test-backend-ops.cpp` copre matmul e attenzione. Il formato
  ggml ha una propria geometria e non sostituisce il payload Flash DS4.

Riferimenti locali: [manifest T00](../performance/native-suite-t00-20261020/sources.json),
[handoff CUDA Gemma](../performance/managed-gemma-context-handoff-20261112.md),
[stress Metal Flash e Qwen27](../performance/metal-apple-silicon-20261031/t08-20261106/README.md)
e [profilo Flash CUDA](../performance/qwen-flash-agent-launcher-20261008.md).
L'intervento abilita la coesistenza dei modelli. I criteri numerici, le
quantizzazioni e le soglie di ammissione esistenti restano quelli dei gate.

## Verifica

- Riproduzione sul Core attivo: `images` ed `embeddings` hanno entrambi
  `available_modes` vuoto per Flash CUDA. Il profilo ha già la visione attiva.
- Build finale `ds4-server-suite` CUDA SM 8.6 riuscita. Passano i test nativi
  HTTP, executor e budget della cache Image.
- Su WSL passano 25 test Core e 21 test Playwright, incluso il menu di una
  Quick Chat con Flash. Ruff lint e format passano sui tre file Agent.
- Il test CUDA `ExmekhNativeLive.test_registered_image_tool_retains_conversation`
  passa con Flash Q2 65.536/2.048, MTP e proiettore vision. Il tool genera il
  PNG da 256 px con variante streamed 8, zero vittime e 120.261.758.136 byte
  H2D. La risposta successiva restituisce `BETA18` e riusa 1.444 token.
  Il prefill della prima richiesta dura 17,638 s, il decode 21,554 s e la
  richiesta 39,382 s. La generazione dura 215,243 s, di cui 9,776 s encoder,
  204,559 s DiT e 0,871 s VAE. Sono campioni singoli su WSL con attività
  browser concorrente e crescita dello swap globale.
- Il primo gate del tool `search-web` con `qwen3.8-flash-next` esaurisce
  i 256 token prima di produrre una chiamata valida, quindi il primo caso
  termina prima dell'inferenza Gemma. Con l'alias chat e input fissati, il
  ciclo controllato produce tre vettori Gemma da 128 dimensioni e ordina
  il testo alpino sopra quello sulle ricette. Passa anche il gate indipendente
  `EmbeddingToolsE2E.test_04_semantic_ranking_uses_real_vectors_and_explicit_cache`.
  L'endpoint riporta 0,231 s, variante 0 e zero vittime. Segue Image 256 px
  in variante streamed 6, 37,920 s, zero vittime. Due avvii con prompt e
  seed 17 identici producono lo stesso PNG SHA-256
  `0f939be3db29502fad3e66623a6abca2edca5f8f5cd3d12598f53064549cbff2`.
  Con replay completo dell'input, la continuazione restituisce `MANGO-1762`
  e riusa 32 token. Prefill, decode e richiesta misurano rispettivamente
  3,574/0,276/3,986 s all'andata e 1,985/1,261/3,357 s al ritorno.
- Un probe nativo separato usa lo stesso modello e configurazione CUDA,
  un prompt da 13 token e le callback reali di `ds4_engine_reclaim_create`.
  Salva il checkpoint, rilascia il grafo, verifica zero byte device per
  la sessione e ripristina sotto una prenotazione pari alla capacità
  liberata. Tutti i 248.320 logits e l'argmax coincidono. I pesi Flash
  restano disponibili e il checkpoint è eliminato dopo il ripristino.
- Log e risultati locali dei tre gate in `/tmp/ds4-flash-suite.kZJE25`,
  `/tmp/ds4-flash-cycle.OnelBR` e `/tmp/ds4-flash-reclaim.pBEPzm`.
  I server di prova sono stati chiusi. Metal è coperto da catalogo e
  discovery simulati in questa sessione.
- La suite Agent completa iniziale raggiunge il 100% di copertura. La
  regressione sullo stile del pulsante è stata corretta riusando il catalogo
  della UI. Al riesame seriale passano entrambi i casi di modifica allegati,
  la regola del catalogo e la nuova prova del menu. La seconda suite completa
  dà 1.308 pass, 50 skip e un timeout del pannello di output di `date-time`.
  Lo stesso test passa in 7,06 s quando eseguito da solo.
