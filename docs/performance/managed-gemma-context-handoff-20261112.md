# EmbeddingGemma 2 con contesto Qwen27 residente · 12 novembre 2026

## Diagnosi e scelta

Riferimento DS4 `f439aec65e0e6813dba0eb1ee2f02dae32309450`. In `ds4.c:engine_vision_suspend_sessions` e `engine_vision_resume_sessions` la visione parcheggia il grafo Qwen27 e conserva i pesi del modello. Il percorso qualificato chiama le stesse primitive di checkpoint usate da `ds4_lifecycle.inc:lifecycle_reclaim_save`, `lifecycle_reclaim_release` e `lifecycle_reclaim_restore`. Il checkpoint su disco contiene KV e stato ricorrente. `ds4_executor.c` conserva il file durante il ripristino e pubblica il risultato dopo il recupero.

In `ds4_server_suite_operations.inc:suite_run_embedding` il planner Gemma può scegliere il caricamento limitato prima che il planner HTTP valuti il rilascio del grafo chat. La suite offre inoltre i pesi del modello chat tra le vittime. La cache dei pesi Gemma viene sempre rilasciata al termine. `ds4_numeric_cuda.cu` aggiorna la memoria globale al rilascio di buffer, mentre `ds4_numeric_refresh_memory` restituisce `UNSUPPORTED` sul backend CUDA prima di un nuovo job. Un altro processo può cambiare lo spazio disponibile dopo l'ultima allocazione della chat.

Riferimento reale su RTX 3090/WSL, Qwen27 con visione, contesto allocato 22.593 e profilo Gemma BF16 `914f7f89142e33e77833254d9c9b90c3cef7303b`. Con limite device di 19,0 GiB, l'input breve usa zero vittime e la richiesta da 4.410 token rilascia 3.885.340.672 byte del grafo chat. Il turno successivo riusa 753 token. Il caso breve ricarica circa 274 MB via H2D a ogni richiesta. Con pressione aggiunta dopo il prefill il riferimento ha ancora scelto zero vittime per l'input breve, con allocazioni virtualizzate da WSL. Questi campioni riproducono la divergenza di ammissione e non rappresentano una misura statistica di latenza.

Fonti confrontate prima della modifica:

- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`, `src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py` e `src/transformers/models/qwen3_5/modeling_qwen3_5.py`. Gemma combina le maschere per tipo di layer e usa il backbone bidirezionale e il pooling per gli embedding. Qwen alterna attenzione e Gated DeltaNet con stato ricorrente. Entrambi richiedono l'aritmetica già qualificata. La scelta di memoria non deve cambiare dtypes, maschere, ordine delle operazioni o punti di arrotondamento.
- llama.cpp `b38960f52079e41f91863dd0905ec13432966b39`, `src/models/gemma-embedding2.cpp`, `ggml/src/ggml-cuda/mmq.cu` e `fattn-tile.cuh` come fissati in [T00](native-suite-t00-20261020/sources.json). Il grafo separa il pooling dall'attenzione e il backend seleziona le forme supportate. Nel checkout locale `7d442abf5c6244117fd5a1dc51a5d19f00792491`, `src/llama-context.cpp` e `src/llama-memory-hybrid.cpp` serializzano la sequenza, inclusi gli stati delle due famiglie di memoria. `ggml/src/ggml-cuda/ggml-cuda.cu` e `tests/test-backend-ops.cpp` verificano capability e operazioni del backend. Il formato stato ggml e la qualifica delle primitive non attestano la compatibilità con il checkpoint DS4.
- Strata `6674a0065fb96bacde33e3eb10f91a1df86f95f2`, `src/prefill/prefill.cpp`, `src/prefill/moe_mmq.cu`, `src/core/native_head.cpp` e `tests/cuda/prefill_mmq_kquant_test.cpp`. I limiti per pesi residenti, buffer e trasferimenti si valutano per fase. I kernel e i test riguardano Flash e gli esperti quantizzati. La tabella di token di Strata non implementa il ranking EmbeddingGemma o il salvataggio del grafo Qwen27.
- [Checkpoint e lifecycle DS4](native-suite-t03-t04-20261020/README.md), [qualifica Gemma BF16](native-suite-t11-t12-20261021/README.md), [test di ranking con il provider nativo](native-suite-t17-t18-20261022/README.md). Riutilizzare il journal e il ripristino del grafo già qualificati.

Aggiornare la capacità osservata prima del planner. Provare il piano completo di Gemma prima del piano limitato. Per gli embedding il piano può scegliere soltanto le sessioni chat, mentre i pesi Qwen restano residenti. Dopo un piano senza vittime, i pesi Gemma possono servire richieste di embedding consecutive. Prima di un altro prefill della chat, liberare quei pesi per conservare il budget dei suoi buffer temporanei. Un piano che usa il checkpoint rilascia Gemma prima del ripristino. Il piano limitato preesistente resta un'opzione quando il piano completo supera il budget anche con il grafo rilasciato.

L'API di aggiornamento della memoria è usata anche da Qwen Image. Il confronto con Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e` copre `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`, `src/diffusers/models/transformers/transformer_qwenimage21.py` e `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`. Il pipeline ordina offload, encoder, DiT e VAE. Dtype, maschere e arrotondamenti appartengono alle singole fasi. L'hook CUDA aggiorna i byte liberi usati dalle verifiche già presenti in `ds4_qwen_image_pipeline.c`. Il flusso di Qwen Image continua a usare le proprie varianti e vittime.

## Verifica isolata

Qwen27 con contesto 22.593, visione e prefill nativo. Le richieste di confronto usano gli stessi input sui due binari. Stato, checkpoint, cache SSD e configurazione sono temporanei.

| Caso | Riferimento | Candidato |
| --- | --- | --- |
| Budget 18,5 GiB, 37 token Gemma | Zero vittime, 492.376.088 byte H2D | Una sessione parcheggiata, 3.885.340.672 byte liberati, 274.145.264 byte H2D |
| Budget 19 GiB, due richieste identiche | I pesi vengono rilasciati a fine richiesta | Zero vittime, trasferimento da 274.145.264 a 575.424 byte H2D |
| Budget 19 GiB, 4.410 token Gemma | Una sessione parcheggiata | Una sessione parcheggiata, anche con pesi Gemma caldi |
| Pressione VRAM dopo il prefill | Zero vittime per la richiesta breve, con memoria virtualizzata da WSL | Una sessione parcheggiata per la richiesta breve |

I tre embedding a 18,5 GiB sono identici byte per byte nella risposta JSON. Dopo il caso a budget stretto, entrambi i binari restituiscono il codice atteso `GEMMA-4821` e il candidato riusa 753 token del prefisso. Le misure H2D descrivono questi campioni e il riuso dei pesi. La prova dinamica mostra l'ammissione con il campione CUDA aggiornato.

Il gate a 18,5 GiB carica il tool `search-web` dal registro ExMekh e usa il provider nativo del client. La raccolta delle pagine restituisce due candidati fissati per isolare la verifica del ranking. Il tool invia una richiesta `/v1/embeddings` con task `similarity`, classifica la pagina sulle Alpi al primo posto e incorpora il contributo semantico nel punteggio 0,0368191433. Il planner parcheggia una sessione, libera 3.885.340.672 byte e ripristina il turno che riusa 652 token.

La prova attraverso il listener Core gestito usa lo stesso Qwen27, il profilo
temporaneo con Gemma e visione e i pesi locali. A budget libero il tool
restituisce gli stessi punteggi, sceglie zero vittime e il turno riusa 652
token. La suite indipendente CUDA passa nove casi, inclusi testo, immagini,
audio, video, continuazione vision e streaming di Qwen Image. Passano anche
il test HTTP nativo, l'executor, i test CPU Gemma/numerici e la suite Agent
con 1.282 test superati. Le prove GPU qui registrate sono CUDA su RTX 3090.

## Revisione del ciclo di vita

In `suite_run_embedding` il planner HTTP rifiutava un piano limitato valido
quando `ds4_gemma_visual_plan` indicava un rilascio dei pesi Gemma prima della
prenotazione. Trasformava inoltre ogni errore del secondo planner in memoria
insufficiente. Il budget dell'executor conta solo memoria già libera o
rilasciata da una vittima dichiarata. La riserva figlia del modello eredita
quel limite.

Per questo caso sono stati riletti l'assemblaggio dei token visuali e audio
in Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
`src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py`,
e la composizione del grafo di llama.cpp `b38960f52079e41f91863dd0905ec13432966b39`,
`src/models/gemma-embedding2.cpp`. I vettori interni delle torri devono
sopravvivere fino al backbone. Il checkpoint ibrido in llama.cpp
`7d442abf5c6244117fd5a1dc51a5d19f00792491`, `src/llama-memory-hybrid.cpp`,
mantiene separati attenzione e stato ricorrente. Strata
`6674a0065fb96bacde33e3eb10f91a1df86f95f2`, `src/prefill/prefill.cpp`,
distingue scratch e pesi residenti per fase. Il journal DS4 ha un formato
proprio e usa il budget definito in
`ds4_resources.c:ds4_resources_can_reserve` e
`ds4_executor.c:ds4_execution_run`.

L'adapter Gemma possiede il rilascio aggregato dei suoi pesi. Quando la suite
chiede il picco da prenotare, l'adapter esegue il rilascio solo se il piano
limitato lo richiede, poi aggiorna la memoria osservata e ricalcola il piano.
La riserva esterna copre così anche le nuove allocazioni che sostituiranno
i pesi liberati.

Il gate CUDA `tests/probe_gemma_visual.c` usa lo stesso input del gate
congelato T15–T16. Con tutte le torri a caldo e un limite di 500.000.000 byte,
il piano richiede di liberare 1.188.866.008 byte e restituisce un picco
prenotabile di 341.479.424 byte. L'allocazione sotto la riserva esterna
riesce e il vettore di testo è identico bit per bit a quello prodotto prima
della riduzione del budget. Anche l'output multimodale della fixture
coincide byte per byte con l'artefatto congelato `faults/vectors.f32`.
