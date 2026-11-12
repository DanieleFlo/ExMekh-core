# Stima del contesto in Core gestito · 12 novembre 2026

## Origine e perimetro

Su RTX 3090, con 23.853.006.848 byte liberi, Core proponeva al massimo 22.593 token per Qwen27. `tools/managed_catalog.py` usava quel tetto e raccomandava al massimo 8.192 token. Metal usava soglie fisse da 4.096 e 8.192. Il credito del worker CUDA era limitato alla dimensione del GGUF e poteva essere incoerente con il campione di memoria. La schermata di configurazione mostrava il limite senza visione anche quando era selezionato l'encoder.

Il checkout nativo di riferimento è `bb30e2a7f4c45398380aa0a2c396f89ef5d64c5f`. I GGUF fissati dal catalogo dichiarano `qwen35.context_length=262144` e `qwen4exp.context_length=262144`. I manifest locali danno 1.042.944.000 byte al più grande binding Qwen27 e 675.430.400 byte a `output.weight` di Flash, il minimo usato dall'ammissione SSD nativa. Questi limiti appartengono alle varianti con digest fissato dal catalogo.

- `ds4.c:qwen_graph_context_memory_estimate`, `ds4_session_create_internal` e `ds4_cuda.cu:qwen35_attention_gemm_rows_launch` descrivono KV FP32, ricorrenza, grafo di prefill e scratch CUDA per le forme lunghe. Il confronto [prefill gestito](managed-qwen27-prefill-20261112.md) conferma la differenza di memoria tra 64 e 2.048 righe. Le stime distinguono il margine per le allocazioni dinamiche dalla dimensione del grafo.
- `ds4.c:ds4_context_memory_estimate_with_prefill_mode`, `qwen_metal_memory_admit` e `ds4_metal_device.h` descrivono la geometria Flash, il doppio grafo necessario al ripristino SSD, il working set Metal e la quota di file cache disponibile oltre il floor. Il caso Flash usa la stima nativa conservativa, che include anche la riserva degli snapshot. Il margine fisso Metal segue `ds4_metal_memory.h` e il runtime nativo conserva la verifica prima dell'allocazione. [Working set Metal](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
- La VRAM per processo viene attribuita soltanto al worker gestito e alla stessa GPU, quando la telemetria lo consente. NVML documenta che la memoria per processo può mancare sotto WDDM. In quel caso e su Metal il credito usa l'aumento osservato durante l'avvio, limitato alle allocazioni note e applicato una sola volta alla memoria disponibile. [Telemetria NVML](https://docs.nvidia.com/deploy/nvml-api/620/api/structnvmlProcessInfo__t.html)
- L'encoder di visione resta parte del budget chat. I modelli autonomi di immagini ed embedding hanno una propria ammissione e possono ottenere memoria dal modello chat tramite `ds4_server_suite.inc:suite_victims`.

La ricerca del massimo usa il limite GGUF e il budget della modalità selezionata. Core ricontrolla la memoria prima dell'allocazione.

## Telemetria del worker su WSL

Nella prima prova CUDA con Qwen27, la chat è arrivata a 37.252 token di input su un contesto di 37.828. La stima è scesa a 35.836 appena avviato il worker e a 32.824 dopo il primo prefill. `nvidia-smi` riportava il PID e la GPU corretti con memoria `[N/A]`. La prova con visione ha completato 28.451 token su 28.961, con un calo da 28.961 a 25.663 dopo la richiesta.

Nel checkout nativo di riferimento, `ds4_cuda_accounting.cuh:cuda_account_allocate` registra le allocazioni fisiche del processo, tra cui i workspace CUDA creati durante l'inferenza. Su Metal `ds4_qwen4_metal_stream.inc:qwen4_stream_alloc` registra i tensori della cache in `DS4_MEMORY_DEVICE` tramite `ds4_metal.m:ds4_gpu_tensor_alloc`. `ds4_resources.c:ds4_resources_totals` legge il registro sotto mutex. Il contatore può omettere le allocazioni del driver, mentre il campione globale include anche gli altri processi. La telemetria interna espone sul socket privato solo i byte effettivamente allocati in `DS4_MEMORY_DEVICE`. In assenza del dato NVML, Core usa la variazione del contatore dal caricamento e conserva una base limitata all'aumento osservato. Su Metal la stessa variazione copre la crescita della cache SSD. Il runtime conserva la verifica finale dopo il rilascio del worker.

## Verifica

RTX 3090, WSL, Qwen3.8-27B in VRAM, binario CUDA `sm_86`, stato e cache temporanei, prefill nativo da 2.048 righe. Ogni prova avvia un worker con il massimo stimato a freddo, legge il catalogo dopo il caricamento e invia una richiesta in streaming a `/v1/responses`. La prova visione include una PNG 56 × 56. L'endpoint interno restituisce 404 attraverso il listener pubblico.

| Profilo | Massimo a freddo | Appena caricato | Dopo la richiesta | Token di input completati |
| --- | ---: | ---: | ---: | ---: |
| Chat | 40.139 | 40.139 | 38.129 | 39.563 |
| Chat con visione | 29.510 | 29.518 | 29.608 | 29.000 |

In entrambe le prove il contatore del worker ha rilevato circa 340 MiB aggiuntivi dopo l'inferenza. Nella prova chat circa 267 MiB della riduzione globale restavano senza attribuzione al worker e sono rimasti fuori dal credito. I test Core coprono il limite del modello, memoria esterna, cambio di GPU, memoria insufficiente, crescita della cache SSD e pressione critica. Il test browser copre l'aggiornamento della stima senza perdere l'input e il passaggio al massimo compatibile con la visione. Su Metal sono stati verificati i casi simulati e la politica nativa con il test CPU. La prova su Apple Silicon resta da eseguire.

## Adattamento all'hardware

In `bb30e2a7f4c45398380aa0a2c396f89ef5d64c5f`, `ds4_cuda.cu:ds4_gpu_init` passa il dispositivo CUDA logico 0 a `ds4_gpu_init_multi`. L'indice di `nvidia-smi` descrive l'inventario fisico e può differire dall'ordinamento CUDA. Il probe interroga la stessa enumerazione del driver e associa la VRAM tramite UUID. Il budget e il credito usano quella GPU anche in presenza di più dispositivi. [Enumerazione CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html) · [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html)

Su Apple Silicon il valore di 24 GiB documenta la macchina di qualifica. `ds4.c:qwen_metal_memory_admit`, `ds4_metal_device.h` e `ds4_metal_memory.h` usano pressione, RAM disponibile, working set e limite per singola allocazione. Core applica questi limiti anche agli altri tagli di memoria e continua a presentare la capacità come stima. I profili gestiti mantengono le proprie combinazioni di modello e backend.

La prova su WSL ha identificato la RTX 3090 tramite l'UUID del driver e l'ha esclusa con `CUDA_VISIBLE_DEVICES` vuoto. Qwen27 ha avviato un contesto di 37.557 token, con la stessa stima prima e subito dopo il caricamento, e ha completato una richiesta da 36.981 token di input. I test simulati coprono due GPU con capacità e ordine diversi e un host Metal da 16 GiB su cui entra il profilo SSD ma la residenza integrale supera il budget.

## Flash Next con cache CUDA a più livelli

Core e il launcher manuale usano lo stesso preset standard Flash CUDA, qualificato sulla RTX 3090 con chunk di prefill da 2.048, MTP integrato e cache KV paginata. Il catalogo usa 65.536 token come valore consigliato quando il budget lo consente e calcola a parte il massimo stimato, anche con l'encoder attivo. [Qualifica del preset Flash](qwen-flash-agent-launcher-20261008.md#promozione-finale-e-uso)

Sul riferimento nativo `bb30e2a7f4c45398380aa0a2c396f89ef5d64c5f`, `ds4.c:qwen4_graph_alloc` e `ds4_qwen4_cuda.cuh:ds4_gpu_qwen4_cache_prepare` riservano i pesi fissi prima della cache degli esperti. `ds4_qwen4_kv.cuh:ds4_gpu_qwen4_kv_alloc` mantiene K/V paginato anche in RAM. Core stima il grafo con K/V completo come margine per i workspace CUDA e verifica a parte che la RAM basti per le pagine K/V. Durante la sostituzione accredita soltanto la RAM anonima del worker, rilevata da `/proc/<pid>/status`.

Su RTX 3090 e WSL, lo stesso Core ha caricato Qwen27, Flash con visione a 65.536 token, Flash con visione al massimo stimato di 262.144 token e di nuovo Qwen27. I piani hanno riusato i file GGUF presenti. Flash ha servito richieste da 56 token dopo entrambi i caricamenti.
