# Qwen Image in batch sul servizio nativo

Baseline core `8625eb7`, binario Metal SHA256
`0542cb214dd8b97d505e4be5ff4979b484207f71ae308d20726f2b5f901f9f9b`.
Profilo 256×256, prompt `A red square on white.`, seed 17 e 18, Qwen27
streamed 512/64, riserva host 0. Il controllo misura `n=2` sul binario
congelato con cache adattiva e una sola istanza. Il profilo a 256×256
precedente misura 4,684 s di conditioning, 53,786 s di DiT e 10,283 s di VAE.
Il DiT domina la richiesta. La misura a due immagini congela i risultati
del percorso seriale per il confronto.

## Audit prima dell'intervento

- In `ds4_server_suite_operations.inc:suite_run_images` la suite passa
  sempre un solo seed per job, ricrea il reclaim e ripete tutti i componenti
  per ciascuna immagine. CUDA e Metal entrano nella stessa funzione.
- Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
  `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py:379–386,711–755,816,853`
  espande gli embedding e le maschere per immagine, mantiene una dimensione
  batch nei latenti e avanza tutte le immagini allo stesso timestep.
  `models/transformers/transformer_qwenimage21.py:909–1019` conserva la
  separazione tra esempi e le maschere del prefisso. Il VAE prevede anche
  decode a slice singole in `models/autoencoders/autoencoder_kl_qwenimage.py:867–886`.
  Le dimensioni del GEMM native al primo passo sono già vincolate dalla
  parità BF16. Impilare le righe di immagini diverse cambierebbe quel
  dispatch. L'RNG nativo è versionato e usa seed consecutivi.
  [Pipeline Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py).
- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
  `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1143–1246`.
  Token, maschere, posizioni e DeepStack dipendono dal prompt e dai
  riferimenti. Sono comuni agli output con seed diversi. L'encoder e i
  suoi dtype restano invariati.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp:33–79` e `metal/tests/dense_mm_parity.cpp`.
  La residenza bounded dei pesi è pertinente. L'expert streaming MoE e le
  geometrie FP16 del test non modellano il batch DiT BF16.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/ggml-metal-ops.cpp:383–387,485,2602,3252` e
  `tests/test-backend-ops.cpp:12349–12366,12704–12710`. Le geometrie batched
  di matmul, attention e convoluzione sono specifiche delle primitive.
  Questi test non coprono scheduler, maschere o rounding Qwen Image 2.1.
- Il [report I-P05](metal-inference-20261102/qwen-image/group-prefetch-20261105/README.md)
  mostra il costo del riacquisto dei gruppi DiT e congela la numerica Image512.
  In `ds4_qwen_image_graph.h:qi_weight` ogni peso è acquisito dal gruppo
  corrente. Visitare un blocco per tutti gli output riusa quel gruppo
  prima di passare al successivo. La cache K/V del prefisso dipende solo
  dalla maschera causale del prefisso e dallo stesso conditioning.
- Metal e CUDA condividono `ds4_numeric` e `ds4_qwen_image_attention.c`.
  Il loro contesto numerico e i proprietari dei pesi non sono adatti
  all'uso da thread di inferenza indipendenti. Il batch riusa il grafo con
  un solo owner e mantiene le forme delle primitive per ciascuna immagine.

Scelta: wave di al massimo due immagini con timestep e visita dei blocchi
in comune. Condizionamento, cache prefisso e pesi sono condivisi. I latenti,
l'attenzione, il rumore e l'output sono distinti. Il VAE decodifica una
immagine per volta. Il planner include lo stato aggiuntivo e può scegliere
la wave singola prima dell'esecuzione se la coppia non entra. Una volta
avviata l'esecuzione, errori e cancellazione seguono il ripristino normale.
Il test confronta i PNG e i seed con il controllo, oltre al gate Image512
indipendente. Una coppia è una misura di fattibilità e non qualifica uno
speedup generale o la concorrenza di kernel GPU.

## Controllo seriale Metal

Due PNG e replay idempotente in 143,610 s. Conditioning 9,048 s, DiT
113,497 s, VAE 21,050 s, H2D 228361068912 B. Due immagini distinte con
seed 17 e 18, chiusura ordinata. Il planner adattivo ha scelto la variante
finale 8. Log e PNG in `build/image-progress-20261110/batch-qualification/serial`.


## Prova batch Metal

| Campione | Richiesta (s) | Conditioning (s) | DiT (s) | VAE (s) | Byte H2D |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seriale | 143,610 | 9,048 | 113,497 | 21,050 | 228361068912 |
| Batch 2 | 123,044 | 4,732 | 97,716 | 20,582 | 120381575096 |

Stesso prompt, geometria e seed. Entrambi i PNG sono bit-exact rispetto al
controllo. SSE e replay JSON idempotente passano e il server esce con codice
0. Intervallo massimo tra eventi 10,081 s. La variante adattiva finale
è 9 invece di 8, con zero vittime in entrambi i campioni. I tempi sono
singole misure end-to-end sul Mac, con disponibilità di memoria adattiva.
La riduzione osservata è 14,3% per la richiesta e 47,3% per H2D.
I kernel di ciascuna lane conservano la geometria originale e operano
sullo stesso stream GPU.

Binario batch SHA256
`0397f3c1a19893357ac016926a1794448f0aed8bd2154e862ba1bbf48d86dc31`.
Log, metriche, eventi e PNG in
`build/image-progress-20261110/batch-qualification`.

## Gate numerico indipendente a 512 px

Il probe ha eseguito due immagini nel batch. La prima usa il rumore FP32
del controllo congelato, la seconda usa il seed consecutivo e il RNG
nativo. Il processo termina con codice 0 in 447,917 s monitorati, senza
cleanup forzato. L'output della seconda immagine è distinto. Il probe
verifica che entrambi i risultati diventino disponibili dopo il ripristino
e che ciascuno possa essere prelevato una sola volta.

La prima immagine riproduce esattamente 11 artefatti nativi. Coincidono
conditioning, tutti gli otto latenti, RGBA e PNG. Il confronto con
l'oracle indipendente congelato CPU encoder/MPS DiT/MPS VAE passa con
RMSE RGBA 0,00500305345, SSIM RGB 0,99716581329 e MAE alpha 0,00043903192.
Fixture SHA256 `bd5f7eb548ffd8ba2768f09da4d5ccdf657af948cd0799f12533e2cee9b53a80`.
Soglie SHA256 `edefea38067da972800a35b107ab78a1f7e349a7d650f0c93857afef63377f87`.
Il probe registra 11 eventi e otto valutazioni batch completate.
Questa è una verifica numerica con trace, fuori dal confronto di latenza
HTTP. Report in `batch-qualification/oracle-comparison/verification.json`.

Passano i test host di HTTP/SSE, executor, risorse, input/output image,
graph CPU, API, riferimento e contratti. Passano 69 test mirati SDK e
cinque test del frontend, inclusi due casi Playwright. La patch SDK si
applica sulla base pulita. Il dry run Linux seleziona CUDA e cuDNN e
ricompila i consumer dell'header. Compilano gli header C++17 e passa il
factory CUDA con stub. Build con nvcc e prova su GPU NVIDIA da eseguire.

## Riesame del ciclo di vita

Il confronto del riesame usa il binario batch già qualificato e i due PNG
congelati del controllo. In `ds4_qwen_image_pipeline.c:ds4_qwen_image_job_run`
l'ammissione è ripetuta dopo l'anteprima chiamata dal server. Una seconda
decisione NOMEM può quindi sfuggire al fallback. `ds4_executor.c:
ds4_execution_run` distingue lo stato NEW dall'inizio della transazione.
Il server deve usare lo stato pubblico del job per riconoscere un rifiuto
senza effetti, senza avviare nuovamente un job entrato nell'executor.
Il buffer e la quota dei PNG restano in proprietà del worker fino alla
pubblicazione. Un errore di generazione, ripristino o ricevuta deve
rilasciare le risorse parziali anche quando il job resta in recovery.

Riletto il confronto Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`
(`pipeline_qwenimage21.py:695–755,816,853`) per il confine tra batch,
sampling e pubblicazione. Transformers, Strata e ggml nei pin sopra
restano riferimenti per forme e pesi. Il riesame mantiene ordine,
geometrie e rounding del forward. La stima dei buffer del DiT viene
spostata al DiT con lo stesso budget. Il test di regressione usa gli
stessi input HTTP e verifica replay e contabilità delle risorse.

La prima ripetizione HTTP sul binario del riesame è stata interrotta dal
monitor per crescita dello swap globale oltre 256 MiB durante la suite
browser. È classificata `SKIP_RESOURCE_LIMIT` ed esclusa dal confronto.

Il DiT possiede il calcolo dello scratch per le lane. L'operazione usa
l'API del job per conoscere lo stato di ammissione. Il worker conserva la
proprietà dei file fino al commit della risposta e riusa lo stesso
metodo per rilasciare output parziali, file transitori e file scaduti.
Il replay SSE riporta il numero di immagini della richiesta validata.
Il modello di progresso restituisce una percentuale solo per indici
appartenenti al batch.

La suite dell'agent passa con 1218 test, 49 live saltati e copertura del
100% su 6880 righe e 2222 rami. La suite SDK completa passa 3686 test
con 88 skip. Riproduce gli stessi cinque fallimenti benchmark del
commit base SDK `4a0d1f7`. La patch si applica alla revisione di origine
e i 285 sorgenti Python installati coincidono con il checkout.

La ripetizione sequenziale completa la richiesta in 124,483 s sul
binario SHA256 `93ad7189aaef6b5594f9874c355369922a48e32122f30393721e6685cc50ceb1`.
Condizionamento 5,073 s, DiT 98,559 s, VAE 20,789 s, variante 8 e
H2D 111118443960 B. I due PNG corrispondono byte per byte al controllo.
Il replay JSON e SSE restituisce gli stessi file. Il terminale del replay
SSE indica due immagini e la generazione pubblica 23 eventi con due
immagini attive. Il monitor passa in 139,869 s senza crescita dello swap,
con arresto del server riuscito. Si tratta di una verifica di regressione
con planner adattivo. Il gate Image512 sopra resta il riferimento
indipendente del forward, che il riesame non cambia.
