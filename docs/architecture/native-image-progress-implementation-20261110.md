# Implementazione del progresso image — 10 novembre 2026

## Audit prima dell'implementazione

Baseline core `4a03d05c94bcaf88ff295875b8b5ea0e2ca467dd`; sorgente SDK locale
`ExMekh-lib`, revisione `ed3a744c9843ce9980f2fdf87db32c73d104cd91`, pulita.
Il progetto è [native-image-progress.md](native-image-progress.md).

Fonti rilette per questo intervento, senza modificare kernel o dispatch:

- Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`, snapshot in
  `build/metal-oracles/sources/diffusers-<revision>/src/diffusers/`:
  `pipelines/qwenimage21/pipeline_qwenimage21.py:768–853`, callback dopo lo
  scheduler e prima della decodifica VAE. Trasferibile la separazione fra
  avanzamento della generazione e risultato finale; callback che ricevono
  e possono cambiare latenti sono incompatibili con osservazione UI sola.
- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`, snapshot omonimo
  in `build/metal-oracles/sources/`,
  `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py:821–842`:
  ordine dei layer, deepstack e norm finale restano invariati. Nessun callback
  UI deve cambiare dtype, rounding o semantica degli input immagine.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`, snapshot
  `build/metal-perf-qwen27-20261102/upstream/Niko1221/Strata/`,
  `metal/src/engine/stream.cpp:ExpertStream::wanted,open`:
  streaming indica caricamento pesi e capability specifiche del modello,
  non progresso di un'operazione image. Non si trasferiscono queue o policy
  Metal in un contratto che deve coprire CPU/CUDA e modelli futuri.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`, snapshot parallelo
  `ggml-org/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:213–215`:
  capability verificate per operazione ggml. Il tree dell'audit non copre
  il DiT image; i suoi kernel non costituiscono un'API di progresso image.
  Provenienza dei due snapshot: audit Q-P15/I-P03 e
  `docs/performance/metal-apple-silicon-20261031/t06-t07/upstream-numerics.md`.
  Le directory esportate non hanno un checkout Git autonomo: `git rev-parse`
  vi risale al core e non è usato come prova della loro revisione.
- Locale: `ds4_qwen_image_observer.c` aggiunge fence solo quando `trace` è
  abilitato; `ds4_qwen_image_dit.c:ds4_qwen_image_dit_forward` ha già un
  completamento sincronizzato. Il nuovo feedback usa questi completamenti,
  senza abilitare trace/readback e senza fence aggiuntivi. Le unità intermedie
  del primo adapter sono valutazioni del denoiser, non scheduler step terminati.
- `ds4_executor.h/.c` e contratto T17.5: journal del lifecycle e successo
  soltanto dopo restore. `ds4_server_suite_http.inc` accoda su worker seriale:
  heartbeat inviati da quel worker si fermerebbero durante compute. Il thread
  della connessione deve essere l'unico writer SSE, con snapshot bounded.
- SDK: `core/models/tool.py:create_execution_with_hooks`, `native_suite.py`
  e `interfaces/tool_hook.py`; il contesto del tool può portare un callback
  infrastrutturale senza esporlo agli argomenti del modello. Il trasporto
  installato è OpenAI/httpx2; verificare parsing incrementale e retry su quello.

Differenze osservate: oggi Images risponde soltanto con JSON finale; la
deadline assoluta include la coda e vale 300 secondi nel profilo Metal.
I retry SDK amplificano gli errori 409/503. Il 256px della chat “ciao” completa
la generazione; l'errore vision seguente riguarda la continuazione LLM.

Approccio scelto: osservatore comune di metadati, adapter Qwen come primo
consumer, SSE opt-in con JSON compatibile, heartbeat indipendente da compute,
callback SDK per invocazione e aggiornamenti mirati allo skeleton. Totali e
percentuali sconosciuti restano indeterminati; nessun requisito pubblico
dipende da Qwen, otto passi, Metal o CUDA. La policy di deadline resta distinta
dal timeout di inattività e viene documentata/configurata esplicitamente.

## Verifiche

### Implementazione e copertura

`ds4_operation_progress.h` definisce metadati e callback comuni senza dipendenze
da modello/backend. Il primo adapter pubblica fasi e otto valutazioni realmente
completate del denoiser, sfruttando il fence già presente nel forward. Il
contratto è verificato anche con token e totale sconosciuto; ogni futuro adapter
image deve implementarlo. Il progresso non modifica tensori, recipe o dispatch.

Un solo thread di connessione scrive SSE; il worker aggiorna uno snapshot
coalescente e consegna un unico corpo terminale, dopo restore. Heartbeat ogni
10 secondi, incluso il tempo in coda. Il client negozia il formato e accetta
anche JSON legacy, verifica sequenza/identità e rifiuta EOF senza terminale.
I retry automatici image sono disabilitati; il replay di un commit resta
idempotente. `image_timeout_seconds` è un limite assoluto opzionale valido su
tutti i backend, include la coda e altrimenti eredita `timeout_seconds`.
L'esempio Metal sceglie 1200 secondi image e mantiene 300 per le altre API.

Il callback SDK è nel contesto dell'invocazione, fuori dagli argomenti LLM.
Bridge e collector preservano lo snapshot; root e skill aggiornano le etichette
senza rifare la galleria per il solo progresso. Come richiesto successivamente
dall'utente, lo skeleton mostra **solo percentuale e tempo**, senza “Generating
image”, fasi o barra. `overall_kind` distingue la stima strutturale del
piano Qwen nei metadati. `—%` indica una percentuale sconosciuta.

I tipi C degli observer sono stati estesi: i consumer C vanno ricompilati con
gli header aggiornati. Le dipendenze del Makefile sono state aggiornate; il
formato dei bundle e il contratto JSON finale restano invariati.

SDK basato su `ed3a744c9843ce9980f2fdf87db32c73d104cd91`, patch riproducibile
in `agent/vendor-patches/exmekh-lib-image-progress.patch`. Applicazione su un
export pulito e ricostruzione esatta di tutti i sorgenti Python: PASS. Wheel
della revisione finale ricostruita e installata, hash `94741ebc1b366c1aef94f22ce3d970fdf5eea378b9a2dfa7d61cb5afc26b22e7`.
METADATA identico byte per byte; nel lockfile cambia solo l'hash della wheel.
Confronto dei sorgenti Python installati contro il sorgente SDK: identico.

### Prove host, SDK e frontend

- Build del server Metal e del probe image: PASS. Risorse, executor e protocollo
  HTTP CPU: PASS; bundle 12 test e testi di prodotto 4 test: PASS.
- Protocollo su socket con produttore lento: eventi prima del risultato,
  heartbeat durante compute, un solo terminale, unità token con totale ignoto.
  Una deadline corta scade comunque ricevendo heartbeat: PASS.
- SDK: 118 test PASS fra native suite, stream frammentato, sequenza/identità,
  errori 409/503 senza retry, compatibilità JSON, hook e capabilities.
  Un socket TCP reale riceve heartbeat mentre supera il timeout di lettura
  configurato per singolo intervallo: PASS.
- Frontend: 78 test unitari PASS; 9 test browser PASS sulla wheel installata,
  inclusi batch, refresh, galleria, secondo evento controllato di progresso,
  assenza di testo/barra nello skeleton e PNG conservati dopo errore LLM.
  Screenshot ispezionato in `build/image-progress-20261110/progress.png`.

Il primo test browser ha identificato un refresh del pannello dovuto al
progresso incluso nell'uguaglianza strutturale; corretto nel modello e nei
consumer UI, anche per skill. Il controllo DOM iniziale era inoltre sensibile
alla prima pubblicazione della history/context: il test definitivo separa
quella pubblicazione da un nuovo evento di progresso controllato e verifica
l'identità del nodo su quest'ultimo. Nessuna asserzione sul progresso è omessa.
Le prove TCP/browser fallite nel sandbox sono state ripetute con accesso locale,
senza disabilitare casi. Il test bundle usa il Python predisposto con jsonschema.

### Confronto numerico isolato e gate indipendente

Fixture Image512 originale, SHA256
`bd5f7eb548ffd8ba2768f09da4d5ccdf657af948cd0799f12533e2cee9b53a80`, seed20261021,
stessi pesi/tokenizer, FP32 noise e criteri congelati. Un'unica istanza nativa
alla volta, riserva0, soglia swap256MiB e deadline600 secondi come nella
qualifica precedente. Risultati:

| Prova | Esito |
| --- | --- |
| Callback disattivato / attivo |11/11artefatti esatti: conditioning,8latenti,RGBA,PNG |
| Progresso nativo |11eventi di metadati,8valutazioni completate |
| Gate sull'oracle indipendente congelato |PASS: conditioning,8latenti,RGBA |
| RGBA rispetto all'oracle |RMSE0.00500305345;SSIM0.99716581329;alpha MAE0.00043903192 |

Il gate ripete il forward nativo completo sullo stesso oracle indipendente
CPUencoder/MPSDiT/MPSVAE congelato; non è un nuovo forward oracle né un PASS
del corpus T00 completo. Tolleranze e geometrie non sono state modificate.
Prima del gate sono stati riletti anche Diffusers, stesso pin sopra,
`models/transformers/transformer_qwenimage21.py:998` (ordine dei blocchi),
`models/autoencoders/autoencoder_kl_qwenimage.py:841–865` (post_quant_conv,
decoder e clamp), Strata `metal/tests/dense_mm_parity.cpp` (FP16 e geometrie
delle proiezioni, non una pipeline image) e ggml `tests/test-backend-ops.cpp`
(operazioni e IMROPE Qwen3-VL, non un gate DiT image). Le assunzioni dei kernel
FP16 Strata non vengono trasferite al percorso BF16.

Il binario copiato prima della ricompilazione non ha aperto il bundle
(`NOMEM6`), quindi non è stato promosso a controllo numerico. Il controllo
valido usa **lo stesso binario attuale** per entrambe le modalità, SHA256
`0443884d5a3ce7f59998348b2f1dc6d99506e78237e3da183adc889b67e79461`.
Entrambi completano senza cleanup forzato. Il monitor misura230.781s off e
229.842s on; i tempi nativi sono217.191s e216.707s. Il planner adattivo sceglie
varianti6/7 nelle due condizioni di memoria: questi tempi non isolano il costo
del callback e **non sono una misura di speedup né un limite sull'overhead**.
I profili per conditioning/DiT/VAE e le condizioni sono conservati nelle evidenze.

### HTTP reale con compute nativo

Server locale isolato con il binario aggiornato, Qwen27 SSD512/64 senza richieste
LLM e bundle Image. Richiesta256px tramite il client tipizzato installato:
PASS in68.764s con timeout di lettura20s. Ricevuti15eventi ordinati: started,
12progress,1heartbeat,1completed; intervallo massimo10.033s. Discovery dichiara
versione1 e deadlineimage1200s. Replay JSON con la stessa chiave restituisce
lo stesso PNG in0.00269s, senza nuova generazione. Arresto ordinato, exit0.
Questa prova verifica il trasporto durante il compute reale; non è un benchmark
di latenza LLM o una nuova qualifica vision del modello chat.

[Evidenze sintetiche](../performance/native-image-progress-20261110/verification.json);
log, fixture e output completi in `build/image-progress-20261110/`.
CUDA resta `NOT_RUN` su questo host; i test host/backend-neutral non sostituiscono
un'esecuzione CUDA. Nessuno speedup attribuito e nessuna qualifica generale
Metal o qualità T00 promossa da queste prove.

### Verifica del percorso CUDA

Nel ramo Linux, `Makefile:ds4-server-suite` collega gli stessi oggetti C del
progresso e di SSE con `ds4_numeric_cuda.o` e `ds4_numeric_cudnn.o`.
`ds4_server_suite.inc:suite_open` seleziona il backend numerico CUDA e
`suite_run_images` gli passa lo stesso observer usato su Metal. La chiamata
a `ds4_qwen_image_dit_forward` termina dopo `ds4_numeric_fence` anche senza
trace. `ds4_numeric_cuda.cu:gpu_fence` attende prima l'eventuale stream di
copia e poi quello di calcolo. Se un'attesa fallisce, il forward restituisce
l'errore prima del callback di progresso. L'heartbeat è emesso dal thread
della connessione, che rimane indipendente dalle attese CUDA.

Il dry run `make -Bn UNAME_S=Linux CUDA_ARCH=sm_86 ds4-server-suite` verifica
la selezione degli oggetti e delle librerie CUDA/cuDNN. Il dry run incrementale
con `-W ds4_qwen_image_observer.h` verifica la ricompilazione dei consumer
dell'header, compresi server e probe. L'header pubblico compila con Clang in
modalità C++17. Passano il test HTTP host e il test host del factory con il
ramo CUDA selezionato e le chiamate CUDA sostituite da stub. Build con `nvcc`
ed esecuzione su GPU NVIDIA restano `NOT_RUN` su questo host Apple Silicon.

Su Linux con la wheel aggiornata e un server avviato con `--cuda`, il test
seguente verifica discovery, ordine e avanzamento degli eventi, risultato
PNG e replay idempotente sul percorso JSON. È incluso anche nel target
`test-suite-agent-live`. Lo stesso test è passato sul server Metal isolato
in 74,013 secondi, con arresto del server riuscito.

```sh
DS4_SUITE_HTTP_URL=http://127.0.0.1:18095/v1 \
  agent/.venv/bin/python -m unittest -v \
  tests.test_suite_image_tools_e2e.NativeImageProgressE2E
```


## Revisione delle responsabilità

- Il worker pubblica un esito tipizzato e il thread HTTP costruisce l'evento
  SSE. Un messaggio di errore con testo `request_cancelled` resta un fallimento.
  La risposta di ripristino per la chat usa il percorso HTTP della chat.
  Il body terminale è inviato dal buffer già pubblicato dal worker.
- Il batch pubblica indice, fase e unità nello stesso snapshot. La natura
  della frazione segue il valore conservato, e il completamento del replay
  pubblica un avanzamento determinato.
- `OperationProgress` è uno snapshot immutabile e possiede le regole di
  ordinamento per SDK, bridge e frontend. Il client pubblica il completamento
  dopo la convalida del batch e della geometria. La discovery usa l'identità
  image selezionata nel client, anche per il modello autoregressivo simulato.
- `ToolEvent`, `SkillNode` e `ChatMessage` espongono confronto e traversata
  dell'esecuzione. Lo skeleton aggiorna soltanto le sue etichette e ricava
  la percentuale per immagine dal batch. Il timer si arresta all'esito.
- La revisione riguarda il trasporto e lo stato di presentazione. Il forward
  Image512, i kernel, il dispatch e le soglie numeriche corrispondono alle
  verifiche precedenti in questo documento.

I test HTTP host coprono il ripristino di una richiesta chat, la classificazione
degli esiti e il framing del risultato. Il dry run Linux, il factory CUDA
con stub e la sintassi C++17 sono stati ripetuti.
Il replay della patch sul commit base ricostruisce tutti i sorgenti SDK.
I 285 sorgenti Python installati corrispondono byte per byte al checkout.
La suite SDK completa registra 3685 pass, 88 skip e cinque fallimenti dei
benchmark riprodotti sul commit base, che registra 3670 pass e 88 skip.
I fallimenti riguardano compaction, interruzione del sandbox e import di un
modulo dal workspace del benchmark.

La seconda suite completa dell'agent registra 1217 pass e 49 skip live.
Due timeout del primo run parallelo passano isolatamente sia sul commit base
sia sulla versione finale. Le sette prove mirate della versione finale,
compresi due casi Playwright e i due timeout ripetuti, passano. Il report
della suite completa raggiunge il 99,88% prima delle regressioni aggiunte
per traversata, snapshot fuori ordine e timer. Il report combinato con
queste prove raggiunge il 100% su 6880 righe e 2222 rami con la soglia
originale invariata.

Con la wheel finale, il server Metal isolato supera la prova SSE e replay in
79,400 secondi, restituisce lo stesso PNG e termina con codice 0. Il tempo
descrive la singola prova di trasporto e include la preparazione del test.
Le verifiche Linux/CUDA restano statiche e con stub.
