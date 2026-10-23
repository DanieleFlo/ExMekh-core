# Suite nativa: build, streaming e accettazione

La suite usa un server C/CUDA e un client configurato con una sola base URL.
ExMekh resta il frontend agentico. Il runtime non avvia Python, Transformers,
Diffusers, ASR o server modello esterni. Python serve agli strumenti offline,
agli oracle e agli esempi SDK.

Il [report T19–T22](../performance/native-suite-t19-t22-20261022/README.md)
distingue implementazione, prove locali e gate di rilascio. La qualità image
T11 resta un prerequisito aperto: una build funzionante non certifica da sola
tipografia, layout, preservation e alpha sul corpus completo.

## Modelli e profili

| Operazione | Modello / profilo | Target locale |
| --- | --- | --- |
| Generazione/editing/PNG RGBA | Qwen/Qwen-Image-2.1-Turbo, `bf16-native-v1` | CUDA sm_86; gate qualità T11 aperti |
| Testo/image/video/audio/interleaving | google/embeddinggemma-2, `bf16-native-v1` | CUDA sm_86; dimensioni 128/256/512/768 |
| Ripresa conversazione | Qwen3.8-27B UD-Q4_K_S | CUDA single GPU; MTP esterno verificato separatamente |

Revisioni, dimensioni e SHA-256 sono nei manifest
[`config/suite-bundles/`](../../config/suite-bundles/) e nel
[lock T00](../performance/native-suite-t00-20261020/models.json).
Il catalogo pubblico conserva BF16. Q8 storage con compute BF16, W8A8,
GGUF DiT e layout misti sono candidati distinti nel
[registro](../../config/native-suite-candidates.json); nessuna conversione o
precisione alternativa viene scelta implicitamente. La valutazione conserva
`KEEP`, `REJECT` e `NEED_MORE_DATA`, con i gate mancanti.

## Preparazione offline e build

Scaricare solo gli artefatti del manifest esplicito; il downloader verifica
dimensioni e checksum e conserva i download incompleti. Per ciascun bundle:

```sh
python3 tools/suite_bundle.py download \
  --manifest config/suite-bundles/qwen-image-2.1-turbo-bf16.json --root models
python3 tools/suite_bundle.py build \
  --manifest config/suite-bundles/qwen-image-2.1-turbo-bf16.json \
  --root models --output bundles/qwen-image
python3 tools/suite_bundle.py download \
  --manifest config/suite-bundles/embeddinggemma-2-bf16.json --root models
python3 tools/suite_bundle.py build \
  --manifest config/suite-bundles/embeddinggemma-2-bf16.json \
  --root models --output bundles/gemma
python3 tools/compile_qwen_image_tokenizer.py \
  models/Qwen/Qwen-Image-2.1-Turbo/processor/tokenizer.json assets/tokenizer.ds4q
python3 tools/compile_embedding_gemma2.py \
  models/google/embeddinggemma-2 assets/gemma.ds4g
```

Il processor media si prepara con `tools/compile_gemma_media.py --prepared`
usando il corpus di sorgenti fissato; la [guida T13/T14](../performance/native-suite-t13-t14-20261021/README.md)
descrive la preparazione e gli asset verificati. Il pacchetto locale include
gli asset già compilati e i due indici, senza duplicare i pesi.

Servono compilatore C11, pthread, toolkit CUDA, cuBLAS/cuBLASLt, cuDNN 9 e le
librerie native FFmpeg (`avformat`, `avcodec`, `avutil`, `swscale`,
`swresample`) e soxr. La build dei codec è riproducibile con
`tools/build_suite_media_dependencies.py`; sorgenti e licenze sono fissati
in `third_party/ffmpeg/` e `third_party/soxr/`.

```sh
make -j2 ds4-server-suite CUDA_ARCH=sm_86 \
  SUITE_MEDIA_PREFIX=/absolute/path/to/native-media \
  SUITE_CUDA_LIBDIR=/absolute/path/to/cuda-libraries \
  CUDNN_HOME=/absolute/path/to/cudnn
./ds4-server-suite -m gguf/Qwen3.8-27B-UD-Q4_K_S.gguf --cuda --no-mtp \
  --ctx 2048 --port 18095 --suite-config config/native-suite.example.json
```

La qualifica locale usa header/toolkit CUDA 12.4 e librerie native CUDA 13;
il manifest `native-dependencies.txt` del pacchetto registra le dipendenze
effettive, incluse quelle legacy. I percorsi locali sotto una virtualenv
contengono file `.so`: il servizio li carica direttamente. Un'installazione
nativa equivalente può fornire gli stessi file senza un interprete Python.

Per l'agente locale, `./agent/start-ds4-server.sh` avvia la suite sulla porta
8080 con contesto 26000, MTP e visione. `DS4_SUITE_CONFIG` sceglie il JSON dei
bundle. Copiare `config/native-suite.env.example` in `config/native-suite.env`
e impostare i tre percorsi delle dipendenze native; le variabili già fornite
dal chiamante hanno precedenza. Il launcher compila lo stesso target che avvia.

Metal, ROCm, TP/distribuito, SSD LLM e batching suite restano fuori dalla nuova
qualifica CUDA. Il launcher LLM della root e quello Flash mantengono default e dipendenze.
Il server suite rifiuta combinazioni non qualificate prima del compute.

## Policy e misure

Il planner valuta l'intera operazione, incluso il ritorno al LLM. Considera
prima i piani senza vittime; poi cache, sessioni e pesi con costi espliciti.
Le varianti image coprono DiT residente, gruppi mantenuti e gruppi streamed,
prefix cache device/host e componenti seriali. I limiti si ricavano dai byte
reali dei gruppi e dal workload; non esiste un numero fisso di blocchi per GPU.
Gemma applica lo stesso contratto ai rami testo, vision e audio.

Nel servizio HTTP i pesi Gemma restano disponibili per tutte le parti della
richiesta e vengono rilasciati prima della pubblicazione. Il solo footprint
della sessione LLM non include i suoi futuri transitori di prefill/visione:
conservarvi anche cache Gemma poteva causare OOM al turno successivo con
contesto 26000 e MTP. La conservazione fra richieste attende un bound completo
del working set LLM; resta uno studio di ottimizzazione aperto. Le cache host
dei media e la residenza del planner nativo standalone conservano i propri
contratti.

Una coda persistente legge il mapping in due slot pinned di massimo 2 MiB
ciascuno. Il budget pinned può ridurre questi slot; sotto 64 KiB per slot resta
il caricamento sincrono da mmap. Eventi CUDA separano disponibilità e riuso
degli slot; il rilascio attende anche l'ultimo uso compute. Il registro addebita
entrambi gli slot prima di usarli. Un hint CPU di massimo 16 MiB riguarda
soltanto un bundle configurato, dopo la validazione completa dell'operazione;
la cancellazione ferma gli hint successivi e attende il worker.

Il percorso capiente conserva il DiT per gli otto passi, senza ritrasferirlo.
Lo streaming conserva i gruppi selezionati dal budget e ricarica gli altri a
ogni visita. La conservazione di cache fra richieste segue i piani ammessi e
la riserva di restore. Non sono promossi wavefront encoder→DiT/VAE→LLM,
step skipping, cache DiT approssimate, KV lossy o CUDA Graph con indirizzi mobili.

Con `DS4_SUITE_RESOURCE_METRICS=1` nel processo server, `ds4.resources`
fotografa le risorse dopo il cleanup della richiesta. La diagnostica è
disattivata per le risposte ordinarie, così i risultati dei tool restano
compatti nel contesto LLM. Gli
array dichiarano l'ordine dei ledger; owner `0` è il totale, `500` il contesto
nativo, `501/502` i bundle, `503` il processor/cache media. MMAP non è RSS;
reserved non è memoria già allocata. Le copie native escludono il traffico
dei kernel e il reload legacy. Le stime dei piani sono bound, non picchi misurati.
Quando un tensore CUDA viene liberato, il backend aggiorna il campione globale
del registro condiviso. Il prefill LLM successivo deve vedere la memoria
fisicamente tornata libera, anche se nessuna richiesta LLM precedeva il tool.
Anche i buffer visivi allocati durante il caricamento di un payload appartengono
alla sessione ripristinata: attribuirli al chiamante riduceva il budget dichiarato
dal successivo park/restore. La regressione CUDA ripete il ripristino tre volte
entro la capienza esatta dichiarata e confronta continuazioni bit-exact.

Per la consegna del 22 ottobre sono completati i controlli funzionali; gli
studi di ottimizzazione restano aperti su richiesta dell'utente: wavefront,
confronti di overlap, quantizzazione, fusioni e benchmark comparativi finali.
Il solo profilo pubblico resta `bf16-native-v1`.

`DS4_SUITE_TRANSFER_TRACE=/path/trace.jsonl` registra frontiere read/copy/riuso
per diagnosi. `DS4_SUITE_TRANSFER_SYNC=1` serve al confronto attribuibile con
il caricamento sincrono. I benchmark finali escludono la trace e riportano
almeno cinque campioni dopo warm-up; CV oltre 5% resta `NEED_MORE_DATA`.
I tempi dello stress funzionale non sostituiscono questi benchmark.

## Ricette e accettazione

La [guida client](native-suite-client.md) contiene generazione, editing,
background e upload ordinato delle quattro modalità con un solo client.
Gli endpoint standard sono Images/Embeddings/Files; `ds4.version=1` dichiara
seed e input embedding multimodali. Input oltre contesto, geometrie non
supportate e parametri ignorati nominalmente vengono rifiutati.

```sh
make test-resources test-executor test-transfer test-numeric-lifetime \
  test-suite-http test-bundle test-suite-release
DS4_SUITE_HTTP_URL=http://127.0.0.1:18095/v1 \
  agent/.venv/bin/python -m unittest tests.test_suite_http_live \
  tests.test_suite_exmekh_live -v
PYTHONPATH=performance-results/native-suite-t13-t14-20261021/development \
  agent/.venv/bin/python tools/qualify_suite_release.py \
  --url http://127.0.0.1:18095/v1 --cycles 50 --output /new/stress-directory
```

Lo stress usa tool registrati, call ID e continuazioni Responses: verifica
LLM → image → LLM → batch delle quattro modalità più composizione → LLM,
PNG e vettori deterministici, cached token, prenotazioni restituite e stato
bounded dell'owner nativo. Una fixture H264 può essere passata con `--video`;
in assenza viene generata dal solo ambiente di sviluppo PyAV.
Per lo stress avviare il server con `DS4_SUITE_RESOURCE_METRICS=1`;
il driver rifiuta di qualificare la memoria se mancano i ledger.

`tests/probe_suite_phase_server` aggiunge soltanto nella build di qualifica
una barriera alle frontiere native. Il driver
`tools/qualify_suite_disconnects.py` chiude il socket a conditioning, DiT,
latenti e frontiere VAE, poi verifica continuazione cached, risorse restituite
e assenza di output parziali. La generazione completa finale deve avere lo
stesso hash della prima. Il server ordinario non compila questa barriera.
Il driver usa gli estremi DiT; `--all-layers` estende il distacco a tutti i
layer e passi, mentre i test nativi coprono già ogni frontiera.

Per confrontare la stessa fixture image senza LLM, usare
`tests/probe_qwen_image_pipeline` con `DS4_SUITE_CASES_ONLY=1` e ripetere con
`DS4_SUITE_DEVICE_LIMIT=4294967296`: il probe registra piano, payload H2D,
picco posseduto e tutti gli otto latenti. Il budget controllato di 4 GiB su
una 3090 non qualifica una scheda fisica da 4 GiB. I test
`tests/test_executor_cuda` e `tests/test_lifecycle_cuda` esercitano parcheggio,
reload e ripresa bit-exact con i LLM locali, separatamente dagli oracle image.

## Pacchetto e recupero

`tools/package_native_suite.py` copia eseguibile, codec nativi, indici,
tokenizer/processor, manifest e un albero sorgenti senza oggetti precompilati.
Non copia i pesi: la directory `models/` deve contenere gli artefatti dei due
manifest. `run-suite` risolve le librerie locali e quelle indicate da
`DS4_NATIVE_LIBDIR`, poi avvia direttamente il server C.

Il packager richiede `PASS` per qualità image/embedding, residenza, guasti,
50 alternanze, regressioni e build pulita. Un pacchetto incompleto è possibile
solo con `--development` e conserva `release_qualified=false` e l'intera matrice.
La preparazione di un pacchetto di sviluppo non chiude T21/T22.
La directory finale viene pubblicata con una rinomina soltanto dopo copia e
manifest completi. Un errore elimina lo staging temporaneo e permette di
riprovare con la stessa destinazione; una destinazione esistente viene rifiutata.

```sh
python3 tools/package_native_suite.py --development \
  --media-prefix /absolute/path/to/native-media \
  --qualification docs/performance/native-suite-t19-t22-20261022/qualification.json \
  --candidates docs/performance/native-suite-t19-t22-20261022/candidates.json \
  --output /absolute/path/to/new-package
ln -s /absolute/path/to/verified-model-root /absolute/path/to/new-package/models
DS4_NATIVE_LIBDIR=/absolute/path/to/cuda-libraries:/absolute/path/to/cudnn/lib \
  /absolute/path/to/new-package/run-suite \
  -m /absolute/path/to/Qwen3.8-27B-UD-Q4_K_S.gguf --cuda --no-mtp \
  --ctx 2048 --port 18095
```

La directory locale verificata è
`performance-results/native-suite-t19-t22-20261022/package-functional/`.
Il collegamento `models` serve alla prova locale e non include i pesi nel
manifest di distribuzione. Le prove e gli hash del binario dello stress e
del binario ricompilato sono conservati separatamente.

In caso di restore fallito il servizio conserva checkpoint e handle, espone
indisponibilità e tenta solo cleanup/restore prima di altro compute. Il client
non deve impartire load/unload né ripetere una generazione parzialmente
avanzata. Non eliminare manualmente i checkpoint di una sessione in recupero.
Per un errore media controllare codec, tipo dichiarato, geometria e quota;
per un errore di ammissione confrontare bound, ledger e memoria globale.

Provenienza: i nuovi moduli C riusano le primitive DS4 e la coda host esistente;
non importano orchestrazione Studio o nuovi kernel esterni. Il
[registro](../performance/native-suite-t19-t22-20261022/provenance.json) conserva
fonti di progetto, confini e riferimenti ai lock delle dipendenze.
