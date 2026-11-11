# Servizio e client multimodale nativo

Implementazione T17/T18 del 22 ottobre 2026. Il servizio seriale CUDA usa lo
stesso worker delle conversazioni; il gestore ammette il lavoro, conserva i
checkpoint necessari e ripristina il LLM prima di pubblicare il risultato.
Qualifica e limiti nel [report](../performance/native-suite-t17-t18-20261022/README.md).

## Avvio

`ds4-server-suite` è un target esplicito, selezionato dal launcher Qwen27
`./agent/start-ds4-server.sh --manual` con MTP, visione e contesto 26000. Il launcher
LLM della root e quello Flash conservano i propri default. Servono i bundle BF16 e gli asset verificati
nei task precedenti; [la configurazione](../../config/native-suite.example.json)
mostra i percorsi della qualifica locale. I percorsi sono relativi alla directory
di avvio. `media_directory` deve esistere ed essere scrivibile; il server crea
una sottodirectory privata e la elimina dopo un arresto ordinato.

```sh
make ds4-server-suite \
  SUITE_MEDIA_PREFIX=performance-results/native-suite-t13-t14-20261021/dependencies/installed \
  SUITE_CUDA_LIBDIR=agent/.venv/lib/python3.13/site-packages/nvidia/cu13/lib \
  CUDNN_HOME=agent/.venv/lib/python3.13/site-packages/nvidia/cudnn
./ds4-server-suite -m gguf/Qwen3.8-27B-UD-Q4_K_S.gguf --cuda --no-mtp \
  --ctx 2048 --port 18091 --suite-config config/native-suite.example.json
```

Il catalogo può configurare uno o entrambi i modelli implementati. Non ammette
identità arbitrarie. Questa integrazione richiede CUDA single GPU e worker
seriale; avvii SSD, TP, distribuiti e batch vengono rifiutati prima dell'apertura
della suite. I percorsi legacy continuano a usare i loro contratti.

## SDK standard

Un solo client e una sola base URL servono chat, immagini ed embedding:

```python
from openai import OpenAI

with OpenAI(base_url="http://localhost:18091/v1", api_key="local", timeout=600) as client:
    vectors = client.embeddings.create(
        model="google/embeddinggemma-2", input=["ciao", "hello"], dimensions=256
    )
    images = client.images.generate(
        model="Qwen/Qwen-Image-2.1-Turbo", prompt="Un quadrato rosso", n=2,
        size="256x256", extra_headers={"Idempotency-Key": "example-image-1"},
        extra_body={"ds4": {"version": 1, "seed": 17}},
    )
    with open("reference.png", "rb") as reference:
        edited = client.images.edit(
            model="Qwen/Qwen-Image-2.1-Turbo", image=reference,
            prompt="Trasforma il quadrato in un cerchio", size="256x256",
        )
```

`size` appartiene al protocollo standard. L'helper ExMekh e i tool espongono
invece `width: ImageDimension` e `height: ImageDimension`, due Literal indipendenti. PNG è l'unico formato output implementato;
`quality`, `style`, `mask` e campi non mappati producono errore esplicito.
Larghezza/altezza devono essere multipli di 32 fra 32 e 2048; `n` è fra 1 e 8,
con al massimo 4.194.304 pixel totali. Una richiesta `n > 1` usa lo stesso
prompt e gli stessi riferimenti, con seed consecutivi e PNG restituiti in
ordine. La pipeline nativa esegue il denoising in gruppi di massimo due
immagini. I gruppi condividono conditioning e pesi e la dimensione scende
a uno se il piano a due supera il budget. La decodifica VAE procede per
singola immagine. Client, helper e tool ExMekh usano 2048×2048 per default; accettano le dimensioni minori richieste esplicitamente.
La descrizione del tool suggerisce 2048×1536 per paesaggi, 1536×2048 per
ritratti e 2048×1152 per 16:9. Per più immagini 2048×2048 servono chiamate
separate, dato il limite totale del batch. Il default del protocollo HTTP
resta indipendente. La recipe Turbo resta otto step e CFG 1.
I riferimenti di editing mantengono la geometria fornita (`reference_resolution=0`):
devono essere allineati e già ammessi dal piano Qwen3-VL. Geometrie che richiedono
un resize implicito vengono rifiutate prima del compute; per esempio usare
riferimenti 256×256, non l'output diagnostico 64×96, per il round trip di editing.

## Client tipizzato ExMekh

`Client.native_suite()` usa il medesimo oggetto OpenAI di `LocalProvider`.
Le identità implementate sono `Literal`; essendoci un solo modello per ciascuna
operazione, i metodi non chiedono di sceglierlo. `models()` restituisce soltanto
le voci native tipizzate; la selezione dei modelli chat filtra le capability.

```python
from pathlib import Path
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.native_suite import EmbeddingItem, TextContent
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.providers.local import LocalProvider

with LocalProvider(model="Qwen3.8-27B-UD-Q4_K_S", base_url="http://localhost:18091/v1") as provider:
    client = Client(provider, SystemMessage("Help the user."))
    suite = client.native_suite()
    image = suite.upload(Path("reference.png"), kind="image")
    video = suite.upload(Path("clip.mp4"), kind="video")
    video.audio = "exclude"
    audio = suite.upload(Path("speech.wav"), kind="audio")
    vectors = suite.embeddings([
        EmbeddingItem(content=[TextContent(text="Descrizione"), image, video, audio])
    ], dimensions=256, task="similarity")
    images = suite.images("", images=[image], n=2, width=256, height=256, seed=17)
    Path("output.png").write_bytes(suite.content(images.data[0].ds4.file_id))
```

Ordine delle parti e dei riferimenti preservato; un vettore per elemento.
Testo/token standard restano usabili senza helper. L'estensione multimodale usa
`ds4.version=1`, non captioning o trascrizione intermedi. Video con audio incluso
richiede una traccia valida; overflow oltre 8192 token produce errore.

## Tool e frontend

`generate-image` è la capability registrata nel catalogo ExMekh. Accetta testo,
riferimenti ordinati `list[str]` di percorsi
nel workspace oppure entrambi; usa `n`, `width`, `height`, PNG, background,
seed uint64 e chiave idempotente. Salva gli output originali in
`.exmekh_lib/generated-media/` e restituisce percorso, file ID, seed, dimensioni
e recipe. Le opzioni native sono validate dal client prima degli upload;
i riferimenti devono essere file esistenti. La risposta deve rispettare numero
e dimensioni richiesti. `GeneratedImage.png_bytes()` possiede decodifica base64
e verifica PNG; il salvataggio usa questa API e verifica l'intero batch prima
di creare la directory degli output, anche con anteprime disabilitate.
`include_image_in_result=True` allega anche le immagini al risultato
tool; `preview_scale` nell'intervallo `(0,1]` riduce solo le anteprime.

Le risposte strutturate passano dal renderer XML esistente. L'allegato usa il
percorso già previsto da `ToolMessage`: la UI mostra miniature nella risposta
assistant, dialog e persistenza dopo reload. Il worker dei tool esegue il lavoro
fuori dal loop UI e propaga il client nativo anche attraverso i wrapper del
provider. Mostrare un'immagine nella UI non avvia automaticamente un encoder LLM.
L'eventuale allegato al risultato può essere elaborato dal provider solo secondo
le sue capability vision, come ogni altro allegato esplicito.

Il ranking web usa gli embedding nativi per testo e immagini nello stesso
spazio; OpenVINO e i due precedenti modelli di ranking non vengono caricati.
Gli embedding sono interni alla skill di ricerca web e all'API tipizzata;
non esiste una capability `embed-content` esposta al modello.
Se il servizio non è disponibile, restano i fallback dichiarati dal ranking.

## Accettazione live della libreria e del frontend

La suite completa usa la wheel installata nell'agente e un solo server reale,
già avviato. La fixture MP4 sintetica con traccia audio è versionata con
checksum in `tests/fixtures/native-suite-e2e/`: PyAV serve soltanto al suo
generatore offline, non ai test. `DS4_SUITE_HTTP_VIDEO` può indicare un altro
MP4 con traccia audio valida.
Non avviare un altro processo modello per questi test.

```sh
DS4_SUITE_HTTP_URL=http://127.0.0.1:8080/v1 \
DS4_SUITE_E2E_ARTIFACTS=/absolute/path/to/new-evidence \
make test-suite-agent-live SUITE_PYTHON=agent/.venv/bin/python
```

`tests/test_suite_image_tools_e2e.py` copre testo, editing con riferimenti
ordinati, input solo immagine, output multipli, alpha, seed, retry idempotente,
anteprime e il paesaggio 1024×768. Un LLM con history nuova riceve soltanto
i pixel generati e deve riconoscere colore e forma, senza conoscere il prompt.
`tests/test_suite_embedding_tools_e2e.py` attraversa `Client.native_suite()` e
i consumer reali del ranking: dimensioni/task, quattro modalità, ordine del
batch e delle parti, audio incluso/escluso dal video, cache esplicita,
validazione/overflow, ripresa cached della conversazione. Le fixture del ranking
sono locali; trasporto ed embedding usano il servizio reale.
`agent/tests/test_native_image_generation_e2e.py` aggiunge browser → DeepAgent
→ tool → GPU → allegati vision, responsività e persistenza dopo reload.
Il target rifiuta l'avvio senza URL; errori dei modelli e del restore fanno
fallire l'accettazione.

Per un controllo funzionale degli allegati su CUDA o Metal, usare un server
nativo già avviato con un LLM vision e Qwen Image abilitato all'editing. Il test
prova le tre destinazioni con un upload, con **Save & regenerate** e con il
pulsante sotto un'immagine generata. Verifica che il turno termini senza errori
e che l'editing si completi fra la richiesta LLM iniziale e la sua continuazione.

```sh
cd agent
DS4_SUITE_HTTP_URL=http://127.0.0.1:8080/v1 \
DS4_SUITE_LLM_MODEL=Qwen3.8-27B-UD-Q4_K_S \
  uv run --no-sync pytest -o addopts= --run-live -x -q -s \
  tests/test_native_image_generation_e2e.py::test_live_image_attachment_routes
```

## Lifetime e osservabilità

Il [contratto del progresso image](native-image-progress.md) definisce SSE,
heartbeat, timeout e percentuale sullo skeleton come contratto comune per
CPU, Metal, CUDA e backend futuri, valido anche per i modelli image diversi
da Qwen Image che verranno integrati. SSE v1 è opt-in tramite
`Accept: text/event-stream`; il percorso JSON resta compatibile. Lo skeleton
mostra solo percentuale e tempo. `image_timeout_seconds` configura la deadline
assoluta image, inclusa la coda, e altrimenti eredita `timeout_seconds`.
Risultati delle prove nel [report](native-image-progress-implementation-20261110.md).

Discovery espone revision/profilo, configured/resident/ready, componenti e limiti.
`GET /v1/models` e `GET /v1/models/{id}` per i modelli nativi rispondono
durante prefill, decode e operazioni della suite, senza entrare nella coda
di inferenza. Entrambi leggono lo stesso snapshot posseduto dal catalogo.
Quando l'inferenza possiede gli adapter, la risposta usa l'ultimo snapshot
coerente del catalogo. Il worker aggiorna lo snapshot dopo ogni operazione
della suite e la discovery lo aggiorna quando gli adapter sono disponibili.
I campi di residenza descrivono lo stato osservato nello snapshot.
Le risposte riportano request ID, piano/vittime, byte device recuperati, cache
hit, token delle modalità, copie H2D/D2H native e tempi. I contatori copie non
includono traffico interno ai kernel o restore legacy; i tempi separati Gemma
non disponibili sono `null`.

File e output sono su disco: massimo 256 file, 512 MiB per file, quota predefinita
4 GiB, TTL 3600 secondi rinnovato quando il file è usato per il compute. Ricevute:
64 slot, massimo 64 MiB ciascuna e 600 secondi dal commit. JSON è limitato a
64 MiB; multipart è spooled su file e mappato, con limite aggregato 512 MiB più
64 KiB di framing. Il servizio ammette al massimo otto connessioni quando la
suite è attiva. I download usano blocchi da 64 KiB.

La chiave idempotente identifica endpoint e corpo canonico; le immagini di
editing sono confrontate per hash dei byte. Il retry di un commit ritorna
risultato e request ID originali; un corpo diverso produce 409. La cache è
valida nella vita del processo, non costituisce persistenza fra riavvii.
L'helper assegna automaticamente una chiave alla chiamata images quando manca,
per consentire i retry di trasporto del medesimo client.

Disconnessione e deadline annullano cooperativamente il lavoro. Non si pubblica
successo prima del restore; se il restore fallisce l'operazione conserva gli
handle e i checkpoint e il worker tenta il recupero prima di accettare altro
compute. Il client non deve gestire load, unload o checkpoint. I file server
sono temporanei: il client conserva su disco gli output necessari oltre il TTL.


## Geometria delle immagini

Il client e il tool `generate-image` espongono due `ImageDimension` Literal
indipendenti, con tutti i 64 multipli di 32 da 32 a 2048 nello schema.
Non ci sono coppie riservate; il batch deve rispettare
`n * width * height <= 4194304`. La validazione locale precede gli upload.
Questi sono limiti operativi di DS4: il checkpoint Qwen pubblica anche preset
più larghi, fino a 2752×1536. Non sono dichiarati dal contratto DS4 corrente.
1920×1080 non è allineato a 32; 2560×1440 supera il limite DS4 per lato.
Per 16:9 sono supportati, per esempio, 1024×576 e 2048×1152.
La regressione host verifica tutte le 4096 coppie contro il parser del server;
la regressione client verifica gli enum del tool caricato e il trasporto.
