# Contratti della suite nativa DS4, versione 1

Stato: specifica congelata per T01, 20 ottobre 2026; implementazione HTTP
T17/T18 del 22 ottobre nel target esplicito `ds4-server-suite`, CUDA single GPU.
Il [client e i limiti operativi](native-suite-client.md) e il
[report di qualifica](../performance/native-suite-t17-t18-20261022/README.md)
distinguono disponibilità del servizio e qualifica numerica dei componenti.
Il contratto C è [ds4_runtime.h](../../ds4_runtime.h); lo schema delle richieste
è [ds4-suite-v1.schema.json](../../schemas/ds4-suite-v1.schema.json).

## Identità e discovery

Il catalogo distingue `model_id`, `revision`, `profile` e `component`. La
revisione identifica i pesi e i processor; il profilo identifica precisione e
recipe. Lo stato `configured` non implica `resident`, `ready` o `qualified`.
`GET /v1/models` mantiene l'envelope OpenAI `object=list,data=[...]`; ogni
elemento aggiunge un oggetto `ds4` con versione 1, revision, profile,
operations, input_modalities, context_tokens, embedding_dimensions,
configured, resident, ready e qualified_backends. Non viene riportata una
capability non implementata come disponibile. Gli ID e i default LLM attuali
rimangono validi. Un profilo lossy si sceglie esplicitamente come ID catalogato.
Lo schema catalogo è `schemas/ds4-suite-catalog-v1.schema.json`. `components`
elenca id, optional, modalità, storage_format, compute_dtype, host_loaded e
device_resident. `resident` significa che tutti i componenti richiesti dal
profilo sono residenti sul device; `ready` significa utilizzabile attraverso
un piano ammesso e può essere true con pesi streamed. La lista contiene i componenti abilitati dal profilo, inclusi quelli opzionali
per singole richieste: tutti contribuiscono a resident. Componenti esclusi dal
profilo non compaiono nella lista. optional non significa resident escluso. `embedding_dimensions` è un array vuoto
per operazioni che non producono vettori. `qualified_backends` identifica anche
la piattaforma qualificata, per esempio `cuda-sm86`, e può essere vuoto.
Il bridge C legacy usa `local-gguf` come etichetta non crittografica: non è una
revisione pubblica certificata. La futura discovery deve usare il manifest
verificato, non pubblicare quell'etichetta come hash dei pesi.

## Operazioni e input standard

| Operazione | Metodo e percorso | Client |
| --- | --- | --- |
| Chat | POST `/v1/chat/completions`, POST `/v1/responses` | Client e contratti attuali |
| Generazione | POST `/v1/images/generations` | `client.images.generate` |
| Editing | POST `/v1/images/edits`, multipart | `client.images.edit` |
| Embedding | POST `/v1/embeddings` | `client.embeddings.create` per testo/token |
| Upload | POST `/v1/files`, multipart con `file` e `purpose=ds4_media` | Helper o client files |
| Download | GET `/v1/files/{id}/content` | Byte originali/output |

Le immagini usano `model`, `prompt`, `n` (default 1, intervallo 1–8), `size` (default
1024x1024), `output_format` (default png), `background` (default auto).
Il supporto concreto a size/background dipende dalla capability del modello:
dimensioni non supportate vengono rifiutate, non arrotondate. `ds4` versione 1
consente seed uint64 (default casuale), profilo di sampling (default
checkpoint), ma non un cambio implicito di precisione. La recipe Turbo usa
gli otto step e CFG 1 del bundle; la recipe effettiva viene restituita.

Editing accetta uno o più file `image`/`image[]`, nell'ordine multipart,
oppure riferimenti DS4 ordinati nel corpo JSON tramite l'estensione.
Il corpo JSON di editing aggiunge `images:["file-id",...]` e richiede
`ds4.version=1`. Il multipart standard viene normalizzato nello stesso ordine;
in assenza di opzioni DS4 il server assegna la versione 1, senza richiederla al
client standard. Gli altri campi e default coincidono con la generazione.
Non accetta una maschera inpainting finché la capability non è implementata:
`mask` non viene mai ignorata. `quality`, `style`, `user` e altri parametri SDK
non mappati ricevono `unsupported_parameter`, non un significato inventato.
JPEG non può essere richiesto con background transparent. Un'opzione di
trasparenza non garantisce che il modello produca ogni pixel alpha atteso:
la recipe e la qualità sono verificate separatamente.

Embedding standard accetta stringa, array di stringhe, array di token o array
di array di token, tutti non vuoti. I token devono appartenere al tokenizer
del modello selezionato. `dimensions` è opzionale: default nativo 768 per
EmbeddingGemma 2; valori consentiti 768/512/256/128. `encoding_format` è float
(default) o base64. Base64 codifica float32 IEEE-754 little-endian, senza
header. Gli embedding vengono L2-normalizzati dopo l'eventuale troncamento.
Le stringhe attraversano tokenizer/processor della revisione fissata, inclusi
i suoi BOS/EOS. Gli array di token sono sequenze complete del modello: DS4 non
aggiunge BOS/EOS o prefissi. Per questo input si ammette solo task raw e nessun
title; task testuali diversi richiedono stringhe. Padding esplicito segue il
pad ID del modello ed è escluso dalla media; input interamente padding è
invalid_request. Token strutturali media senza i relativi contenuti vengono
rifiutati prima del compute, non trasformati in media impliciti.

## Estensione embedding multimodali

La versione è dichiarata da `ds4.version=1`; il campo `input` contiene un
array non vuoto di elementi `{content:[...]}`. Questa è un'estensione DS4,
non un input standard dell'API OpenAI. L'helper ExMekh nasconde upload e
serializzazione; una chiamata HTTP valida non richiede testo o placeholder
fittizi per gli input media-only.

```json
{
  "model": "google/embeddinggemma-2",
  "input": [
    {"content": [
      {"type": "text", "text": "Descrizione del prodotto"},
      {"type": "image", "file_id": "file-123"},
      {"type": "video", "file_id": "file-456", "audio": "exclude"},
      {"type": "audio", "file_id": "file-789"}
    ]}
  ],
  "dimensions": 768,
  "ds4": {"version": 1, "task": "similarity", "overflow": "error"}
}
```

Ogni elemento produce un vettore; i contenuti sono ordinati e compongono una
sola sequenza. Il video viene campionato secondo il processor della revisione
fissata; `audio=exclude` è il default. `audio=include` aggiunge esplicitamente
anche la traccia sonora completa. Nella versione 1 ogni parte video viene
espansa in sequenza frame video, poi audio, prima della parte content successiva;
entrambe le tracce conservano l'origine temporale del contenitore. Non viene
inventato un interleaving temporale frame/finestra audio. Se il file non
contiene audio la richiesta viene rifiutata prima del compute. Non si
mescolano implicitamente audio e video, né si sostituisce un encoder con ASR
o captioning. Un video composto non equivale alla media di vettori di frame.

`ds4.task` è raw (default, nessun prefisso), search_query, document,
code_query o similarity. I prefissi si applicano una sola volta ai contenuti
testuali; document consente `ds4.title` e usa `none` in sua assenza. Con raw,
il testo è fornito esattamente al processor. Con task diversi da raw, il
client fornisce testo non preformattato: prefissi già presenti vengono
rifiutati per evitare duplicazione. Media-only non richiede un task testuale.

`overflow=error` è il default e l'unico comportamento della versione 1:
superare 8192 token operativi, incluse le modalità media, produce errore.
Segmentazione futura richiederà un contratto versionato con segmenti,
timestamp e una risposta esplicita. Non sono previste medie o troncamenti
silenziosi. Il fatto che config.json riporti una capacità posizionale maggiore
non estende il limite operativo del bundle.

## Risultati, idempotenza e completamento

Embedding mantiene `{object:"list",data:[{object:"embedding",index,embedding}],
model,usage}`. L'indice corrisponde all'ordine degli elementi. Usage conta il
lavoro effettivo del processor e distingue metadati media in `ds4.usage`.
Immagini mantengono `{created,data:[{b64_json}],...}`; metadati DS4 includono
file_id, revision, profile, seed e recipe. URL opzionali devono riferirsi a
media riutilizzabili, non a percorsi locali arbitrari del server.

Ogni risposta possiede `x-request-id`. `Idempotency-Key` è opzionale:
stessa chiave, stesso modello/profilo e corpo canonico restituiscono lo stesso
risultato committato; corpo diverso produce HTTP 409. Un risultato resta
memorizzato per 600 secondi dopo il commit, poi il riuso della chiave è una
nuova richiesta. Il limite di cache deve essere esplicito e bounded;
l'esaurimento prima del commit produce errore, non perdita dell'idempotenza.
Il media referenziato deve vivere almeno quanto la ricevuta. Una richiesta
in corso con stessa chiave non viene eseguita due volte.

La transazione termina con successo solo dopo il ripristino dello stato
sospeso. Progresso/preview non costituiscono commit; il risultato finale non
viene inviato prima del restore. Il client non controlla checkpoint o memoria.
La cancellazione è cooperativa alle frontiere valide, drena le copie e tenta
il ripristino. Il fallimento di restore conserva checkpoint e identità della
sessione e ne impedisce l'uso finché non è recuperata.

| Errore | HTTP | Codice |
| --- | --- | --- |
| Campo/schema non valido | 400 | invalid_request |
| Modello inesistente | 404 | model_not_found |
| Modello noto ma non pronto/artefatti invalidi | 503 | model_not_ready |
| Operazione o backend non supportato dal modello | 400 | unsupported_operation / unsupported_model_backend |
| Capability o parametro mancante | 400 | unsupported_parameter / unsupported_modality |
| Codec/contenitore non decodificabile | 400 | unsupported_media_format |
| Input oltre contesto | 400 | context_length_exceeded |
| Budget non ammissibile prima di mutazioni | 503 | insufficient_resources |
| Chiave idempotente in conflitto | 409 | idempotency_conflict |
| Inferenza fallita | 500 | inference_failed |
| Ripristino fallito | 503 | restore_failed |
| Richiesta annullata | 409 | request_cancelled |

Errori usano `{error:{message,type,code,param},ds4:{version:1,request_id,
recovery_required}}`. Testi del prodotto appartengono alle risorse del dominio;
enum di stato interni non incorporano messaggi. I campi non riconosciuti sono
rifiutati. Errori di validazione e ammissione non mutano sessioni.
NOT_RESIDENT è uno stato interno che induce il piano a caricare i gruppi:
non è un errore HTTP quando il caricamento è ammissibile. Un fallimento di
caricamento diventa model_not_ready o insufficient_resources secondo la causa.

Upload accetta fino a 512 MiB per file. I MIME v1 sono PNG, JPEG, WebP,
MP4, WebM, WAV, MPEG audio, FLAC e Ogg, elencati nello schema; il decoder
verifica i byte, non si fida dell'estensione. Le capability dichiarano i
codec effettivamente decodificabili. Un formato ammesso ma non disponibile
produce `unsupported_media_format`. Il nome è solo metadato, mai un percorso.
Il media vive 3600 secondi dall'upload, con scadenza rinnovata quando è usato
dal compute. La ricevuta dura meno del TTL del media; l'operazione seriale
impedisce che la scadenza elimini file in uso. Il frontend conserva su disco
gli output e nelle proprie history gli allegati necessari oltre il TTL server.
Il servizio non conserva riferimenti file ID in conversazioni persistenti. File scaduti
producono HTTP 404 `media_not_found`. Quota e limite richieste sono
configurazioni server; la discovery li restituisce esplicitamente.

## Esempi client e matrice delle capability

Le chiamate seguenti usano il servizio T17. Esempi completi con una sola base
URL e con il client tipizzato sono nella [guida client](native-suite-client.md).
Con un client OpenAI configurato su DS4:

```python
client.images.generate(model="Qwen/Qwen-Image-2.1-Turbo", prompt="Un quadrato rosso")
with open("input.png", "rb") as image:
    client.images.edit(model="Qwen/Qwen-Image-2.1-Turbo", image=image,
                       prompt="Trasforma il quadrato in un cerchio")
client.embeddings.create(model="google/embeddinggemma-2", input=["ciao", "hello"],
                         dimensions=512, encoding_format="float")
with open("clip.mp4", "rb") as video:
    media = client.files.create(file=video, purpose="ds4_media")
client.post("/embeddings", cast_to=dict[str, object], body={
    "model": "google/embeddinggemma-2",
    "input": [{"content": [{"type": "video", "file_id": media.id,
                             "audio": "include"}]}],
    "ds4": {"version": 1}})
client.files.content(media.id)
client.models.list()
```

Il metodo generico HTTP serve per l'estensione multimodale, che non rientra
nei tipi standard dell'SDK. Nessun SDK è una dipendenza del runtime C.
Riferimenti: [OpenAI Python API library](https://developers.openai.com/api/reference/python),
[Create embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create),
[Create image](https://developers.openai.com/api/reference/resources/images/methods/generate).
I limiti e l'estensione DS4 descritti qui sono decisioni del servizio locale.

| Identità | Operazione prevista | Modalità input | Stato attuale |
| --- | --- | --- | --- |
| LLM catalogati esistenti | chat | Quelle già qualificate per ciascun LLM/backend | Invariato |
| Qwen/Qwen-Image-2.1-Turbo | image_generate, image_edit | Testo, immagini di riferimento | Servizio seriale CUDA, PNG; gate qualità T11 ancora aperti |
| google/embeddinggemma-2 | embed | Testo, immagini, video, audio, interleaving | Servizio seriale CUDA sm_86 |

La discovery aggiunge solo i modelli configurati con bundle verificato;
la qualifica del trasporto non promuove i gate qualità T11. I default dei launcher restano invariati. Gli esempi JSON sono
validati da `tests/test_suite_contracts.py`; token-vocabulary, codec, limite
operativo e capability richiedono anche validazione semantica dell'adapter.

## Confine C e lifecycle

`ds4_runtime` possiede serializzazione e teardown condiviso. I model handle
possiedono descrittori immutabili e callback di attivazione/rilascio; tengono
vivo il runtime. Il chiamante termina/join tutti gli utilizzatori prima di
chiudere un handle. Le scope sono sul medesimo thread, bilanciate, ricorsive
solo sullo stesso modello. Una chiamata ricorsiva a un altro modello ritorna
BUSY prima di mutare lo stato. Una chiamata da un altro thread attende.

Ogni cambio fra scope di proprietari diversi esegue una fence; un'attivazione
fallita invalida il contesto attivo e richiede una nuova attivazione. Il
rilascio di un modello libera solo le risorse del suo adapter; lo shutdown
runtime avviene soltanto dopo l'ultima reference. Stime ignote non valgono
zero; le fasi dichiarano gruppi e dipendenze senza esporre tensor al server.
Checkpoint, ammissione e scheduling delle fasi appartengono a T03–T05.

L'apertura CUDA single GPU può precedere il LLM con
`ds4_compute_runtime_open`; `ds4_engine_open_with_runtime` collega un engine
all'istanza viva senza ricreare stream o cuBLAS. Il servizio possiede una
reference esplicita e la rilascia con `ds4_runtime_release`. La convenience
API legacy continua a creare un runtime privato. Una scope update esclusiva
serializza costruzione e modifica prima della registrazione del nuovo modello;
non ammette nesting di compute e invalida il contesto attivo prima del rientro.

`prepare` restituisce fasi con ID unici, dipendenze nello stesso piano e un DAG
aciclico; gli ID gruppo sono locali al modello proprietario. In caso di errore
restituisce piano NULL e conteggio zero. `execute` esegue una fase del piano;
`release_plan` libera il piano dopo successo/errore/cancellazione. Le stime
sconosciute richiedono known=false, mai byte zero inventati. La validazione
operativa del DAG e l'ammissione dei gruppi appartengono a T03/T05.
