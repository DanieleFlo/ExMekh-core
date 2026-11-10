# Progresso della generazione immagini

Data: 10 novembre 2026. Stato: implementazione v1. Contratto e gate sono
definiti qui; prove eseguite e limiti di qualifica sono registrati nel
[report di implementazione](native-image-progress-implementation-20261110.md).

## Requisito di portabilità

**Progresso, heartbeat e visualizzazione sullo skeleton appartengono al
contratto comune delle operazioni image. Devono funzionare su CPU, Metal,
CUDA e sui backend futuri, per generazione ed editing, con Qwen Image e con
ogni altro modello image che verrà integrato.**

L'integrazione di un nuovo modello deve includere questo contratto e i suoi
test. Non è ammessa un'implementazione del progresso disponibile solo per
Qwen Image o attivata attraverso macro Metal/CUDA. Non si presume che un
modello usi diffusione, otto step, un encoder separato o un VAE. Un modello
autoregressivo può esporre token; un altro può esporre passi, tile o soltanto
transizioni di fase. Totali sconosciuti restano sconosciuti.

Runtime e servizio possiedono lifecycle, heartbeat, timeout e trasporto.
L'adapter del modello traduce le proprie unità di lavoro nel contratto comune.
Client, bridge e UI consumano lo stesso schema senza selezionare il nome del
modello o il backend di calcolo. Eventuali stime temporali possono usare il
profilo di esecuzione come dato, senza creare percorsi UI dedicati.

## Motivazione e punti di integrazione verificati

Baseline ispezionata: ExMekh-core
`4a03d05c94bcaf88ff295875b8b5ea0e2ca467dd`.

- `ds4_server_suite_http.inc:suite_route` assegna la deadline assoluta prima
  dell'accodamento. `config/native-suite-metal.example.json` imposta 300 secondi.
  `ds4_server_suite.inc:suite_cancel` verifica quella deadline anche con socket
  ancora connesso: inviare heartbeat non la proroga.
- `ds4_server_suite_operations.inc:suite_run_images` e il client installato
  `exmekh_lib/core/models/native_suite.py:NativeSuiteClient.images` consegnano
  alla baseline un risultato finale JSON, senza canale di progresso.
- `ds4_executor.h:ds4_execution_ops.journal` espone già il lifecycle comune
  dell'esecuzione. È un punto da valutare per gli eventi di gestione risorse,
  distinto dai contatori interni del modello.
- `ds4_qwen_image_observer.[ch]` espone trace e cancellazione. Abilitare `trace`
  aggiunge `ds4_numeric_fence` prima dei callback. Non va riutilizzato per la
  UI attivando il tracciamento di tutti i tensori.
- `ds4_qwen_image_dit.c:ds4_qwen_image_dit_generate` contiene il loop Turbo
  corrente di otto passi. È un possibile produttore per un adapter Qwen,
  non la definizione del progresso comune.
- `agent/.../exmekh_bridge/tools/tool_feedback.py`, `models/execution.py:ToolEvent`
  e `ui/chat/chat_view.py:_render_tool_images` già correlano stato, immagini
  e skeleton all'invocazione. Alla baseline mancava il feedback durante il tool;
  i soli hook prima/dopo non bastano.

La ricerca locale T17/T18 richiede già di separare progresso e successo dopo
restore: vedere [contratto client](native-suite-client.md#lifetime-e-osservabilità)
e [T17.5](../roadmaps/native-multimodal-suite-implementation-20261020.md).
Il [report Image512](../performance/metal-apple-silicon-20261031/t08-20261106/README.md)
separa tempi di conditioning, generazione e decodifica; le sue misure di un caso
non sono pesi universali per tutte le immagini.

Riferimenti esterni: [SSE WHATWG](https://html.spec.whatwg.org/dev/server-sent-events.html),
[timeout HTTPX](https://www.python-httpx.org/advanced/timeouts/) e
[callback Diffusers](https://huggingface.co/docs/diffusers/main/using-diffusers/callback).
Il callback di fine step Diffusers è un esempio di adapter, non un requisito
per gli altri modelli. Le API del trasporto installato, incluso `httpx2`, vanno
verificate nell'implementazione: non si presume identità con HTTPX.

Prima di intervenire sul runtime o sull'inferenza si applica nuovamente
l'audit richiesto da `AGENTS.md`: sorgenti upstream e ricerca locale con
revisioni, percorsi, applicabilità e alternative nel report dell'intervento.
I pin storici riportati nei report non sostituiscono questa nuova verifica.

## Contratto comune degli eventi

Il servizio pubblica eventi versionati `operation.started`,
`operation.progress`, `operation.heartbeat`, `operation.completed`,
`operation.failed` e `operation.cancelled`. Un evento terminale chiude
l'operazione; la chiusura del socket da sola non significa successo.

Ogni evento include versione, `operation_id`, `request_id`, sequenza monotona
`seq`, stato e tempo trascorso monotono. Il bridge associa l'operazione alla
chiave completa dell'invocazione: turno, nodo/scope e `tool_call_id`.

Il payload di progresso può includere:

| Campo | Significato |
| --- | --- |
| `phase` | Identificatore della fase attuale, estensibile |
| `item_index`, `item_count` | Primo indice attivo e numero totale di immagini |
| `active_items` | Numero di immagini attive a partire da `item_index`. Default 1 |
| `completed_units`, `total_units` | Lavoro completato nella fase; totale nullable |
| `unit` | Per esempio `step`, `token`, `tile`, `work_unit` |
| `overall_fraction` | Frazione dell’intero batch fra 0 e 1, oppure `null` |
| `overall_kind` | `work`, `estimated` oppure `indeterminate` |

Le fasi comuni comprendono attesa, preparazione, generazione, finalizzazione
e ripristino. Conditioning e decodifica sono opzionali: non si inventano
fasi che il modello non esegue. Gli identificatori restano metadati diagnostici:
la UI mostra solo percentuale e tempo e conserva il risultato anche quando
riceve una fase sconosciuta.

Le unità descrivono lavoro completato, non comandi semplicemente accodati alla
GPU. L'adapter usa i confini di completamento già disponibili; eventuale
granularità più fine va valutata senza fence aggiuntive per aggiornare la UI,
readback dei latenti o decodifiche intermedie. I callback pubblicano soltanto
metadati in uno stato bounded, senza I/O di rete sul thread di compute.

## Percentuale e skeleton

Lo skeleton mostra **soltanto percentuale e tempo trascorso**, come richiesto
dall'utente: per esempio `42% · 12s`. Non mostra scritte come “Generating
image”, nomi delle fasi, contatori dei passi o una barra di progresso. Fasi e
unità restano disponibili nei metadati diagnostici; `4/8` valutazioni del
denoiser non equivale da solo al 50% dell'intera operazione.

La percentuale complessiva deriva dal piano di lavoro dichiarato dall'adapter
oppure da pesi temporali calibrati su misure per profilo, geometria, batch e
tipo di richiesta. Il campo `overall_kind` distingue le stime.
I pesi si congelano all'inizio dell'operazione per evitare salti
all'indietro. Non si divide indiscriminatamente il numero dei passi per otto,
né si applicano i tempi di Qwen ad altri modelli.

Ogni skeleton ricava la propria percentuale dalla frazione del batch e
dall'intervallo delle immagini attive. Le immagini della stessa wave avanzano
insieme. Le immagini in attesa mostrano `—%`. Quelle già elaborate mostrano
`99%` fino alla consegna del batch.
Quando manca una stima attendibile, la UI mostra `—%` e il tempo trascorso.
Un heartbeat aggiorna il tempo e mantiene la percentuale. Rimane lo shimmer
dello skeleton; il numero non avanza in assenza di lavoro osservato. La UI
aggiorna le etichette in posizione, con coalescenza degli eventi, evitando di
ricostruire la galleria a ogni callback.

`100%` è riservato al risultato completato: il servizio ha terminato compute,
salvataggio e restore; il frontend ha ricevuto, verificato e salvato il PNG.
Fino alla consegna del PNG lo skeleton mostra al massimo `99%`.
Il tempo si ferma quando arriva l'esito del servizio. L'eventuale
continuazione LLM è un'attività distinta: un suo errore, incluso vision non
abilitata, non deve sostituire l'immagine riuscita con uno stato di generazione
fallita. In caso di errore image restano visibili fase e causa, con eventuali
risultati parziali identificati esplicitamente e senza falso successo del batch.

## Trasporto v1 e timeout

La prima implementazione usa SSE opt-in sulle chiamate image native,
negoziato esplicitamente dal client, mantenendo il JSON corrente per i client
standard. `Accept: text/event-stream` negozia il trasporto; discovery
e client devono riconoscere la versione realmente supportata. Il formato SSE
è letto dal client Python: il browser resta sul collegamento NiceGUI esistente.
Il risultato terminale contiene il batch finale con lo stesso significato
del contratto JSON; dopo gli header streaming gli errori sono eventi terminali
strutturati, non una seconda risposta HTTP.

Il percorso è:

```text
adapter modello -> osservatore comune -> servizio SSE -> ExMekh-Lib
                -> ToolFeedback -> stato chat -> skeleton NiceGUI
```

Un writer di trasporto separato dal compute invia heartbeat indicativamente
ogni 10 secondi, anche durante attesa, caricamento, kernel lunghi e restore.
Un solo writer possiede il socket; worker e writer non possono intercalare
byte di risposta. Stato e coda sono bounded: gli aggiornamenti intermedi
possono essere accorpati, l'evento terminale deve essere conservato. Un client
lento non deve bloccare il compute indefinitamente.

Il flusso inizia all'accodamento, dopo la lettura della richiesta HTTP.
Il worker valida il payload prima di avviare il calcolo e comunica eventuali
errori di validazione con l'evento terminale.
Il writer non può essere un altro lavoro sulla stessa coda seriale che sta
eseguendo la generazione. Gli header disabilitano il buffering dove applicabile
e ogni heartbeat viene effettivamente scritto. Il timeout di lettura del
client è maggiore dell'intervallo heartbeat con un margine esplicito.

Si distinguono tre limiti:

1. **Inattività di rete:** gli heartbeat mantengono attiva una connessione il
   cui timeout misura l'attesa fra dati ricevuti. Non aggirano limiti assoluti
   imposti da un proxy o dal client.
2. **Deadline totale:** resta assoluta e include il tempo in coda. Il parametro
   opzionale `image_timeout_seconds` (1–86400) vale per generazione/editing,
   con SSE o JSON, su tutti i backend; se assente eredita `timeout_seconds`.
   Il profilo Metal ora seleziona 1200 secondi per image e conserva 300 per
   gli altri endpoint. Gli heartbeat non prorogano nessuno dei due limiti.
3. **Stallo del lavoro:** un watchdog, se introdotto, misura l'assenza di
   avanzamento reale rispetto a fasi note. Non si resetta con heartbeat e
   non confonde un kernel lungo ma valido con una connessione morta.

Cancellazione e cleanup restano cooperativi. La prima versione conserva la
semantica attuale: perdere la connessione client-server richiede cancellazione
e restore. Un refresh del browser può ricollegarsi allo stato del task Python
ancora attivo, senza ricreare il tool. La ripresa dopo perdita del trasporto
native o riavvio richiederebbe un registro job e un endpoint di osservazione
separati; SSE e `Last-Event-ID` da soli non la implementano.

La chiave idempotente resta stabile per l'operazione. Non si ritenta
automaticamente una generazione fallita per cancellazione/deadline o risorse
insufficienti. Le ricevute attuali proteggono il risultato già committato;
non provano la ripresa di un lavoro in corso. Non si aggiunge un retry che
duplichi il compute per recuperare soltanto la visualizzazione del progresso.

## Responsabilità dell'implementazione

1. Osservatore e schema comuni nel core/servizio; collegare il journal
   lifecycle e lo stato bounded posseduto dall'operazione. Nessun tipo pubblico
   di progresso deve richiedere header Metal/CUDA o Qwen.
2. Adapter modello: Qwen Image come primo consumer, con fasi
   e passi reali; usare anche un adapter di test non Qwen con unità diverse
   e totale sconosciuto. L'adapter non possiede heartbeat o socket.
3. Negoziazione, writer SSE, heartbeat, errori e policy timeout.
   Conservare il percorso JSON e le invarianti di restore/commit.
4. ExMekh-Lib possiede la lettura incrementale e un callback di progresso
   associato al contesto del tool. Il callback è infrastruttura, non un nuovo
   argomento che il modello deve compilare nello schema `generate-image`.
   Aggiornare sorgente, wheel e lockfile attraverso il flusso documentato.
5. Snapshot propagato attraverso bridge, collector e stato della chat;
   il renderer aggiorna skeleton e percentuale nel contesto NiceGUI corretto.
   Gli eventi transitori non entrano nella history inviata al LLM. Conservare
   l'ultimo snapshot durante una riconnessione UI e il riepilogo terminale.

## Gate di accettazione

- Contratto comune verificato senza GPU con adapter deterministici Qwen-like
  e non Qwen: passi, token, fasi mancanti e totali sconosciuti.
- Stesso schema e comportamento sui percorsi CPU, Metal e CUDA supportati.
  Un backend non disponibile è `NOT_RUN`, non un PASS dedotto da un altro.
  Ogni futuro modello richiede questi gate nel proprio piano di integrazione.
- Writer e heartbeat restano attivi mentre il compute è deliberatamente lento;
  ordine, coalescenza, memoria bounded e client lento/disconnesso sono verificati.
- Prova di trasporto con timeout di lettura corto: ricevere heartbeat evita
  il timeout di inattività; una deadline totale corta scade comunque. Una
  richiesta lunga completa quando il budget totale configurato lo consente.
- Singolo evento terminale; cancellazione, errore e restore fallito non
  pubblicano successo. Nessuna seconda generazione per retry/ripristino UI.
- Browser: percentuale coerente per invocazione e immagine, aggiornamenti
  durante il tool, cambio chat, refresh, batch, errore image e errore LLM
  successivo con immagine conservata. Sola percentuale/tempo, indicazione `—%` e aggiornamenti accessibili.
- Sugli identici input isolati, con seed/recipe congelati, progresso attivo
  e disattivo producono gli stessi artefatti. Ripetere poi il gate numerico
  indipendente del modello, senza allargare criteri o cambiare oracle.
- Profilare prima/dopo: distinguere costo dei callback, trasporto e rendering
  dal tempo dei componenti e dalla latenza totale. Nessuno speedup dichiarato
  senza misure; nessuna sincronizzazione GPU introdotta per il solo progresso.
