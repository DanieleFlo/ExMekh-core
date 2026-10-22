# Revisione clean code del servizio e dell'integrazione ExMekh

Revisione del 22 ottobre 2026 delle modifiche T17/T18, dell'audit immagini e
dei percorsi collegati di ricerca web, provider, cache e frontend. Le correzioni
seguono le responsabilità esistenti e mantengono distinti trasporto, persistenza,
inferenza e presentazione. L'inventario non pretende di certificare ogni file
della repository: i percorsi esaminati e i limiti sono espliciti qui sotto.

## Problemi corretti

| Problema osservato | Correzione e responsabilità |
| --- | --- |
| Il fallback standalone creava client OpenAI senza chiuderli | `native_suite_client()` è un context manager: prende in prestito il trasporto del provider oppure apre e chiude quello standalone anche in caso di errore |
| La suite esponeva il trasporto e le prove accedevano a `_client` dello SDK | `NativeSuiteClient` conserva il trasporto privato; `LocalProvider` espone l'iniezione pubblica `http_client`, usata anche dalle prove live |
| Le stesse immagini tool venivano salvate in due formati nuovi diversi | Gli export nuovi usano il codec cache della libreria; il frontend conserva la lettura del formato UI precedente |
| Il collector conosceva il layout interno del codec della libreria | La lettura delle URL senza decode è posseduta da `user_message_codec`; la UI ne usa la funzione pubblica |
| Il ranking aveva cache globali abilitate in base alla classe concreta del backend | Un backend stateless condiviso per testo e immagini; cache per chiamata, con riuso testuale esplicito soltanto per il chiamante che ne gestisce la durata |
| Nomi e descrizioni continuavano a citare SigLIP/Granite | `compute_multimodal_ranking` e testi del dominio descrivono il servizio nativo; aggiornati ricerca, fallback, chiamanti e test |
| Un riferimento successivo fuori workspace poteva essere scoperto dopo il primo upload | Tutti i percorsi e la destinazione degli output sono risolti prima di iniziare le chiamate native |
| `created_at` era ricavato dalla scadenza e cambiava al rinnovo del TTL | Il file conserva separatamente creazione e scadenza; la regressione host verifica entrambi dopo un rinnovo |
| Una chiusura parziale lasciava puntatori a componenti già liberati | `suite_model_close` azzera ogni proprietario dopo il rilascio riuscito, permettendo di riprovare dopo un errore successivo |

Il salvataggio degli output e la costruzione delle anteprime sono separati
dall'orchestrazione HTTP. Le operazioni native usano i metodi del client; il
frontend usa il codec e le API dei messaggi. Restano pubblici soltanto i confini
necessari. Nessuna nuova gerarchia generica o implementazione alternativa.

## Scope degli embedding

Dopo il chiarimento dell'utente, `embed-content` è eliminato da sorgenti,
catalogo, default e HDS. Gli embedding restano nell'API tipizzata e nel ranking
interno della ricerca web. La scansione dei sorgenti di libreria e frontend
non ha trovato altri percorsi di inferenza embedding locale da migrare.
Le dipendenze Transformers/PyTorch rimangono per le altre funzionalità già
esistenti: la rimozione riguarda il caricamento e l'inferenza dei vecchi ranker.

Le prove sui default verificano l'insieme completo delle capability, mentre
quelle native verificano richieste reali al servizio per ranking testuale e
visuale. Nessun tool embedding separato è inviato al modello. I report T17/T18
e dell'audit precedente conservano le loro fotografie storiche; questa patch
descrive l'implementazione finale.

## Principi applicati e verifica

Responsabilità singole: provider e context manager gestiscono le connessioni;
codec e helper degli output gestiscono serializzazione/persistenza; ranking
gestisce i punteggi; la UI presenta gli allegati. Le dipendenze attraversano
metodi e funzioni pubbliche, senza ispezione dei dettagli dello SDK. I parser
e gli adapter mantengono errori espliciti e rilascio delle risorse. Le opzioni
booleane richieste dal contratto del tool restano parametri del prodotto.

La revisione comprende anche i chiamanti web e i wrapper del provider fuori
dal diff iniziale. Sono state eliminate ramificazioni e cache inutilizzate,
uniformati gli import del modulo search e centralizzati la creazione delle
patch di evidenza e il controllo dell'intero contenuto della wheel. La verifica
source/wheel/installazione funziona anche dopo il commit della libreria.

[Risultati, hash e log](verification.json) e [patch completa della libreria](exmekh-lib.patch)
registrano i gate finali. I log e le schermate locali sono sotto
`performance-results/native-suite-clean-review-20261022/`. La patch si applica
alla libreria dal commit base indicato nel manifest e viene verificata con
reverse-apply prima della consegna. La wheel deriva dai sorgenti, con lock
aggiornato; tutte le risorse del pacchetto vengono confrontate byte per byte
con sorgenti e installazione.

Sono eseguiti la suite offline completa della libreria e del frontend, Ruff,
Pyright, testi prodotto, documentazione, build CUDA del servizio, protocollo
host, risorse/runtime/executor/numeric lifetime, server legacy e 44 regressioni
CPU. Le prove live ripetono i tre scenari tool/ranking e il browser con LLM,
generazione GPU, due anteprime, dialog, reload e secondo turno reali.

Le prove live usano lo stesso [avvio e comandi dell'audit immagini](../native-suite-image-e2e-20261022/README.md),
con directory artefatti `native-suite-clean-review-20261022`. Il browser usa
`wait_seconds=300`; resta documentato nel report precedente il replay lento
oltre l'attesa predefinita di 60 secondi. Non sono qualificati qui recupero
asincrono del modello, qualità immagini T11, altri backend o distribuzione.
Non sono modificati kernel CUDA, Metal, SSD o TP.

| Gate finale | Esito |
| --- | --- |
| Libreria, suite offline completa | 3629 PASS, 76 live esclusi |
| Frontend, suite offline completa inclusi browser scripted | 1115 PASS, 61 live esclusi |
| Browser con LLM e generazione GPU reali | 1 PASS, 83.26 s |
| Tool immagini e ranking nativo | 3 PASS, 79.392 s |
| Contratti, manifest, API e processor CPU | 44 PASS |
| Documentazione | 10 PASS |
| Protocollo HTTP, risorse, executor, numeric lifetime, runtime e server legacy | PASS |
| Ruff, Pyright e testi prodotto | PASS, zero errori e warning Pyright |
| Wheel/source/installazione/lock | PASS, 288 risorse confrontate |

Commit della dipendenza esterna: `bd908cf`. Le quattro modifiche di sola
formattazione a configurazione, context usage e relativi test UI non cambiano
il comportamento. I log integrali includono tutti i gate e le durate.
