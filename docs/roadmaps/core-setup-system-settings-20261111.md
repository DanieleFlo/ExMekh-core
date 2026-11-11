# Avvio guidato e impostazioni di sistema di ExMekh

**Proposta di prodotto e piano di implementazione · 11 novembre 2026**

## Esperienza prevista

Al primo avvio `./start-ds4-server.sh` rende Core raggiungibile con zero modelli caricati. L'utente avvia Agent separatamente, collega il server e sceglie un modello chat, la modalità di memoria e il contesto. Può aggiungere visione, generazione/modifica di immagini ed embedding se il server li supporta. Core scarica e verifica i file, carica il modello chat e conferma che è pronto. Solo allora Agent apre la chat e rende disponibili le funzioni aggiuntive che Core dichiara utilizzabili.

Core salva l'ultimo profilo attivato con successo. Agli avvii successivi rilegge il profilo, rivaluta le risorse disponibili e prova a ripristinarlo. Agent mostra la stessa pagina **Avvio** se il modello si sta caricando. Se il server è già pronto apre direttamente la chat. Un file mancante o una configurazione che supera la memoria disponibile porta a una schermata di ripristino con l'azione pertinente, mantenendo il profilo per un nuovo tentativo. Il cambio di modello, modalità o contesto richiede una scelta esplicita.

La gestione di sistema si apre come pagina dedicata tramite l'ingranaggio e il link **Gestisci server** nel menu di stato del server. **Riconfigura avvio** apre il percorso guidato con il profilo corrente precompilato. Le stesse pagine di catalogo, modalità e attività servono al primo avvio e alla manutenzione.

## Navigazione e contenuto delle pagine

### Percorso guidato

L'intestazione mostra il server connesso e il passaggio corrente. **Indietro** conserva le scelte. Su schermi stretti il riepilogo scende sotto il contenuto e le righe dei modelli si impilano. La pagina **Connessione** entra nel percorso solo quando serve collegare o correggere il server.

| Pagina | Domanda a cui risponde | Contenuto e azioni |
| --- | --- | --- |
| **Connessione** · `/settings/connection` | A quale Core mi collego? | URL precompilato con `http://127.0.0.1:8080/v1`, credenziale di gestione se richiesta, **Verifica connessione**, risultato con identità e versione. In caso di assenza del server mostra il comando del README, **Copia comando** e **Riprova**. |
| **Modello chat** · `/setup/model` | Quale modello uso? | Catalogo con varianti, quantizzazione, dimensione e funzioni. Sintesi dell'hardware **del server**, backend, memoria utile e spazio su disco. Ogni riga mostra **Scarica**, **Già scaricato**, **In download** o un motivo di incompatibilità. **Scarica** avvia il trasferimento e mostra subito barra e numeri sulla riga. Selezionare un modello consente di scegliere la configurazione mentre il download procede. |
| **Memoria e contesto** · `/setup/runtime` | Come deve funzionare quel modello? | Per il modello scelto mostra **Streaming da SSD**, **Interamente in RAM** e **Interamente in VRAM**, secondo il supporto effettivo. Su hardware a memoria unificata usa l'etichetta **Residente in memoria unificata** quando applicabile. Ogni modalità ha contesto massimo stimato, contesto consigliato, compromesso pratico e motivo se indisponibile. Campo in token con **Usa il consigliato**, limite esplicito e **Aggiorna stima**. |
| **Funzioni aggiuntive** · `/setup/extensions` | Quali altre funzioni mi servono? | Scelte per **Analizzare immagini in chat**, **Generare e modificare immagini** ed **Embedding per ricerca**. Per ognuna mostra modello o dipendenza, spazio richiesto, compatibilità con il profilo scelto e stato del download. L'encoder di visione è abbinato alla variante LLM. **Continua con la sola chat** è sempre disponibile. |
| **Riepilogo** · `/setup/review` | Che cosa verrà applicato? | Modello, modalità, contesto, funzioni, file mancanti, spazio temporaneo e totale, comportamento al prossimo avvio. Ogni gruppo ha **Modifica** verso la pagina corrispondente. L'azione primaria è **Scarica e avvia** oppure **Avvia modello**. Da un profilo in uso è **Applica profilo**. |
| **Avvio** · `/startup` | Quando posso usare la chat? | Fase corrente e progresso del modello principale. Azioni per riprovare, correggere il profilo o aprire i dettagli dell'operazione. Al raggiungimento dello stato pronto **Apri chat** riprende la destinazione iniziale, se presente. Durante il ripristino ordinario la pagina torna automaticamente alla destinazione quando Core è pronto. |

### Impostazioni dopo l'avvio

Da desktop una navigazione laterale identifica la pagina attiva. Su mobile un indice porta alle stesse route. Ogni pagina ha un solo gruppo di decisioni e salva il proprio ambito. Tornare alla chat ripristina conversazione, bozza e allegati locali.

| Pagina | Responsabilità |
| --- | --- |
| **Server** · `/settings/server` | Stato Core, modello attivo, modalità, contesto, hardware e profilo al prossimo avvio. **Riconfigura avvio** e **Riprova avvio** quando pertinente. **Copia diagnostica** riporta versione, profilo, codice errore e ID operazione con dati sensibili rimossi. |
| **Connessione** · `/settings/connection` | URL, credenziali, protocollo di inferenza e verifica. Le modifiche si salvano dopo un test esplicito. Distingue server irraggiungibile, accesso negato, API incompatibile e Core raggiungibile ma non pronto. |
| **Modelli** · `/settings/models` | Libreria del server con due viste **Installati** e **Catalogo**. **Scarica**, **Importa dai file del server**, **Imposta come modello chat** e **Rimuovi**. Rimozione con spazio recuperabile e controllo di riferimenti del profilo attivo o salvato. Il download di più LLM è possibile, con un solo LLM chat attivo nella prima versione. |
| **Funzioni** · `/settings/extensions` | Disponibilità e gestione di visione, immagini ed embedding per il profilo attuale. Mostra il motivo di ogni funzione indisponibile. Rimanda ai file nella libreria e alle operazioni in Attività. |
| **Attività** · `/settings/activity` | Coda e storico recente di download, verifica, preparazione e caricamento. Righe espandibili con tempi, fase, errore e azioni di recupero. Lo stato nell'intestazione rimanda all'operazione in corso. |
| **Conversazioni** · `/settings/conversations` | Preferenze del client oggi nel menu globale, tra cui ragionamento, soglia di compattazione e invio delle immagini generate all'agente. Il limite del contesto è letto da Core. Le regole di approvazione e le altre preferenze specifiche di un progetto restano nelle impostazioni di quel progetto. |
| **File temporanei** · `/settings/storage` | Statistiche e pulizia della cache di Agent. Mostra quantità e dimensione prima della conferma. I file dei modelli si gestiscono dalla libreria sul server. |

## Stati che governano la navigazione

| Stato osservato | Destinazione o comportamento |
| --- | --- |
| Core irraggiungibile o credenziali respinte | Connessione, con istruzioni e causa specifica. |
| Core raggiungibile, nessun LLM verificato | Modello chat. Download incompleti e bundle senza verifica rimangono in preparazione. |
| LLM già scaricati, nessun profilo attivo o in ripristino | Modello chat con la vista Installati in evidenza e azione **Configura e avvia**. |
| Profilo confermato, nessun LLM pronto e attività in corso | Avvio. Un accesso diretto a una chat viene conservato come destinazione locale e il client recupera lo stato dal server anche dopo refresh. Se un LLM precedente serve ancora le richieste, la chat resta disponibile fino alla transizione. |
| Attivazione fallita | Avvio con **Riprova** per errori transitori, **Modifica profilo** per memoria insufficiente e **Trova o scarica il file** per artefatti mancanti. |
| LLM chat pronto e profilo effettivo pubblicato | Chat. I moduli facoltativi ancora in preparazione compaiono in Attività e si abilitano al termine se operativi. |
| Prontezza persa durante l'uso | Salva la bozza, blocca nuovi invii e mostra Avvio. Un turno interrotto è recuperabile tramite **Riprova** esplicito, così una richiesta con strumenti non viene ripetuta automaticamente. |

Core applica il vincolo anche alle API di inferenza. Agent usa lo stesso controllo per `/`, URL delle chat, invio dal compositore e avvio dei tool dipendenti da Core. Se il server propone un nuovo limite del contesto dopo un cambio di profilo, Agent aggiorna il budget prima di inviare e usa la normale procedura di compattazione quando applicabile. Le rotte di gestione restano sempre raggiungibili. Il solo elenco `/v1/models` non è sufficiente per dichiarare pronto un Core gestito.

### Download, verifica e caricamento

- **Download** mostra byte trasferiti e totali se noti, percentuale, velocità, tempo trascorso, stima residua e file corrente per pacchetti con più artefatti. **Pausa** e **Riprendi** agiscono sul job di Core. Se la sorgente non supporta la ripresa, il server lo segnala prima di ricominciare. **Annulla** esplicita se conserva il parziale e quanto spazio occupa. Lasciare la pagina non interrompe il lavoro.
- **Verifica e preparazione** sono fasi esplicite. Raggiungere il 100% del trasferimento non equivale a **Già scaricato**. Un bundle è installato quando file, revisioni e checksum sono validati e gli asset derivati necessari risultano pronti.
- **Caricamento** mostra fasi osservabili come apertura dei file, preparazione del runtime, inizializzazione della sessione e disponibilità HTTP interna. La percentuale è misurata quando ci sono unità contabili, oppure etichettata **stimata** se supportata da dati di caricamenti comparabili. Una fase opaca mostra attività indeterminata, fase e tempo trascorso. La stima del tempo residuo appare solo con dati sufficienti. Nel profilo SSD il completamento indica che il runtime può servire richieste con la sua politica di streaming.
- **Riconnessione** rilegge stato e job autorevoli con lo stesso ID. La perdita di rete mostra **Riconnessione al server** mantenendo l'ultimo valore con indicazione che non è aggiornato. **Annulla avvio** compare solo quando il worker può fermarsi e rilasciare le risorse in modo sicuro.

## Confine tra Core e Agent

| Core | Agent |
| --- | --- |
| Rileva hardware e limiti, mantiene catalogo e inventario, valida profili, esegue download e preparazione, attiva il runtime e pubblica lo stato. Mantiene l'ultimo profilo riuscito anche in assenza di Agent. | Raccoglie scelte, mostra lo stato HTTP, gestisce navigazione e preferenze di conversazione. Non accede ai pesi, non esegue comandi sul server e non decide i limiti di memoria. |
| I binari e i file del server sono installabili e gestibili da riga di comando. Le operazioni di configurazione sono accessibili anche via HTTP a client autorizzati. | NiceGUI interroga Core lato Python con un client HTTP tipizzato. Il browser comunica con Agent, e Agent usa il normale percorso ExMekh-Lib per l'inferenza. |

### Avvio del processo

Il launcher documentato entra in **modalità gestita** per default e accetta una modalità manuale esplicita per chi usa già gli argomenti del server o gli script di qualifica. In modalità gestita un servizio di controllo appartenente a Core possiede la porta pubblica fin dall'inizio. Avvia il server nativo come processo figlio su trasporto locale privato, con configurazione derivata da ID di catalogo e profilo validato. La scelta del processo figlio limita l'impatto sul server attuale, che apre l'engine prima del listener, e rende recuperabile un'uscita fatale durante l'inizializzazione.

Il servizio di controllo serve il contratto di gestione e inoltra le richieste d'inferenza quando il figlio è pronto. Verifica iniziale obbligatoria su streaming, backpressure, cancellazione alla disconnessione, upload multipart, errori e API agentiche. Il processo figlio comunica eventi di caricamento strutturati su un canale privato e segnala prontezza dopo l'inizializzazione del runtime e del worker. L'uscita o il riavvio del servizio deve terminare e raccogliere i figli prima di riutilizzare le risorse. Le richieste d'inferenza senza runtime pronto ricevono `503 MODEL_NOT_READY`.

Il profilo nuovo viene validato prima di intervenire su quello attivo. Il download avviene mentre il modello corrente può ancora rispondere. Quando si applica il profilo, Core serializza la transizione e attende la fine dei turni e delle operazioni ammesse, mostrando **In attesa di fine attività**. Il server usa un'identità del runtime per invalidare cache effimere e handle interattivi di un'istanza sostituita. Se l'avvio del candidato fallisce, tenta il ripristino dell'ultimo profilo riuscito, espone anche il risultato del ripristino e conserva la modifica proposta come scelta da correggere. Non servono due runtime residenti contemporaneamente.

### Contratto HTTP proposto

Un prefisso separato `/v1/manage` distingue il provisioning dal contratto di inferenza esistente. Gli endpoint di mutazione richiedono una credenziale di gestione. Il default è loopback. L'accesso remoto richiede configurazione esplicita e trasporto protetto, e i segreti non entrano in URL, eventi o log. Per la gestione dei file il server accetta ID di manifest oppure asset riconosciuti in directory autorizzate sul server, mai URL arbitrari, comandi shell o percorsi forniti dal browser.

| Endpoint proposto | Risultato |
| --- | --- |
| `GET /v1/manage/status` | Versione del contratto, identità Core, `boot_id`, revisione dello snapshot, stato del runtime, profili attivo e salvato, prontezza chat e disponibilità delle operazioni opzionali. Include ID operazione in corso o motivo del blocco. |
| `GET /v1/manage/hardware` e `GET /v1/manage/catalog` | Snapshot risorse, backend effettivamente compilati e dispositivi, inventario verificato, varianti e dipendenze. Filtri di catalogo calcolabili da questi dati. |
| `POST /v1/manage/plans` | Anteprima senza mutazioni. Il client propone modelli e opzioni, Core restituisce compatibilità, stime, download mancanti, scadenza e revisione su cui si basa il piano. |
| `POST /v1/manage/jobs` | Accoda download, importazione, rimozione o applicazione di un piano. Risposta `202` e ID stabile. Riutilizzare una chiave idempotente con lo stesso payload recupera il job, un payload diverso restituisce conflitto. |
| `GET /v1/manage/jobs` e `GET /v1/manage/jobs/{id}` | Snapshot di attività e stato per ripresa, inclusi fase, unità, qualità del progresso, tempi, azioni ammesse ed errore con codice stabile. |
| `POST /v1/manage/jobs/{id}/actions` | Pausa, ripresa, nuovo tentativo o annullamento secondo lo stato del job. |

Il contratto iniziale funziona con polling asincrono e frequenza adattiva. Aggiornamenti SSE possono seguire se utili, usando gli stessi snapshot per la riconnessione. `boot_id` invalida le risposte rimaste in coda dopo un riavvio, e la revisione del profilo evita sovrascritture da due client. In caso di stima scaduta o profilo cambiato il server richiede un piano aggiornato. Risposte tipizzate, per esempio `INSUFFICIENT_VRAM`, `DISK_SPACE_REQUIRED`, `ARTIFACT_MISSING` e `BACKEND_UNSUPPORTED`, alimentano messaggi locali e azioni precise. Per un endpoint compatibile esistente senza API di gestione, Agent conserva la connessione manuale e la discovery standard. In quel percorso caricare il modello resta un'operazione del gestore del server. Un errore di autenticazione o un problema di rete non attiva la compatibilità.

Ogni modalità nel piano espone `mode_id`, disponibilità e relativo motivo, `estimated_max_context_tokens` e `recommended_context_tokens`, con valori nulli quando ignoti. Lo stato distingue quel massimo stimato dal `context_tokens` effettivo del profilo caricato. Ogni job fornisce almeno `phase`, unità elaborate e totali, frazione e qualità del dato, millisecondi trascorsi, stima residua facoltativa e azioni ammesse. L'inventario espone la revisione verificata per ogni variante, indipendentemente dallo stato di caricamento.

### Catalogo, stime e persistenza

1. **Catalogo versionato.** Modello e variante hanno identità stabili, limite architetturale, dimensioni e digest degli artefatti, riferimenti a eventuali MTP, encoder o bundle e matrice di modalità qualificate per build/backend/topologia. Riutilizzare le identità già fissate in [download_model.sh](../../download_model.sh), i [manifest suite](../../config/suite-bundles/) e la verifica di [suite_bundle.py](../../tools/suite_bundle.py). Aggiungere a questi percorsi telemetria strutturata per job, evitando di interpretare le barre testuali. Riconoscere installazioni esistenti tramite inventario e verifica degli artefatti. Le versioni in valutazione possono essere indicate come tali e non ricevono una raccomandazione.
2. **Stima senza pesi residenti.** Core usa metadati affidabili del catalogo prima del download e i metadati reali dopo. Calcola il massimo stimato per coppia modello/modalità con il planner di famiglia, tenendo conto di pesi, KV e stato ricorrente, prefill, buffer, MTP, encoder, sessioni, cache SSD e riserva. Considera memoria e spazio effettivamente utilizzabili, staging, asset derivati e spazio per checkpoint quando richiesto. Su Apple Silicon contabilizza la memoria condivisa una sola volta e rispetta il limite di working set. Per una combinazione senza dati affidabili risponde **Stima non disponibile** con il motivo. Il consigliato è un profilo qualificato che rientra nel budget con margine, altrimenti compare solo il massimo stimato se disponibile. L'ammissione reale ripete i controlli al caricamento. I valori dei file di esempio Metal sono evidenze di profili specifici. [Memoria Metal](../../ds4_metal_memory.h) · [API di stima](../../ds4.h) · [Riesame T08](../architecture/metal-t08-clean-code-review-20261106.md)
3. **Download durevoli.** Uno stesso artefatto ha un solo job di trasferimento anche con più richieste. Download in file parziali, ripresa con verifica dell'identità remota, spazio ricontrollato prima della scrittura, verifica finale e pubblicazione atomica. Il riavvio riconcilia coda, inventario e file parziali. Per gli asset con licenza o accesso vincolati, il piano espone il prerequisito e il server usa credenziali configurate localmente.
4. **Profilo durevole.** In directory dati Core, file versionato e scritto atomicamente con modello, modalità, parametri effettivi, contesto, funzioni e revisione. Separare piano richiesto, profilo attivo e ultimo profilo confermato. Confermare l'ultimo profilo solo dopo prontezza chat, registrando a parte esito e desiderata delle funzioni facoltative. Per profilo mancante, corrotto o incompatibile, mantenere i dati recuperabili e presentare una correzione esplicita.
5. **Funzioni per capacità effettive.** Per ciascuna, Core distingue asset verificati, configurazione abilitata e operazione disponibile per il profilo. Visione dipende dall'encoder corretto per il LLM. Immagini ed embedding possono essere disponibili con caricamento su richiesta secondo le regole della suite, quindi non richiedono residenza permanente. Un download da solo non sblocca controlli o strumenti. Agent aggiorna allegati, strumenti immagine e ricerca semantica al cambio di snapshot. [Contratto suite](../architecture/native-suite-v1-contract.md) · [Configurazione suite Metal](../../config/native-suite-metal.example.json)

## Fasi di implementazione e accettazione

| Fase | Consegna | Verifica minima |
| --- | --- | --- |
| 1. Core raggiungibile senza modello | Launcher gestito, servizio di controllo, stato e avvio manuale compatibile. Spike del passaggio HTTP verso il worker su un profilo già supportato. | Checkout senza GGUF, stato HTTP raggiungibile, inferenza rifiutata con errore strutturato. Test di processo figlio guasto e di chiusura. Con worker pronto, confronto di Responses in streaming e non streaming, multipart, cancellazione e codici HTTP con il percorso diretto. |
| 2. Modello chat gestito | Catalogo iniziale curato, inventario, download duraturo, stima e attivazione. Partire dai profili Flash Next Q2 e Qwen3.8-27B già documentati, abilitando le sole combinazioni con evidenze. | Interruzione a metà, ripresa, checksum errato, disco pieno, due client che chiedono lo stesso file e risorse diminuite tra piano e avvio. Verificare nessuna allocazione dei pesi durante la stima. |
| 3. Percorso guidato completo | Route NiceGUI, stime, Avvio e blocco delle chat, ripristino del profilo. | Playwright con Core simulato per primo avvio, deep link, refresh durante download/caricamento, server offline, ripartenza, errore e cambio profilo durante un turno. Gate reale con profilo supportato e prova di riavvio senza Agent. |
| 4. Impostazioni a pagine | Sostituire il drawer globale, migrare i controlli esistenti, aggiungere libreria e Attività. | Verificare persistenza delle preferenze esistenti, riapertura della configurazione, conservazione bozza, navigazione da tastiera, layout mobile e competizione tra client. |
| 5. Funzioni aggiuntive e documentazione | Visione abbinata al LLM, immagini ed embedding per combinazioni qualificate. Quickstart Core e Agent, connessione remota e uso manuale. | Test installato ma indisponibile, pronto su richiesta, errore solo di un modulo, passaggio di profilo e discovery legacy. Ripetere una prova reale per ogni combinazione promossa e ogni operazione dichiarata. |

Il primo rilascio utile comprende le fasi 1–4 per la chat. La fase 5 aggiunge installazione e attivazione guidata dei moduli. Prima di promuovere nuove modalità o toccare ammissione, scheduling o inferenza, seguire le verifiche e gli audit per intervento previsti da [AGENTS.md](../../AGENTS.md). Il progetto UI non attribuisce nuovi limiti di contesto o miglioramenti prestazionali.

## Implementazione e verifiche

- Il launcher gestito usa il servizio Python di Core con un listener Unix privato e un canale di stato separato verso il binario nativo. `--manual` e le invocazioni esplicite mantengono il percorso precedente. Il listener di gestione è locale e supporta un token facoltativo. Per accesso remoto è documentato il port forwarding. [Supervisore](../../tools/managed_server.py) · [Quickstart](../../README.md#getting-started)
- Il catalogo offre SSD su Metal e i profili residenti qualificati per Qwen27. Le sessioni Qwen27 e Flash del catalogo richiedono un backend grafico. La modalità interamente in RAM compare come indisponibile. Il limite CUDA di Qwen27 usa il costo FP32 del KV dalle 16 attenzioni complete, i pesi, la memoria libera, una riserva conservativa e il tetto del launcher. La stima Metal rimane circoscritta ai profili documentati e l'ammissione nativa misura il budget all'avvio. [Catalogo e stime](../../tools/managed_assets.py) · [Profili Metal](../performance/metal-apple-silicon-20261031/t06-t07/README.md)
- I moduli selezionati vengono preparati durante l'applicazione del profilo. Se un modulo fallisce, Core prova ad avviare la chat senza il modulo e pubblica la causa. EmbeddingGemma richiede un asset processore già preparato. Gli strumenti esistenti usano la discovery nativa della suite. Agent protegge l'invio di immagini al modello e l'uso di riferimenti immagine in base alle capacità pubblicate.
- Verificati in WSL avvio senza pesi, prontezza e ripristino con Qwen3.8-27B UD-Q4_K_S a 512 token in VRAM su RTX 3090. Verificate Responses native con e senza SSE, riavvio con lo stesso contesto e modalità, e una richiesta con immagine dopo verifica e abilitazione del proiettore corrispondente a 4096 token. I test locali coprono ripresa HTTP Range, verifica checksum, richiesta multipart e cancellazione alla disconnessione. Il browser usa un Core simulato per esercitare l'intero percorso di setup. Le combinazioni Metal e la preparazione completa dei bundle Image ed EmbeddingGemma non sono state eseguite in questa verifica.

### Riesame delle responsabilità

- Il supervisore decide il ciclo di vita del profilo. Il catalogo valida configurazioni e inventario, il processo nativo possiede socket e configurazione temporanei, e il trasporto HTTP usa le API pubbliche di Core. Le azioni consentite sui job arrivano nel relativo snapshot. [Core](../../tools/managed_server.py) · [Catalogo](../../tools/managed_catalog.py) · [Processo nativo](../../tools/managed_worker.py) · [Trasporto HTTP](../../tools/managed_http.py)
- Il cambio di profilo somma al campione corrente soltanto i byte dei pesi residenti recuperabili. La verifica del budget viene ripetuta dopo l'arresto del processo precedente. Un artefatto inattivo che ha perso la verifica può essere rimosso e riscaricato anche se il profilo salvato lo indica. Due istanze Core che condividono la directory non possono scrivere contemporaneamente sullo stesso artefatto.
- Le viste di Agent usano operazioni pubbliche del client Core e ricevono oggetti tipizzati per stato, piano e job. Il controller della chat possiede la sospensione e il ripristino delle bozze. La rimozione del drawer elimina un percorso duplicato. I salvataggi concorrenti della configurazione sono serializzati e la discovery usa il contesto dell'endpoint selezionato. [Client HTTP](../../agent/src/exmekh_agent/exmekh_bridge/providers/core_management.py) · [Pagine](../../agent/src/exmekh_agent/ui/settings/)
- Sul checkout di partenza `6c23de1` con la nuova implementazione, il riesame ha ripetuto l'avvio di Qwen3.8-27B su RTX 3090, il cambio di contesto da 512 a 768 token, Responses SSE e il ripristino di 768 token al riavvio. Il riesame non cambia kernel o dispatch di inferenza e non attribuisce variazioni prestazionali.

## Riferimenti nel checkout

- Avvio attuale, profili e downloader. [Getting started](../../README.md#getting-started) · [Launcher](../../start-ds4-server.sh) · [Download](../../download_model.sh)
- Apertura del modello prima del listener, discovery e routing dell'inferenza. [ds4_server.c](../../ds4_server.c)
- Contatori e stato di operazioni native, base per eventi di caricamento. [ds4_operation_progress.h](../../ds4_operation_progress.h) · [ds4_server_suite_http.inc](../../ds4_server_suite_http.inc)
- Entry point NiceGUI, discovery e impostazioni. [main.py](../../agent/src/exmekh_agent/main.py) · [provider_health.py](../../agent/src/exmekh_agent/exmekh_bridge/providers/provider_health.py) · [system_pages.py](../../agent/src/exmekh_agent/ui/settings/system_pages.py)
