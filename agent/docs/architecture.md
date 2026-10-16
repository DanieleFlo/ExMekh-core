# Responsabilità e API dell’applicazione

La revisione ha considerato UI, stato, integrazione con ExMekh-Lib e i flussi di
streaming, approvazione, persistenza, allegati e visualizzazione degli output.
Gli interventi si concentrano sulle responsabilità che erano mescolate e sui
comportamenti incoerenti verificabili con test.

| Componente | Responsabilità | API da usare |
| --- | --- | --- |
| `AgentDesign` | Struttura della lavagna e invarianti di nodi/collegamenti | `add_node`, `add_positioned_node`, `replace_node`, `move_node`, `connect`, `disconnect`, `remove_node`, `set_root`, `set_persistent_history`, `consolidate_capabilities`, `capability`, `children`, `connections`, `connection_targets` |
| `CapabilityRepository` | Catalogo, copie e configurazioni su disco per ExMekh-Lib | `folder_reference`, `catalog`, `default_design`, `expand_skill_connections`, `add_to_design`, `editable_values`, `copy_for_design`, `copy_capability`, `compile` |
| `CompiledAgentDesign` | Accesso alla configurazione generata | `capabilities_for`, `default_names_for`; i chiamanti non devono conoscere i file interni |
| `ProjectSettings` / `ChatSettings` | Configurazione e risoluzione dei default/override | `capability_paths`, `with_additional_capability_paths`, `with_agent`, `with_agent_override`, `with_project` |
| `SessionManager` | Persistenza e aggiornamento dello stato gestito; selezione per pagina | `for_view`, `save_chat_agent`, `save_project_settings`, `save_project_settings_async`, `execution_settings`, `edit_message`, `update_title`, `record_command_approval` |
| `RecordedHistory` | Attraversamento, correlazione e ripristino delle history registrate | `scopes`, `arguments_for_call`, `output_for_call`, `restored_snapshot`, `restore_user_content` |
| `ContextUsageUpdater` | Snapshot del conteggio, coalescenza e consegna nel contesto del client | `request`, `schedule`, `wait_idle`, `close` |
| `AppConfig` | Configurazione runtime, override di ambiente e persistenza | `update`, `update_async`, `save_to_env`, `select_available_model`; `save_to_env` e `update_async` restituiscono l'esito |
| `config.catalogs` | Caricamento delle impostazioni e dei testi suddivisi per competenza | `settings`, `presentation_settings`, `message`; snapshot indipendenti delle configurazioni |
| Risorse UI | Accesso alla presentazione e rendering dei template HTML/SVG | `ui_text`, `presentation_config`, `render_html`; delega ai cataloghi, escaping automatico |
| `utils` | Operazioni generiche condivise | `atomic_write_text`, normalizzazione dei nomi, formatter, catene di eccezioni, lettori di risorse e parser Markdown/YAML |
| `ChatMessage` | Ricerca di argomenti e risultati nelle history registrate | `arguments_for_tool`, `output_for_call`, `execution_timeline`; i chiamanti non devono attraversare le history per correlare invocazioni |
| `ExecutionTimeline` | Ordine e collocazione delle conferme dello steering | `root_activity`, `skill_activity`; il modello risolve scope, call ID, tempi e invocazioni ripetute |
| `CommandApprovalEvaluator` / `CommandApprovalHook` | Valutazione e accettazione della policy corrente | `evaluate` non modifica la history; `record_approved_command` registra solo la decisione accettata dal hook; `ApprovalModeChanged` riavvia la valutazione manuale |
| `DesignHistory` / `DesignSnapshot` | Snapshot indipendenti e annullamento delle modifiche locali alla lavagna | `capture`, `record`, `undo`, `can_undo`, `clear` |
| `AgentDesigner` | Lavagna, selezione e applicazione delle modifiche | `build`, `open`; controlli e stato interno restano privati |
| `CapabilityEditor` | Form di modifica di una capability su uno snapshot indipendente | `open`, callback tipizzata; interpreta e salva tramite `CapabilityRepository` |
| `validate_agent_design` | Validazione delle capabilities reali al confine con ExMekh-Lib | Costruzione e caricamento tramite le API pubbliche di `Node` |
| Costruttori della gerarchia | Nodi, figli, capability bundled e identità dell'agente | `build_generic_agent`, `build_designed_agent`; il runner coordina i turni senza costruire direttamente i nodi |
| `AgentSelector` | Scelta e personalizzazione del profilo nella UI | `set_value`, `profile`, `design_snapshot`, `apply_to`; i controlli NiceGUI restano privati |
| `CapabilityPathsEditor` | Cartella predefinita immutabile, riferimenti aggiuntivi e navigazione fuori dal thread UI | `build`, `set_paths`, `all_paths`, `additional_paths`, `browse` |
| `MessageEditor` | Dialog di modifica e durata degli upload temporanei | `open`, callback di salvataggio tipizzata; la chat non gestisce il ciclo di vita del form |
| `Disclosure` / `DisclosureState` | Interazione coerente e scelte di apertura locali alla vista | Clic, Enter e Spazio, visibilità, chevron e `aria-expanded` |
| `ChatRoutes` | URL delle conversazioni e risoluzione dei bookmark | `path_for`, `resolve`; repository esplicito e ID stabile anche dopo rinomina, refresh e navigazione Indietro/Avanti |
| `RecordedToolOutput` | Interpretazione dei metadati degli output registrati | `parse`, `cancelled_before_execution`; include corpo, riferimenti e thinking già interpretati |
| `ProviderResponse` (ExMekh-Lib) | Evidenze della risposta da registrare, distinte dalle chiamate eseguibili | `history_messages`; il client non costruisce ricevute di cancellazione |
| `package_paths` / `utils.resources` | Percorsi del pacchetto e lettura dei cataloghi, disponibili anche nel bootstrap | Costanti condivise, `catalog_message`; configurazione runtime e sync delegano allo stesso proprietario |
| `ToolEvent` | Identità, stato e durata di una chiamata | `execution_status`, `matches_update`, `merge_update`, `normalized_copy`; `record_tool_event` mantiene una sola voce per invocazione |
| `TurnStreamCollector` | Accumulo e snapshot del turno, compreso lo steering | `can_steer`, `steer`, `snapshot_message`, `build_message`; il registro non attraversa lo steering interno |
| `ActiveTurnRegistry` | Task attivi, revisioni, cancellazione e passaggio delle code | `start`, `finish`, `snapshot`, `can_steer`, `steer`, `stop_and_wait`; delega al raccoglitore la consegna della guida |

## Organizzazione dei moduli e delle risorse

La compattazione automatica separa il contesto di esecuzione dalla cronologia
ispezionabile. `AutoCompactionPolicy` confronta la prossima richiesta e gli ultimi
totali del provider con la soglia globale e il limite scoperto per endpoint/modello.
Il runtime del nodo esegue la sostituzione a batch completato e ricostruisce la
richiesta prima dell'inferenza. Le chiamate interne di scoring e riassunto non
attraversano la policy. User/system e ancore delle skill restano protetti: la
compattazione non garantisce una riduzione sotto soglia per qualsiasi input.

`HistoryArchive` possiede aggiornamenti, porzioni sostituite e confini del turno
per root e figli; espone snapshot indipendenti come `ModelContextCheckpoint`.
Il runner e gli osservatori usano questi metodi senza modificare i dati interni.
Le porzioni vengono espanse per Graph/Raw e normalizzate
con i lettori condivisi, anche negli alberi delle skill. Il checkpoint viene salvato
nel messaggio del turno: la ripresa e il conteggio usano il contesto effettivo,
mentre il transcript visibile conserva gli originali. I nodi figli hanno checkpoint
indipendenti. Replay, conteggio e vista Current context usano la stessa selezione
del checkpoint e includono tutti i messaggi successivi. Il provider mantiene lo
stato DS4 di ogni nodo anche senza osservatore UI; solo il root pubblica i propri
conteggi alla chat. La soglia è persistita; i limiti scoperti restano runtime e
sono attribuiti all'endpoint effettivamente interrogato.
La stima Qwen usa il testo letterale del client, come il server dopo la rimozione
delle istruzioni operative sintetiche. La query non conserva prefissi modificati
né aggiunge scope o messaggi; solo le risposte aggiornano i totali dei checkpoint.

La UI è divisa in `agents`, `chat`, `graph`, `history`, `navigation`, `settings`
e `shared`. Il bridge separa `agents`, `capabilities`, `history`, `providers`
e `tools`. Il pacchetto `models` separa `execution`, `messages`, `settings` e
`sessions`; il suo `__init__` mantiene gli import pubblici precedenti.
Il ciclo di vita degli eventi resta nei modelli di esecuzione, il branching e
le code nella sessione, gli override nelle impostazioni e il contenuto nel messaggio.
Anche il contesto sintetizzato usa `ToolEvent.is_running`: gli avvii legacy non
diventano risultati completati e gli errori terminali restano nel conteggio.
`state` resta una cartella coesa per stato e persistenza, senza aggiungere
sottopacchetti che conterrebbero un solo modulo.

`config` è un pacchetto con moduli separati per provider, server, agenti, storage,
allegati, esecuzione, formattazione, ambiente, percorsi e approvazioni.
`application.py` compone i default e mantiene `AppConfig` e l'istanza `config`,
con gli stessi import pubblici e la stessa persistenza degli override di ambiente.

`config/resources` contiene i valori modificabili divisi per competenza.
`config/texts` raccoglie etichette, suggerimenti, messaggi di errore e di log,
approvazioni e testi con parametri nominati. `config/presentation` contiene
geometria del grafo e della lavagna, icone, ordine dei fogli CSS e sanitizzazione
Markdown; `controls` separa gli stili dei componenti. Gli identificatori dei
protocolli, i campi degli schemi e le operazioni matematiche restano nel codice.

HTML, SVG e il template JavaScript per il testo incollato appartengono a
`ui/resources/templates`; CSS e JavaScript hanno le proprie cartelle.
Le istruzioni DS4, della quick chat, dei profili, delle approvazioni e della
generazione dei titoli sono in `exmekh_bridge/resources/prompts`.
Il fallback della quick chat usa lo stesso prompt della chiamata al modello e
del conteggio del contesto. Le risorse sono risolte dal pacchetto e incluse nella wheel.

`utils` è divisa in `files`, `filenames`, `formatting`, `exceptions`, `markdown`
e `resources`. La UI importa i formatter dal loro modulo proprietario; il bridge
e lo storage conservano gli alias pubblici precedenti per compatibilità.
I lettori condividono la cache, mentre i cataloghi restituiscono copie indipendenti
delle impostazioni. Il parser Markdown/YAML non dipende dal repository delle
capability. Stato e configurazione usano la stessa utility di scrittura atomica:
i file temporanei predefiniti sono univoci e vengono rimossi anche in caso di errore.

## Regole per le modifiche successive

- Il modello della lavagna non importa ExMekh-Lib e non esegue operazioni su file.
  I cambiamenti strutturali passano dai suoi metodi: aggiunta, sostituzione e cambio
  del root ripristinano lo stato precedente se la validazione fallisce.
- I nomi delle capability devono essere univoci per tipo. Le vecchie schede
  duplicate con sorgente e contenuto identici vengono consolidate conservando tutti
  i collegamenti. Le definizioni in conflitto e i cicli creati da una consolidazione
  vengono rifiutati senza modifiche parziali. Il rollback di un’aggiunta fallita
  conserva anche l’ordine originale dei nodi, inclusa l’aggiunta della stessa istanza.
- Le capabilities originali non vengono modificate. Il repository interpreta il
  nome dichiarato in `@deep_tool`, anziché presumere che coincida con la cartella;
  rinomina solo quel valore, conservando commenti, altre stringhe e testo Unicode.
  I nomi già presenti nel catalogo non possono essere riutilizzati per una copia.
- La lavagna include le dipendenze annidate delle skill e riutilizza i nodi condivisi.
  Le configurazioni precedenti importano le dipendenze YAML una sola volta; da quel
  momento i collegamenti sono la fonte delle dipendenze, anche per le skill bundled.
  L’editor espone descrizione e istruzioni separate: il repository interpreta il
  front matter e genera `sub-skills` e `sub-tools` dai collegamenti. I metadati non
  gestiti dal form restano conservati. Le copie mantengono identità e collegamenti
  del nodo; i cicli tra skill vengono rifiutati senza mutazioni parziali.
- Un tool fornito dall’utente non diventa un tool bundled solo perché ne condivide
  il nome. La provenienza della capability fa parte della decisione di compilazione.
  I tool runtime provengono dall’elenco pubblico esposto dalla libreria.
- La UI usa le API dei componenti e snapshot indipendenti delle configurazioni.
  La selezione Custom si apre nel contesto NiceGUI dell’evento e l’annullamento
  ripristina il profilo precedente. Le operazioni su file rimangono nei worker.
- L’editor conserva gli allegati esistenti se il chiamante aggiunge upload senza
  specificare una selezione. Durante il salvataggio non permette rimozioni o
  annullamenti che cancellerebbero file ancora utilizzati dalla callback.
- Gli output rimangono compatti fino all’interazione dell’utente. Tool e skill
  condividono la stessa interazione da tastiera e gli stessi metadati di accessibilità.
  La ricerca degli output appartiene al messaggio; argomenti e risultati condividono
  lo stesso attraversamento delle history, compresi gli output legacy di tipo
  `function`. L’apertura di righe e risultati appartiene alla vista: non modifica
  il modello dell’esecuzione e sopravvive ai suoi ridisegni. Ogni skill completata,
  incluse quelle annidate, resta compatta fino al clic dell’utente.
- Il raw inspector conserva l'header XML nel content originale e ne formatta
  l'indentazione; la vista chat usa invece i metadati grafici dell'output. La copia
  del JSON mantiene sempre il contenuto registrato. I risultati di `read_tool_output`
  mantengono il nome del tool originario nell'header: la correlazione considera
  output ID e modalità di sintesi, senza riscrivere le conversazioni salvate.
  Un risultato cached si associa prima per identità, poi per nome: un'altra
  chiamata in attesa allo stesso tool non può assorbirlo. Il grafo normalizza uno
  snapshot indipendente tramite l'adapter della history, come la vista chat,
  includendo le risposte legacy di ruolo function.
- Le opzioni Connect to appartengono al modello: sono verificabili con le stesse
  regole di validazione dei collegamenti e non includono archi già presenti.
  Skill e tool condivisi restano collegabili; HDS già assegnati e cicli sono esclusi.
- L'annullamento conserva design, catalogo locale e selezione prima di ogni modifica
  riuscita. Ctrl+Z / Cmd+Z e il pulsante Undo ripristinano gli snapshot in ordine
  inverso; modifiche senza effetto e navigazione non aggiungono passi. Il testo
  nei form conserva l'undo del browser. Alla riapertura dell'editor la cronologia
  riparte da zero; le copie su disco restano nella cache, gli originali intatti.
- Durante il trascinamento la geometria delle linee viene aggiornata nel browser
  a ogni movimento. Lo spostamento viene registrato al rilascio; l'annullamento
  durante il drag ripristina nodo e linee senza modificare il modello. Le frecce
  usano un passo configurabile di 48 px e scorrimento animato, rispettando la
  preferenza di sistema per la riduzione delle animazioni.
- Il salvataggio della lavagna valida e restituisce uno snapshot indipendente.
  Durante la validazione i controlli sono bloccati; richieste ripetute vengono
  ignorate e gli errori restituiscono una lavagna ancora modificabile.
- La configurazione globale applica le modifiche nel loop chiamante; `update_async`
  consegna al worker uno snapshot indipendente. Solo i campi dichiarati sono
  aggiornabili: un argomento non può sostituire un metodo. Letture e scritture
  fallite conservano il file esistente, e la UI mostra un errore con possibilità
  di riprovare. Il form impedisce invii ripetuti e non tocca controlli di un client
  eliminato durante la persistenza o la callback. Il refresh dei modelli evita
  richieste sovrapposte e scarta risultati arrivati dopo un cambio di endpoint;
  il provider conserva l'autenticazione esplicitamente vuota.
- La persistenza asincrona delle impostazioni del progetto esegue solo le operazioni
  su file nel worker. Poi aggiorna le chat nel loop chiamante usando il loro stato
  più recente, conservando le approvazioni arrivate durante il salvataggio. Se il
  progetto viene eliminato nel frattempo, rimuove l’eventuale record ricreato.
  Il refresh del provider non aggiorna controlli appartenenti a un client eliminato.
- Il conteggio usa copie indipendenti dello stato; un worker non osserva le
  mutazioni dello streaming. Pubblica i conteggi intermedi dello stesso target,
  scarta quelli di conversazioni o generazioni superate e riconteggia alla fine.
  Errori di conteggio non impediscono le richieste successive; la chiusura del
  client disattiva consegne e richieste in sospeso.
- Le invocazioni ripetute della stessa skill hanno scelte di apertura separate
  anche nel raw inspector. Expand/Collapse all conserva lo stato delle skill
  annidate; un errore di esecuzione viene mostrato come errore.
- Il ripristino multimodale e delle identità dei tool visita ogni history una sola
  volta, su uno snapshot indipendente. Risposte legacy function e risposte
  tool condividono lookup e presentazione. Gli eventi terminali conservano
  le durate registrate anche se non è stato osservato il loro evento iniziale.
- Il runner costruisce soltanto la gerarchia selezionata; non crea più una gerarchia
  legacy per poi scartarla quando viene richiesto un profilo o design personalizzato.

## Limiti della valutazione

Clean Code è un insieme di criteri di progettazione, non un elenco certificabile
tramite la copertura dei test. I componenti UI storici (`ChatViewComponent`,
`InputBarComponent`, `ChatPageController`) e `graph_data.py` restano estesi e
possono essere ulteriormente suddivisi per funzionalità. In questa revisione sono stati separati i modelli per dominio, il conteggio asincrono del contesto,
la lettura delle history e il formatter JSON. Sono stati estratti il form di modifica dei messaggi e quello delle capability e sono state rimosse le dipendenze
più dirette dalle strutture interne introdotte con la personalizzazione degli agenti.
I modelli Pydantic espongono i dati per serializzazione; le mutazioni della lavagna
in produzione ora usano i metodi del modello invece di ricostruirne le invarianti nella UI.

## Revisione delle Correzioni 3, 14 ottobre 2026

Ogni pagina riceve una vista di `SessionManager`: i record e gli store sono
condivisi, mentre la selezione appartiene alla pagina. La pagina iniziale conserva
l'ultima chat visitata. Aprire un bookmark in un'altra scheda non modifica la
conversazione mostrata, il destinatario degli input o la cartella degli upload
nella pagina già aperta. Il fallback dopo un'eliminazione usa l'ordine dei record.

Chat e grafo consumano `ExecutionTimeline` per la history strutturata. Conferme
senza un call ID disponibile usano il tempo di conferma; gli ID ripetuti sono
risolti per scope e le invocazioni ripetute conservano identità distinta. Le
conferme allo stesso confine mantengono l'ordine FIFO. `TurnSteering` delega il
ripristino multimodale a `RecordedHistory`, senza conoscere la struttura dei
transcript delle skill. Il ripristino completo visita ogni scope una sola volta.

Titoli, profili, modalità e history delle approvazioni pubblicano lo snapshot
solo dopo un salvataggio riuscito. Il hook gestisce tutte le modalità attraverso
una decisione strutturata; una valutazione LLM obsoleta non entra nella history.
Un cambio di modalità interrompe una richiesta manuale con un segnale esplicito
di rivalutazione. La sincronizzazione del select della UI non simula una scelta
dell'utente e non risalva le impostazioni. Normalizzazione e identificatori
appartengono a `config.approvals`, senza dipendenza dal client LLM.

La geometria resta responsabilità di `AgentDesign`: l'aggiunta automatica è
atomica anche rispetto alle coordinate della scheda del chiamante e usa un gap
esplicito. I nodi spostati manualmente conservano le loro posizioni.

La lettura si è estesa alle API `CalledTool`, `ToolExecutionOperation` e
`FileWriteTarget` della libreria e ai confini tra documento, parser e runtime
interattivo del server. In quei percorsi non sono state introdotte riscritture
senza un difetto dimostrato. Restano candidati a una revisione dedicata i grandi
componenti UI e la decomposizione dei moduli C; non si certifica l'intero
repository tramite la copertura del frontend. I kernel non sono stati modificati.

Verifica conclusiva di questa revisione: `uv run pytest -n 4` nel frontend,
**984 passed, 43 skipped**, copertura configurata di istruzioni e rami **100%**.
La selezione deterministica sui contratti di tool, history, file e steering della
libreria passa con **171 test**. Nel repository principale passano la build
`ds4_server_test`, `test-server-interactive` e i **5 test** di
`test-server-sampling-matrix-static`, senza caricare modelli. `ruff check src tests`,
la verifica di formattazione dei sorgenti e dei test modificati e `git diff --check`
non segnalano errori. Le modifiche preesistenti ai test live sono escluse dal commit;
la formattazione preesistente di `test_live_e2e.py` rimane da uniformare nel suo lavoro.

## Verifica

`uv run pytest -n 4`: **907 passed, 40 skipped**.
Copertura configurata di istruzioni e rami: **100.00%**. I test con provider reale
richiedono opt-in; i test e2e con mock sono stati eseguiti.
La selezione live di questa revisione passa con **4 test**: health check,
risposta e reasoning DeepAgent, esecuzione dei tool e history multi-turno.
I test completi includono gli e2e su personalizzazione, coda, modifica dei messaggi,
impostazioni su desktop e telefono e le sei sequenze complesse di Ctrl+Z / Cmd+Z.
`uv run ruff check src tests`, `uv run ruff format --check src tests` e
`git diff --check -- .` non segnalano errori.

Le regressioni aggiunte verificano le mutazioni atomiche della lavagna, i nomi
espliciti e impliciti dei tool, le copie senza alterare gli originali, i metadati
YAML invalidi, gli override di profilo invalidi, la conservazione degli allegati,
l’annullamento di Custom e la tastiera nelle righe di tool e skill. Le nuove
regressioni verificano inoltre il form senza YAML, la descrizione separata, i
collegamenti annidati e condivisi, le dipendenze generate dalle connessioni anche
per le skill bundled e la conservazione delle rimozioni dopo la riapertura. I
test e2e verificano il trascinamento oltre le dimensioni iniziali e il mantenimento
della posizione di scorrimento durante la selezione.


Le regressioni sulla seconda revisione coprono i salvataggi ripetuti, gli snapshot
indipendenti, il recupero dopo errori di validazione, le schede duplicate legacy,
il rollback delle mutazioni, le etichette accessibili con virgolette e Unicode,
le approvazioni concorrenti e l’eliminazione di progetti durante il salvataggio.
I test e2e verificano le scelte di apertura dopo il ridisegno della chat e l’apertura
esplicita delle skill annidate, senza modificare i dati dell’esecuzione.


La riorganizzazione verifica inoltre il conteggio su snapshot indipendenti, la
coalescenza senza risultati obsoleti, il recupero dopo errori di conteggio e la
chiusura dei client durante un worker attivo. I cataloghi vengono controllati
per chiavi e parametri mancanti; i template verificano escaping, frammenti
sanitizzati e variabili obbligatorie. La wheel contiene le risorse di configurazione, UI e prompt; la verifica confronta
ogni file con il sorgente e importa il pacchetto estratto da una directory diversa.
La dipendenza installata corrisponde alla wheel versionata.

Le regressioni della lavagna coprono Connect to per HDS, skill e tool, dipendenze
condivise, cicli, connessioni esistenti e nessun target disponibile; verificano
l'annullamento di spostamenti, collegamenti, rimozioni, root, flag di history,
aggiunte e copie. Playwright controlla le linee prima del rilascio, l'annullamento
del drag in corso, i menu con focus e i dialoghi in chiusura. Le history legacy
verificano tre modalità di lettura dello stesso output e risultati fuori ordine,
conservando i dati salvati; il raw inspector mantiene l'XML anche se incompleto.

I test e2e delle sequenze di undo eseguono gli stessi tre percorsi con Control+Z
e Meta+Z (Cmd+Z): sette modifiche miste annullate in ordine inverso, due undo
seguiti da una nuova aggiunta con dipendenze condivise, e dialoghi con testo,
focus sui pulsanti, modifiche rifiutate, menu aperti e riapertura dell'editor.
Ogni checkpoint confronta nodi, posizioni, root, selezione, curve SVG e catalogo;
il form verifica anche il ripristino delle istruzioni dopo la copia. Il drag
annullato non consuma le modifiche precedenti. Lo scorrimento non aggiunge passi,
Shift+Z con il modificatore di undo non viene interpretato come undo e l'undo a
cronologia vuota non modifica il grafo. Le azioni sono eventi reali del browser;
gli helper leggono il DOM senza cambiare il modello o invocare callback private.
Meta viene verificato su Chromium nel runner disponibile, senza simulare una
validazione nativa macOS.

La configurazione verifica inoltre l'assenza di chiavi JSON duplicate, gli override
di ambiente, la conservazione di commenti e variabili estranee durante il
salvataggio, le copie indipendenti e i riferimenti ai testi e agli stili.
I template di presentazione devono ricevere esattamente i parametri dichiarati.
Un controllo AST impedisce di reintrodurre testi, icone e stili letterali nei
controlli NiceGUI. Le utility verificano scritture concorrenti con file temporanei
distinti e la conservazione della destinazione quando scrittura o sostituzione falliscono.

Le regressioni della revisione finale verificano anche thread e snapshot del
salvataggio globale, errori e retry, richieste ripetute e client eliminati durante
persistenza, callback e lettura delle statistiche. La discovery scarta metadata
di capacità arrivati dal provider precedente, conservando l'owner nel suo endpoint.
Le letture cached sono verificate con tool ordinari aventi lo stesso nome,
ordine delle risposte invertito, due modalità di sintesi e ruoli `tool` / `function`.

Gli helper e2e attendono la pubblicazione dell'input nella chat o nella coda prima
di proseguire; la fine del turno richiede sia il compositore pronto sia l'assenza
di righe in streaming. Questo elimina la race tra l'evento di invio e il suo
ricevimento sul server, senza assumere che un ritardo fisso basti.

Il designer mantiene il catalogo scorrevole a tutte le larghezze; le proprietà
passano sotto la lavagna sui viewport intermedi e il catalogo diventa una fascia
orizzontale sui telefoni. Le connessioni hanno icona e testo su una sola riga,
allineamento a sinistra ed ellissi, conservando nome accessibile e tooltip completi.
Il nuovo e2e ridimensiona lo stesso editor da 1440 a 320 pixel, include il formato
orizzontale e il ritorno al desktop, controlla sbordamenti e sovrapposizioni e
verifica aggiunta, disconnessione e undo dopo il resize.

Le impostazioni di progetto ordinano la scelta dell'agente prima delle approvazioni,
lasciando le cartelle delle capability in una sezione espandibile. Quattro icone
distinte espongono nomi accessibili, tooltip e lo stato di selezione; Custom apre
direttamente il designer. Il flag di history si modifica sulla lavagna.
Il picker nativo è condiviso tra creazione del progetto e cartelle delle capability;
la validazione del riferimento appartiene a CapabilityRepository. ProjectSettings
risolve la lista ordinata e senza duplicati per runtime, catalogo, prompt e conteggio.
I nuovi e2e verificano il percorso completo su desktop e telefono, inclusi
annullamento, duplicati, rimozione, salvataggio, esecuzione e refresh.
