# Responsabilità e API dell’applicazione

La revisione ha considerato UI, stato, integrazione con ExMekh-Lib e i flussi di
streaming, approvazione, persistenza, allegati e visualizzazione degli output.
Gli interventi si concentrano sulle responsabilità che erano mescolate e sui
comportamenti incoerenti verificabili con test.

| Componente | Responsabilità | API da usare |
| --- | --- | --- |
| `AgentDesign` | Struttura della lavagna e invarianti di nodi/collegamenti | `add_node`, `replace_node`, `move_node`, `connect`, `disconnect`, `remove_node`, `set_root`, `set_persistent_history`, `consolidate_capabilities`, `capability`, `children`, `connections`, `connection_targets` |
| `CapabilityRepository` | Catalogo, copie e configurazioni su disco per ExMekh-Lib | `catalog`, `default_design`, `expand_skill_connections`, `add_to_design`, `editable_values`, `copy_for_design`, `copy_capability`, `compile` |
| `CompiledAgentDesign` | Accesso alla configurazione generata | `capabilities_for`, `default_names_for`; i chiamanti non devono conoscere i file interni |
| `ProjectSettings` / `ChatSettings` | Configurazione e risoluzione dei default/override | `with_agent`, `with_agent_override`, `with_project` |
| `SessionManager` | Persistenza e aggiornamento dello stato gestito | `save_chat_agent`, `save_project_settings`, `save_project_settings_async`, `execution_settings`, `edit_message` |
| `RecordedHistory` | Attraversamento, correlazione e ripristino delle history registrate | `scopes`, `arguments_for_call`, `output_for_call`, `restored_snapshot` |
| `ContextUsageUpdater` | Snapshot del conteggio, coalescenza e consegna nel contesto del client | `request`, `schedule`, `wait_idle`, `close` |
| Risorse UI | Cataloghi dei testi, configurazioni e template HTML/SVG | `ui_text`, `presentation_config`, `render_html`; escaping automatico, frammenti sanitizzati espliciti |
| `ChatMessage` | Ricerca di argomenti e risultati nelle history registrate | `arguments_for_tool`, `output_for_call`; i chiamanti non devono attraversare le history per correlare invocazioni |
| `DesignHistory` / `DesignSnapshot` | Snapshot indipendenti e annullamento delle modifiche locali alla lavagna | `capture`, `record`, `undo`, `can_undo`, `clear` |
| `AgentDesigner` | Lavagna, selezione e applicazione delle modifiche | `build`, `open`; controlli e stato interno restano privati |
| `CapabilityEditor` | Form di modifica di una capability su uno snapshot indipendente | `open`, callback tipizzata; interpreta e salva tramite `CapabilityRepository` |
| `validate_agent_design` | Validazione delle capabilities reali al confine con ExMekh-Lib | Costruzione e caricamento tramite le API pubbliche di `Node` |
| `AgentSelector` | Scelta e personalizzazione del profilo nella UI | `set_value`, `profile`, `design_snapshot`, `apply_to`; i controlli NiceGUI restano privati |
| `MessageEditor` | Dialog di modifica e durata degli upload temporanei | `open`, callback di salvataggio tipizzata; la chat non gestisce il ciclo di vita del form |
| `Disclosure` / `DisclosureState` | Interazione coerente e scelte di apertura locali alla vista | Clic, Enter e Spazio, visibilità, chevron e `aria-expanded` |
| `RecordedToolOutput` | Interpretazione dei metadati degli output registrati | `parse`; include corpo, riferimenti e thinking già interpretati |
| `ToolEvent` | Identità, stato e durata di una chiamata | `execution_status`, `matches_update`, `merge_update`, `normalized_copy`; `record_tool_event` mantiene una sola voce per invocazione |

## Organizzazione dei moduli e delle risorse

La UI è divisa in `agents`, `chat`, `graph`, `history`, `navigation`, `settings`
e `shared`. Il bridge separa `agents`, `capabilities`, `history`, `providers`
e `tools`; i modelli condivisi rimangono nel suo livello principale.
`state` resta una cartella coesa per stato e persistenza, senza aggiungere
sottopacchetti che conterrebbero un solo modulo.

I testi della UI risiedono in `ui/resources/text`, suddivisi per funzionalità.
I valori dinamici sono parametri nominati. Icone dei file, colori del grafo,
geometria della lavagna, ordine dei fogli CSS e policy di sanitizzazione Markdown
risiedono in `ui/resources/config`. HTML e SVG appartengono ai template;
CSS e JavaScript hanno le proprie cartelle. Le istruzioni DS4, della quick chat
e della generazione dei titoli sono in `exmekh_bridge/resources/prompts`.
Le risorse vengono risolte dal pacchetto e sono incluse nella wheel.

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
possono essere ulteriormente suddivisi per funzionalità. In questa revisione sono stati separati il conteggio asincrono del contesto,
la lettura delle history e il formatter JSON. Sono stati estratti il form di modifica dei messaggi e quello delle capability e sono state rimosse le dipendenze
più dirette dalle strutture interne introdotte con la personalizzazione degli agenti.
I modelli Pydantic espongono i dati per serializzazione; le mutazioni della lavagna
in produzione ora usano i metodi del modello invece di ricostruirne le invarianti nella UI.

## Verifica

`uv run pytest -n 4`: **858 passed, 40 skipped**.
Copertura configurata di istruzioni e rami: **100.00%**. I test con provider reale
richiedono opt-in; i test e2e con mock sono stati eseguiti.
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
sanitizzati e variabili obbligatorie. La wheel contiene tutte le 61 risorse UI
e dei prompt; la dipendenza installata corrisponde alla wheel versionata.

Le regressioni della lavagna coprono Connect to per HDS, skill e tool, dipendenze
condivise, cicli, connessioni esistenti e nessun target disponibile; verificano
l'annullamento di spostamenti, collegamenti, rimozioni, root, flag di history,
aggiunte e copie. Playwright controlla le linee prima del rilascio, l'annullamento
del drag in corso, i menu con focus e i dialoghi in chiusura. Le history legacy
verificano tre modalità di lettura dello stesso output e risultati fuori ordine,
conservando i dati salvati; il raw inspector mantiene l'XML anche se incompleto.
