# Semplificare i tool di scrittura, modifica e cancellazione — 13 ottobre 2026

Ricerca sul contratto attuale DS4/ExMekh e sulle alternative per ridurre errori del modello e complessità del formato. La proposta principale è separare `write-file`, `edit-file` e `delete-file`, usare argomenti strutturati ordinari e conservare il vincolo dinamico di selezione sullo snapshot. Questo documento propone esperimenti e una direzione; non modifica il protocollo operativo.

## 1. Risultato della lettura

Il punto fragile è l'accoppiamento fra operazione, selezione e rappresentazione del contenuto. Un'unica stringa deve descrivere creazione, riscrittura, sostituzione, inserimento, selezione per righe, svuotamento e cancellazione. Nei corpi convivono testo del file, comandi XML e due rappresentazioni del testo letterale: entità e CDATA.

Il modello deve quindi decidere sia cosa cambiare, sia in quale sottolinguaggio esprimerlo. Il motore verifica che il risultato appartenga al linguaggio ammesso, ma una riscrittura valida può ancora essere la risposta sbagliata a una richiesta di cancellazione. Aggiungere istruzioni per scegliere il ramo giusto lascia intatta questa ambiguità.

La lunghezza della skill, da sola, non dimostra che un tool sia sbagliato: parte delle istruzioni riguarda lettura, PDF, immagini e policy. La sezione di editing, però, descrive diverse eccezioni del formato; le prove live documentano errori coerenti con quelle eccezioni.

### Codice e documentazione esaminati

- [Descrizione e adattatore server](../../ds4_interactive_edit.h): descrizione aggiunta al tool, scambio degli snapshot, composizione degli escape nativi e del parser XML.
- [Documento incrementale server](../../ds4_edit_document.h): CDATA/entità, chiusure, comandi, intervalli, checkpoint e proposte di copia dello snapshot.
- [Vincolo dinamico](../../ds4_edit_constraint.h): inizi di riga, candidati, prefissi globalmente univoci e indice dei suffissi.
- [Skill file-processing](../../exmekh_lib/src/exmekh_lib/capabilities/deep-skill/file-processing.md), [configurazione](../../exmekh_lib/src/exmekh_lib/config/file_editing.py) e [tool](../../exmekh_lib/src/exmekh_lib/capabilities/tools/edit-file/main.py): contratto presentato al modello in più punti.
- [Parser client](../../exmekh_lib/src/exmekh_lib/core/models/edit_document.py), [selezioni](../../exmekh_lib/src/exmekh_lib/core/file_inspection/edit_selection.py), [autorizzazione](../../exmekh_lib/src/exmekh_lib/core/file_inspection/editing.py) e [filesystem](../../exmekh_lib/src/exmekh_lib/core/file_inspection/edit_filesystem.py).
- [Contratto interattivo](../agentic/interactive-calls.md), [piano e storico](../roadmaps/interactive-edit-file-server-plan.md), [qualifica client](client-xml-edit-v2-20261012.md), [revisione successiva](client-xml-clean-review-20261012.md) e [copia speculativa dello snapshot](../performance/old-text-snapshot-chunks-20261012.md).

### Evidenze locali

La revisione client più recente riporta **13 PASS e 2 FAIL su 15 casi target-only**, senza skip. Delete e svuotamento esauriscono entrambi 20 turni. Per delete il file finale contiene `\n<delete/>\n`; per lo svuotamento contiene un LF. I casi MTP selezionati passano, ma non qualificano quelle due operazioni. Fonte: [revisione client](client-xml-clean-review-20261012.md) e [trace target-only](../performance/client-xml-clean-review-20261012-target-wire.json).

Anche il [trace della qualifica precedente](../performance/client-xml-edit-v2-20261012-a3-wire.json) conserva chiamate come questa:

```xml
<edit path="notes.txt"><old_text><all/></old_text><new_text>
<![CDATA[<delete/>]]>
</new_text></edit>
```

Il client la interpreta correttamente come testo da scrivere. Trasformarla automaticamente in unlink renderebbe impossibile scrivere quel testo letterale. Il problema è che una cancellazione viene rappresentata attraverso un campo che serve anche a generare contenuto.

La prova sulla copia dello snapshot ha inoltre trovato un ciclo di CDATA vuoti a EOF: la selezione era già corretta, ma il modello continuava ad aggiungere involucri fino al budget. Il guard specifico ora impedisce quel ciclo. È un esempio di complessità introdotta dalla rappresentazione, ulteriore rispetto alla selezione stessa. Fonte: [report snapshot](../performance/old-text-snapshot-chunks-20261012.md).

Queste evidenze non identificano da sole la causa interna nei logits, nel framing o nello stato ripristinato. Sostengono però la necessità di confrontare un'interfaccia con meno scelte equivalenti e meno operazioni implicite.

## 2. Cosa fanno altri strumenti

Questa sezione segue il processamento dei file per agente: scoperta, ispezione, lettura, creazione, modifica, inserimento, rimozione, rinomina e verifica. Distingue tool del modello, comandi dell'interfaccia utente e contratti API. Gli esempi sono casi illustrativi originali, non trace di prove eseguite.

Fonti primarie consultate il 13 ottobre 2026. I repository sono descritti nell'istantanea di `main` consultata, senza commit fissato. Per Codex sono verificate anche le dichiarazioni dei tool esposte in questa chat. Dove è disponibile solo il contratto pubblico non viene inventato l'algoritmo interno. Non sono state riprodotte le misure esterne su DS4.

### 2.1 Codex: ricerca e lettura tramite strumenti disponibili, modifiche tramite patch

Il catalogo di tool dipende dall'harness, non dal solo nome del modello. La documentazione CLI descrive un agente che ispeziona il repository locale, lo modifica ed esegue comandi. Lettori dedicati, shell e strumenti MCP possono coesistere. Fonte: [Codex CLI](https://learn.chatgpt.com/docs/codex/cli).

#### Scoperta, ispezione e lettura

Nel set di questa chat ricerca e lettura testuale passano soprattutto per `exec_command`: il modello genera un comando, il runtime avvia il processo e restituisce output, exit code o session ID se continua. Il risultato è quello del programma eseguito; non uno schema universale di documento.

| Tool esposto qui | Input rilevante | Funzione sui file |
| --- | --- | --- |
| `exec_command` | `cmd`, `workdir`, `shell`, budget di output/attesa, `tty`, eventuali permessi | Ricerca, listing, metadati, lettura, script, conversioni, Git e test |
| `write_stdin` | `session_id`, `chars`, budget di output/attesa | Input o altro output di un processo già avviato |
| `apply_patch` | Payload testuale freeform | Creazione, aggiornamento, spostamento e cancellazione |
| `view_image` | `path`, `detail` | Ispezione visiva di un'immagine locale |

Queste firme vengono dalle dichiarazioni locali: non sono una lista identica per ogni versione di Codex. `functions.exec` orchestra le chiamate in JavaScript; non aggiunge una selezione di testo del file.

Esempio del percorso via shell:

```text
exec_command(cmd="rg --files -g '*.py'", workdir="/work/project")
exec_command(cmd="rg -n 'parse_config' src", workdir="/work/project")
exec_command(cmd="sed -n '40,100p' src/config.py", workdir="/work/project")
```

Listing e metadati si ottengono con `ls`, `stat`, `file`, `wc` o equivalenti PowerShell. Un PDF, archivio o formato binario può essere analizzato invocando programmi/librerie adatti. `view_image` non è un parser universale di documenti. Se l'output è tagliato, il modello deve richiedere un altro intervallo o cercare più precisamente.

La guida ufficiale suggerisce `rg`/`rg --files` e lettori dedicati quando presenti. Cita `read_file`, `list_dir` e `glob_file_search` come possibili strumenti dell'harness; non ne garantisce il caricamento in ogni sessione. Descrive `shell_command` e la coppia `exec_command`/`write_stdin` per processi interattivi. Fonte: [Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide).

#### Creazione, edit, insert, rimozione e rename

`apply_patch` separa l'operazione nell'intestazione dal corpo orientato alle righe. Il modello produce una patch, senza incapsularla in XML. Esempio:

```text
*** Begin Patch
*** Add File: notes.txt
+prima nota
*** Update File: config.py
@@
-enabled = False
+enabled = True
+verbose = False
*** Delete File: obsolete.txt
*** End Patch
```

| Operazione | Forma | Effetto |
| --- | --- | --- |
| Creare | `Add File`, righe `+` | Contenuto del nuovo file |
| Sostituire | `Update File`, righe `-` e `+` | Rimozione del vecchio tratto e aggiunta del nuovo |
| Inserire | Righe `+` vicino al contesto invariato | Inserimento localizzato dal contesto |
| Rimuovere testo | Righe `-` senza aggiunte | Il file rimane |
| Delete del file | `Delete File` senza corpo | Cancellazione esplicita |
| Rename/move | `Move to` nell'update | Destinazione e possibile edit |
| Localizzare | Contesto, ancora `@@`, eventuale marcatore EOF | Non richiede i conteggi standard degli hunk unified diff |

La grammatica freeform è pubblicata nella [guida Codex](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide). I prefissi del diff alterano la rappresentazione visiva di old; il contenuto, però, non richiede entità XML o CDATA. Rimangono framing e delimitatori: freeform significa una sintassi testuale, non assenza di sintassi.

#### Contratto API, risultati e garanzie

Il tool **Responses API** `apply_patch` è distinto dalla forma freeform locale. Restituisce operazioni `create_file`, `update_file`, `delete_file`, con `path` ed eventualmente `diff`. Delete non porta diff. Il chiamante applica e restituisce `apply_patch_call_output` con call ID e successo/errore. L'atomicità complessiva o per file è una scelta del chiamante, non una garanzia implicita del formato. Fonte: [Apply Patch API](https://developers.openai.com/api/docs/guides/tools-apply-patch).

Anche il tool shell API è diverso dalla firma locale: il modello propone comandi, l'ambiente li esegue e restituisce stdout/stderr e stato. Fonte: [Shell API](https://developers.openai.com/api/docs/guides/tools-shell).

Nel set locale esaminato non è esposto uno scambio pubblico `path → snapshot → old vincolato` come in DS4. La validità della sintassi non dimostra quindi che sia stato impedito al modello di generare old inesistente. Per la nostra progettazione il precedente utile è la dichiarazione esplicita delle operazioni. Verifica e recupero possono usare rilettura, diff e comandi di test; una trasformazione molto ripetitiva può essere espressa da uno script anziché da una grande patch.

### 2.2 Claude Code: Read, Edit e Write, ricerca e shell

Claude Code cerca, legge, modifica ed esegue verifiche nel proprio loop. L'Agent SDK incorpora lo stesso agente e i suoi strumenti; un Client SDK dell'API richiede invece il loop e le implementazioni del chiamante. Il text editor API del paragrafo successivo è un contratto ulteriore. Fonti: [ciclo di Claude Code](https://code.claude.com/docs/en/how-claude-code-works), [Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).

#### Tool e argomenti

| Tool | Input | Lavoro svolto |
| --- | --- | --- |
| `Read` | `file_path`, `offset?`, `limit?` | File o porzione di righe |
| `Glob` | `pattern`, `path?` | Percorsi per glob |
| `Grep` | `pattern`, `path?`, `glob?`, `output_mode?`, `-i?`, `multiline?` | Ricerca di contenuto; righe, percorsi o conteggi |
| `Bash` / `PowerShell` | `command` e opzioni di esecuzione | Listing, metadati, conversioni, Git, comandi su file e test |
| `Edit` | `file_path`, `old_string`, `new_string`, `replace_all` | Sostituzione testuale |
| `Write` | `file_path`, `content` | Creazione o contenuto completo |

Le firme sono nella [reference degli input degli hook](https://code.claude.com/docs/en/hooks#pretooluse-input), che documenta le chiamate reali. Non richiedono documenti XML dentro gli argomenti.

Read restituisce testo numerato e gestisce anche immagini, PDF e notebook; per directory si usa la shell. Nella documentazione corrente Glob/Grep sono normalmente esposti su Windows; macOS/Linux/WSL usano di default la ricerca via Bash, con condizioni per riabilitare i due tool. Fonte: [reference dei tool](https://code.claude.com/docs/en/tools-reference).

Per file lunghi offset/limit permette letture mirate. Il changelog 2.1.296 aggiunge `allow_large`, quando il contesto può contenere una lettura oltre i limiti ordinari. Fonte: [changelog ufficiale](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md).

#### Edit: sostituzione, inserimento e rimozione di testo

```json
{"file_path":"/work/config.py","old_string":"enabled = True\n","new_string":"enabled = True\nverbose = False\n","replace_all":false}
```

Questo inserisce dopo l'ancora mantenendola in new. Per inserire prima si inverte l'ordine del nuovo testo; per togliere un blocco si usa new vuoto. `replace_all` cambia tutte le occorrenze, non il file intero.

Edit richiede match esatto e, senza replace_all, univoco; non usa regex/fuzzy. La lettura preventiva dipende da modello/versione: i modelli meno recenti la richiedono; quelli nuovi hanno eccezioni quando Read è disponibile e non richiederebbe permesso. Un file cambiato può essere ammesso se old continua a coincidere esattamente e univocamente. Fonte: [Edit](https://code.claude.com/docs/en/tools-reference#edit-tool-behavior).

La conseguenza progettuale è una verifica al momento dell'esecuzione, diversa dal congelamento del file dalla lettura. Per DS4 dobbiamo decidere separatamente quali cambiamenti esterni accettare.

#### Write: creazione, riscrittura e svuotamento

Write crea o sovrascrive con il contenuto intero; non appende o fonde. Content vuoto lascia un file vuoto. La lettura preventiva di un file esistente ha eccezioni analoghe a Edit, con casi speciali per notebook e letture parziali. Fonte: [Write](https://code.claude.com/docs/en/tools-reference#write-tool-behavior).

Non ci sono old e marcatore all: il nome identifica già il tipo di cambiamento. Questa separazione è vicina al candidato write/edit locale.

#### Delete, rename, notebook e ispezione strutturale

Il catalogo pubblico non elenca un tool Delete filesystem: rimozioni e spostamenti passano dalla shell. Write vuoto non fa unlink. La documentazione del rewind cita proprio rm/mv/cp come cambiamenti da shell non ripristinati dai checkpoint ordinari. Fonte: [checkpointing](https://code.claude.com/docs/en/checkpointing).

NotebookEdit interviene sulle celle, con replace/insert/delete, anziché sostituire old nel JSON completo. Fonte: [NotebookEdit](https://code.claude.com/docs/en/tools-reference#notebookedit-tool-behavior).

I plugin di code intelligence aggiungono definizioni, riferimenti e diagnostica. Non costituiscono un inspect obbligatorio prima di ogni edit. Fonte: [categorie di tool](https://code.claude.com/docs/en/how-claude-code-works#tools).

#### Controlli e risultati

Permessi su file e shell, normalizzazione dei path e hook sono responsabilità del runtime. Un deny di lettura può impedire anche edit/write. Il modello riceve l'esito e può rileggere o ritentare. Fonti: [permessi](https://code.claude.com/docs/en/permissions), [hook](https://code.claude.com/docs/en/hooks#pretooluse-input).

I checkpoint offrono recupero delle modifiche tracciate, non una transazione universale. Le firme Read/Edit/Write non dimostrano da sole conservazione di ogni encoding o terminatore: quelle garanzie richiedono verifiche dell'implementazione specifica. La semplificazione dell'API e le policy di filesystem restano decisioni distinte.

### 2.3 Claude via Messages API: text editor con comandi espliciti

L'editor API `str_replace_based_edit_tool` ha schema definito da Anthropic. Il modello emette tool_use, ma lettura/scrittura sono implementate dal chiamante. Non è il catalogo Claude Code. Fonte: [text editor API](https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool).

| Comando | Argomenti | Operazione |
| --- | --- | --- |
| `view` | `path`, `view_range?` | File/directory; righe 1-based, -1 per EOF |
| `str_replace` | `path`, `old_str`, `new_str` | Sostituzione esatta e univoca |
| `create` | `path`, `file_text` | File nuovo con contenuto completo |
| `insert` | `path`, `insert_line`, `insert_text` | Dopo riga; 0 significa inizio |

La configurazione max_characters controlla la troncatura di view nelle versioni supportate. Il chiamante restituisce tool_result associato all'ID e is_error per errori, inclusi match zero/multipli. Fonte: [comandi e risultati](https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool).

```json
{"command":"insert","path":"/work/notes.txt","insert_line":0,"insert_text":"Titolo\n"}
```

La lista documentata non include delete del file. New vuoto elimina testo; unlink richiede un altro strumento del chiamante. Non gli vanno attribuiti hook o rewind di Claude Code. La variante mostra che anche un unico tool con campo comando può separare l'operazione dai dati: non è necessario nasconderla nel contenuto.

### 2.4 Aider: file nel contesto, repo map e coders di editing

Il percorso ordinario di Aider organizza i file nel contesto e interpreta la risposta del modello. L'utente nomina file all'avvio o li aggiunge nella chat; un coder trasforma le istruzioni di edit in cambiamenti. È diverso dal ciclo di chiamate Read → Edit degli altri agenti. Fonte: [uso di Aider](https://aider.chat/docs/usage.html).

#### Scoperta, ispezione e lettura

La repo map presenta percorsi, simboli e firme pertinenti entro un budget. Parsing e relazioni nel repository aiutano a selezionare contesto: la mappa non contiene necessariamente il file completo. Fonte: [repository map](https://aider.chat/docs/repomap.html).

| Comando dell'utente | Effetto |
| --- | --- |
| `/ls` | File conosciuti e presenti nel contesto |
| `/map`, `/map-refresh` | Visualizzazione/aggiornamento della mappa |
| `/add` | Caricamento di file editabili |
| `/read-only` | Caricamento di riferimenti non editabili |
| `/drop` | Rimozione dal contesto, senza cancellazione sul disco |
| `/run` | Comando shell e possibile inclusione dell'output |
| `/diff` | Differenze prodotte |
| `/lint`, `/test` | Verifiche e recupero |
| `/undo` | Annullamento dell'ultimo commit di Aider |

Questi sono comandi dell'interfaccia, non una lista di function tool autonomi del modello. Fonte: [comandi Aider](https://aider.chat/docs/usage/commands.html).

L'ispezione del codice avviene quindi attraverso mappe e file caricati, oppure output di comandi. Non emerge un passaggio obbligatorio inspect che restituisca prima numero di righe e metadati del file.

#### SEARCH/REPLACE: edit, insert, append e creazione

Il formato diff è percorso più fence e blocchi SEARCH/REPLACE. Il modello descrive i tratti cambiati e il coder produce il nuovo file; esistono varianti con il percorso dentro la fence, oltre a whole e udiff. Fonte: [formati di edit](https://aider.chat/docs/more/edit-formats.html).

````text
config.py
```
<<<<<<< SEARCH
enabled = False
=======
enabled = True
verbose = False
>>>>>>> REPLACE
```
````

Nel coder corrente SEARCH vuoto crea un file assente o appende a un file presente. REPLACE vuoto elimina il blocco e conserva il file. Il match prova l'esatto e recuperi mirati sull'indentazione; gestisce alcuni `...`. Il matching generico per distanza compare dopo un return e non è il fallback attivo. L'esatto ordinario seleziona il primo blocco corrispondente: non va attribuito il vincolo DS4 di unicità globale. Fonte: [editblock_coder.py](https://github.com/Aider-AI/aider/blob/main/aider/coders/editblock_coder.py).

Un insert interno mantiene un'ancora in REPLACE. Old vuoto è quindi una scorciatoia con semantica diversa dal nostro candidato. Il coder può togliere involucri e aggiungere newline; non è un riferimento per conservazione byte-esatta. Alcuni blocchi possono risultare applicati quando altri falliscono: il recupero comunica quelli riusciti affinché il modello non li ripeta. Fonte: [applicazione dei blocchi](https://github.com/Aider-AI/aider/blob/main/aider/coders/editblock_coder.py).

#### Whole e udiff, delete e rename

Il coder whole riceve il contenuto completo del file e lo riscrive. Udiff usa hunk orientati alle righe ed è implementato da un altro coder. Sono formati scelti dall'applicazione, non tag all nel campo vecchio testo. Fonti: [wholefile coder](https://github.com/Aider-AI/aider/blob/main/aider/coders/wholefile_coder.py), [udiff coder](https://github.com/Aider-AI/aider/blob/main/aider/coders/udiff_coder.py).

Nel percorso whole/SEARCH-REPLACE, eliminare tutto il contenuto produce un file vuoto. Delete e rename fisici richiedono un'operazione filesystem nel workflow, per esempio una shell. `/drop` non fa unlink. Non viene documentato un tool dedicato delete-file nel percorso ordinario descritto.

#### Risultati e verifica

L'integrazione Git offre diff, commit e undo, con auto-commit configurabile. Non equivale a rollback transazionale di un batch. Fonte: [Git integration](https://aider.chat/docs/git.html).

Nel modo architect un modello propone la soluzione e un editor la traduce nel formato di edit. È una strategia che aggiunge una richiesta/modello, diversa dal nostro vincolo dinamico sullo snapshot durante la generazione. Fonte: [chat modes](https://aider.chat/docs/usage/modes.html).

### 2.5 Pi: set piccolo ed edit strutturati sul file originale

Pi è distinto dal fork oh-my-pi. La factory coding corrente costruisce `read`, `bash`, `edit`, `write`; il catalogo contiene anche `grep`, `find`, `ls`, `powershell`. La factory read-only seleziona read/grep/find/ls. Presenza nel codice e caricamento di default non sono equivalenti. Fonte: [registro Pi](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/index.ts).

#### Lettura e ispezione

`read(path, offset?, limit?)` accetta path relativo/assoluto e righe 1-based. Legge il file e restituisce testo o allegato immagine, con troncatura; il trattamento delle immagini considera capacità del modello e ridimensionamento. Fonte: [read.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/read.ts).

L'output comune è limitato normalmente a 2.000 righe o 50 KiB. Il modello continua con offset invece di assumere di aver visto tutto. Fonte: [truncate.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/truncate.ts).

Quando disponibili, gli altri strumenti sono:

| Tool | Argomenti | Backend/risultato |
| --- | --- | --- |
| `grep` | `pattern`, `path?`, `glob?`, `ignoreCase?`, `literal?`, `context?`, `limit?` | Ripgrep e righe di contesto, ignore nelle scansioni |
| `find` | `pattern`, `path?`, `limit?` | Percorsi per glob, normalmente tramite fd |
| `ls` | `path?`, `limit?` | Listing ordinato, dotfile e directory marcate |

Fonti: [grep.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/grep.ts), [find.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/find.ts), [ls.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/ls.ts).

Con il set coding piccolo, la shell copre ricerca, listing e metadati. Non c'è un inspect obbligatorio distinto dalla lettura.

#### Edit, inserimento e rimozione di testo

La firma corrente è `edit(path, edits=[{oldText,newText}, ...])`. Le sostituzioni si riferiscono all'originale, devono essere disgiunte e precedono una scrittura finale. Il risultato comprende diff, patch e prima riga modificata. Fonte: [edit.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/edit.ts).

```json
{"path":"config.py","edits":[{"oldText":"enabled = False","newText":"enabled = True"},{"oldText":"obsolete = True\n","newText":""}]}
```

L'insert conserva un'ancora in newText, la rimozione usa newText vuoto. Old vuoto è rifiutato; non si impone partenza da inizio riga. Il backend verifica match, duplicati, sovrapposizioni e cambiamenti effettivi; applica in ordine inverso per mantenere gli offset. Fonte: [edit-diff.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/edit-diff.ts).

**La firma dice exact, il backend ammette anche recuperi.** Normalizza i terminatori e può usare fallback su Unicode, virgolette e spazi, cercando di conservare le righe invariate. Fonte: [matching e normalizzazione](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/edit-diff.ts).

Il modulo edit separa/ripristina BOM e terminatori rilevati e scrive UTF-8. Serializza le mutazioni tramite una coda per file. Il corpo esaminato non richiede una precedente evidenza di Read in conversazione: legge il file per applicare old. Fonte: [esecuzione di edit](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/edit.ts).

Queste scelte non garantiscono ogni combinazione di CR/LF e encoding del contratto DS4. La coda interna non impedisce tutte le scritture di processi esterni.

#### Write, delete, rename e verifica

`write(path, content)` crea/sovrascrive, crea parent mancanti e scrive UTF-8. Content vuoto lascia un file vuoto; non richiede all. Fonte: [write.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/write.ts).

Nel set ordinario non c'è delete/rename dedicato. `bash(command, timeout?)` copre quelle operazioni, conversioni e test, restituendo output/stato e informazioni di troncatura. Fonte: [bash.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/bash.ts).

Il precedente più utile è il batch di sostituzioni sullo stesso originale con una firma piccola. Fallback, creazione dei parent e scrittura diretta sono decisioni aggiuntive che non servono per adottare quella firma.

### 2.6 Oh-my-pi: lettura unificata, snapshot e linguaggio hashline

Il lavoro originale hashline proponeva brevi riferimenti per riga; l'API corrente usa anche un tag di snapshot per file e un linguaggio più ampio. Gli esempi storici e attuali non vanno mescolati. Fonti: [ricerca originale](https://stencil.so/blog/the-harness-problem), [edit corrente](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/edit.md).

#### Read: ispezione e lettura

`read(path)` può indirizzare file, directory, URL o risorse interne. Selettori come `:50-100`, `:-20`, `:raw` e intervalli multipli scelgono la vista; path letterali esistenti hanno precedenza sui suffissi. Può mostrare alberi, sintesi strutturali, testo, immagini, documenti, archivi e SQLite. Restituisce range/troncatura e registra snapshot/righe viste quando il target è editabile. Fonte: [read](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/read.md).

Non richiede una coppia inspect/read: il renderer sceglie la rappresentazione e il modello può richiedere raw o una porzione. Il costo è un linguaggio nel path e risultati che possono essere proiezioni del file, non byte da riscrivere integralmente.

#### Grep e Find

`grep(pattern, path?, case?, gitignore?, skip?)` cerca contenuti con regex, contesto e gruppi per file. In hashline i risultati possono portare tag e righe registrate per edit; skip pagina i file. Fonte: [grep](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/grep.md).

`find(query, grep_keywords, path?)` è invece ricerca semantica: pre-ranking lessicale, valutazioni di file/passaggi con un modello e risultati con range/rilevanza. Non è il find per glob di Pi. Fonte: [find](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/find.md).

Grep è diretto per identificatori; find aggiunge giudizi per concetti. Un tag rende verificabile la versione, non l'intento della scelta.

#### Edit hashline: replace, insert, cut, delete e move

`edit(input)` usa sezioni `[PATH#TAG]` e numeri dello snapshot originale. Le righe nuove hanno prefisso `+`, separato dai comandi. Fonte: [hashline](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/edit.md).

| Forma | Effetto |
| --- | --- |
| `PUT N.=M:` | Sostituzione del range inclusivo |
| `PUT N*:` | Sostituzione di blocco sintattico |
| `PUT <N:` / `PUT >N:` | Inserimento prima/dopo |
| `PUT >$:` | Append a EOF |
| `CUT N.=M` / `CUT N*` | Rimozione e cattura nel registro |
| `PUT ... @nome` | Incolla registro |
| `REM` | Delete del file senza corpo |
| `MV destinazione` | Move del risultato |

```text
[config.py#4D2A]
PUT 7.=7:
+enabled = True
```

I blocchi sono risolti sintatticamente. Il runtime verifica tag, range, righe viste e conflitti; può recuperare tag obsoleti da snapshot conservati. Prepara i file prima delle scritture, ma un errore operativo può lasciare risultati parziali. Fonte: [validazione di edit](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/edit.md).

Sono disponibili anche replace/patch/apply_patch/sloppy, con prompt e schema diversi. Per DS4 la parte utile è selezionare blocchi lunghi senza copiarne l'interno e separare delete dal corpo; registri e modalità multiple sono funzionalità ulteriori.

#### Write e risorse speciali

`write(path, content)` crea/sovrascrive file e può delegare a handler per risorse interne, archivi, SQLite e conflitti. Per un file normale content vuoto significa vuoto; per una riga SQLite identificata può significare cancellazione. Restituisce ricevuta ed eventualmente tag/diagnostica. Un tag dopo write non rende automaticamente viste tutte le righe. Fonte: [write](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/write.md).

Due campi non significano necessariamente poca semantica: qui essa cambia anche col path. Nel confronto DS4 conviene isolare il caso file regolare.

#### AST, LSP e trasformazioni strutturali

`ast_grep(pat, path?, lang?, skip?)` cerca nodi e catture sintattiche, usando linguaggio inferito o esplicito. Fonte: [ast_grep](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/ast-grep.md).

`ast_edit(ops=[{pat,out}], paths=[...])` prepara un preview; un write ai dispositivi resolve/reject applica/scarta. Out vuoto elimina un nodo. Il percorso di apply ricalcola la proposta; non va equiparato a una transazione universale o a snapshot byte-esatto. Fonte: [ast_edit](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/ast-edit.md).

`lsp(action, file?, line?, symbol?, query?, new_name?, apply?)` offre diagnostica, definizioni/riferimenti, simboli, rename, rename_file e code actions, secondo capacità dei server linguistici. Fonte: [lsp](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/lsp.md).

Il sistema copre molti casi: file, range, blocchi, registri, AST e semantica. È una fonte per i casi lunghi, con molte più regole del problema new/edit/delete.

### 2.7 Qwen Code: lettori dedicati, write/edit e stato di lettura

Qwen Code è un harness completo, distinto dall'utilizzo di un modello Qwen dentro DS4. Il catalogo filesystem separa listing, lettura, ricerca, scrittura ed editing; la configurazione può cambiare quali strumenti sono esposti. Fonte: [filesystem Qwen Code](https://qwenlm.github.io/qwen-code-docs/en/developers/tools/file-system/).

#### Scoperta e lettura

| Tool | Argomenti principali | Operazione |
| --- | --- | --- |
| `list_directory` | `path`, `ignore?`, `respect_git_ignore?` | Listing diretto, directory prima dei file; opt-in |
| `glob` | `pattern`, `path?` | Percorsi, ignore configurati, risultati limitati/ordinati |
| `grep_search` | `pattern`, `path?`, `glob?`, `limit?` | Regex, percorsi e righe dei match |
| `read_file` | `file_path`, `offset?`, `limit?`, `pages?` | Testo/media oppure intervalli |
| `read_many_files` | `paths`, opzioni include/exclude/ignore | Raccolta di più file |

Le prime quattro firme sono nella [reference filesystem](https://qwenlm.github.io/qwen-code-docs/en/developers/tools/file-system/); la lettura multipla ha una [reference propria](https://qwenlm.github.io/qwen-code-docs/en/developers/tools/multi-file/).

ReadFile usa offset di testo **0-based**, mentre pages dei PDF è 1-based. La pipeline distingue testo, media e notebook, considera modalità del modello e segnala la troncatura. Una richiesta senza offset/limit/pages conta come lettura completa solo quando il risultato non è tagliato; la cache può evitare di reinviare un contenuto invariato ancora presente in history. Fonte: [read-file.ts](https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/tools/read-file.ts).

ReadManyFiles prende path/glob, applica include/exclude e concatena i testi con separatori per file; per una directory occorre un pattern, non il solo nome. Media esplicitamente richiesti hanno trattamento dedicato. Fonte: [read_many_files](https://qwenlm.github.io/qwen-code-docs/en/developers/tools/multi-file/).

Questa ispezione è distribuita fra listing/ricerca/lettura. Non emerge un inspect separato obbligatorio; il controllo importante è lo stato della lettura registrata dal runtime.

#### Edit e inserimento

`edit(file_path, old_string, new_string, replace_all?)` richiede un match univoco salvo sostituzione globale. Old vuoto crea soltanto un file assente; su uno presente fallisce. Insert conserva un'ancora in new; new vuoto rimuove contenuto. Il modulo controlla lettura/freschezza, prepara il diff e ricontrolla prima della scrittura. Fonte: [edit.ts](https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/tools/edit.ts).

```json
{"file_path":"/work/config.py","old_string":"enabled = True\n","new_string":"enabled = True\nverbose = False\n","replace_all":false}
```

Il backend non è soltanto un confronto dei byte emessi: gli helper cercano il letterale, poi equivalenze deterministiche su caratteri/whitespace, recuperando il tratto del disco. Per delete di testo possono includere una newline successiva per non lasciare una riga vuota. Questo modifica la semantica di una rimozione rispetto a «solo i byte selezionati». Fonte: [editHelper.ts](https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/utils/editHelper.ts).

**Documentazione e codice vanno distinti:** la guida descrive anche correzione multi-stage tramite modello; nel calculateEdit del modulo esaminato compaiono normalizzazioni e errori espliciti per match zero/multiplo. Non è stato verificato un passaggio di correzione LLM in altri livelli. Non si può quindi dichiararlo una garanzia corrente dell'intera applicazione. Fonti: [guida](https://qwenlm.github.io/qwen-code-docs/en/developers/tools/file-system/), [modulo edit](https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/tools/edit.ts).

#### Write, notebook, delete e rename

`write_file(file_path, content)` crea o sostituisce il contenuto completo. Il codice conserva metadata di encoding/BOM del file presente, crea i parent quando necessario, controlla freschezza prima della scrittura e aggiorna la cache dopo. I controlli non eliminano ogni corsa con writer esterni. Fonte: [write-file.ts](https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/tools/write-file.ts).

`notebook_edit(notebook_path, cell_id?, new_source?, cell_type?, edit_mode?)` lavora su celle con replace/insert/delete. Richiede lettura e ID non ambiguo; un cambio del codice invalida gli output della cella. Fonte: [notebook-edit.ts](https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/tools/notebook-edit.ts).

Nel catalogo descritto non c'è un delete/rename dedicato per file ordinari. `run_shell_command(command, description?, directory?, is_background?)` copre operazioni filesystem e verifiche. La guida indirizza ricerca/lettura ai tool dedicati; la shell restituisce output, exit code e informazioni sui processi. Fonte: [shell Qwen Code](https://qwenlm.github.io/qwen-code-docs/en/developers/tools/shell/).

#### Utilità per DS4

Write/edit separati e stato di lettura sono precedenti pertinenti. Creazione tramite old vuoto, replace_all, normalizzazioni e consumo implicito di newline sono invece scelte aggiuntive. La lunghezza di una firma non rivela da sola quanta semantica il backend introduca.

### 2.8 DeepSeek Tool Calls: protocollo di funzioni, non catalogo filesystem

La guida DeepSeek definisce funzioni del chiamante con nome, descrizione e JSON Schema. Il modello restituisce tool_calls; il client esegue e aggiunge risultati collegati da tool_call_id, poi richiama il modello. Il servizio non esegue quelle funzioni al posto del client. Fonte: [Tool Calls DeepSeek](https://api-docs.deepseek.com/guides/tool_calls/).

#### Cosa è definito e cosa no

| Operazione di file | Evidenza nel contratto consultato |
| --- | --- |
| Listing/ispezione | Nessuna funzione filesystem standard definita |
| Lettura/ricerca | Il chiamante deve fornire implementazione e schema |
| Create/edit/insert | Semantica scelta dall'harness che espone la funzione |
| Delete/rename | Non dedotti da stringhe speciali del protocollo |
| Risultato/errore | Contenuto della risposta del tool associata all'ID |

Non è corretto inventare un read_file universale di DeepSeek a partire da questo protocollo. Se un agente usa DeepSeek insieme ai tool di Pi, Qwen Code o DS4, è quel catalogo a determinare il processamento dei file. Il nome del provider non stabilisce se old vuoto crei, se si usino righe o se delete passi da shell.

#### Schema e controllo dell'inferenza

La modalità strict beta vincola la forma alla funzione definita dal client; richiede gli oggetti con campi required e additionalProperties false, con limitazioni sui tipi supportati. Fonte: [strict DeepSeek](https://api-docs.deepseek.com/guides/tool_calls/#strict-mode-beta).

È una garanzia di forma. Per DS4 old deve anche esistere nello snapshot ed essere univoco, e il filesystem deve essere ricontrollato: sono vincoli ulteriori. Framing nativo DeepSeek e XML dentro edits sono livelli distinti. Il protocollo non obbliga a incorporare XML negli argomenti e non giustifica togliere le validazioni del client. La qualifica dei nostri backend/modelli resta separata dalla disponibilità del tool calling del provider.

### 2.9 Llguidance: controllo della generazione, non agente di file

Llguidance non definisce Read/Edit/Write e non decide come leggere o cancellare. Dati grammatica, tokenizer e prefisso, calcola token che possono proseguire verso una stringa valida; supporta avanzamento nei tratti deterministici. Filesystem, autorizzazione, snapshot e commit appartengono all'harness. Fonte: [llguidance](https://github.com/guidance-ai/llguidance).

Serve come riferimento per il nostro controllo dell'inferenza: grammatica per la forma, vincolo dinamico per old reale. Nessuno dei due determina da solo il blocco semanticamente giusto. Confrontarlo con un agente come se avesse un diverso tool delete confonderebbe i livelli.

### 2.10 Mappa comparativa delle operazioni

La tabella sintetizza i paragrafi precedenti nel medesimo perimetro/versione; non introduce garanzie ulteriori.

| Operazione | Codex locale | Claude Code | Claude editor API | Aider | Pi | Oh-my-pi | Qwen Code |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scoperta | Shell/tool esposti | Shell/Glob | view/altri tool | Repo map | Bash/find/ls | read/grep/find | glob/list_directory |
| Lettura | Shell/lettore/view_image | Read | view | File nel contesto | read | read e selettori | read_file/read_many_files |
| Ispezione strutturale | Programmi/plugin | Plugin/LSP | Chiamante | Repo map | Estensioni/programmi | Sintesi/AST/LSP | Lettori/ricerche |
| Creazione | Add File | Write | create | SEARCH vuoto/whole | write | write | write_file/old vuoto |
| Edit | Update File | Edit | str_replace | SEARCH-REPLACE/udiff | edits[] | PUT/CUT/varianti | edit |
| Insert | Aggiunte con contesto | Ancora in new | insert dopo riga | Ancora/append | Ancora | PUT prima/dopo/EOF | Ancora |
| Togliere testo | Righe rimosse | new vuoto | new vuoto | REPLACE vuoto | newText vuoto | CUT/range | new vuoto + helper |
| File intero/vuoto | Patch/shell | Write | Sostituzione/chiamante | Whole/edit integrale | write | write | write_file |
| Delete del file | Delete File/shell | Shell | Altro tool | Workflow filesystem | Shell | REM | Shell |
| Rename/move | Move to/shell | Shell/plugin | Altro tool | Workflow filesystem | Shell | MV/LSP | Shell |

DeepSeek è nella sezione precedente come protocollo, llguidance come motore di vincoli: non hanno un catalogo filesystem equivalente alle colonne della tabella.

Per DS4 emergono quattro decisioni indipendenti:

1. **Contesto:** lettori mirati, file nel prompt, mappe di simboli o ricerca semantica.
2. **Selezione:** old letterale, contesto di patch, righe associate a snapshot o nodi.
3. **Operazione:** nome del tool, campo comando o intestazione.
4. **Controlli:** vincolo durante generazione, validazione all'esecuzione e preflight filesystem.

Il nostro vantaggio è il vincolo dinamico di old. Adottare una firma più corta non richiede di adottare matching permissivo, delete via shell o perdita dello snapshot.

Diff-XYZ trova risultati dipendenti da compito/modello; EDIT-Bench sottolinea il contesto delle richieste reali. Nessuno determina quale catalogo vinca su DS4. Fonti: [Diff-XYZ v2](https://arxiv.org/abs/2510.12487v2), [EDIT-Bench v2](https://arxiv.org/abs/2511.04486v2).

## 3. Le protezioni da conservare

La maggior parte della complessità del filesystem ha una responsabilità reale e può restare interna al runtime:

- Autorizzazione del percorso prima di generare contenuto e snapshot iniziale immutabile.
- Vincolo di `old_text` al testo effettivo; chiusura della selezione solo quando individua una posizione ammessa univoca.
- Verifica indipendente nel client, anche dopo gli hook.
- Workroot, path canonici/alias, permessi, identità del file e rilevamento di modifiche concorrenti.
- Encoding/BOM, conservazione di CR/LF e nessuna aggiunta implicita di newline.
- Cancellazione/steering e cambi di scope che impediscono il commit.
- Preflight, conflitti fra selezioni e pubblicazione del risultato per file.
- Risultati espliciti sulle scritture già applicate in caso di errore fra file distinti; nessuna promessa nuova di rollback globale.

Si può semplificare ciò che il modello deve scrivere senza eliminare questi controlli. Un formato più facile non autorizza selezioni approssimate, correzioni dell'intento o scritture su snapshot obsoleti.

## 4. Contratto candidato: write, edit, delete

Il primo candidato da confrontare con v2 usa tre strumenti con argomenti ordinari:

```text
write-file(path, content)
edit-file(path, old_text, new_text)
delete-file(path)
```

`write-file` significa sempre scrivere **il contenuto completo**. Crea un file assente e sostituisce un file presente, dopo le verifiche previste per quel caso. `content=""` crea o lascia un file vuoto. Non contiene una scelta implicita fra sostituzione parziale e totale.

`edit-file` significa sostituire un blocco esatto. `new_text=""` rimuove il blocco, conservando il file. Per un file non vuoto `old_text` deve essere non vuoto e selezionare una sola posizione ammessa. Si può usare lo stesso strumento per inserire prima o dopo un'ancora, mantenendo quell'ancora in `new_text`.

`delete-file` significa cancellare il file indicato. Non ha `old_text`, `new_text`, CDATA o body. Il client mantiene la verifica del target e della sua identità prima dell'unlink. Un percorso non autorizzato non diventa valido perché il tool è più corto.

La scrittura completa permette di eliminare `<all/>` senza perdere creazione, riscrittura e svuotamento. Delete smette di competere con un campo di testo libero. Gli intervalli e gli inserimenti per numero di riga escono dal percorso ordinario.

### Se new deve essere distinto da write

Esiste un compromesso reale. Se `new-file` deve rifiutare ogni file già esistente, la coppia `new-file`/`edit-file` non offre una riscrittura completa economica: con edit bisogna inviare tutto il vecchio testo, oppure aggiungere un'altra modalità. Non conviene nasconderla in `old_text=""`, `"*"` o in un nuovo equivalente di `<all/>`.

Le alternative coerenti sono:

1. Tre tool `write/edit/delete`, con la semantica di write esplicita sopra.
2. Quattro tool `new/write/edit/delete`, dove new crea soltanto e write sostituisce soltanto. Offre una barriera più netta contro sovrascritture accidentali, con una scelta in più per il modello.
3. Tre tool `new/edit/delete`, accettando il costo di selezionare l'intero vecchio contenuto per la riscrittura e senza inventare scorciatoie.

Partirei dalla prima opzione per il confronto. La seconda ha senso se il requisito di creazione senza sovrascrittura è prioritario; eviterei un flag `overwrite` che trasformi new in due operazioni.

### Esempi delle operazioni

Questi sono oggetti logici, non una nuova sintassi da copiare dentro una stringa XML:

```json
{"path":"notes.txt","content":""}
```

Con write questo svuota il file. Con delete basta il solo percorso. Per aggiungere una riga dopo un'ancora:

```json
{"path":"settings.py","old_text":"enabled = True\n","new_text":"enabled = True\nverbose = False\n"}
```

Per eliminare il blocco `obsolete = True\n`, edit usa quel blocco come old e una stringa vuota come new. Per cambiare tutto il file si usa write. Nessuna stringa di contenuto ha semantica di comando.

### Un file per chiamata e gestione dei batch

Una chiamata con tre campi piatti evita il documento di frammenti, i path ripetuti e le pause a metà XML. Il costo è che più chiamate non ereditano automaticamente il preflight globale della chiamata v2.

Per il confronto iniziale partirei da un file per chiamata. Prima di una migrazione bisogna stabilire esplicitamente la semantica dei batch del turno. Se il requisito è preparare più modifiche sullo stesso snapshot e pubblicare una volta sola, il candidato può usare `edit-file(path, replacements=[{old_text,new_text}, ...])`, con una sola autorizzazione per path. L'array aggiunge struttura ordinaria; non deve aggiungere comandi nei corpi.

Non equiparare una serie di tool indipendenti alla vecchia chiamata multi-file: il rifiuto del secondo file oggi invalida anche il primo prima della scrittura. Questa garanzia va conservata nel coordinatore oppure dichiarata come cambiamento di comportamento e valutata separatamente.

## 5. Eliminare CDATA: distinguere dati e trasporto

CDATA non cambia il valore logico dopo il parsing, ma cambia ciò che il modello deve generare. Anche le entità XML trasformano il testo apparente. Il matching attuale opera già sui byte logici decodificati: non è corretto dire che confronti direttamente gli involucri CDATA con il file. Il problema è il carico di rappresentazione e le scelte equivalenti lungo la generazione.

Con argomenti ordinari si elimina **lo strato XML aggiunto dall'edit**. HTML come `<button title="a & b">` resta contenuto del campo. Non servono entità XML, comandi protetti, sezioni CDATA adiacenti o la regola della chiusura finale.

Rimangono gli escape richiesti dal trasporto nativo del modello o dal JSON: virgolette, backslash, newline e delimitatori speciali non spariscono. Il server deve applicare il vincolo sul valore decodificato del campo, con una sola rappresentazione canonica dove possibile. La policy di newline del documento non va dedotta dalla forma serializzata.

Quindi «basta passare a JSON» sarebbe una conclusione incompleta. Va misurato il costo del trasporto nativo sulle grammatiche supportate. Il vantaggio specifico del candidato è smettere di serializzare un documento XML dentro un argomento stringa, continuando a usare l'adattatore nativo già necessario per gli altri tool.

### Perché non partire da un formato raw nuovo

Un payload raw sarebbe più vicino al file, ma ha comunque bisogno di distinguere contenuto e confini. Delimitatori fissi possono comparire nel testo; conteggi di byte sono difficili da produrre; delimitatori casuali aggiungono stato e istruzioni. Token di controllo realmente fuori banda richiedono compatibilità con tokenizer, template, history, streaming e replay; non basta inventare una stringa che somigli a un token speciale.

Lo considererei soltanto se il confronto dimostrasse che l'escaping nativo resta una causa importante di fallimento. Deve sostituire una rappresentazione esistente, senza creare una modalità permanente ulteriore.

## 6. Usare il controllo dell'inferenza senza ampliare l'API

Il motore può seguire un flusso piccolo e distinto per operazione:

```text
tool scelto → path → autorizzazione/snapshot
  write  → content → validazione/commit
  edit   → old vincolato → new libero → validazione/commit
  delete → validazione/commit
```

Sono ruoli interni. Il modello non deve conoscere exchange ID, checkpoint, hash, line_count, stato di creazione o dettagli di KV. Per write/edit la pausa resta utile; per delete non c'è generazione di contenuto da riprendere, quindi non serve il suo riconoscitore testuale.

Le opportunità principali sono:

1. **Struttura canonica.** Ordine dei campi, separatori e chiusure sono gestiti dalla grammatica. Non lasciare più codifiche XML equivalenti della stessa selezione.
2. **Old vincolato allo snapshot.** Riutilizzare i candidati e la verifica dei prefissi, mantenendo una chiusura possibile solo in una posizione univoca. Il modello decide quanto testo sostituire.
3. **Proposte di copia.** Conservare l'ottimizzazione che propone continuazioni dello snapshot e le verifica. Riduce decode, ma non elimina il numero di byte rappresentati nel risultato.
4. **Fasi indipendenti.** Struttura/old e nuovo contenuto conservano le rispettive politiche di sampling; verifier, rollback e MTP rispettano la stessa frontiera.
5. **Esistenza coerente con l'operazione.** Il runtime rifiuta edit su file assente, delete su target non valido e new su file esistente se si sceglie quella variante. Il modello riceve un errore utile, non una reinterpretazione.

llguidance documenta sia il masking basato su grammatica/tokenizer sia i tratti deterministici. È un riferimento per questi meccanismi, non una raccomandazione ad aggiungere una dipendenza al motore DS4. Fonte: [llguidance](https://github.com/guidance-ai/llguidance).

Non si deve chiudere automaticamente old appena il prefisso diventa univoco: l'unicità individua la partenza, ma non dice dove finisce il blocco che l'utente vuole sostituire. Né si deve copiare forzatamente fino a EOF per ogni selezione. Le strutture determinate dal contratto sono diverse dalle decisioni di editing.

Il masking può impedire un vecchio testo inesistente, non la scelta di un blocco esistente ma sbagliato. Anche delete separato può essere chiamato sul path sbagliato o il modello può scegliere write quando l'utente voleva delete. La separazione rimuove una confusione strutturale; il risultato semantico resta da qualificare.

## 7. Una restrizione da riesaminare: unicità globale

Il contratto impone contemporaneamente partenza da inizio riga e unicità globale, comprese occorrenze a metà riga. Il client usa `text.find(old)` e una seconda ricerca globale prima di verificare l'inizio riga; il server prepara `unique_after` rispetto ai suffissi di tutto il testo.

Considerare questo snapshot:

```text
foo
prefix foo
bar
```

`old_text="foo\n"` compare due volte nel file, ma soltanto la prima occorrenza è una partenza selezionabile. Oggi viene rifiutato. Una seconda occorrenza a metà riga non aggiunge una seconda destinazione ammessa dal contratto.

Se si mantiene la selezione da inizio riga, propongo **unicità fra le partenze ammesse**. Il client deve cercare tutte e sole quelle posizioni e il server deve calcolare/prunare gli stessi candidati. Non basta cambiare la descrizione o ignorare un errore client. Righe davvero duplicate restano ambigue e richiedono contesto maggiore.

È una proposta di semplificazione coerente, non un difetto di implementazione: il codice applica la regola attuale. Il cambiamento amplia intenzionalmente le selezioni consentite. Va misurato anche il costo dell'indice e non si può promettere di eliminare l'indice dei suffissi senza un'alternativa prestazionalmente verificata.

L'altra opzione è permettere partenze a qualunque posizione e mantenere unicità globale. È più flessibile ma amplia i candidati dinamici. Non adotterei entrambe le modifiche nello stesso confronto: partirei da inizi di riga e unicità coerente con essi.

## 8. Blocchi lunghi, inserimenti e file vuoti

### Inserimenti

Per un file non vuoto una sostituzione che conserva l'ancora rappresenta già l'inserimento prima/dopo di essa. Si può quindi togliere `<line/>` senza aggiungere `insert_before` e `insert_after` come nuove modalità. Occorre verificare EOF senza newline e ancore duplicate.

Il file vuoto si gestisce con write. Non darei a old vuoto il significato ambiguo di «crea», «inserisci», «sostituisci tutto» o «appendi». Questa regola evita un nuovo `<all/>` sotto altro nome.

### Intervalli numerici attuali

Il ramo `start_line/end_line` viene controllato solo lessicalmente dal server: positività, ordine e limiti sono verificati dal client. È una scorciatoia che evita la copia, ma anche un percorso che aggira il vincolo sul vecchio testo.

Lo toglierei dal contratto ordinario del primo candidato. Bisogna però riconoscere ciò che si perde: selezionare un lungo tratto interno richiede più output. Write può sostituire l'intero file, ma richiede di rigenerare anche contenuto che dovrebbe restare uguale. La copia speculativa accelera old, senza farlo sparire dal payload.

### Alternative da valutare solo se emerge il bisogno

| Idea | Vantaggio | Costo che introduce |
| --- | --- | --- |
| Due ancore testuali di confine | Selezione lunga senza copiare l'interno | Due selezioni, ordine e regole precise di inclusione dei confini |
| Riferimento a blocco già letto | Il runtime conosce già testo e identità | Modifica dei risultati di read, gestione/lifetime dei riferimenti e scelta del blocco |
| Righe associate a snapshot verificato | Evita la copia e rende esplicita la versione | Numeri/riferimenti da riportare, rischio di scegliere righe semanticamente sbagliate |
| Riscrittura completa | API semplice | Output e rischio di omissione del contenuto invariato |

Fra queste, due ancore letterali potrebbero riutilizzare meglio il masking esistente; riferimenti a blocchi possono essere migliori se la lettura espone già unità adeguate. Sono ipotesi, non funzionalità da aggiungere tutte. Serve un corpus di modifiche lunghe per capire se il costo della selezione letterale giustifica un secondo primitivo.

## 9. Cosa togliere, cosa spostare, cosa rinviare

| Elemento attuale | Proposta | Motivo |
| --- | --- | --- |
| Delete dentro `new_text` | Togliere; delete separato | Comando e dati competono nello stesso campo |
| `<all/>` | Togliere; write completo | Operazione esplicita, file vuoto incluso |
| CDATA ed entità XML nei corpi edit | Togliere con il documento XML | Secondo livello di rappresentazione non necessario agli argomenti ordinari |
| Bare close letterale e chiusura finale speciale | Togliere | Complicazione del dialetto rispetto a campi strutturati |
| `<line/>` | Togliere dal candidato | Inserimento tramite ancora o write su vuoto |
| `start_line/end_line` senza old | Togliere dal percorso ordinario e misurare i casi lunghi | Scorciatoia con verifica più debole lato server |
| Unicità delle occorrenze non selezionabili | Rivedere | Restringe old senza una seconda destinazione ammessa |
| Snapshot/hash/encoding/alias/hook/steering | Conservare nel runtime | Protezioni indipendenti dal formato scelto |
| Descrizione ripetuta in skill, campo e server | Centralizzare e accorciare | Il modello deve conoscere semantica e dati, non il parser |
| Nuovi marcatori, lunghezze, fuzzy matching, auto-repair | Evitare come prima risposta | Rischiano di spostare o nascondere la stessa complessità |
| Tool AST, move, clipboard, registri | Rinviare | Non necessari per risolvere new/edit/delete |

La separazione di delete consente anche di riesaminare la dipendenza dalla lettura del contenuto: cancellare un file non richiede decodificarne il testo. Nella prima migrazione manterrei la policy di autorizzazione attuale per non confondere due cambiamenti. In seguito si può valutare una prova di identità/metadata specifica per delete, distinta dall'evidenza di lettura necessaria per edit. Questo può rendere delete applicabile a file non testuali senza ampliare edit.

## 10. Alternative di formato e priorità

| Candidato | Beneficio | Limite | Priorità |
| --- | --- | --- | --- |
| Tre tool con campi ordinari | Operazioni chiare; riuso degli adattatori nativi; elimina XML annidato | Escaping nativo da qualificare; gestione batch da decidere | Primo confronto |
| Un tool con `operation` e payload discriminato | Mantiene un ingresso unico e operazioni esplicite | Varianti di schema e scelta del comando; batch più articolato | Secondo candidato se serve un batch eterogeneo |
| XML con write/edit/delete esterni | Corregge la posizione di delete e all | Conserva CDATA/entità e documento custom | Solo termine di confronto, soluzione incompleta |
| Patch testuale orientata alle righe | Contenuto leggibile e operazioni distinte | Prefissi, framing, EOF/CRLF e grammatica da apprendere | Solo se batte i campi ordinari sui nostri modelli |
| Hashline/linguaggio di ancore | Old compatto | Nuovi riferimenti e nuova semantica; vantaggio parzialmente sovrapposto al masking | Esperimento per blocchi lunghi |

L'XML standard dei **risultati** e dei metadati del runtime è un confine diverso. Non è necessario cambiare quel formato per eliminare il documento XML negli **argomenti** di edit. Le direttive della libreria già distinguono risultati presentati al modello, valori Python e protocollo degli argomenti del provider.

## 11. Skill candidata più corta

Una descrizione candidata dell'editing, da adattare alla semantica batch scelta, potrebbe essere:

```text
Read an existing file before changing or deleting it.
Use write-file for complete contents, including creation and empty files.
Use edit-file to replace one exact, unambiguous block starting at a line boundary.
To insert, keep an existing anchor in the replacement. Empty replacement removes the block.
Use delete-file to remove a file. File contents never act as commands.
If the file changes, read it again. Check the tool result before claiming success.
```

Non è un obiettivo di lunghezza arbitrario: è una conseguenza della rimozione dei rami speciali. Il contratto effettivo va esposto dal tool e verificato dal runtime; la skill deve spiegare il flusso di lavoro. I dettagli su TTL, exchange, hook, indici, sampling e alias appartengono alla documentazione tecnica e al codice, salvo effetti che il modello deve considerare nella propria scelta.

Le raccomandazioni Anthropic richiedono scopi distinti, parametri chiari e valutazioni di token/errori, senza assumere che più tool siano sempre migliori. Per questo la separazione proposta va confrontata con il candidato a comando discriminato. Fonte: [principi per tool di agenti](https://www.anthropic.com/engineering/writing-tools-for-agents).

## 12. Confronto sperimentale e migrazione

Il prossimo lavoro dovrebbe confrontare v2 e un solo candidato minimale con richieste naturali, senza dettare nel prompt i tag o gli argomenti da emettere. Conservare gli assert sui byte quando il compito richiede byte precisi e sul comportamento quando il compito ammette più implementazioni.

Misurare almeno:

- Successo al primo tentativo e finale entro un budget uguale; numero di chiamate e tentativi falliti.
- Errori di formato, errori di selezione e operazione sbagliata come categorie separate.
- Token di descrizioni/skill e output generato, inclusa la copia di old.
- Tempo di generazione, costo del masking e costo degli scambi; target-only e MTP separati.
- Scritture effettive, cancellazioni e casi di contenuto parziale.

La matrice minima deve includere create/write/delete/svuotamento, inserimento su file vuoto e non vuoto, EOF con/senza newline, CRLF/CR/LF, Unicode, HTML, `]]>`, delimitatori nativi letterali, righe duplicate, occorrenze interne non selezionabili, selezioni lunghe e file modificato durante la generazione. Conservare i casi di alias, hook, scope, rifiuto del secondo file e assenza di commit prima del preflight.

Non aggiungere nuove ottimizzazioni prestazionali mentre si confrontano i formati: usare il masking e le proposte già disponibili, adattandone il confine. La riduzione del testo istruttivo è una misura; non sostituisce il successo del compito.

Se il candidato vince, la migrazione richiede schema/tool registry, ruoli del comportamento server, client interattivo, planner/filesystem, replay/history, skill, fixture e wheel distribuita. Il motore generico conserva checkpoint, IDs e lifecycle; non deve conoscere nomi hardcoded dei tool. Serve una versione coordinata, senza mantenere formati permanenti selezionabili tramite flag.

Il primo esperimento consigliato è dunque **write/edit/delete con campi ordinari e old vincolato**, seguito dal confronto dell'unicità fra inizi di riga. Il bisogno di un selettore per blocchi lunghi si decide sui risultati, evitando di aggiungere preventivamente un nuovo linguaggio.

## 13. Stato delle verifiche di questa ricerca

Sono stati letti codice, skill e report locali e confrontate fonti primarie esterne. Le percentuali e gli esiti live citati sono quelli dei report esistenti, non test rieseguiti in questa ricerca. Non è stata implementata né qualificata una nuova API; nessun modello è stato avviato. Indici rigenerati e verifica documentale tramite `python -m unittest tests.test_research_corpus tests.test_docs_wiki -v`: **10 PASS**, inclusi link, copertura e intervalli delle sezioni.
