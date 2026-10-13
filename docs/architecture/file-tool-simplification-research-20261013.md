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

Le fonti esterne sono primarie e sono state consultate il 13 ottobre 2026. Le pagine dei repository si riferiscono all'istantanea web di `main` consultata in quella data; non sono commit fissati. Prima di portarne codice nel progetto va fissata la revisione. Non sono state riprodotte le loro misure su DS4.

| Approccio | Forma e osservazione dalla fonte | Utilità per DS4 |
| --- | --- | --- |
| Anthropic text editor | Operazione esplicita; sostituzione tramite `path`, `old_str`, `new_str`; creazione tramite contenuto separato | Il vecchio testo resta una stringa di dati, senza comandi incorporati |
| Aider | Blocchi SEARCH/REPLACE; oppure file completo; formati differenti per modelli differenti | La sostituzione è semplice da capire; riscrivere tutto costa output anche per piccoli cambiamenti |
| Codex apply_patch | Intestazioni distinte per aggiunta, aggiornamento e cancellazione; corpi orientati alle righe | Delete è un'operazione senza corpo; i dettagli del diff aggiungono altre regole da apprendere |
| Qwen Code | `write_file(file_path, content)` e `edit(file_path, old_string, new_string, replace_all)` | Distingue scrittura completa e sostituzione; conserva però la creazione implicita con old vuoto |
| DeepSeek Tool Calls | Funzioni definite dal client tramite JSON Schema; modalità strict per la forma degli argomenti | Il protocollo non decide la semantica dei file: selezione, autorizzazione e commit restano nel runtime |
| Hashline | Riferimenti verificabili associati alla lettura, invece di ricopiare tutto il vecchio testo | Alternativa per selezioni lunghe, ma parte del suo vantaggio è già coperta dal nostro masking |
| llguidance | Maschere di token da grammatica/tokenizer e avanzamento su tratti deterministici | La struttura può essere responsabilità del motore; non occorre esporre nuovi comandi al modello |

La documentazione Anthropic richiede corrispondenza esatta per `old_str`, inclusi spazi e indentazione, e distingue creazione e sostituzione con un campo comando. Non documenta una cancellazione tramite contenuto speciale. Non è una prova che tre tool separati siano sempre migliori, ma mostra una separazione chiara fra comando e testo. Fonte: [text editor ufficiale](https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool).

Aider documenta SEARCH/REPLACE come modo per produrre solo i tratti cambiati. Descrive il formato whole come semplice ma costoso, e varianti adattate a famiglie di modelli. La lezione trasferibile è valutare il formato sul modello effettivo, anziché assumere che un formato diff sia universalmente naturale. Fonte: [formati Aider](https://aider.chat/docs/more/edit-formats.html).

Il parser pubblico di Codex rappresenta aggiunta, cancellazione e aggiornamento come varianti distinte. La cancellazione contiene il percorso, senza un corpo di sostituzione. Questa separazione si può adottare senza importare l'intera grammatica patch. Fonte: [parser apply_patch](https://github.com/openai/codex/blob/main/codex-rs/apply-patch/src/parser.rs).

Qwen Code documenta `write_file` per creare o sovrascrivere il contenuto completo e `edit` per una sostituzione univoca, oppure tutte le occorrenze con `replace_all`. Old vuoto crea soltanto un file assente. Il runtime può chiedere al modello di correggere una selezione non trovata o ambigua prima della verifica finale e mostra il diff per l'approvazione. La pagina consultata non espone un tool dedicato alla cancellazione di file. Fonte: [filesystem di Qwen Code](https://qwenlm.github.io/qwen-code-docs/en/developers/tools/file-system/).

Per DS4 prenderei la distinzione write/edit come candidato da misurare. Eviterei di trasferire creazione implicita, sostituzione globale e correzione automatica: aggiungono decisioni che il contratto proposto vuole eliminare. Questa è una scelta progettuale locale, non una prova di superiorità rispetto a Qwen Code né una qualifica del nostro Qwen27.

DeepSeek documenta tool definiti dall'applicazione, argomenti JSON e risultati associati al `tool_call_id`: il modello propone la chiamata, il client esegue la funzione. La modalità strict beta vincola gli argomenti allo schema; gli oggetti richiedono tutti i campi e `additionalProperties: false`. La guida consultata non definisce un editor filesystem standard o un contratto universale write/edit/delete. Fonte: [Tool Calls DeepSeek](https://api-docs.deepseek.com/guides/tool_calls/).

L'inferenza per DS4 è che uno schema piatto può semplificare la forma della chiamata, ma non certifica che old esista nello snapshot, sia univoco o che il file non sia cambiato. Il server conserva framing e vincoli di generazione; il client conserva autorizzazioni, verifiche sul filesystem e pubblicazione. Il framing nativo DeepSeek e l'XML attualmente contenuto nell'argomento `edits` sono due livelli distinti: semplificare il secondo non richiede di eliminare il primo. Questa proposta resta da qualificare separatamente sui modelli e backend DeepSeek disponibili.

Il lavoro originale su hashline associa riferimenti di riga verificabili al testo letto. Il benchmark dell'autore usa mutazioni di file React e mostra differenze fra modelli, compresi casi in cui hashline non supera replace. È una motivazione per un confronto locale, non una previsione numerica per Qwen27 o Flash. Fonte: [ricerca dell'autore](https://stencil.so/blog/the-harness-problem).

La documentazione corrente di oh-my-pi mostra anche quanto possa crescere questa idea: tag di snapshot, operazioni di riga e blocco, registri, spostamenti, recupero di snapshot e modalità multiple. Per la nostra ricerca è utile la separazione fra selezione e contenuto; copiare quel linguaggio amplierebbe il problema che stiamo cercando di ridurre. Fonte: [tool edit di oh-my-pi](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/edit.md).

Diff-XYZ confronta comprensione/applicazione/generazione di diff e trova risultati dipendenti da compito e dimensione del modello. EDIT-Bench valuta editing da richieste reali e mostra quanto conti il contesto disponibile. Sono ragioni per misurare sia correttezza meccanica sia risultato della richiesta: nessuno dei due benchmark dimostra quale API vinca nel nostro motore. Fonti: [Diff-XYZ v2](https://arxiv.org/abs/2510.12487v2), [EDIT-Bench v2](https://arxiv.org/abs/2511.04486v2).

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
