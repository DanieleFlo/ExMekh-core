# Piano server: edit XML v2 e storico del masking v1

Aggiornamento 2026-10-12: XML v2 senza code è il formato predefinito. Il
[contratto attuale](../agentic/interactive-calls.md) sostituisce v1.
La migrazione ExMekh-Lib è implementata e installata nel frontend come wheel
predefinita. La [qualifica client](../architecture/client-xml-edit-v2-20261012.md)
conserva gli esiti di intento del modello; la
[revisione successiva](../architecture/client-xml-clean-review-20261012.md)
documenta responsabilità, refactoring e regressioni di cancellazione.

## Implementazione v2

- Un argomento stringa associato da `fields.document`; pause dentro il documento
  dopo ogni path, una call ID e nuovi exchange ID/indici per i singoli edit.
- Dopo provide/create il modello completa l'apertura con `>` senza prefill
  o attributo code. L'ultimo `</new_text>` prima di `</edit>` chiude il singolo
  elemento; i precedenti sono testo. La sequenza completa letterale richiede
  CDATA/entità. La forma con code non è più accettata e non esiste un flag.
- Riconoscimento ristretto con entità/CDATA e CR/LF preservati. I comandi
  all/line/delete sostituiscono i marcatori @. Snapshot ripetuti identici sono
  condivisi; un rifiuto invalida tutta la chiamata XML.
- La selezione letterale mantiene inizio riga e unicità globale. Gli intervalli
  start_line/end_line hanno corpo vuoto: il server controlla solo la sintassi,
  il client controlla i numeri e il file.
- I callback del comportamento governano frontiere e fase/indice MTP;
  il protocollo possiede lifecycle, KV, budget e ricevute. La descrizione del
  tool resa al modello viene integrata dal server con il contratto di chiusura.
- Enumerazioni/const/combinazioni sul documento sono rifiutate alla configure,
  perché le alternative statiche non si adattano in generale allo snapshot autorizzato.
- Il client conserva filesystem, workroot, autorizzazioni, inspect/read, hash,
  alias, intervalli, conflitti, hook e commit. Non è implementato sul server
  alcun controllo di sovrapposizione tra edit né una transazione filesystem.

I report del 11 ottobre qui sotto conservano lo storico della variante con code;
non qualificano automaticamente il formato aggiornato. Le
[verifiche mirate del 12 ottobre](../performance/interactive-edit-xml-plain-smoke-20261012.json)
registrano solo build, fixture server/XML, estrazione indipendente e smoke HTTP
target/MTP pertinenti alla modifica. Creazione, stream con chiusure letterali e
edit multipli passano; le asserzioni delete/CDATA di contenuto rimangono fallite.
Il lookahead mantiene validi anche i prefissi intermedi dei token combinati;
le chiusure ancora ambigue conservano la temperatura del corpo e sospendono MTP.
Il client mantiene le responsabilità elencate sopra.

La [qualifica v2 con code](../performance/interactive-edit-xml-v2-qualification-20261011.json)
registra suite C XML con oracolo indipendente su 400 snapshot, test server
ordinari, ASan/UBSan, SDK OpenAI e build di produzione CUDA/frontend CPU.
Le prove HTTP aggiornate sono opt-in e non eseguono scritture filesystem.
Il prefill reale e la chiusura codificata sono verificati su CUDA/Qwen27;
la cancellazione resta non qualificata perché il modello genera una riscrittura
valida. La maschera ammette anche il comando delete alla frontiera del corpo.
Il confronto completo maschera/oracolo non dimostra parità numerica delle
inferenze target/MTP, né qualifica backend diversi. La matrice v1 sotto è storica.

La configurazione e il replay delle interruzioni accettano soltanto v2.
I test di prefisso procedono byte per byte anche con snapshot vuoto/create:
le forme complete da sole non rilevavano la potatura prematura di all/line.
Entità, escape nativi e Unicode sono controllati prima di ammettere prefissi
che non potrebbero più chiudersi validamente.

La [raccolta completa aggiornata](../performance/interactive-edit-xml-v2-full-tests-20261011.md)
include anche tutte le suite client live disponibili e distingue gli errori di
protocollo dagli esiti di contenuto. Il batch ha trovato un ulteriore prefisso
senza prosecuzione: un opener XML incompleto seguito dall'escape nativo del
delimitatore del tool. Il server ora lo rifiuta prima del commit, insieme ai
casi analoghi di entità e scalar UTF-8 incompleti. Le regressioni native e le
ripetizioni target/MTP/batch conservano separatamente gli esiti prima e dopo
la correzione. Alla data di quella raccolta la migrazione client era ancora
pendente; è stata completata con il contratto XML v2 il 12 ottobre.

## Storico v1 del 9 ottobre 2026

Le sezioni seguenti conservano il piano, il wire e la qualifica precedenti.
I marcatori, first_fields e tre ruoli descritti sotto non sono più accettati
dal server v2.

## 1. Obiettivo e confini

DS4 deve generare prima i campi dichiarati nella configurazione, sospendere la tool call quando sono tutti completi, ricevere dati di controllo dal runtime e completare la stessa chiamata con vincoli dinamici sugli argomenti. Per edit il gruppo prioritario contiene `path`; il protocollo generico non conosce questo nome. `edit-file` è il primo comportamento di una categoria estendibile di chiamate interattive.

La fonte delle attività client è [exmekh_lib/todo.md](../../exmekh_lib/todo.md), macro task «Manipolazione file». Sanificazione persistita, inspect/read, hash filesystem, autorizzazione, hook, scrittura e cancellazione appartengono alla libreria. Il server non apre il path ricevuto e non applica modifiche ai file.

Prima di edit su un file esistente, il modello deve averne visto il contenuto tramite una lettura valida nella history. Lo snapshot ricevuto durante la sospensione serve al masking: non viene inserito nel prompt, non produce un nuovo tool result e non richiede un prefill aggiuntivo del file. Per un file inesistente il client autorizza direttamente la creazione; un file esistente vuoto resta distinto dalla creazione.

## 2. Invarianti del comportamento

- Nell'istanza edit il tool espone soltanto `path`, `old_text`, `new_text`, tutti stringhe. Il protocollo associa ruoli ai nomi reali attraverso `fields`; non contiene confronti hardcoded con questi nomi o con il nome del tool.
- `first_fields` è una lista ordinata di nomi reali del tool: questi campi devono precedere tutti gli altri per un vincolo di generazione, non per l'ordine delle proprietà JSON.
- Il comportamento dichiara anche le dipendenze fra ruoli dopo la pausa: per edit selection deve essere completa prima di replacement. Il motore risolve questa precedenza tramite `fields`, senza dipendere dall'ordine delle proprietà nello schema.
- La sospensione avviene dopo che tutti i campi prioritari sono completi, validi e decodificati, prima di qualsiasi campo successivo o finalizzazione della chiamata. Per edit `first_fields=["path"]`, quindi old/new attendono l'autorizzazione.
- La chiamata conserva la stessa call ID attraverso gli scambi HTTP. Una chiamata sospesa non è eseguibile.
- Il ramo testuale di `old_text` parte esclusivamente dall'inizio fisico di una riga, inclusa l'indentazione.
- Le occorrenze a metà riga o sovrapposte contano per l'unicità globale, ma non diventano candidate di generazione.
- La chiusura testuale richiede un blocco non vuoto e univoco nell'intero snapshot. Dopo l'unicità restano ammesse sia la chiusura sia ulteriori continuazioni compatibili, anche multilinea.
- Fin dal primo token sono ammessi anche i marcatori validi. Un marcatore completo non richiede unicità testuale.
- Il server non corregge o tronca argomenti e non usa fallback a chiamate ordinarie se il protocollo non è supportato.
- Le richieste prive dell'estensione conservano il comportamento esistente.

## 3. Componenti da riutilizzare e responsabilità

L'orchestrazione attuale vive soprattutto in [ds4_server.c](../../ds4_server.c); filtri, analisi del vocabolario e snapshot della sessione sono esposti da [ds4.h](../../ds4.h).

| Componente | Intervento previsto |
| --- | --- |
| Parsing Responses e struttura `request` | Configurazione interattiva, messaggi di ripresa, validazione anticipata e ownership dei payload |
| Registro/schema e scope `agentic` | Risoluzione del nome effettivo, verifica dei campi stringa e disponibilità corrente |
| Tracker della sintassi nativa e filtro agentico | Ordine di `first_fields`, completamento dei valori e punto di sospensione configurato |
| Analisi dei vincoli e trie del vocabolario | Composizione della grammatica esistente con la maschera dinamica del passo corrente |
| Job, slot, contesti agentici e KV live | Continuazioni isolate, stato congelato, rilascio del worker e ripresa della sessione |
| Sampling, sequenze forzate e verifier MTP | Temperatura locale, gate MTP, aggiornamento dei vincoli per ciascun token |
| Rendering e streaming Responses | Eventi di controllo, finalizzazione, errori e replay delle chiamate interrotte |

Separare protocollo, storage delle continuazioni, comportamento `edit_file.v1` e riconoscitore del valore vincolato. Usare strutture e interfacce C piccole, senza C++, scripting nel payload o un secondo agent loop. Estrarre moduli soltanto quando hanno una responsabilità autonoma; riutilizzare le primitive numeriche e KV esistenti.

## 4. Contratto wire da congelare nella prima fase

Il server implementa il contratto v1. Per payload completi, campi obbligatori e limiti usare il [contratto wire](../agentic/interactive-calls.md); gli esempi seguenti descrivono l'architettura. La libreria deve adottarlo senza fallback silenziosi.

### 4.1 Riutilizzo di extra_body.agentic, configurazione e negoziazione

La struttura attuale è già adatta: `AgenticScope.build_extra()` costruisce `{"agentic": {...}}` con `agent_context_id`, `allowed_tools`, `allowed_skills` ed eventualmente `operation=return` e `skill_call_id`. Il mapper Responses passa questo oggetto in `extra_body`; il server legge `agentic` con `parse_agentic_value()`.

Estendere questa struttura con `extra_body.agentic.interactive`, che sul wire diventa `agentic.interactive`. Non aggiungere un namespace `interactive` parallelo alla radice, non duplicare contesto/scope e non alterare il registro statico o il prompt. Il parser attuale rifiuta chiavi agentic sconosciute: va esteso esplicitamente per la chiave opzionale `interactive`, conservando validazioni e comportamento delle richieste precedenti.

`agentic.operation` mantiene esclusivamente la semantica di ritorno dalle skill; configure/resume/cancel appartengono a `agentic.interactive.operation`. Una richiesta ordinaria può applicare un return valido e configurare le chiamate dello scope risultante; non combinare return con la ripresa/annullamento di una continuazione sospesa.

Prevedere una capacità esplicita interrogabile, proposta `GET /v1/agentic/capabilities`, che dichiari versione, comportamenti, API e sintassi di modello supportati, oltre ai limiti configurati. La prima versione usa `/v1/responses`; Chat Completions e Anthropic non sono implicitamente qualificati.

Esempio di oggetto passato dal client come `extra_body` (lo SDK ne porta le proprietà nel body HTTP):

```json
{
  "agentic": {
    "agent_context_id": "run-42:root",
    "allowed_tools": ["edit-file"],
    "allowed_skills": [],
    "interactive": {
      "version": 1,
      "operation": "configure",
      "request_id": "turn-1",
      "tools": [
        {
          "name": "edit-file",
          "behavior": "edit_file.v1",
          "fields": {
            "resource": "path",
            "selection": "old_text",
            "replacement": "new_text"
          },
          "first_fields": ["path"],
          "markers": {
            "line": "@ex_meckh_line[<N>]",
            "all": "@ex_meckh_all",
            "delete": "@ex_meckh_delete"
          }
        }
      ]
    }
  }
}
```

`fields` mappa i ruoli del comportamento ai nomi effettivi degli argomenti; `first_fields` stabilisce separatamente l'ordine iniziale e il confine di sospensione. Il parser generico usa indici di proprietà risolti dal registro, non nomi C fissi come `path`. Anche il comportamento edit risolve resource/selection/replacement dalla mappa ricevuta.

La specifica deve includere grammatica della famiglia di riga, campi ammessi, riserva `@ex_meckh_` e combinazioni; gli esempi non sostituiscono la grammatica. Il comportamento è registrato nel server e il payload non può ridefinirne arbitrariamente la semantica. Il JSON Schema del tool mantiene i tipi standard.

Validare prima del decode versione, duplicati, nomi, schema, scope e limiti, compresi `first_fields` non vuoto, nomi distinti presenti fra gli argomenti obbligatori dello schema e compatibilità con il comportamento. Edit richiede tre ruoli associati a campi distinti, resource nel gruppo prioritario e selection/replacement successivi, nell'ordine selection → replacement. Rifiutare alias fra ruoli, tipi incompatibili e configurazioni che contraddicono queste dipendenze. Le altre implementazioni registrate possono richiedere uno o più campi e tipi differenti, fra quelli supportati dal parser e dallo schema; non limitare il protocollo generico a una stringa path.

Rendere esplicita l'accettazione del protocollo prima che il client possa utilizzare una chiamata generata; un server vecchio che ignora campi sconosciuti non deve essere scambiato per un server compatibile.

### 4.1.1 Campi prioritari e sospensione generale

Generare i campi di `first_fields` nell'ordine dichiarato. Ogni valore deve superare il normale riconoscimento e la validazione del suo tipo prima di passare al successivo. Il completamento di un primo campo non sospende prematuramente la chiamata se ne restano altri nel gruppo. Vietare campi successivi, duplicati, riordinamenti e chiusura della chiamata prima dello scambio.

Quando l'ultimo campo prioritario è completo, sospendere e pubblicare una mappa dei valori decodificati, indicizzata con i nomi reali degli argomenti. Questi valori restano congelati per l'intera continuazione: non vengono rigenerati né modificati dal runtime. Anche se il gruppo comprende tutti gli argomenti, la finalizzazione della chiamata attende la risposta del runtime.

Per esempio, un comportamento futuro di un altro tool potrebbe dichiarare `first_fields=["resource_id","format"]`: entrambi precedono il resto, la pausa avviene dopo format e l'evento contiene entrambi i valori. È un esempio del protocollo e una fixture di test, non una nuova implementazione produttiva da aggiungere in questo task.

I ruoli e le esigenze del comportamento specifico restano separati dalla macchina generica. Eventuali scambi successivi usano nuovi exchange ID e dipendenze dichiarate; non aggiungere rami basati su nomi di campi nel loop di inferenza.

### 4.2 Sospensione e risposta del runtime

```json
{
  "type": "interactive.await_input",
  "version": 1,
  "continuation_id": "cont_...",
  "exchange_id": "exchange_...",
  "call_id": "call_...",
  "tool_name": "edit-file",
  "behavior": "edit_file.v1",
  "request": {
    "kind": "file_snapshot",
    "values": {"path": "src/example.py"}
  }
}
```

Lo stream HTTP termina dopo l'evento di controllo; il worker viene liberato, mentre lo stato logico rimane conservato. Non emettere `response.completed` o un evento di completamento degli argomenti per la chiamata parziale. Per `stream=false` restituire un involucro dedicato con stato `awaiting_input` e gli stessi identificatori.

La nuova richiesta conserva il medesimo oggetto `agentic` per identità e scope e porta `agentic.interactive.operation = resume`, versione e ID di continuazione/scambio, senza duplicarli in un namespace root. Il runtime controlla la mappa `request.values` e, per edit, ricava il path dal campo associato al ruolo resource. Il comportamento edit ammette uno dei payload seguenti:

| Esito | Dati e conseguenza |
| --- | --- |
| `provide` | Testo esatto, encoding originario e numero di righe fisiche; ripresa per un file esistente, anche vuoto |
| `create` | Autorizzazione del path inesistente; nessun vecchio contenuto, soltanto all o line[1] in old |
| `reject` | Codice e messaggio di rifiuto; nessun contenuto del file, nessuna generazione di old/new |

Gli esiti provide/create/reject e il payload file appartengono al comportamento edit; il protocollo generico trasporta il tipo e i dati della risposta prevista dal comportamento registrato senza assumere un filesystem. Il testo di controllo è una stringa Unicode trasportata in JSON/UTF-8; encoding e BOM originali servono al client per ricostruire i byte del file. Il server ricava autonomamente inizi di riga e conteggi, verificando i metadati ricevuti, e usa offset interni UTF-8 distinti dagli offset del file originario. Gestire lunghezze esplicite per NUL oppure rifiutarlo prima del decode con un errore di formato non supportato: mai troncare stringhe C silenziosamente. Gli hash restano interni al client. Il server verifica coerenza del payload e dei metadati, non l'identità del filesystem. Definire nel contratto i nomi finali dei campi, la risposta di accettazione e i messaggi terminali `interactive.rejected`, `interactive.cancelled`, `interactive.expired` e `interactive.failed`.

Le riprese non sono normali richieste con una nuova history: il parser individua `agentic.interactive` e valida identità/scope prima del rendering del prompt, usando lo stato conservato. Non chiamare genericamente tutte le operazioni agentic «resume»: mantenere il return esistente distinto. Una ripresa non può modificare valori dei campi prioritari, ordine, mappa dei ruoli, tool, registro o impostazioni del turno originale.

### 4.3 Completamento, contabilità e replay

Al completamento produrre una normale function call con argomenti JSON completi e la call ID originaria. I delta della ripresa riguardano il nuovo suffisso; la risposta finale contiene gli argomenti completi, senza duplicazioni. Separare gli ID delle singole risposte HTTP dall'ID della chiamata logica.

Contabilizzare una sola volta i token generati attraverso gli scambi, senza conteggiare lo snapshot di controllo come token di prompt. `max_output_tokens` resta il budget complessivo del turno sospeso, non si azzera a ogni resume. Esporre limiti separati per bytes del payload e stato conservato.

Le chiamate già complete nello stesso turno restano conservate durante la pausa. Il client le accumula e le esegue soltanto quando il turno logico termina; una sospensione non è un confine di dispatch. Nella v1 un reject termina la generazione del turno, conservando le precedenti chiamate complete e il record del rifiuto. Su cancel non eseguire chiamate accumulate e registrare gli esiti di annullamento necessari per quelle complete, senza fingere risultati per la chiamata parziale.

Un rifiuto conserva call ID, nome, mappa dei campi prioritari e motivo in un record di chiamata interrotta non eseguibile; per edit la mappa contiene il path. Proiettarlo nel replay come evento runtime non eseguibile con valori escapati secondo il renderer condiviso, distinto da function_call/function_call_output; congelare tale rappresentazione nelle fixture: non inventare old/new vuoti, non dichiarare completa la tool call e non creare un tool result orfano. La successiva inferenza ordinaria deve poter vedere il rifiuto e le eventuali ispezioni sintetiche aggiunte dal client senza ereditare un parser bloccato nella chiamata incompleta. Dopo un rifiuto invalidare la frontiera contenente il prefisso nativo parziale e riallineare la KV alla history canonica visibile, riusando solo il prefisso effettivamente comune. Questo eventuale prefill ripara il replay del rifiuto; non inserisce lo snapshot del file nel prompt e non si applica alla ripresa riuscita.

## 5. Sospensione e vita della continuazione

```text
generating → awaiting_input → generating → completed
                         ↘ rejected / cancelled / expired / failed
```

Lo stesso protocollo può prevedere più scambi per futuri comportamenti. `edit_file.v1` ne richiede uno dopo il gruppo prioritario configurato, che nell'istanza attuale contiene path.

- Assegnare la call ID quando la chiamata interattiva viene riconosciuta, prima di pubblicare la sospensione.
- Conservare `first_fields`, avanzamento nel gruppo, valori decodificati congelati, mappa dei ruoli risolta, prefisso nativo esatto, offset, tracker/escape, chiamate complete precedenti nello stesso turno, thinking/streaming, RNG, budget residuo, override locali e stato necessario della sessione e del drafter.
- Legare lo stato a contesto agentico opaco, registro, scope, modello e sessione; non ricostruire la gerarchia delle skill sul server.
- Congelare la frontiera KV e preservare sessione e metadati prima di pubblicare `await_input` e liberare il worker; se la conservazione fallisce, emettere un errore terminale. Riutilizzare il checkpoint dei contesti estendendolo esplicitamente alle continuazioni interattive: oggi `agent_context_has_pending_tool_call()` considera soltanto `responses_live.call_ids` delle chiamate complete e non protegge questo caso. Non aggiungere una call ID parziale alla lista delle chiamate eseguibili per aggirare tale condizione.
- Conservare anche identità del prefisso di sistema, immagini e relativa ownership, stato interattivo e riferimento alla KV. Il routing resume deve trovare il contesto attraverso la continuazione prima della normale materializzazione di input/prompt e della verifica del prefisso di una nuova richiesta. Ripristinare la sessione senza un input fittizio e senza introdurre il file nel prompt.
- Impedire decode concorrenti sullo stesso stato. La ripresa deve acquisire lo scambio una sola volta anche quando due job sono già in coda. Durante la sospensione rifiutare nuove inferenze ordinarie sullo stesso contesto con conflitto di stato, finché la continuazione non termina o viene annullata; gli altri contesti possono procedere. Usare una generazione del contesto per invalidare anche resume già accodati prima di cancel/steering.
- Usare TTL e budget di numero/memoria per le continuazioni. In questa v1 non promettere ripresa dopo un riavvio: restituire stato sconosciuto esplicitamente.
- Richiedere un ID stabile per ogni operazione prima di avviare la generazione e una lettura di stato. Un retry con lo stesso ID e payload canonico restituisce lo stato o il risultato già prodotto, senza risampling; stesso ID con payload diverso produce conflitto. Una ripresa di altro contesto non è autorizzata. Conservare ricevute terminali e risultato recuperabile per un TTL dichiarato, separato dalla KV, così il cleanup della sessione non impedisce il recupero di una risposta HTTP persa.
- Annullamento e steering invalidano la continuazione; cambio di scope e return di skill non possono riutilizzarla. Liberare payload voluminosi e stato di inferenza in ogni percorso terminale; conservare soltanto le ricevute bounded previste per i retry.
- Una disconnessione dopo la sospensione conserva lo stato soltanto entro il TTL. Specificare come il client ritrova lo scambio attraverso un ID di richiesta stabile se l'evento non è arrivato.

## 6. Confine esatto del gruppo prioritario

Estendere il tracker della sintassi realmente generata da ciascun modello e la conversione in argomenti JSON. Il riconoscimento deve essere strutturale, incrementale e limitato alle chiamate effettive: testo citato nel reasoning o nei valori non apre una continuazione.

Applicare gli indici compilati da `first_fields`, registrare ciascun valore completo e sospendere dopo l'ultimo. Gestire stringhe, escape, Unicode e gli altri tipi supportati senza ricerche testuali su frammenti arbitrari. Nell'istanza edit il valore prioritario è il path, ma il tracker generico non lo identifica attraverso un confronto con quel nome.

Un token può contenere insieme chiusura dell'ultimo valore prioritario e inizio di un argomento successivo: vietare prima del sampling i token che oltrepassano il confine consentito. Non emettere soltanto una parte di un token già committato. Limitare anche builder delle sequenze forzate, batching e MTP; invalidare eventuale next token speculativo non più valido. Durante l'avvicinamento alla sospensione preferire decode senza speculazione finché il confine non è garantito.

## 7. Tipo dinamico di old_text

Comporre tre rami nella grammatica dell'argomento: selezione letterale da inizio riga, all e line[N]. Delete appartiene soltanto alla grammatica di new.

### 7.1 Maschera del prossimo passo

1. Dal testo ricevuto ricavare gli inizi fisici delle righe CR/LF/CRLF, senza inventare una riga per il newline finale.
2. Mantenere prefisso logico generato e soli offset di inizio riga ancora compatibili.
3. Per ogni token candidato simulare il decoding del valore e l'avanzamento della grammatica; ammetterlo se rimane almeno un ramo valido.
4. Rispettare UTF-8 parziale, escape e segmentazioni alternative: non usare soltanto una tokenizzazione precalcolata delle righe.
5. Valutare separatamente eventuale chiusura: selezione testuale non vuota, esatta, univoca nell'intero snapshot e iniziata a un offset di riga.
6. Committare lo stato soltanto per il token selezionato. Dopo l'unicità mantenere ammesse le estensioni compatibili oltre alla chiusura.

Scartare anche candidati testuali che non potranno mai raggiungere una chiusura valida: se perfino la massima estensione letterale consentita fino a EOF (o prima della famiglia riservata) resta ambigua globalmente, nessun suo prefisso può diventare univoco. Effettuare questa verifica con indici/ricerche e cache per offset, anche lazy, senza espandere scenari di token futuri né copiare tutti i suffissi. Esempio: in `xabc\nabc` il candidato dell'ultima riga `abc` è impossibile, perché a EOF coincide ancora con un'occorrenza interna alla prima riga; non consentire il token che lascerebbe soltanto quel ramo. I marcatori rimangono disponibili all'inizio, ma non costituiscono un fallback dopo aver iniziato una selezione testuale.

Riutilizzare filtro e trie del vocabolario per la sola maschera del passo corrente. Non materializzare tutte le sottostringhe, i suffissi o i percorsi futuri. Per il conteggio globale, distinto dai candidati di generazione, basta distinguere 0, 1 e più occorrenze, includendo quelle sovrapposte.

Esempio: nello snapshot `foo\nprefix foo\nbar`, `foo` e `foo\n` restano ambigui perché compaiono anche dentro la seconda riga. `foo\np` è univoco; il modello può chiuderlo oppure continuare con testo compatibile con la prima posizione. La seconda occorrenza non autorizza mai una partenza a metà riga.

### 7.2 Composizione con i percorsi ottimizzati esistenti

Il vincolo dinamico deve intersecare ogni percorso di accettazione: maschere statiche, fast path delle stringhe libere, token dichiarati unconditional, sequenze forzate e verifier. Un token ammesso dallo schema non è automaticamente ammesso dallo snapshot. Le sonde del trie con `token=-1` chiedono la vitalità di un prefisso di token, non la validità di un token completo: non potare prefissi UTF-8/escape ancora completabili.

Tenere distinto l'hash del registro statico dall'identità del vincolo dinamico. Le cache devono includere configurazione compilata, versione dello snapshot/ramo create, prefisso e stato corrente; invalidarle su ogni transizione pertinente. `ds4_constraint_analysis` appartiene alla sessione ed è valido soltanto fino alla successiva analisi: ricostruirlo dopo restore e per il passo corretto, senza conservarne puntatori nelle continuazioni o riutilizzarlo tra righe speculative.

### 7.3 Marcatori e new_text

| Caso | old ammesso | Vincolo su new |
| --- | --- | --- |
| File esistente non vuoto | Selezione univoca da inizio riga, all, line[1..L+1] | Testo ordinario; delete soltanto dopo all |
| File esistente vuoto | all, line[1] | Testo ordinario; delete soltanto dopo all |
| Creazione | all, line[1] | Testo ordinario, anche vuoto; delete vietato |

I marcatori devono occupare l'intero campo, senza spazi, ripetizioni o aggiunte. Una famiglia riservata sconosciuta o incompleta non diventa testo ordinario. Dopo all, il ramo delete compete inizialmente con il testo ordinario di new; una volta scelto delete non sono ammessi altri caratteri. Fuori da questo caso non generarlo come contenuto da scrivere.

La sanificazione preventiva appartiene al client; il server non deve rimuovere i marcatori dagli argomenti generati. Il testo nuovo ordinario conserva contenuto e terminatori esatti e può superare la lunghezza di old entro il budget. Vietare EOS/stop che finalizzerebbero prematuramente una chiamata interattiva incompleta.

## 8. Temperatura zero e MTP locale

Per old usare temperatura effettiva 0 dopo il masking, selezionando il migliore token ammesso. Se la maschera non ammette token validi, terminare con errore esplicito anziché campionare senza vincoli.

Disattivare MTP finché più righe di partenza restano candidate. La v1 mantiene inoltre target-only finché il prefisso letterale non è globalmente univoco: il confronto locale ha rilevato una diversa estensione greedy con il verifier attivato quando esisteva una sola partenza ma restavano match interni. Riattivarlo dopo entrambe le condizioni, sempre aggiornando e verificando il vincolo per ogni passo proposto. L'unicità della riga candidata non sostituisce il conteggio globale richiesto per chiudere il blocco.

Il verifier deve usare copie/rollback dello stato del riconoscitore e includere la possibilità di chiusura. Non applicare la stessa maschera a un intero draft. La sequenza greedy accettata deve coincidere con quella senza MTP, compresi i token di confine.

La chiusura di selection è anche una frontiera di sampling: impedire che token atomici o batch forzati/speculativi contengano insieme contenuto di old e contenuto di new campionato con la politica sbagliata. Nella v1 fermare tali percorsi al confine, scartare carry obsoleti e ripristinare la politica ordinaria prima del primo token di replacement; la grammatica/tokenizer supportata deve consentire questa separazione, altrimenti la combinazione non è qualificata.

Gli override sono locali alla chiamata: non modificare variabili d'ambiente o impostazioni globali dell'engine. Ripristinare sampling ordinario per new e dopo ogni uscita, compresi rami dei marcatori, errori e annullamenti. Gestire carry/next token e sincronizzazione del drafter prima di riattivarlo; rispettare i gate esistenti delle sessioni batch e non attivare MTP su un backend che lo aveva disabilitato.

## 9. Ordine di realizzazione e criteri di completamento

### Fase 1 — Congelare il contratto

- [x] Fissare capability, namespace, messaggi, schemi, errori, limiti e TTL, sincronizzandoli con il TODO client.
- [x] Preparare fixture di provide, create, reject, streaming/non streaming, retry e replay della chiamata interrotta, riuso di agentic con e senza return e gruppi prioritari singoli/multipli con nomi diversi da path.
- [x] Stabilire la matrice iniziale API/modelli; le API/sintassi non implementate sono rifiutate esplicitamente. Le grammatiche implementate e la qualifica con modello reale sono dichiarate separatamente nel contratto.

Completamento: le fixture descrivono ogni transizione e non esiste ambiguità fra tool completo, sospensione e rifiuto.

### Fase 2 — Protocollo e storage senza masking testuale

- [x] Aggiungere parsing, negoziazione e validazione senza modificare richieste ordinarie.
- [x] Implementare registro dei comportamenti, dipendenze fra ruoli, storage bounded delle continuazioni e ricevute idempotenti.
- [x] Compilare `fields`/`first_fields` dal registro e integrare ordine configurato, frontiera dell'ultimo campo prioritario, rilascio del worker e ripresa della stessa call ID.
- [x] Estendere checkpoint/routing ai prefissi interattivi incompleti; verificare salvataggio prima dell'evento, ripresa dopo riuso dello slot e isolamento del contesto.
- [x] Implementare rifiuto e replay canonico, timeout, annullamento, isolamento e cleanup.

Completamento: un runtime simulato può sospendere/riprendere/rifiutare senza argomenti anticipati né snapshot file nel prompt; il rifiuto lascia una history canonica riproducibile.

### Fase 3 — Vincoli dinamici e marcatori

- [x] Implementare riconoscitore del valore, offset di riga, controllo globale di unicità e pruning dei candidati senza chiusura possibile.
- [x] Comporlo con sintassi nativa/JSON, trie e maschera del vocabolario esistente, invalidando cache e fast path incompatibili.
- [x] Coprire all, line, delete, vuoto, creazione, famiglie riservate e limiti.
- [x] Applicare temperatura 0 e mantenere MTP disabilitato nel tratto ambiguo.

Completamento: nessun token accettato viola inizio riga, compatibilità letterale o regole di chiusura; il modello può continuare dopo l'unicità.

### Fase 4 — MTP e flussi multipli

- [x] Integrare il riconoscitore nel verifier passo per passo e verificare riattivazione locale, carry e rollback.
- [x] Coprire più tool nello stesso turno, pause successive e richieste concorrenti senza duplicazioni di call ID/esiti nelle fixture host.
- [x] Verificare passaggi di contesto, skill return, riuso dello slot e budget cumulativi nelle prove locali, anche con due sessioni residenti CUDA. Le altre piattaforme restano da qualificare come indicato nella matrice.

Completamento: parità greedy con MTP spento e corrispondenza esatta di stream, chiamate ed esiti. I cambiamenti del filesystem fra operazioni rimangono verificati dal client.

### Fase 5 — Qualifica e documentazione finale

- [x] Eseguire test deterministici, build e regressioni pertinenti.
- [x] Eseguire integrazione con il motore reale e registrare la matrice PASS/FAIL/NOT_VERIFIED.
- [x] Misurare costo del masking, memoria, latenza di sospensione/ripresa e riuso KV, includendo snapshot ripetitivi e blocchi lunghi.
- [x] Aggiornare contratto implementato, indice progetto e indici wiki; non dichiarare supporto a backend non verificati.

L'integrazione con la libreria richiede il lavoro client del TODO e resta una verifica successiva; non fa parte di questa consegna server.

## 10. Matrice di verifica

| Gruppo | Casi obbligatori |
| --- | --- |
| Protocollo | Versioni/capacità incompatibili, agentic precedente invariato, namespace nested, distinzione return/resume, override del nome, schema, scope, unknown/expired/consumed, doppia ripresa concorrente, riavvio |
| Campi prioritari | Nomi arbitrari e override, gruppi singoli/multipli, tipi supportati, duplicati/campi assenti/ordine errato, pausa solo dopo l'ultimo valore, valori congelati, escape/Unicode, token oltre la frontiera, sequenze forzate e MTP |
| Testo | Indentazione, partenza a metà riga vietata, righe identiche, sovrapposizioni, occorrenze interne, multilinea, continuazione dopo unicità, suffisso ambiguo fino a EOF, selezione prima di replacement indipendentemente dallo schema |
| Marcatori | Rami disponibili dal primo token, prefissi condivisi, indici limite, incompleti/sconosciuti, combinazioni errate, vuoto e creazione |
| Stato | Pausa senza call complete, errore di checkpoint prima dell'evento, restore senza nuovo prompt, conflitto sul contesto sospeso, cancel con resume accodato, Thinking, batch di tool, passaggi di nodo/scope, slot eviction, disconnessioni, budget, ripristino sampling/MTP, assenza di stato condiviso |
| Cache e sampling | Stesso registro con snapshot/configurazione diversi, sonde trie e UTF-8 parziale, fast path stringa libera, analisi session-owned, confine old/new e carry speculativo |
| Risultati | Retry stesso ID/payload e payload diverso, risposta persa e ricevuta dopo cleanup KV, batch prima di reject/cancel, Call ID invariata, nessun completamento parziale, esito rifiutato riproducibile, nessun argomento inventato, usage senza duplicazioni |
| Regressioni | Tool ordinari, JSON Schema, continuazioni Responses e checkpoint agentici, target-only/MTP, SSD e backend disponibili |

Usare come riferimento un filtro semplice della maschera corrente e confrontarlo con il percorso ottimizzato sul vocabolario intero; non scrivere test che si limitino a replicare l'implementazione. Riutilizzare i test server/agentici esistenti e aggiungere fixture specifiche dove servono.

Non avviare modelli grandi in parallelo. Rispettare i gate del progetto per Metal, SSD, CUDA e distribuito; registrare i test non eseguiti e richiedere l'accesso hardware previsto dalle istruzioni quando necessario. Non sono previste modifiche ai kernel numerici per questo piano.

## 11. Definizione di completamento

Il lavoro server è completo quando tutte le fasi sono verificate per la matrice dichiarata, la libreria può usare il contratto senza fallback silenziosi e una chiamata può essere sospesa, rifiutata o completata con identità e stato coerenti. Validazioni e scritture client rimangono obbligatorie anche dopo la qualifica del masking.

## 12. Qualifica locale del 9 ottobre 2026

Implementazione disponibile in `ds4_server.c`, `ds4_interactive_config.h`, `ds4_interactive_grammar.h`, `ds4_interactive_runtime.h`, `ds4_interactive_edit.h` e `ds4_edit_constraint.h`. Il [contratto wire](../agentic/interactive-calls.md) e il TODO client sono sincronizzati. Il [report JSON](../performance/interactive-edit-file-qualification-20261009.json) conserva la matrice, il confronto delle chiamate e i limiti della verifica.

Ambiente: Ubuntu 24.04 in WSL2, NVIDIA RTX 3090, Qwen3.8-27B UD-Q4_K_S con sidecar MTP Q4_0. Le prove HTTP usano contesto 22.593 e, per due sessioni residenti, 4.096 per sessione. Sono processi sequenziali: non sono stati avviati modelli grandi contemporaneamente.

| Verifica | Esito |
| --- | --- |
| Build di produzione e front-end CPU | PASS |
| `make test`, con il modello Qwen3.8 supportato | PASS; i gate Metal/fixture mancanti restano saltati dal runner |
| `test-server-interactive` e AddressSanitizer/UndefinedBehaviorSanitizer | PASS |
| Raccolta Python locale con dipendenze temporanee uv | PASS: 267 test, 57 skip; i gate opt-in sono eseguiti separatamente quando applicabili |
| Compatibilità Qwen3.8, CLI target/MTP, performance e checksum | PASS: 20/20, senza skip |
| HTTP interattivo target-only | PASS: 10/10, inclusi i casi greedy/protocollo e replacement a temperatura positiva |
| HTTP interattivo MTP | PASS: 9/9 casi greedy/protocollo e replacement a temperatura positiva |
| HTTP interattivo con due sessioni residenti e MTP caricato | PASS: 10/10; speculazione interattiva disabilitata nel batch |
| HTTP JSON Schema/tool ordinari | PASS: 20/20 |
| HTTP skill/scope/return/checkpoint agentici | PASS: 17/17, incluso il gate extreme sul report numerico completo |
| Checkpoint e frontiere CUDA/MTP: restore, corruzione, troncatura, annullamento, nesting e continuazione | PASS bit-exact nella variante breve con MTP, nella prova completa da 10.000 token e nelle frontiere residenti |
| Matrice sampling ordinario, thinking e MTP | PASS: 12 scenari |
| Wiki, indici e collegamenti | PASS |
| Metal, ROCm, distribuito, streaming SSD degli esperti e altri modelli | NOT_VERIFIED: non qualificati da queste prove server |
| Adattatore interattivo ExMekh, filesystem e hook client | NOT_VERIFIED: implementazione client ancora nel TODO |

Il trie è confrontato con il filtro esaustivo sul vocabolario corrente, sia in decode sia nel verifier: zero divergenze osservate. Le fixture testuali usano inoltre un oracolo indipendente su 400 snapshot ripetitivi/CR/LF. Per creazione, inserimento e selezione multilinea gli argomenti JSON greedy coincidono fra target-only e MTP, mantenendo call ID, usage cumulativa e assenza di prefill del file. La prova multilinea riusa lo slot con un altro contesto prima della ripresa.

Benchmark host: snapshot di 1.048.520 byte e 18.590 righe, costruzione dell'indice circa 67–87 ms, lookup della selezione usata circa 1,4–1,9 µs. Gli indici occupano 11.195.721 byte e il picco degli array durante la costruzione è stimato in 32.167.149 byte, esclusi allocator e memoria del server. Queste dimensioni sono derivate dalle allocazioni, non misure RSS.

Con il confronto esaustivo diagnostico attivo, una ripresa completa impiega circa 5,9–8,9 s nei tre casi registrati. Include generazione, restore KV e riallineamento del transcript; non misura soltanto il checkpoint. Non è un claim di accelerazione MTP: la correttezza dei vincoli ha precedenza e le finestre vengono annullate ai confini. In produzione il confronto esaustivo diagnostico è disabilitato.

Correzioni emerse durante la qualifica: separazione strutturale del reasoning anche con tag citati; `string=false` non può bypassare i vincoli stringa; stato terminale pubblicato prima della risposta; ricevute distinte per generazione/cancel; cleanup delle richieste disconnesse prima del dispatch; sequenze SSE monotone dopo l'accettazione; policy di temperatura condivisa con il verifier; decode ordinario anche per singleton, per evitare variazioni di logits indotte dal prefill di un token. La riattivazione MTP resta conservativa fino all'unicità globale del prefisso letterale.

I default operativi dei test sono allineati al modello supportato disponibile; sono stati rimossi i riferimenti operativi obsoleti incontrati nei piani. I risultati storici e i test che verificano il rifiuto dei modelli ritirati non costituiscono supporto a tali modelli.

## 13. Revisione delle responsabilità

Il protocollo ora conserva configurazione e stato opachi del comportamento. La grammatica usa policy dichiarative per dipendenze, campi dopo la pausa, temperatura greedy e frontiere di sampling; i dettagli di edit non sono conosciuti dal dispatcher o dal runtime generico.

| Componente | Responsabilità dopo la revisione |
| --- | --- |
| `ds4_interactive_config.h` | Registro, compilazione e interrogazione delle policy dei campi; elenco unico delle sintassi accettate |
| `ds4_interactive_edit.h` | Ruoli edit, marcatori, interpretazione dello snapshot, vincoli e gate di sampling; creazione e distruzione dei dati propri |
| `ds4_interactive_grammar.h` | Composizione delle policy e dei callback con la grammatica nativa e la validazione JSON Schema |
| `ds4_interactive_runtime.h` | Lifecycle dei job e delle continuazioni, trasferimento dello stato di generazione, ricevute, budget, capabilities ed eventi di interruzione |
| `ds4_server.c` | Routing HTTP, scheduling e orchestrazione tramite le operazioni dei moduli |

Gli eventi failed/rejected/cancelled/expired condividono il renderer dei metadati. Le failures di generazione espongono sempre `output:[]`, come le altre interruzioni senza tool eseguibili. Capabilities e validazione derivano dallo stesso registro di sintassi.

I valori prioritari congelati sono confrontati con `schema_json_equal()`, già usato dal validatore: oggetti JSON con chiavi riordinate restano equivalenti; un valore diverso viene rifiutato. Lo snapshot viene preparato separatamente e sostituito soltanto dopo una validazione completa, così un input invalido preserva lo stato precedente. I distruttori del comportamento sono chiamati prima di liberarne la configurazione.

Le fixture aggiuntive verificano l'uguaglianza strutturale degli oggetti prioritari, il rifiuto dei valori modificati e la conservazione dello snapshot dopo una risposta invalida. La revisione non modifica kernel o formati dei checkpoint; la matrice hardware della sezione precedente continua a delimitare la qualifica disponibile.

Verifica finale della revisione: `make test`, build CUDA/CPU e AddressSanitizer/UndefinedBehaviorSanitizer passati; suite HTTP combinata target-only 47/47 senza skip; interattivo MTP 10/10; due sessioni residenti 10/10; matrice ordinaria MTP 12 scenari. Il confronto con l'oracolo non registra divergenze nelle tre modalità e i tre casi greedy registrati mantengono gli stessi argomenti fra target-only e MTP. Esiti e ownership sono conservati anche nel report JSON.
