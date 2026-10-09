# Chiamate interattive v1

Contratto server implementato il 9 ottobre 2026. Il [piano e la matrice di verifica](../roadmaps/interactive-edit-file-server-plan.md) descrivono obiettivi e qualifica. Le attività client restano in [exmekh_lib/todo.md](../../exmekh_lib/todo.md). Il server genera argomenti: non apre, sanifica, scrive o cancella file.

## Trasporto e configurazione

Interrogare `GET /v1/agentic/capabilities`, verificando anche `model_supported` e `model_syntax` del modello caricato. L'estensione usa esclusivamente `POST /v1/responses`; sulle altre API è rifiutata. Le grammatiche implementate sono Qwen27 DSML, Qwen4 nativa e DSML DeepSeek 4/4.1; GLM è rifiutato. La qualifica con modello reale riguarda Qwen3.8-27B Q4_K_S su CUDA RTX 3090, con e senza MTP Q4_0. Gli altri modelli/backend restano **NOT_VERIFIED**.

Il client OpenAI passa questa configurazione in `extra_body`. Sul wire `agentic` è una proprietà del body della normale richiesta Responses, insieme a `model`, `input`, `tools`, sampling e budget:

```json
{
  "agentic": {
    "agent_context_id": "run:root",
    "allowed_tools": ["edit-file"],
    "allowed_skills": [],
    "interactive": {
      "version": 1,
      "operation": "configure",
      "request_id": "turn-1",
      "tools": [{
        "name": "edit-file",
        "behavior": "edit_file.v1",
        "fields": {"resource": "path", "selection": "old_text", "replacement": "new_text"},
        "first_fields": ["path"],
        "markers": {"line": "@ex_meckh_line[<N>]", "all": "@ex_meckh_all", "delete": "@ex_meckh_delete"}
      }]
    }
  }
}
```

I nomi del tool e degli argomenti sono quelli effettivamente registrati, inclusi gli override. Non sono fissati nel loop di inferenza. `first_fields` è ordinato, non vuoto, senza duplicati e composto da proprietà obbligatorie dello schema. Può includere più campi e tipi standard, per esempio `target` stringa seguito da `format` booleano. I ruoli edit devono riferirsi a tre stringhe obbligatorie distinte; resource deve appartenere al gruppo prioritario, selection/replacement devono seguirlo. La grammatica impone selection prima di replacement anche con proprietà schema riordinate.

La configurazione non modifica il registro o il prompt. `agentic.operation=return` conserva la semantica delle skill e può accompagnare una configurazione ordinaria valida. Non può accompagnare operazioni di controllo di una continuazione. `stop` e `ignore_eos` sono incompatibili con v1.

In streaming il server conferma con `interactive.accepted` dopo `response.created`. Le risposte finali normali includono `interactive.version` e `interactive.continuation_id`; una pausa conferma esplicitamente la versione nel suo involucro. Il client deve interpretare questi eventi tramite trasporto raw/custom, non aspettarsi che i tipi standard dell'SDK riconoscano l'estensione.

## Pausa e ripresa

Dopo la chiusura dell'ultimo campo prioritario il server salva il checkpoint, pubblica questo evento e termina la connessione HTTP, liberando il worker:

```json
{
  "type": "interactive.await_input",
  "version": 1,
  "status": "awaiting_input",
  "continuation_id": "cont_...",
  "exchange_id": "exchange_...",
  "call_id": "call_...",
  "tool_name": "edit-file",
  "behavior": "edit_file.v1",
  "request": {"kind": "file_snapshot", "values": {"path": "src.py"}},
  "usage": {"input_tokens": 420, "output_tokens": 60, "total_tokens": 480}
}
```

L'esempio abbrevia i dettagli di usage. Con `stream=false` lo stesso JSON è il body della risposta. Con SSE l'evento è `interactive.await_input`; non viene emesso `response.completed`. Le chiamate precedenti nello stesso turno rimangono accumulate: la pausa non autorizza il dispatch dei tool.

Inviare una nuova richiesta contenente soltanto `agentic` ed eventualmente `stream`:

```json
{
  "agentic": {
    "agent_context_id": "run:root",
    "allowed_tools": ["edit-file"],
    "allowed_skills": [],
    "interactive": {
      "version": 1,
      "operation": "resume",
      "request_id": "turn-1-snapshot",
      "continuation_id": "cont_...",
      "exchange_id": "exchange_...",
      "reply": {"status": "provide", "text": "foo\nprefix foo\nbar", "encoding": "utf-8", "line_count": 3}
    }
  }
}
```

Il testo è Unicode esatto in JSON/UTF-8. `encoding` è una stringa non vuota che descrive l'encoding originario al client; il server non ricodifica il file. NUL e UTF-8 invalido sono rifiutati. Il conteggio delle righe viene ricavato e verificato: CRLF conta una volta, CR/LF sono terminatori fisici, il terminatore finale non inventa una riga aggiuntiva. Un file vuoto ha `line_count=0`.

Gli altri payload edit sono `{"status":"create"}` per un path inesistente e `{"status":"reject","code":"unread","message":"Read the current file first"}`. Nessuno dei due invia contenuto. Un file esistente vuoto usa provide con testo vuoto; resta distinto da create.

La ripresa eredita modello, schema, scope, valori prioritari, immagini, sampling, RNG, budget e modalità streaming. Se `stream` è indicato deve coincidere con il turno originale; se omesso viene ereditato anche nei retry. Non inviare nuova history o nuove impostazioni. Lo snapshot alimenta soltanto il riconoscitore e non viene tokenizzato/inserito nel prompt. La KV viene ripristinata anche dopo il riuso dello slot da parte di un altro contesto.

Il completamento produce normali `function_call` con argomenti completi e le call ID pubblicate nelle pause. Le risposte HTTP hanno ID propri. Il percorso Responses esistente sopprime i delta parziali degli argomenti: emette gli item completi alla fine; i cursori di testo/reasoning preservati impediscono di duplicare il prefisso. Usage e `max_output_tokens` sono cumulativi, non si azzerano alla ripresa; non sommare la usage delle pause a quella finale.

## Vincoli edit e campionamento

La selezione letterale parte sempre da un inizio di riga fisico, inclusa l'indentazione. Ogni token deve proseguire almeno una riga candidata compatibile. La chiusura richiede un blocco non vuoto con una sola occorrenza nell'intero snapshot, contando anche occorrenze sovrapposte e interne alle righe. Queste ultime non diventano partenze. Dopo l'unicità il modello può continuare attraverso altre righe.

I marcatori sono disponibili fin dal primo token:

| Campo | Valori speciali |
| --- | --- |
| selection | `@ex_meckh_all`, `@ex_meckh_line[N]` con N decimale senza zeri iniziali, 1 ≤ N ≤ L+1 |
| replacement | `@ex_meckh_delete` soltanto con selection all e file esistente |

La famiglia `@ex_meckh_` è riservata e non può essere inserita nel testo ordinario. Vuoto/create consentono selection all o line[1]; replacement può essere vuoto. Gli escape dei delimitatori nativi sono reversibili: il controllo globale usa i byte logici del testo, preservando newline e spazi. DSML `string=false` non può aggirare i vincoli dei campi stringa.

Durante selection la temperatura effettiva è zero. MTP resta disabilitato finché manca una sola riga candidata **o** il prefisso è ancora globalmente ambiguo; può ripartire dopo entrambe le condizioni e nel corpo replacement, usando una maschera aggiornata per ogni riga del verifier. Questo gate conservativo preserva la parità greedy nei casi verificati: anticipare la riattivazione alla sola riga candidata aveva prodotto una diversa estensione ammessa del blocco. Il cambio di parametro/interruzione della finestra provoca rollback dei draft e del relativo RNG. Token che oltrepassano la pausa o attraversano selection/replacement sono vietati prima del commit. Anche l'apertura di replacement deve chiudersi prima del suo primo token di contenuto, per applicare il sampling ordinario al nuovo testo. Nelle sessioni batch le chiamate interattive usano il decode target-only. Gli override sono locali al turno e non modificano l'engine o l'ambiente.

## Stato, retry e interruzioni

`request_id` è obbligatorio per configure/resume/cancel, non vuoto e lungo al massimo 128 byte. Lo stesso ID con payload JSON canonico identico restituisce il risultato conservato o lo stato in corso; un payload differente produce HTTP 409. L'ordine delle chiavi degli oggetti è irrilevante; l'ordine degli array rimane significativo. Ogni nuova operazione usa un nuovo ID.

Per interrogare lo stato usare lo stesso scope e `interactive={"version":1,"operation":"status","continuation_id":"cont_..."}`. La risposta `interactive.status` contiene `status` e `result`, anche dopo il cleanup della KV. Un retry della configure ritrova la continuazione se l'evento di pausa è andato perso. Una continuazione di un altro contesto/scope, già consumata o sconosciuta produce 409. Non è riprendibile dopo il riavvio.

Per annullare usare `operation=cancel`, continuation ID e un nuovo request ID. Il rifiuto emette `interactive.rejected`, conserva gli item precedenti completi in `output` e include call ID, nome, valori prioritari e motivo. L'annullamento non autorizza l'esecuzione delle chiamate accumulate. Scadenza ed errori producono rispettivamente `interactive.expired` e `interactive.failed`. Gli errori anteriori al riconoscimento della chiamata possono non avere metadati di call.

Questi record si possono includere nell'input della successiva inferenza ordinaria. Il renderer li proietta come messaggi runtime non eseguibili `<interactive_call_interrupted>`, con valori JSON escapati; non inventa old/new, una function call completa o un tool result orfano. Per un reject con item precedenti il client aggiunge separatamente quegli item e i loro risultati alla history. La frontiera interrotta viene invalidata; il prossimo turno riusa soltanto il prefisso comune con la history canonica.

Limiti v1 pubblicati nelle capabilities: snapshot 1 MiB, testo generato per turno 1 MiB, valori prioritari serializzati 16 KiB, 16 tool configurati, 64 proprietà/campi, 8 continuazioni attive, 64 record e 64 scambi oltre alla configure per continuazione; budget globale snapshot 8 MiB e checkpoint 8 GiB. I reject ammettono codice di massimo 128 byte e messaggio di massimo 16 KiB. Un cancel resta possibile al limite degli scambi. Una pausa scade dopo 300 secondi; le ricevute terminali restano 600 secondi. Il listener esegue manutenzione periodica anche senza nuove richieste. I limiti di testo, numero e TTL rendono bounded anche le ricevute e gli indici derivati. Capacità esaurita: HTTP 429; formato/configurazione invalida: 400; conflitto di stato/ID/scope: 409; errore di generazione/checkpoint: record terminale esplicito.

## Estensione e test

`ds4_interactive_config.h` contiene il registro C dei comportamenti e compila lo schema e le policy dei campi. `ds4_interactive_grammar.h` applica queste policy e i callback pubblicati dal comportamento; non interpreta ruoli edit o marcatori. `ds4_interactive_edit.h` possiede ruoli, snapshot, marcatori e gate di sampling di `edit_file.v1`. Configurazione e stato del comportamento sono opachi per il protocollo; i rispettivi distruttori accettano anche NULL e preservano l'ordine delle ownership. Una risposta di controllo invalida lascia intatto lo stato precedente. `ds4_interactive_runtime.h` possiede lifecycle delle continuazioni, restore del testo/RNG/stream, budget, capabilities, ricevute e rendering delle interruzioni.

`ds4_edit_constraint.h` indicizza le partenze e l'unicità tramite suffix array/LCP e seleziona candidati per byte iniziale; non enumera tutte le selezioni o le traiettorie future di token. L'analisi del trie esistente è limitata al logit corrente; anche i token singleton sono committati attraverso decode ordinario, perché un prefill di un singolo token può modificare i logits di Qwen. Un nuovo comportamento deve registrare handler, dipendenze, policy dei campi e dati propri con fixture dedicate; v1 non introduce uno script nel payload né implementazioni fittizie di altri comportamenti. Il confronto dei valori prioritari congelati usa l'uguaglianza strutturale JSON: l'ordine delle chiavi di un oggetto non altera il valore, mentre ordine degli array e contenuto rimangono vincolanti.

`make test-server-interactive` esegue le fixture host e l'oracolo indipendente; `make test-interactive-api-live` usa il server indicato da `DS4_INTERACTIVE_BASE_URL`. La matrice di esecuzione è nel piano. L'integrazione ExMekh rimane da svolgere dopo l'implementazione client.
