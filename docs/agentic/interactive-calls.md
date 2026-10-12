# Chiamate interattive v2: edit document con chiusura finale

Contratto server aggiornato al 12 ottobre 2026. Sostituisce v1; le configurazioni
`edit_file.v1` sono rifiutate. Il [piano e lo storico di qualifica](../roadmaps/interactive-edit-file-server-plan.md)
e il [TODO client](../../exmekh_lib/todo.md) distinguono i due componenti.
Il server genera argomenti: non apre, sanifica, scrive o cancella file.

## Configurazione e descrizione del tool

Usare GET /v1/agentic/capabilities e POST /v1/responses. Verificare versione 2,
`model_supported`, sintassi e comportamento `edit_file.v2`. Sono supportate
le grammatiche Qwen27, Qwen4 e DeepSeek 4/4.1; GLM resta rifiutato.
Il JSON Schema espone una sola stringa obbligatoria, chiamata `edits` per
default; tool e campo possono essere rinominati.

~~~json
{
  "agentic": {
    "agent_context_id": "run:root",
    "allowed_tools": ["edit-file"],
    "allowed_skills": [],
    "interactive": {
      "version": 2,
      "operation": "configure",
      "request_id": "turn-1",
      "tools": [{
        "name": "edit-file",
        "behavior": "edit_file.v2",
        "fields": {"document": "edits"}
      }]
    }
  }
}
~~~

Non inviare `markers`, `first_fields` o tre ruoli separati. Il documento
supporta tipo stringa, limiti di lunghezza e le annotazioni standard; enum/const
e combinazioni di schema sul campo sono rifiutati prima dell'inferenza:
le alternative statiche non possono in generale adattarsi allo snapshot autorizzato.
`stop` e `ignore_eos` restano incompatibili. Il ritorno dalle skill conserva
`agentic.operation=return`, distinto dai controlli interattivi.

Il server aggiunge alla descrizione del tool inviata al modello le istruzioni
per la grammatica. Non modifica la firma del tool o l'identità
del registro statico del client. La libreria deve riportare la stessa
descrizione nelle proprie istruzioni quando migra a v2.

## Documento e chiusura finale

Il modello genera soltanto il prefisso `<edit path="example1.txt"`.
Dopo l'autorizzazione dello snapshot il modello prosegue con `>` e i corpi.
Non esistono attributi code, codici casuali o prefill del server per edit.
Questo è il formato predefinito, senza flag o compatibilità con la forma codificata.

Esempio della stringa finale:

~~~text
<edit path="example1.txt"><old_text><![CDATA[vecchio testo]]></old_text><new_text>nuovo testo con </new_text> letterale</new_text></edit>
<edit path="example2.md"><old_text><all/></old_text><new_text><delete/></new_text></edit>
~~~

Il riconoscitore usa l'ultimo `</new_text>` del singolo edit, prima di `</edit>`
con eventuale whitespace di framing fra le due chiusure. Le occorrenze precedenti
sono contenuto letterale. Il whitespace prima del tag new_text finale resta testo.
Questa estensione richiede il parser del protocollo quando contiene chiusure
letterali: un parser XML standard non le interpreta allo stesso modo.
La forma generata usa doppi apici, il solo attributo path e ordine old/new.
Whitespace è consentito fra edit consecutivi e fra le due chiusure finali,
non fra old_text e new_text.

- Ogni apertura richiede il solo path non vuoto. Gli attributi code sono rifiutati.
- `</new_text></edit>` (anche con whitespace intermedio) termina l'elemento;
  non si cerca l'ultimo tag dell'intera
  chiamata, che può contenere più edit con autorizzazioni distinte.
- Per la sequenza completa `</new_text></edit>` letterale, o per altri tag,
  usare CDATA oppure entità XML. I delimitatori dentro CDATA non chiudono il corpo.

Old_text è seguito esattamente da new_text. Non ammettere DTD, entità
esterne, dichiarazioni, commenti, namespace, attributi o comandi sconosciuti.
Le entità ammesse sono amp, lt, gt, quot, apos e riferimenti numerici a
caratteri XML validi. NUL è rifiutato; UTF-8 deve essere valido alla chiusura.

Testo escapato e CDATA possono alternarsi. Spazi, indentazione, CR/LF/CRLF
sono byte logici preservati, senza normalizzazione XML. Rappresentare `]]>`
in CDATA dividendo il contenuto in sezioni adiacenti.
Gli escape del delimitatore nativo del modello sono uno strato separato:
un delimitatore nativo letterale dentro CDATA va escapato per il trasporto
nativo; all'esterno occorre preservare anche lo strato di entità XML.
Il parser dei risultati rimuove lo strato nativo una sola volta.

| Contenuto | Semantica |
| --- | --- |
| old_text letterale | Blocco non vuoto da un inizio di riga fisico, globalmente univoco |
| old_text con solo all/ | Riscrittura completa o creazione |
| old_text con solo line number="N"/ | Inserimento prima della riga N, 1..L+1 |
| new_text con solo delete/ | Cancellazione del file, solo con all e file esistente |
| new_text vuoto | Testo sostituito con vuoto, senza cancellazione del file |
| old_text con start_line/end_line e corpo vuoto | Righe inclusive, validazione semantica del client |

I comandi sono tag completi, per esempio `<all/>`, `<line number="4"/>`
e `<delete/>`, e devono occupare tutto il corpo. Le stesse stringhe dentro
CDATA o escapate sono testo. Una newline prima di delete non viene eliminata
e non può trasformare testo in un comando. Le stringhe `@ex_meckh…` non
hanno più semantica riservata.

Per intervalli molto lunghi:

~~~text
<old_text start_line="120" end_line="900"></old_text>
~~~

Il server controlla esclusivamente forma, ordine degli attributi e cifre.
Non converte i numeri e non verifica positività, ordine, limiti, righe
esistenti o corrispondenza con lo snapshot. Un intervallo come 999..0 può
arrivare al client e deve ottenere un errore del tool senza scritture.

## Pausa, ripresa, rifiuto e campionamento

La pausa avviene **dentro la stringa**, subito dopo il doppio apice che
chiude path e prima di `>`. Ogni elemento genera uno scambio,
anche se ripete un path:

~~~json
{
  "type": "interactive.await_input",
  "version": 2,
  "status": "awaiting_input",
  "continuation_id": "cont_...",
  "exchange_id": "exchange_...",
  "call_id": "call_...",
  "tool_name": "edit-file",
  "behavior": "edit_file.v2",
  "request": {
    "kind": "file_snapshot",
    "edit_index": 1,
    "values": {"path": "example1.txt"}
  },
  "usage": {"input_tokens": 420, "output_tokens": 60, "total_tokens": 480}
}
~~~

L'esempio abbrevia usage. La call ID è unica per l'intera chiamata,
l'exchange ID è nuovo a ogni pausa, l'indice parte da 1.
Una pausa non abilita il dispatch degli elementi precedenti.

Resume porta soltanto agentic ed eventualmente stream, con versione 2,
request ID nuovo, continuation ID, exchange ID e reply:

- provide: `{"status":"provide","text":"testo","encoding":"utf-8","line_count":1}`;
- create: `{"status":"create"}`, distinto dal file esistente vuoto;
- reject: `{"status":"reject","code":"unread","message":"Read the current file"}`.

Il server verifica line_count: CRLF conta una volta, CR e LF sono terminatori
fisici, un terminatore finale non crea un'altra riga; il vuoto conta zero.
Lo snapshot è controllo per il masking e non viene aggiunto al prompt.
Path identici nella stessa chiamata condividono lo snapshot iniziale;
un provide diverso o create incompatibile viene rifiutato. Gli alias
canonici sono responsabilità del client.

Durante selezione letterale ogni prefisso deve proseguire una riga candidata.
La chiusura richiede una sola occorrenza globale, anche contando quelle
interne e sovrapposte; dopo l'unicità è possibile continuare.
Temperatura zero vale per struttura e old_text; il sampling richiesto torna
prima del primo token di contenuto di new_text. MTP può operare su old
globalmente univoco e new, con maschera aggiornata per ogni verifier row.
Il cambio di fase XML o indice dell'elemento interrompe la finestra e scarta
draft/RNG obsoleti. Il batch interattivo resta target-only.

Il completamento emette una normale function_call con la stringa completa,
senza code. Il client verifica il framing finale, poi interpreta le operazioni.
Budget e usage sono cumulativi; non sommare pause e risultato finale.

Un reject al secondo edit rende **non eseguibile l'intera chiamata**.
L'interruzione include call ID, exchange ID, edit_index, values e motivo.
Altre tool call già complete nello stesso turno possono essere preservate;
i figli della chiamata interrotta non diventano chiamate indipendenti.

Retry idempotenti, scope congelato, stream ereditato, cancel, status, scadenza
e ripristino KV seguono il lifecycle già esistente. Request ID riutilizzato
con payload diverso, scambi consumati e scope diverso producono 409.
Dopo un riavvio le continuazioni non sono riprendibili.
Replay proietta l'interruzione come record runtime non eseguibile, senza
inventare function_call o tool result orfani.
Anche il replay richiede record v2; configurazioni, continuazioni e record
di interruzione v1 sono rifiutati. Il passaggio server/client è coordinato.

## Responsabilità e limiti

**Server:** grammatica, masking, unicità della selezione letterale, combinazioni
dei comandi, chiusura finale, frontiere di sampling/MTP, congelamento
dei path, IDs, checkpoint, retry e limiti del protocollo.

**Client:** filesystem/workroot, path canonici, permessi, parent, inspect/read
nella history visibile, hash e modifiche concorrenti, intervalli start/end,
conflitti, hook, encoding/BOM, scritture/cancellazioni e risultati per file.
Il masking non sostituisce la validazione client prima di qualunque effetto.

Più edit sullo stesso file si riferiscono allo snapshot iniziale. Il client
rifiuta sovrapposizioni, inserimenti interni a sostituzioni e due inserimenti
nella stessa posizione. All/delete devono essere l'unica modifica del file.
Non è promessa una transazione filesystem con rollback tra file distinti.

Il server non può riconoscere dal solo snapshot che l'utente volesse delete:
una riscrittura valida resta un risultato generato valido, ma fallisce il
test applicativo della richiesta. Non convertire stringhe simili a comandi,
riparare chiusure o nascondere errori con fallback.

Capabilities: snapshot 1 MiB, generazione 1 MiB, valori risorsa 16 KiB,
16 tool, 64 campi, 64 edit per chiamata, 64 scambi per continuazione,
8 continuazioni attive; testo snapshot globale 8 MiB e budget conservativo
per indici/scratch 128 MiB. Snapshot ripetuti identici non sono addebitati
due volte. Checkpoint 8 GiB, pausa 300 s e ricevute terminali 600 s.
La prenotazione degli indici include anche il picco della costruzione.

La migrazione del client v1 resta necessaria prima dell'uso applicativo.
Le verifiche deterministiche non qualificano automaticamente modello reale,
prefill GPU, prestazioni o integrazione filesystem.

## Esecuzione dei test

`make test` esegue la suite C, i test del server e i gate disponibili del motore.
Per le fixture Python usare l'ambiente con tokenizers, Torch, Pillow e SDK:

```bash
exmekh_lib/.venv/bin/python -m tests.run_server_qualification --report /tmp/server-fixtures.json
```

Per una build v2 attiva con contesto 22593, eseguire la matrice HTTP completa:

```bash
DS4_INTERACTIVE_BASE_URL=http://127.0.0.1:18091 \
DS4_INTERACTIVE_REPORT=/tmp/edit-wire.json \
exmekh_lib/.venv/bin/python -m tests.run_server_qualification \
  tests.test_interactive_api --report /tmp/edit-live.json
```

Ripetere con target-only e MTP, avviando un solo processo modello alla volta.
Per MTP omettere `--batched-session`, anche il valore `1`: il batching nativo
disabilita la speculazione. Verificare i contatori MTP del run effettivo.
Con encoder vision impostare anche `DS4_INTERACTIVE_VISION=1`. Il report
conserva i fallimenti dei sottocasi; timeout e skip non diventano successi.
`DS4_INTERACTIVE_HTTP_TIMEOUT_SECONDS` configura il timeout HTTP (default 90 s).
Il confronto esaustivo diagnostico può richiedere un valore maggiore; mantenere
il primo timeout nel report e distinguere il tentativo diagnostico successivo.
Le suite client v1 restano distinte dalla matrice server v2 finché la migrazione
applicativa non è completata.

La [raccolta completa del 11 ottobre](../performance/interactive-edit-xml-v2-full-tests-20261011.md)
conserva lo storico della variante con code, fallimenti, skip e responsabilità.
Non qualifica automaticamente il nuovo formato senza code.
Le regressioni native verificano anche che un escape del trasporto non possa
proseguire da un opener XML, un'entità o uno scalar UTF-8 incompleto. Il confronto
maschera/oracolo controlla l'accordo tra i filtri; da solo non dimostra che ogni
prefisso ammesso abbia una prosecuzione completa.
