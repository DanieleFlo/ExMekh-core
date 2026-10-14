# Chiamate interattive: edit letterali v3

Il comportamento attuale è `edit_file.v3`; il trasporto Responses resta alla
versione 2. Server e client negoziano esplicitamente il comportamento: v1 e v2
non sono fallback. Il server genera gli argomenti e non accede al filesystem
del client. Creazione e cancellazione sono tool ordinari separati.

## Tool e responsabilità

| Tool | Argomenti | Requisiti |
| --- | --- | --- |
| new-file | path, content | Path assente, padre esistente nel workroot; content obbligatorio di almeno due caratteri Unicode; nessun inspect/read |
| delete-file | path | File regolare esistente nel workroot; nessun inspect/read o decoding del contenuto |
| edit-file | edits | File di testo esistenti, inspect e read successivi dello stesso contenuto e ancora visibili |

Il client conserva i controlli sul workroot, sui permessi, sulle identità dei
file, sugli alias e sui cambiamenti concorrenti. La creazione è esclusiva;
la cancellazione usa metadata e descriptor di directory e non ha il limite
di decoding degli snapshot di testo.

## Configurazione

```json
{
  "version": 2,
  "operation": "configure",
  "request_id": "turn-1",
  "tools": [{
    "name": "edit-file",
    "behavior": "edit_file.v3",
    "fields": {"document": "edits"}
  }]
}
```

La configurazione vive in `agentic.interactive`. Si usa
`GET /v1/agentic/capabilities` per verificarne il supporto e
`POST /v1/responses` per l'inferenza. Il campo documento è una sola stringa
obbligatoria; il nome del tool e quello del campo possono essere rinominati.
Le forme finite enum/const non sono compatibili con il vincolo dinamico.
Il modello, lo scope e la sintassi devono essere supportati.

## Due marcatori e corpi letterali

```text
<exmekh_edit path="notes.txt" id="427" />vecchio testo<exmekh_new_text id="427"/>nuovo testo
```

Il modello genera il path. Il server si ferma dopo l'apice finale del path,
richiede lo snapshot autorizzato al client e assegna l'ID. Il suffisso
` id="427" />` è imposto dal masking: entra nella generazione e nel contesto
effettivo del modello, senza riparazione successiva degli argomenti.
Il modello può ricopiare l'ID nel separatore o ometterlo.

Gli ID sono tre cifre ASCII, da 000 a 999, senza ripetizioni nella chiamata.
Sono stabili nelle riprese e nei retry. Il server evita ID già presenti come
attributi nello snapshot; se non esiste un ID disponibile rifiuta lo scambio.
L'unicità è locale alla chiamata, non globale.

I corpi preservano esattamente Unicode, spazi, indentazione, CR, LF e CRLF.
Non sono XML: CDATA, entità, vecchi tag all/line/delete e chiusure XML sono
testo letterale. Solo il path mantiene l'escaping delle entità di un attributo.
Gli escape JSON e quelli dei delimitatori nativi del modello restano strati
di trasporto, rimossi una volta.

Il testo nuovo termina con la stringa oppure con il blocco successivo.
Non ci sono chiusure del corpo o tag edit finali. Il whitespace prima di un
blocco successivo appartiene al testo nuovo precedente.

## Selezione e ID omessi

La selezione letterale deve partire da un confine di riga fisica, comprendere
l'indentazione e comparire una sola volta globalmente, contando anche
occorrenze sovrapposte e nel mezzo delle righe. Il masking server conserva
questo vincolo durante la generazione; il client lo verifica nuovamente.
Soltanto un file esistente vuoto permette una selezione vuota.

Per inserire testo si conserva un riferimento esistente nel testo nuovo e si
aggiunge contenuto prima o dopo. Una sostituzione vuota elimina la selezione
ma mantiene il file. Non esistono comandi per righe, intervalli, all,
creazione o cancellazione dentro edit.

Il client accetta i marcatori senza ID quando esiste una sola interpretazione.
Un separatore con l'ID esplicito corrispondente prevale sui marcatori senza ID
letterali. In assenza di una scelta univoca, gli snapshot autorizzati
disambiguano le possibili selezioni; se resta più di una scelta, errore prima
di scrivere. Un ID di separatore diverso non chiude la selezione.
La presenza di tre cifre non costituisce una garanzia contro qualsiasi
collisione con testo che riproduce i marcatori completi del protocollo.

## Autorizzazione, applicazione e trasporto

Ogni path richiede uno scambio `file_snapshot`, con call ID, exchange ID
e indice consecutivo. Il client risponde provide oppure reject; create non
è una risposta valida per edit. Rifiutare una risorsa invalida l'intera
chiamata, anche se contiene elementi già generati.

Il documento finale viene verificato dopo gli hook. I path devono corrispondere
nello stesso ordine agli scambi autorizzati. Alias e modifiche dello stesso
file condividono lo snapshot iniziale; selezioni sovrapposte sono rifiutate.
Tutte le risorse vengono pianificate e verificate prima della prima scrittura
e ogni file canonico è pubblicato una volta.

Restano encoding/BOM, lock cooperativi, controllo delle evidenze visibili,
cancellazione, scope, commit atomico per file e rendicontazione del progresso
parziale. Non è garantito il rollback fra file distinti.
Il runtime filesystem qualificato è Linux/WSL.

Il server preserva checkpoint, KV, retry idempotenti e receipts, scope e
limiti bounded. Il campionamento del testo vecchio resta greedy e vincolato;
il testo nuovo usa la policy richiesta. Il parser incrementale conserva
solo frontiere accettate e viene confrontato con il parser completo;
MTP e draft del suffisso dello snapshot non attraversano i cambi di fase.