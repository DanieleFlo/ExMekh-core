# Tool file letterali v3 — 13 ottobre 2026

La migrazione separa creazione, modifica e cancellazione in ExMekh-Lib e nel
frontend ExMekh Agent. Il server applica il nuovo comportamento `edit_file.v3`
sul trasporto Responses v2. Il contratto è in
[interactive-calls.md](../agentic/interactive-calls.md).

## Contratto implementato

| Tool | Input | Comportamento |
| --- | --- | --- |
| `new-file` | `path`, `content` | Creazione esclusiva di un path assente, content obbligatorio di almeno due caratteri Unicode; nessun inspect/read |
| `delete-file` | `path` | Rimozione di un file regolare esistente; nessun inspect/read e nessun decoding del contenuto |
| `edit-file` | `edits` | Modifica di file esistenti autorizzati dalle evidenze inspect/read ancora visibili; nessuna creazione o cancellazione del file |

Edit usa soltanto due marcatori:

```text
<exmekh_edit path="notes.txt" id="427" />old text<exmekh_new_text id="427"/>new text
```

Il server impone un ID casuale di tre cifre, unico nella chiamata, dentro la
generazione effettiva e quindi nel contesto del modello. Il modello lo copia
nel separatore. L'omissione è accettata quando la selezione e il framing sono
univoci; gli snapshot possono disambiguare separatori concorrenti.
CDATA, entità e vecchi tag di selezione non hanno semantica nei corpi.
Spazi, CR/LF, Unicode e markup rimangono testo letterale.

La selezione conserva unicità globale e inizio a confine di riga fisica.
Un file esistente vuoto ammette old text vuoto. Per inserire si conserva
un riferimento esistente nella sostituzione; una sostituzione vuota mantiene
il file. Alias, encoding/BOM, hook, cancellazione, scope, controlli di identità
e cambiamenti concorrenti conservano il controllo lato client.
Non esiste un rollback transazionale fra file distinti.

## Distribuzione

La wheel `agent/wheels/exmekh_lib-0.1.0-py3-none-any.whl` è stata ricostruita
con `agent/sync_exmekh_lib.sh` e installata nell'ambiente dell'agente.
Gli schemi installati espongono i tre tool e `minLength: 2` per content.
Il server CUDA su porta 8080 è stato ricompilato e riavviato mantenendo modello,
vision, contesto e MTP. Non sono stati modificati i kernel di inferenza.
Le modifiche contemporanee dell'altra chat nella libreria sono state preservate.

## Verifiche e limiti osservati

| Verifica | Esito |
| --- | --- |
| Suite deterministica ExMekh-Lib | 2.635 passati, 55 saltati |
| Suite completa ExMekh Agent | 940 passati, 41 saltati, copertura 100% |
| Integrazione dopo l'ultima wheel | 39 passati |
| Contratti dei file e calcolo dopo gli import condivisi | 76 passati |
| Parser e runtime C | PASS, incluse quattro sintassi native, parser completo/incrementale e checkpoint |
| ASan/UBSan e lifetime della memoria | PASS |
| Prove HTTP con modello reale | 6 passate anche sul binario finale |
| Prove agentiche su file reali | 11 passate, 3 fallite su 14 |
| Ripetizione delle sette operazioni con richieste dirette | 4 passate, gli stessi 3 casi falliti |

I tre fallimenti live sono svuotamento dell'intero file, inserimento con CRLF
esatto e creazione di markup letterale. Nelle tracce iniziali il modello
seleziona soltanto parte del file, genera LF al posto di CRLF oppure trasforma
il markup in entità HTML. Con richieste dirette lo svuotamento genera anche
chiusure XML nel testo nuovo e supera il budget di 20 turni tentando correzioni.
Il runtime scrive il testo effettivamente ricevuto: queste chiamate non sono
state riparate e i test conservano gli stessi byte attesi.

Il masking garantisce una selezione valida nello snapshot autorizzato;
non garantisce che la selezione sia quella intesa dall'utente né che il testo
nuovo rispetti l'intento. La qualifica agentica resta quindi parziale.
Le prove reali riguardano Qwen3.8-27B Q4_K_S su CUDA/WSL; le altre sintassi
native sono verificate con fixture C, senza qualifica live di altri backend.
Tre cifre non eliminano ogni collisione possibile con marcatori completi.

I log di questa sessione sono conservati in `/tmp/ds4-v3-*`, in particolare
`client-live-initial.log`, `client-live-operations.log`, `agent-final-full.log`,
`http-live-current.log` e `asan-final.log` con il prefisso `ds4-v3-`.