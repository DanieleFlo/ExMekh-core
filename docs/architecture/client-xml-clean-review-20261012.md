# Revisione del client XML e delle responsabilità — 12 ottobre 2026

La revisione riguarda le modifiche della chat, i compositori dei tool, il runtime
dei file, i provider, il server C e l'integrazione nel frontend. Le correzioni
concrete sono nel client ExMekh-Lib e nella sua distribuzione predefinita.

## Responsabilità e refactoring

| Componente | Responsabilità verificata o corretta |
| --- | --- |
| Server C | Grammatica e masking XML, sampling/MTP, pause, ricevute e KV; gli adattatori usano le API del documento e dei vincoli |
| `Tool` | Possiede la copia della politica di esecuzione; i compositori chiamano `with_execution_policy_of` per mantenere contratto interattivo e vincoli sincroni degli override |
| `SourceLines` | Possiede offset, conteggio delle righe fisiche e punto di inserimento a EOF; le selezioni usano questa API invece di una seconda lista di offset |
| `InteractiveContract` | Congela nome, percorsi ordinati e ultimo exchange in un record immutabile per chiamata, con metodi di confronto; elimina tuple posizionali e dizionari paralleli |
| `FileInteractiveCalls` | Rileva il cambio di scope anche quando la history resta identica e chiede la cancellazione prima di preparare una risposta alla pausa |
| `EditApproval` | Verifica le evidenze inspect/read del proprio snapshot |
| Edit preparato | Possiede preflight di tutti gli alias, commit e risultato; l'orchestratore mantiene ordine dei lock, verifica globale e gestione degli errori parziali |
| `InteractiveTransport` | Classifica JSON o encoding malformati come errori di protocollo in HTTP e SSE, chiude la risposta e non ritenta l'inferenza come per un guasto di rete |

Il bug sullo scope si manifestava quando una skill veniva chiusa e riaperta
durante una pausa senza cambiare i messaggi. La scrittura aveva già un controllo
tardivo, ma la risposta alla pausa poteva ancora essere preparata. Ora il provider
usa il controllo di cancellazione del runtime prima di inviare lo snapshot.
Le regressioni verificano questo comportamento con e senza streaming.

Il preflight di tutti i file precede la prima scrittura. Le evidenze e lo stato
di cancellazione vengono controllati nuovamente fra i commit. Gli alias condividono
lo snapshot iniziale e ogni file canonico riceve un solo commit. I risultati
parziali dichiarano i file già applicati: non viene promesso un rollback tra file.

## Criteri di leggibilità

Le operazioni di copia mantengono sottoclasse e metadati senza modificare il tool
fornito dal chiamante. Il codice di composizione usa l'API del tool invece di
assegnarne la politica. L'orchestrazione degli edit usa operazioni del dominio,
mentre il servizio filesystem conserva identità, permessi, encoding/BOM e
pubblicazione. I record immutabili rendono espliciti i dati congelati e impediscono
che una pausa con exchange duplicato modifichi le autorizzazioni precedenti.

Le righe fisiche hanno un'API comune: CR/LF delimitano le posizioni, mentre
separatori Unicode di visualizzazione restano contenuto. I test verificano file
vuoti, newline finali, CR/LF misti, inserimento a EOF e intervalli. Queste regole
descrivono posizioni nel file; non aggiungono limitazioni al testo generato.

Gli errori del protocollo restano distinti dai guasti di rete e dagli errori
applicativi dei tool. La serializzazione XML dei risultati continua a essere
responsabilità del confine di presentazione; filesystem, cache e argomenti del
provider mantengono le proprie rappresentazioni strutturate.

## Verifiche

La suite completa ExMekh-Lib passa **2.611 test**, con **55 SKIP live opt-in**
e copertura **100%** di statement e branch. Ruff verifica **412 file**; Pyright
riporta **0 errori e 0 avvisi**. Le regressioni mirate iniziali passano **645 test**.
Le fixture native del server interattivo passano.

Il frontend con la wheel aggiornata passa **938 test**, inclusi Playwright,
con **40 SKIP live opt-in** e copertura **100%**. Tutti i **238 file** distribuiti
nella wheel coincidono con i sorgenti e con l'installazione predefinita.

La suite Python del server esegue **275 test**, con **214 PASS e 61 SKIP**;
le fixture native interattive passano. `make docs-check` passa **10 test** e
indicizza **66 documenti**. I primi avvii della suite Python hanno evidenziato
dipendenze mancanti nel Python di sistema, il lock del server live e indici
documentali ancora da rigenerare. La prova finale usa l'ambiente completo,
il runner come modulo dalla radice e i server live già fermati.

| Prova live | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: |
| Target-only, tutti i 15 casi XML e lettura testuale | 13 | 2 | 0 |
| MTP: creazione, alias, rifiuto e cambio scope, con e senza streaming | 6 | 0 | 0 |

Passano le due nuove regressioni sul cambio di scope sia target-only sia MTP.
La telemetria MTP registra **15 cicli e 5 token draft accettati**. I due
fallimenti target-only esauriscono entrambi il limite di 20 turni: delete
genera un tag letterale in CDATA con newline, mentre lo svuotamento genera
un LF. Il file contiene rispettivamente `\n<delete/>\n` e `\n`.
Questi documenti XML validi vengono applicati letteralmente dal client.

Il [report JSON](../performance/client-xml-clean-review-20261012.json)
conserva hash dei sorgenti, wheel e risultati. Le evidenze
[target-only](../performance/client-xml-clean-review-20261012-target-wire.json) e
[MTP](../performance/client-xml-clean-review-20261012-mtp-wire.json) conservano
argomenti, scambi, cancellazioni e byte finali. I report precedenti restano
disponibili nella [qualifica della migrazione](client-xml-edit-v2-20261012.md).

La suite frontend è stata eseguita nel workspace corrente, che contiene anche
modifiche UI preesistenti estranee a questa revisione. La nuova wheel è
installata per default; i server di test sono stati fermati, uno alla volta.

## Contratto e limiti

Il protocollo resta XML v2 senza attributo `code`. I marcatori precedenti
restano abbandonati. L'editing preserva i byte richiesti e distingue i comandi
XML dal contenuto escapato o in CDATA.

Il masking non può garantire l'intento dell'utente quando il documento generato
è valido ma sceglie un'operazione diversa. Un `<delete/>` dentro CDATA è testo;
interpretarlo come comando renderebbe impossibile scrivere quel contenuto
letterale. Le prove live conservano assert su file e byte, oltre agli argomenti
effettivamente generati, per distinguere questi errori dai difetti del protocollo.

La qualifica usa Qwen27 CUDA su RTX 3090, contesto 22593, un modello grande alla
volta. Non riqualifica altri backend o i percorsi vision.
