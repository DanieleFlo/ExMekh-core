# Revisione del codice e qualifica live — 14 ottobre 2026

La revisione comprende server, libreria e agente. Sono stati riletti i diff e
le responsabilità dei percorsi file, protocollo, checkpoint, output, storia,
configurazione e sincronizzazione del wheel. Lo screening Python comprende
114 moduli dell’agente e 254 della libreria. I risultati e gli hash sono nel
[report verificabile](../performance/clean-code-live-review-20261014.json).

## Responsabilità, incapsulamento e argomenti

`FileWriteTarget` raggruppa workspace, risorsa, path canonico e snapshot.
Espone `validate()` e `commit(content)` e possiede il locking e la pubblicazione
atomica. Sostituisce funzioni con quattro/cinque parametri ripetuti; il servizio
edit usa i suoi metodi. `EditApproval` continua a possedere evidenze, selezione
ed encoding; `EditDocument` possiede la grammatica, senza accedere al filesystem.
`FileInteractiveCalls` conserva lo stato delle autorizzazioni e dello scope:
è stato eliminato soltanto il registro di dispatch con un unico handler.

`ds4_interactive_checkpoint.h` possiede cattura, staging, budget, rilascio e
pubblicazione del rollback; il runtime del protocollo richiede queste operazioni
senza interpretare i payload del modello. `interactive_checkpoint_operation`
riunisce server, slot, entry e buffer di errore per le operazioni del worker.
`reserved_bytes` rende esplicito il conteggio del budget anche quando un payload
è temporaneamente staccato dalla lista dei checkpoint.

La build del wheel, la sua installazione e la loro orchestrazione sono tre
funzioni distinte. La struttura dei percorsi viene dal config, senza risalire
separatamente il file del modulo. I test sostituiscono il percorso applicativo
configurato e restano confinati ai propri workspace temporanei.

Lo screening ha trovato 63 funzioni dell’agente e 101 della libreria con più di
tre argomenti, esclusi `self` e `cls`. Sono candidati, non violazioni automatiche:
API pubbliche, callback e adattatori conservano i contratti compatibili. I valori
che rappresentano una stessa richiesta o destinazione sono raggruppati quando
questo riduce la duplicazione e consente metodi del dominio. Gli accessi privati
individuati appartengono a istanze della stessa classe o all’API `_replace` delle
namedtuple; non è stata applicata una rinomina meccanica.

## Principi controllati

| Principio | Esito e perimetro |
| --- | --- |
| Responsabilità unica | Scrittura, evidenze, protocollo, checkpoint e packaging hanno proprietari espliciti. |
| Incapsulamento | I consumatori del target usano `validate()`/`commit()`; le approvazioni espongono metodi per alias, selezione e verifica. |
| Nomi e leggibilità | Il destinatario della scrittura e i byte riservati indicano il concetto; le operazioni del checkpoint sono isolate dal lifecycle HTTP. |
| Funzioni e argomenti | Raggruppati dati che viaggiano insieme; callback e compatibilità non sono state trasformate in contenitori senza responsabilità. |
| Duplicazione | Versione wire, endpoint, politica dei temporanei e messaggi condivisi hanno un solo proprietario. |
| Comandi e query | La validazione legge, il commit scrive; cattura, pubblicazione e restore hanno nomi e side effect espliciti. I tool restituiscono ricevute delle proprie operazioni. |
| Errori e coerenza | Fallimenti restano espliciti; commit parziali elencano le scritture riuscite e consegne fallite rimangono ritentabili. |
| Classi e astrazioni | Mantenuti gli oggetti che possiedono stato o comportamento; tolto un registro con una sola voce e sostituite le funzioni che ripetevano i dati del target. |
| Costanti e testo | Centralizzati i valori del perimetro modificato; screening dei messaggi di eccezione Python senza nuove esclusioni. |
| Concorrenza e ownership | Il rollback pubblicato non può essere sostituito dall’evizione o eliminato da finalizzazioni ripetute; il worker rimane proprietario della sessione. |
| Test e mantenibilità | Regressioni di errore, copertura dei rami, API reali e frontend; le limitazioni di piattaforma sono distinte dalle prove passate. |

## Configurazione e testo

`config/interactive.py` è il proprietario della versione wire, del tipo di
snapshot e degli endpoint condivisi da negoziazione, validazione e trasporto.
Il formato edit e le sue regole restano in `config/file_editing.py`, che ora
contiene anche prefisso e permessi dei file temporanei.

`ds4_agentic_settings.h` contiene istruzioni del runtime, ritorno dalle skill e
diagnostica dei checkpoint. Lo spostamento delle istruzioni conserva i byte del
prompt. `config/packaging.py` contiene politica e percorsi del wheel; i messaggi
CLI sono nei cataloghi `configuration` ed `errors_exmekh_agent`.

Non sono state trovate eccezioni con messaggi letterali fuori dai config e dalle
risorse delle capability nel perimetro Python analizzato. I nomi dei campi JSON,
le operazioni e gli identificatori del protocollo restano parte dei rispettivi
adattatori. I payload letterali nei test sono fixture, non testo del prodotto.
La centralizzazione non costituisce una dichiarazione che ogni messaggio del
server C e degli altri componenti storici sia già stato migrato.

## Difetto aggiuntivo trovato dai test live

Dopo cancel, il checkpoint dell’ultimo turno completato era pubblicato, mentre
la sessione residente conteneva ancora la pausa interrotta. Una richiesta
ausiliaria poteva salvare quella sessione e sostituire il rollback corretto.
Anche una finalizzazione ripetuta poteva eliminare il checkpoint già pubblicato.

Il checkpoint ora indica quando il ripristino è pendente: un salvataggio per
evizione lo conserva, ricontrollando anche dopo la serializzazione per gestire
una pubblicazione concorrente. Il worker toglie questa condizione soltanto dopo
aver caricato il payload. La pubblicazione del rollback è idempotente. La thread
di controllo continua a trasferire ownership senza modificare la sessione GPU.

La nuova regressione riproduce il problema sul vecchio processo: 596 token
riusati, 2169 rivalutati. Sul binario corretto: 2627 riusati e 329 rivalutati.
Con storia lunga, reasoning e SSE, la stessa regressione conserva 12961 token e
ne valuta 324. Le richieste sintetiche non eseguono tool né scrivono file utente.

## Verifiche e distribuzione

Passano 2767 test della libreria, 927 test paralleli dell’agente e 3 test UI
esclusivi. I gate di copertura dichiarati dai due progetti raggiungono il 100%
delle linee e dei rami; le esclusioni preesistenti dell’agente sono mantenute.
Sono stati aggiunti casi di errore di filesystem, commit parziale, cancellazione,
protocollo, metadati degli output, esportazione e consegna ritentabile, senza
abbassare la soglia di copertura né rimuovere regressioni preesistenti.

Passano Ruff, Pyright della libreria, compilazione CUDA, test nativi del server,
parser interattivo, lifetime host e AddressSanitizer/UndefinedBehaviorSanitizer.
Il wheel è ricostruito e reinstallato tramite il launcher reale; tutti i suoi
254 moduli Python sono stati confrontati byte per byte con i sorgenti.

Passano tutti i 18 scenari live della cache, i 6 test HTTP live dell’edit con
3 controlli del parser e i 5 test del frontend reale. Questi ultimi verificano
creazione, inserimento in un file vuoto, append e delete senza inspect/read.
I log completi sono in `performance-results/clean-code-live-review-20261014/`.
Il server temporaneo è stato terminato e la porta 8080 risulta libera.

## Limiti e lavoro strutturale restante

Il server C, alcuni moduli di inferenza e alcune funzioni di orchestrazione UI
restano grandi. Una loro decomposizione ulteriore richiede confini verificabili
e prove dedicate: questa revisione non certifica ogni classe dell’intera
repository né ogni principio clean code come definitivamente soddisfatto.
La qualifica live riguarda Qwen3.8-27B, CUDA/RTX 3090, MTP, contesto 26000 e
encoder vision caricato. Metal, SSD streaming, distribuito e altri modelli non
sono qualificati da queste prove; i kernel non sono stati modificati.
La cancellazione verificata è una pausa `file_snapshot`, non un fault GPU o un
prefill interrotto. L’adattatore dei lease Windows è verificato con un’API di
locking simulata; le operazioni reali sui file sono state provate su Linux/WSL.


## Verifica successiva della collocazione dei testi

Il controllo è stato esteso oltre le sole eccezioni: restavano istruzioni del
modello, descrizioni degli schemi, ricostruzioni della storia, export, tooltip e
log inline. Sono ora nei config del rispettivo dominio. Il server usa i template
completi di `ds4_prompt_settings.h`, la diagnostica di `ds4_server_text.h` e i
config agentico/interattivo già presenti. L’agente usa i cataloghi dei testi.

Il [report della migrazione](../performance/text-configuration-20261014.json)
conserva la verifica dei token C preprocessati: identici nei percorsi CPU e GPU,
compresi stringhe, placeholder e istruzioni. Passano i 930 test dell’agente con
copertura configurata al 100%, i test nativi e il nuovo gate architetturale.
La regola è esplicita in `AGENT.md`; `make test` esegue anche il controllo contro
la reintroduzione di testo inline. Questa migrazione non ha avviato un modello:
la qualifica live precedente resta un risultato distinto.
