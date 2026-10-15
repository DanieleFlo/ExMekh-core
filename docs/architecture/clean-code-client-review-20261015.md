# Revisione del client e dell'agente — 15 ottobre 2026

La revisione comprende le modifiche allo steering, la cancellazione multipla
del client, il bootstrap del wheel, i test live e la loro integrazione nel
frontend. Lo screening AST ha attraversato 117 moduli dell'agente e 255 della
libreria. I [risultati e gli hash](../performance/clean-code-client-review-20261015.json)
conservano il perimetro verificato. Il motore C e i kernel non cambiano.

## Responsabilità e comportamento

`ProviderResponse.history_messages()` possiede la costruzione della history
osservabile: risposta completata ma superata, ricevute dei tool non eseguiti
e risposta corrente. `Client` registra il risultato dell'API pubblica senza
conoscere il formato delle cancellazioni. Rimane la distinzione fra chiamate
eseguibili e evidenza conservata; `exit_agent` scartato non chiude una skill.
Anche il ripristino dopo reload usa la query
`RecordedInvocation.cancelled_before_execution`: la chiamata rimane nella raw
history ma non viene registrata come un'esecuzione mai avvenuta. La regressione
ha riprodotto il falso inserimento prima della correzione.

`FileDeletionTarget.resolve(...).remove()` possiede il vincolo del workspace,
il tipo della risorsa, il lock, il controllo dell'identità e la rimozione tramite
descrittore della directory. L'adattatore del tool mantiene l'intento delle
modifiche precedenti: `paths` non vuoto, riconoscimento automatico di file e
directory, rimozione ricorsiva, esiti ordinati e prosecuzione dopo un fallimento.
I successi precedenti non vengono annullati; un errore ricorsivo può lasciare
una directory parzialmente rimossa, come documentato dal contratto.

È stata riprodotta una cancellazione inattesa: un link simbolico esplicito verso
una risorsa interna eliminava il bersaglio. Ora quel percorso è rifiutato, come
già accadeva ai link esterni e agli alias del workroot. I link contenuti negli
alberi rimossi non vengono seguiti. Le regressioni coprono bersagli file e directory.

`ToolEvent.is_running` è ora usato anche per costruire il contesto sintetico.
Prima, `tool_start/status=completed` poteva inventare un risultato, mentre
`execution/status=error` poteva perderlo. Le otto combinazioni di stato e scope
nelle nuove regressioni verificano la stessa interpretazione usata dalle viste.

`package_paths` è il proprietario unico dei percorsi necessari a runtime e
bootstrap. `utils.resources.catalog_message` risolve i messaggi per entrambi;
il sync rimane avviabile senza NiceGUI, ExMekh-Lib o altre dipendenze applicative.
Build, installazione e orchestrazione conservano funzioni distinte.

La verifica live del frontend usa il nuovo `paths`, legge i risultati per
call ID attraverso `ChatMessage.output_for_call`, e controlla separatamente
le voci del batch. Un risultato parziale non viene trasformato in una cancellazione
riuscita di tutti i percorsi. Le prove distinguono gli argomenti emessi dal modello
dagli effetti del tool e dall'aderenza alla richiesta dell'utente.

## Principi controllati

| Principio | Evidenza e decisione |
| --- | --- |
| Responsabilità unica | Risposta, persistenza, risorsa filesystem, batch, cataloghi e visualizzazione hanno proprietari distinti. |
| Incapsulamento | Client e viste usano `history_messages`, `remove`, `is_running`, `output_for_call`; non interpretano lo stato privato di altre classi. |
| Coesione e nomi | I nuovi nomi identificano la risorsa o l'operazione; nessun contenitore astratto privo di comportamento. |
| Funzioni e argomenti | La rimozione è divisa fra risoluzione, controllo e operazione; il target raggruppa dati che viaggiano insieme. |
| Duplicazione | Un solo calcolo dei percorsi e un solo lettore dei cataloghi, mantenendo gli import pubblici usati dal prodotto. |
| Comandi e query | `history_messages` descrive la history; Client la registra; `resolve` legge e valida, `remove` modifica. |
| Errori e side effect | Nessuna falsa atomicità del batch; errori e successi rimangono associati ai rispettivi percorsi. |
| Null e compatibilità | La lista vuota delle evidenze è gestita dal client; le interruzioni reali non acquisiscono chiamate sintetiche. |
| Configurazione | Testi del tool e diagnostica restano nei cataloghi; i nomi di campo wire restano nel protocollo. |
| Semplicità | Nessun nuovo flag di comportamento, nessuna rinomina meccanica di metodi privati leciti. |
| Test | Bug riprodotti prima dei fix, suite complete, copertura di linee/rami, SDK reale e prove live locali. |

Lo screening dei richiami privati ha trovato soltanto accessi fra istanze della
stessa classe, richiami interni alla classe e l'API `_replace` dei named tuple.
Questi casi non sono stati trasformati artificialmente in metodi pubblici.

Restano funzioni lunghe nei costruttori UI, nel rendering di chat/history e nel
layout del grafo. Sono candidati alla decomposizione, non dimostrazioni automatiche
di responsabilità errate: separarli senza un problema verificabile aumenterebbe
il perimetro dei cambiamenti. La revisione non certifica che ogni funzione delle
due repository soddisfi ogni principio; registra gli invarianti controllati e i
problemi concreti corretti.

## Verifiche

Le suite complete e i controlli statici sono registrati nel JSON collegato.
Passano 2.928 test della libreria (63 skip opt-in) e 1.000 dell'agente (43 skip),
incluse le prove browser. Dopo l'ultima correzione al ripristino passano anche
40 test mirati e l'intera selezione non E2E: 922 pass, 33 skip. La copertura
finale di linee e rami è 100% in entrambe le repository; Ruff e Pyright passano.
Le prove live usano Qwen3.8-27B UD-Q4_K_S, CUDA RTX 3090, MTP e contesto 26.000:
due steering consecutivi in JSON e SSE mantengono 2.222/2.483 token in cache,
con suffissi nuovi da 159 token. La rimozione singola e il batch con file binario
e albero di directory passano sul client reale in workspace temporanei.

La wheel è ricostruita e reinstallata dal sync di produzione. Il server temporaneo
viene terminato dopo le prove. Il report dello steering precedente conserva le
misure originali; questa revisione documenta l'API e la qualifica successive.
