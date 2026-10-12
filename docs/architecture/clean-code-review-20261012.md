# Revisione del server XML e dei contratti tool

Data: 12 ottobre 2026. Ambito: modifiche della chat, decoder interattivo,
estrazione delle chiamate, scheduling MTP, lifecycle delle continuazioni e
confini collegati di ExMekh-Lib e ExMekh Agent.

## Responsabilità e interfacce

| Componente | Responsabilità e contratto |
| --- | --- |
| `ds4_edit_constraint.h` | Possiede testo e indice delle selezioni. Espone conteggio delle righe, matching, candidati e stima della memoria; il parser non legge bucket, suffix array o offset interni. |
| `ds4_edit_document.h` | Possiede percorsi autorizzati e snapshot unici. Gli elementi prendono in prestito gli snapshot; il documento li libera una sola volta. Gestisce riuso e rifiuto delle risposte contraddittorie. |
| `ds4_interactive_edit.h` | Traduce configurazione e risposte JSON nelle operazioni del documento. Definisce descrizione e policy di sampling del comportamento edit. |
| `ds4_interactive_config.h` / `ds4_interactive_grammar.h` | Validano configurazione, stringhe UTF-8, schema e chiamate complete. Prefix e completamento usano lo stesso validatore; la decodifica della stringa non viene ripetuta nella stessa verifica. |
| `ds4_interactive_runtime.h` | Possiede lifecycle, ricevute, budget, pause e riprese. Consuma i callback del comportamento senza conoscere gli indici dello snapshot. |
| Scheduler del server | Consuma una policy con temperatura, possibilità di speculare e identità opaca della frontiera. Non interpreta i bit della fase XML. |
| Client | Possiede workroot, evidenza inspect/read, identità e hash del file, range, alias, conflitti, hook e scritture. Il server riceve solo lo snapshot autorizzato. |

I quattro callback di sampling sono riuniti in un'unica policy. La fase e
l'indice dell'edit formano un'epoca opaca: il verifier deve cambiare frontiera
quando cambia la policy, mentre una chiusura ancora ambigua mantiene la
temperatura del replacement e sospende la speculazione.

Il prefill specifico dell'edit e il relativo callback sono eliminati. Rimane
il contatore generale `constrained_prefill_tokens`, utilizzato dal decoder
ordinario per i token forzati; il formato edit senza code non li inietta.

## Coerenza dei comportamenti

Il formato predefinito è v2 senza code: pausa dopo il path, autorizzazione,
poi generazione di `>` e del contenuto. La chiusura finale viene riconosciuta
per ciascun edit. Le chiusure precedenti sono testo; la coppia completa
letterale richiede CDATA o entità. Non ci sono flag per scegliere una variante.

Maschera ed estrazione finale condividono il riconoscitore. Reply malformate
o snapshot ripetuti contraddittori lasciano l'exchange in attesa. I test
verificano riuso e invarianti attraverso le API, senza cambiare gli internals
del documento. Gli spazi del framing usano un solo helper, mentre i byte del
contenuto restano preservati.

Sono stati riletti anche esecuzione dei tool, presentazione e persistenza dei
risultati, contesto di ispezione, trasporto interattivo e integrazione del
frontend. Questi percorsi usano le operazioni pubbliche di `CalledTool`,
`ExecutionSnapshot`, `FileToolRuntime` e `ToolResult`. Le differenze fra output
attivo, salvato e preformattato mantengono i contratti già documentati.
Gli esempi Python di sei guide sono uniformati al formatter del progetto.

I test del frontend hanno trovato un provider finto lasciato installato da un
test precedente: le prove di steering successive non raggiungevano il provider
previsto. Il supporto ai test ora espone `using_mock_provider`, che ripristina
il contesto precedente anche dopo errori e contesti annidati. Test e fixture
usano questa operazione; la regressione esegue installazione e recupero dello
steering nello stesso processo.

Il live ha trovato anche un difetto del parser: `aa<![CDATA[]]>]]` era ammesso
come prefisso old_text su uno snapshot la cui continuazione era `aaa`.
La gestione del terminatore CDATA lasciava i `]` pendenti fuori dal matching.
Il parser ora li include come byte letterali prima di interrogare lo snapshot;
le regressioni verificano sia il rifiuto sia un vero snapshot che contiene `]]`.

La migrazione del client v1 è ancora il task distinto in `exmekh_lib/todo.md`.
La sintassi valida non garantisce che il contenuto generato realizzi l'intento
dell'utente: i fallimenti live su delete e CDATA vanno conservati e analizzati,
senza convertirli in operazioni filesystem differenti.

## Verifica

Gli esiti e i limiti delle verifiche sono conservati nel
[report della revisione](../performance/interactive-edit-clean-review-20261012.json).
I report del 11 ottobre e della precedente prova senza code rimangono evidenze
storiche, con gli hash e i fallimenti delle rispettive esecuzioni.

Questa revisione applica responsabilità singola, interfacce ristrette,
encapsulamento, eliminazione delle duplicazioni e verifiche osservabili ai
percorsi esaminati. Non costituisce una certificazione automatica di ogni
modulo o backend del repository; le modifiche ai kernel richiedono i gate
numerici e prestazionali specifici di ciascuna piattaforma.
