# Revisione del budget di compattazione — 16 ottobre 2026

La revisione copre le modifiche al budget e i confini collegati fra libreria,
provider, cancellazione, conteggio del contesto e policy automatica. La ricerca
degli accessi privati fra oggetti è stata estesa ai sorgenti Python delle due
repository; gli accessi rimasti riguardano principalmente oggetti della stessa
classe, estensioni tramite superclassi e API standard come `_replace`.

## Difetti e correzioni

- Il wrapper di cancellazione ricostruiva le richieste dalla conversazione già
  espansa: gli allegati dei risultati dei tool comparivano due volte. Due test
  hanno riprodotto il difetto prima della correzione, sia per richieste normali
  sia strutturate. L'adattatore usa ora `ProviderRequest.with_event_handler()`.
- La stima del budget contava lo schema strutturato separatamente dalla UI e
  ignorava l'override del thinking. Entrambi i consumatori usano ora la stessa
  stima del prompt completo. Un checkpoint skill viene riusato soltanto con
  modalità thinking compatibile e richiesta senza schema strutturato.
- Le risposte strutturate non aggiornavano i totali autorevoli ricordati per
  la policy. Seguono ora lo stesso percorso delle risposte normali.
- Il riarmo basato sull'identità degli oggetti confondeva il ricaricamento di
  messaggi equivalenti con nuove informazioni e ignorava modifiche sullo stesso
  oggetto. Il confronto usa snapshot del contenuto serializzato.
- La suddivisione dei riassunti non deve assumere che il costo del prompt sia
  additivo. Il budget verifica ogni frammento nella richiesta completa e ferma
  esplicitamente un input irriducibile. Un test copre costi di wrapping non lineari.

## Proprietari delle responsabilità

| Classe | Responsabilità |
| --- | --- |
| `RequestBudget` | Capienza, riserva di generazione, ammissione e suddivisione del testo in richieste che entrano nel budget. |
| `TextSummarizer` | Inferenza isolata dei riassunti, combinazione e verifica del progresso. |
| `ToolOutputPreparer` | Materializzazione su file e scelta di una rappresentazione realmente più breve. |
| `OutputCandidate` | Aggiornamento della presentazione conservando corpo originale e recuperabilità. |
| `CompactionContext` | Copia temporanea dei messaggi e sostituzione progressiva dei candidati eleggibili. |
| `CompactionScorer` | Lotti distinti, schema dei voti e validazione delle risposte. |
| `ProviderRequest` | Adattamento degli eventi preservando sottotipo, conversazione espansa e opzioni. |
| `ContextTrackingProvider` / `Ds4ContextEstimator` | Osservazione, totali del provider e stima coerente del prompt effettivo. |
| `AutoCompactionPolicy` / `NodeRuntime` | Rispettivamente soglia e riarmo applicativi; esecuzione, applicazione e verifica finale prima dell'inferenza. |

I chiamanti usano le API pubbliche invece di ricostruire dettagli dei dati di
un'altra classe. Non sono state aggiunte gerarchie o interfacce senza un
consumatore concreto. Errori, prompt e testo del prodotto restano nelle risorse
e nella configurazione del rispettivo dominio.

## Verifica e limiti

Le suite complete delle due repository, i test browser, lint, tipi della
libreria, gate dei testi e prova CUDA live sono registrati nel
[report di verifica](../performance/compaction-budget-clean-review-20261016.json).
La prova live parte da una history sintetica oltre capienza, verifica
l'esportazione prima dello scoring e riprende mantenendo il vincolo CP-7.

Responsabilità, incapsulamento, nomenclatura, duplicazione, gestione degli errori
e testabilità sono stati controllati nei percorsi letti. Questo non certifica
ogni principio per ogni modulo dell'intera repository. Le grandi orchestrazioni
UI e il server C richiederebbero revisioni dedicate; qui non sono state riscritte.
Kernel e protocolli server non sono cambiati; Metal, SSD streaming e inferenza
distribuita non sono stati riprovati. Input protetti più grandi della capienza
restano un errore esplicito dopo il tentativo di compattazione.
