# Revisione clean code di T11/T12

21 ottobre 2026. Riesame della pipeline image, adapter Gemma, tokenizer,
executor, primitive numeriche, tool di qualifica e test. Controllati anche i
confini adiacenti di runtime, registro risorse, gruppi pesi e build legacy.
Il lavoro applica i principi alle responsabilità effettive dei moduli C;
non certifica ogni componente dell'intera repository.

## Correzioni e responsabilità

| Problema | Correzione | Proprietario |
| --- | --- | --- |
| Il costruttore del job copiava reference, prenotava staging, preparava conditioning, costruiva varianti e metadati in un unico flusso | Funzioni distinte per copia, preparazione, piano, executor e report; un solo percorso di cleanup del resize | Pipeline per orchestrazione; ogni fase possiede i propri temporanei |
| La pipeline conosceva dimensioni dei buffer interni del resize e dell'encoder VAE | Query di workspace nel resize e query dei transitori nel VAE; nessun accesso alle strutture opache | Modulo che effettua le allocazioni |
| Costruzione dei callback dell'executor duplicata nel fallback delle cache | Un solo helper di creazione dell'executor | Job image |
| Un contatore poteva suggerire più job simultanei, mentre il contratto ammette un solo job | Stato booleano esplicito; BUSY fino alla chiusura del job | Pipeline |
| ID numerici di fase e componente mescolati con bitmask e policy delle varianti | Enumerazioni, helper per bit e nomi per cache host/residenza | Planner image |
| Un ID di fase invalido poteva indicizzare il report fuori dai limiti nel cleanup | Validazione prima di qualsiasi accesso; rimosso un ramo di errore irraggiungibile | Dispatcher delle fasi |
| Alcuni costruttori lasciavano un puntatore precedente nell'output dopo un errore di validazione | Output NULL prima della validazione; tokenizer azzerano separatamente ogni uscita fornita; Gemma pubblica l'input solo dopo il cleanup dello staging | API che trasferisce ownership |
| La query del piano dopo un run lo ricalcolava su un inventario di cache ormai cambiato; un secondo run poteva sovrascrivere i tempi | Snapshot del piano dell'executor dopo l'avvio e rifiuto immediato di un secondo run | Job per il contratto pubblico; executor per lo stato e il piano effettivo |
| Il codice dell'executor duplicava la transizione verso RECOVERABLE/UNAVAILABLE | Un solo helper di fallimento del cleanup; indentazione uniforme | Executor comune |
| Il forward dell'encoder ritrovava per nome due tabelle mmap già validate dal costruttore | Indici registrati durante il binding; forward usa l'invariante del proprio adapter | Encoder, senza esporre layout ai chiamanti |
| Il probe riconosceva BF16 tramite il numero grezzo 30 | Uso dell'identificativo pubblico DS4_NUM_BF16 | Probe tramite API numerica |

## Contratti conservati

Runtime ed executor possiedono serializzazione, admission, recupero e commit.
Gli adapter possiedono pesi e grafo; il registro possiede accounting e lifetime.
La pipeline usa soltanto operazioni e query esposte dai componenti. I tipi
pubblici di input, conditioning e risultato restano viste esplicitamente
documentate, senza accesso alle strutture private.

Il refactoring conserva ordine e aritmetica dei forward, sigmas, numero di step,
RNG, dtype e politica dei kernel qualificati. I bound device delle fasi sono
invariati. Il resize senza cambi di geometria ora dichiara workspace temporaneo
zero, coerente con il suo percorso di sola copia; le copie persistenti restano
contabilizzate. Non sono introdotti flag numerici alternativi.

## Verifica

Il [report T11/T12](../performance/native-suite-t11-t12-20261021/README.md)
conserva la qualifica originale e la regressione successiva al refactoring,
con hash di sorgenti, binari, log e output. `tests/test_suite_api.py` verifica
il comportamento su puntatori inizialmente non NULL e uscite mancanti dei
tokenizer. Il probe image verifica snapshot del piano e secondo run rifiutato,
oltre a cancellazione, admission, restore fallito, recupero e lifetime.

La regressione riesegue 54 testi Gemma e otto richieste image, confrontando i
byte degli output con gli artefatti qualificati precedenti. Copre anche la
cache K/V host con budget controllato, i tre percorsi executor del LLM Qwen27
e ASan/UBSan sulla matrice host dell'executor. Tutti passano; inventario e hash
sono nel report collegato. Non è una nuova qualifica completa dei 56 casi T00.

L'analisi GCC dei sette moduli C riesaminati non segnala problemi in pipeline,
Gemma, tokenizer, executor, encoder o VAE. Nel resize restano due diagnostiche
`-Wanalyzer-imprecise-fp-arithmetic`: riguardano il numero intero di tap derivato
da `ceil(support)`. Le dimensioni sono limitate, il numero di tap arrotondato
verso l'alto e il percorso è confrontato pixel per pixel con Pillow. Le
diagnostiche sono conservate, senza soppressioni né dichiarazione di zero warning.

## Debito e limiti

- I bound di attivazioni dei modelli restano conservativi. Una futura query
  completa di workspace nel backend numerico potrebbe eliminare la duplicazione
  della soglia della geometria BF16 nei planner, con test di admission per
  entrambe le geometrie. Nessun cambio di kernel è necessario per questa revisione.
- I forward descrivono molte operazioni per layer. La lunghezza da sola non
  giustifica astrarre equazioni differenti in un framework generico: nomi,
  commenti di forma e confini osservabili mantengono verificabile il grafo.
- Il monolite LLM e i backend Metal/ROCm richiedono revisioni mirate e hardware
  dedicato prima di refactoring strutturali. Qui il codice numerico della suite
  resta separato dal link legacy; la regressione disponibile è CUDA locale.
- T11 conserva il gate di qualità aperto: conteggio/layout della recipe 14
  falliscono anche nell'oracle congelato; OCR, preservation e review umana non
  sono promossi dal superamento del gate numerico. T12 resta qualificato sul
  percorso testuale locale.
