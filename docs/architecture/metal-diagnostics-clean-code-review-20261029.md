# Riesame della continuazione Metal: binding, ownership e diagnostici

29 ottobre 2026. Perimetro: tutte le modifiche pendenti della continuazione,
il wrapper numerico usato da test e tool, i suoi chiamanti, factory dei probe,
Makefile, documentazione e coerenza con runtime/backend Metal già presenti.
Le parti di produzione pertinenti sono state rilette: registry unificato,
allocazione/rilascio numerico, classe `DS4MetalNumeric`, workspace MPS e
binding della cache Flash. Non sono stati modificati kernel, formule o
tolleranze. Il [registro del riesame](../performance/metal-apple-silicon-20261029/continuation/cleanup-review.json)
conserva sorgenti, comandi, esiti e SHA; il precedente pacchetto di evidenze
della continuazione mantiene separati i suoi risultati storici.

## Problemi corretti

| Problema | Responsabilità dopo il riesame |
| --- | --- |
| Il diagnostico Gemma importava un modulo unittest per accedere al backend e ne ridefiniva l'ABI ctypes | `tests/suite_numeric_api.py` contiene il solo binding condiviso. L'import non apre librerie e non configura PyTorch; la policy dell'oracolo viene applicata esplicitamente dal chiamante |
| Budget e query memoria richiedevano di conoscere `lib`, `ctx`, `resources`, layout di `Budget` e codici numerici | `Numeric` espone `configure_metal_budget`, `memory_usage`, `linear_workspace` e `fence`; il diagnostico usa anche `linear`, `rms_norm` e `scale` |
| Un `free` fallito rimuoveva comunque l'handle dalla lista Python | L'owner mantiene l'handle fino al rilascio riuscito, come richiesto dal contratto C. `close` è ripetibile; un errore mantiene il contesto o registry necessario per il retry |
| Un errore nella creazione del contesto lasciava il registry del wrapper aperto | Il costruttore chiude le risorse già acquisite. Il context manager protegge tutti i diagnostici e il benchmark anche quando una verifica solleva un'eccezione |
| La factory dei probe poteva fallire sul budget dopo aver creato un contesto | La factory rilascia l'inizializzazione parziale, azzera l'output dopo cleanup riuscito e conserva esplicitamente l'owner recuperabile se il cleanup fallisce |
| Query memoria fallite potevano lasciare output precedenti | Gli output validi sono azzerati prima della validazione, anche se l'altro puntatore è nullo; il contratto è coerente su Metal e sul ramo CUDA |
| Due diagnostici usavano `Numeric(True)` e il benchmark `Numeric(cuda=True)` | Backend esplicito `cuda`, con configurazione dell'oracolo esplicita. I vecchi argomenti non vengono reinterpretati silenziosamente come un backend diverso |
| Il wrapper convertiva implicitamente FP16, FP64 e interi in F32 | Sono ammessi soltanto input F32/BF16, corrispondenti all'API nativa; gli altri dtype sono rifiutati prima dell'allocazione |
| La CLI Gemma organizzava e validava array anonimi e caricava tutte le trace insieme | `PLETrace` possiede lettura, compatibilità degli embedding e conteggi dei token; il replay mantiene una trace per volta. La CLI orchestra, l'owner numerico esegue e gestisce la memoria |
| I test ripetevano dichiarazioni della query memoria e interpretavano codici di errore anonimi | ABI dichiarata una volta nel binding; `MemoryUsage`, `Status` e `NativeError` rendono espliciti dati ed errori. I test negativi del contratto C mantengono accesso intenzionale all'ABI grezza |

La semantica Gemma resta nel diagnostico: scala PLE, ordine delle frontiere,
oracle PyTorch e testimoni FP64. Il wrapper non conosce checkpoint Gemma,
file, percorsi di output o modello image. `PLETrace` non conosce GPU o budget.
Le vecchie funzioni di analisi image ricevono l'owner nel loro entry point,
mentre la CLI possiede la durata del contesto. La suite di test conserva
criteri e assertion; non è più una dipendenza di esecuzione dei tool.

## Verifiche dopo il refactoring

- Suite numerica CPU: 37 test, PASS, 5 skip specifici di altri backend.
- Suite numerica Metal reale: 37 test, PASS, 3 skip specifici CUDA. Comprende
  cap sintetico di 1 MiB, NOMEM seguito da nuova allocazione riuscita, wrapper,
  workspace, BF16, attenzione e convoluzioni. Tolleranze originali conservate.
- Factory: fixture host sia per il ramo Apple sia per la selezione CUDA
  simulata, compilata anche con `-Werror`; nessun device CUDA coinvolto.
- Runtime: risorse, alias, eventi, numeric lifetime e transfer con fault e
  retry, PASS. Fixture reali Metal Qwen27, streaming esperti Flash e lifecycle
  Flash, PASS; non sono inferenza di un LLM completo.
- Gemma reale breve: due ripetizioni, budget insufficiente, cancellazione e
  retry, PASS. Tutte le 70 frontiere BF16 e i due file di vettori sono
  bit-exact rispetto al run precedente. I 16 array del replay isolato sono
  anch'essi bit-exact. Picco del replay 38.363.136 byte, storage finale zero.
- Preprocessing reference RGB/RGBA, contratti e manifest: 13 test, PASS.
  CLI `--help` dei tre tool migrati e Ruff, PASS. Build ordinaria e suite,
  compilazione dei probe e controlli documentali verificati nel registro.

Il primo run del nuovo test di budget attendeva erroneamente il codice 3
anziché `DS4_STATUS_NOMEM` (6); il test è stato corretto usando `Status.NOMEM`.
Nessuna soglia o risultato numerico è stato modificato per ottenere un PASS.
I report del primo tentativo e del run definitivo restano distinguibili.

Il gate BF16 inter-backend della PLE resta FAIL, identico a prima del
refactoring. Non è stata rieseguita la pipeline image completa in questo
riesame: il precedente smoke RGBA e i suoi hash restano evidenza separata.
Il comando aggregato `make test` non viene rivendicato come PASS: il precedente
run documentato falliva sui modelli LLM non ammissibili, mentre qui sono
state eseguite selezioni esplicite con modelli e costi noti. CUDA hardware
non disponibile, già confermato dall'utente; nessun nuovo gate CUDA/ROCm,
distribuito, grande LLM o frontend ExMekh.

## Valutazione Clean Code e debito residuo

Nel perimetro rivisto sono stati applicati responsabilità singola,
incapsulamento degli owner, dipendenze esplicite, nomi coerenti, assenza di
duplicazione dell'ABI, fallimenti recuperabili e teardown deterministico.
I test aggiunti controllano ownership e contratti osservabili; non replicano
gli algoritmi dei kernel.

**Non tutti i principi Clean Code sono soddisfatti globalmente.** `ds4.c` e
`ds4_server.c` restano monoliti con stato legacy, il runtime Metal ha stato
globale e la cache Flash è seriale. Questi confini erano già segnalati nel
[riesame del porting](metal-clean-code-review-20261029.md); una migrazione
richiede oracle per famiglie e backend, inclusi quelli non disponibili qui.
Il riesame non giustifica una riorganizzazione meccanica di quei moduli o una
qualifica del piano Metal completo.
