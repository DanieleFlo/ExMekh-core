# Riproduzione del prefill dopo steering — 14 ottobre 2026

La perdita del riuso della conversazione è riprodotta su Qwen3.8-27B Q4_K_S,
CUDA/RTX 3090, contesto 26.000, MTP e vision encoder caricati. Sono stati aggiunti
[test HTTP opt-in](../../tests/test_steering_cache_live.py) e conservate
[misure e hash](../performance/steering-cache-reproduction-20261014.json).
Questo intervento aggiunge test e documentazione; i percorsi di produzione
per cache, steering e approvazione dei comandi conservano il codice precedente.

La correzione successiva è descritta nel [rapporto del fix](steering-cache-fix-20261014.md); le misure qui sotto restano quelle della riproduzione originale.

## Risultati

Ogni caso apre una skill sintetica, scalda una conversazione con un tool,
poi confronta `usage.input_tokens_details.cached_tokens` con la frontiera già
elaborata. I nomi delle chiamate, lo stato delle risposte, la pausa richiesta
ed il ritorno della skill sono verificati prima di classificare il risultato.
I tool non vengono eseguiti sul filesystem. Il replay delle interruzioni usa
i record JSON `interactive.cancelled` inviati dal mapper reale del client.

| Caso, reasoning none, non-streaming, 96 righe di archivio | Token riusati | Token di prefill | Esito del riuso |
| --- | ---: | ---: | --- |
| Tool completato, replay completo, nessuno steering | 2.627 | 116 | PASS |
| Tool completato, poi messaggio di steering prima di exit_agent | 596 | 1.975 | FAIL |
| Edit sospeso, cancellazione e steering prima di exit_agent | 596 | 2.170 | FAIL |
| Richiesta ausiliaria con proprio agent_context_id | 2.627 | 116 | PASS |
| Richiesta ausiliaria senza agent_context_id | 596 | 1.959 | FAIL |

La storia viene riusata prima della cancellazione: il test lo richiede anche
sulla risposta `interactive.await_input`. I tre FAIL della matrice sono
asserzioni sul riuso della KV, con tutti i passaggi HTTP validi e cleanup
completato. Gli errori iniziali di preparazione dell'harness sono esclusi
dalle misure archiviate.

La prova SSE su 640 righe di archivio, reasoning none, riusa solo 596 token e
ne precompila **12.499** dopo la cancellazione della pausa edit. Il totale di
input della ripresa è 13.095 token. Riproduce il tipo di ricalcolo mostrato nel
log originale, con un prefisso di sistema più piccolo della configurazione
completa dell'agente.

Con SSE e thinking low, su 96 righe, vengono ricalcolati **1.981 token** dopo
un tool completato e **2.176 token** dopo la cancellazione dell'edit. Entrambi
i casi conservano soltanto i 596 token di sistema.

Un'ulteriore prova SSE con thinking low disabilita la registrazione del
comportamento edit interattivo. Lo steering dopo un tool ordinario conserva
solo **443 token di sistema** e ricalcola **1.977 token**. La perdita si manifesta
quindi anche senza attivare il nuovo protocollo edit.

Sono state eseguite nove prove live: due controlli positivi passati e sette
riproduzioni del mancato riuso, su tre scenari problematici e più configurazioni.
Passano anche i due controlli unitari dei contatori. La qualifica riguarda
questa combinazione modello/backend; gli altri backend non sono stati provati.

## Conseguenze per la diagnosi

La diagnosi iniziale limitata alla cancellazione interattiva era incompleta:
il messaggio di steering provoca il problema anche dopo un tool completato,
prima dell'uscita dalla skill. Il normale ritorno della skill nei test riesce
comunque a ripristinare il checkpoint del parent.

`responses_prepare_live_continuation` riconosce i tool output in coda. Un
messaggio user di steering aggiunto dopo di essi non rientra in quel suffisso.
I test mostrano il successivo fallback al prefisso di sistema; la causa
puntuale del mancato matching della history visibile richiede una correzione
separata. La cancellazione della pausa è un altro percorso da proteggere.

L'isolamento senza comportamento edit e l'assenza di modifiche ai percorsi
server interessati escludono che la riproduzione dipenda dal nuovo formato
letterale dei file. Non è stato eseguito un bisect con un binario storico.

## Comandi ripetibili

Su una sola istanza server inattiva, dalla radice del repository:

```bash
DS4_STEERING_CACHE_BASE_URL=http://127.0.0.1:8080 \
DS4_STEERING_CACHE_REPORT=/tmp/steering-cache.json \
python3 -m unittest tests.test_steering_cache_live -v
```

Nella riproduzione originale tre casi di regressione fallivano. Il fix successivo
aggiunge anche il caso di resume: senza BASE_URL le sei prove live sono saltate e i due
controlli unitari vengono eseguiti.

Le variabili `DS4_STEERING_CACHE_STREAM=1`, `DS4_STEERING_CACHE_REASONING=low`
e `DS4_STEERING_CACHE_ARCHIVE_LINES=640` selezionano flusso SSE, thinking e
lunghezza della storia. Con `DS4_STEERING_CACHE_INTERACTIVE=0` si possono
isolare i casi di tool ordinari; il caso di pausa edit viene saltato.

Le richieste complete e i log sono conservati localmente in
`performance-results/steering-cache-reproduction-20261014/`; il riepilogo
JSON nel repository riporta contatori, configurazioni e hash di sorgenti e
binario. Il server temporaneo viene terminato alla conclusione delle prove.