# Revisione del codice Flash e dei confini dell'agente — 4 ottobre 2026

## Perimetro ed esito

Revisione delle modifiche pendenti sul branch `codex/qwen-flash-3090`, a partire
da `bd89b13f0c839d3d9e17c7afab1f8a4dcf1b98e6`. Sono stati controllati anche i
confini fra runner, conversione della history, modelli di dominio, UI, provider,
parser dei tool, schema e cache. I refactoring risolvono accoppiamenti concreti
senza imporre classi alle API C procedurali o riscrivere i backend.

Questo audit non certifica che ogni principio di clean code sia soddisfatto in
ogni file. Il server C e il backend Flash restano moduli grandi: un'eventuale
separazione fisica richiede un lavoro dedicato con gli stessi gate. Il verdetto
prestazionale della [qualifica Flash](../performance/qwen38-flash-pipeline-20261004.md)
resta **NEED_MORE_DATA**; questi interventi non costituiscono nuovi benchmark A/B.

## Responsabilità e contratti corretti

| Responsabile | Comportamento |
| --- | --- |
| `converter.py` | Serializza le esecuzioni skill, converte la history e deduplica il prompt tramite `Message.get_role_name()` e `get_message()`, anche per messaggi già tipizzati. Il runner coordina l'esecuzione. |
| `provider_health.py` | Legge e valida i metadati `/models`; scarta liste malformate, ID invalidi e contesti non finiti, booleani o non positivi. |
| `AppConfig` | Seleziona il modello disponibile e aggiorna il contesto in memoria; la UI usa il metodo pubblico e non persiste preferenze durante il probe. |
| `ChatMessage` | Gestisce aggiunta dei token e segmentazione del reasoning; `SubAgentConversation` delega senza manipolare i blocchi interni. |
| Handler degli eventi | Quick Chat e DeepAgent emettono reasoning con lo stesso scope. I callback legacy a un argomento sono adattati una volta; un `TypeError` nel corpo non causa una seconda chiamata. |
| `ContextTrackingProvider` | Usa gli stessi confini per `call` e `call_structured`, isola i contesti figli e conserva i totali autorevoli del provider senza cambiare richiesta o risultato. |
| Ordine/schema dei tool | La proprietà dello schema viene letta da un helper unico; il parser finale e il filtro Flash condividono la distinzione fra stringa nativa e JSON. |
| Policy della cache | Heap e scansione lineare condividono il confronto frequenza, ultimo utilizzo e indice. La scelta della vittima conserva l'ordinamento precedente. |

Gli import pubblici precedenti del runner sono mantenuti per compatibilità;
la UI importa il probe dal modulo che ne ha la responsabilità. Le richieste
strutturate continuano a usare il client tipizzato e gli oggetti pubblici
ExMekh-Lib, senza leggere lo storage privato del provider.

Nel protocollo nativo Flash, uno schema che ammette solo stringhe conserva il
testo grezzo anche quando sembra JSON (`123`, `null`). Per schemi misti viene
preservata la regola del parser nativo: un valore JSON completo non quotato è
JSON, gli altri valori sono testo. Ora filtro, validazione dell'invocazione e
tracker riconoscono gli stessi valori e delimitatori, anche con `anyOf`, `allOf`,
`enum` e tipi nullable. I test verificano il valore finale e ogni possibile
confine fra token; i valori di tipo non ammesso continuano a essere respinti.

Il test TCP confronta il timeout ripristinato con quello effettivamente applicato
dal kernel prima dello scambio. Il controllo resta esatto e non assume che
Linux conservi il valore richiesto senza arrotondarlo al tick del kernel.

## Verifiche della revisione

| Gate | Esito |
| --- | --- |
| Suite completa agente, inclusi Playwright mock | 739 passati, 40 saltati, copertura 100% |
| Suite Python principale nel virtualenv dell'agente | 199 passati, 41 saltati |
| Ruff, sintassi dei tre launcher e controllo whitespace | PASS |
| Test C server, parser, vincoli, layer pattern e unit test locali | PASS |
| Policy esperti, ranking e ripristino timeout TCP | PASS |
| Build CPU e `--help` dei quattro eseguibili | PASS |
| Build CUDA sulla RTX 3090 | PASS |
| Primitive CUDA, cache piccole e gruppi parziali | PASS |
| 200 posizioni × 248320 logits contro baseline congelata | PASS, identità byte per byte su 198656000 byte |
| Flash HTTP: registry padre/figlio, skill/return, nesting, corruzione | 4 passati |
| Client tipizzato Flash: chat/responses, stream/non-stream, greedy senza thinking | 4 passati |
| Qwen27 checkpoint con vision, MTP e flag Flash ereditato | PASS, logits e restore bit-exact; tre livelli annidati e corruzione |
| Indici e test documentazione | PASS, 10 test |

Le prove HTTP usano il server compilato con la correzione degli schemi nativi.
Il successivo refactoring del confronto cache, semanticamente equivalente, è
verificato sul binario CUDA finale con primitive, full-vocabulary e checkpoint
27B. Un solo processo modello viene eseguito alla volta.

I log, manifest, comandi, hash e output binari sono in
`performance-results/flash-pipeline-20261004/clean-review-*`, esclusi dal Git.
Gli hash e i campioni della precedente qualifica prestazionale sono conservati:
non vengono sostituiti con quelli del refactoring.

I tentativi con interprete privo delle dipendenze e comando di discovery errato
sono **INVALID_SETUP**, conservati nei log e ripetuti con il virtualenv corretto.
Anche il primo tentativo 27B con ctx 256 era troppo piccolo per la fixture vision;
la ripetizione a ctx 2048 passa con gli stessi assert e 808 token nel parent.
I test saltati non sono considerati passati. Metal, ROCm, RDMA con hardware reale,
distribuito multi-GPU, browser con provider live e matrice completa di contesti
lunghi/sampling sono **NOT_VERIFIED** in questa revisione.

## Riproduzione dei gate statici

```sh
(cd agent && .venv/bin/ruff check src tests && .venv/bin/python -m pytest)
agent/.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
make -j2 ds4 ds4-server ds4-bench ds4-eval ds4_test ds4_server_test
make test-qwen4-expert-cache test-session-state
./ds4_server_test
./ds4_test --server --constraint-trie --qwen35-layer-pattern
make docs-check
bash -n start-ds4-server.sh agent/start-ds4-server.sh agent/start-ds4-server-flash.sh
```

Per i test GPU servono i GGUF locali, il proiettore del modello e memoria
sufficiente; i test live sono opt-in. Non viene cambiato il modello, la
quantizzazione, l'aritmetica dei kernel o il formato dei checkpoint.
