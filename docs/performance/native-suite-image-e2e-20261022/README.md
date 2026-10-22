# Audit E2E della generazione immagini e di «Funzionalità extra»

Il task risulta completato nella qualifica locale CUDA del 22 ottobre 2026,
dopo due correzioni emerse dalla verifica. L'export della history DeepAgent
perdeva gli allegati tool: i pixel arrivavano al modello ma mancavano nelle
miniature e nella cronologia salvata. Lo schema JSON dell'editing rifiutava
inoltre il prompt vuoto, supportato dal server per l'input solo immagine.
Entrambe le regressioni hanno una riproduzione negativa conservata e un test
che passa dopo la correzione.

## Requisiti verificati

| Requisito di `exmekh_lib/todo.md` | Prova effettiva |
| --- | --- |
| Tool di generazione | Browser → LocalProvider reale → DeepAgent → `generate-image` registrato → servizio GPU → continuazione del modello |
| Immagine nella risposta, default True | Due allegati PIL 128×128 nella chiamata successiva al modello; codice `IMAGE-E2E-17` ricordato; il caso senza flag verifica il default True |
| Flag False e rescale | False restituisce solo metadati; scale 0.5 e 1.0 verificate; originali sempre 256×256 e byte identici al download del servizio; scale invalide rifiutate prima del compute |
| Testo, immagini o entrambi | Tre generazioni reali: solo testo, due riferimenti con prompt omesso, testo più riferimento; ordine dei file verificato confrontando i byte caricati |
| Parametri del motore implementati | Count, width/height interi, PNG, background auto/opaque/transparent, seed e idempotency key; recipe fissa 8 step/CFG 1 e versione 1 controllate sul wire e nel risultato |
| Più immagini | Una sola richiesta HTTP produce due PNG con seed consecutivi; retry della stessa richiesta restituisce gli stessi file ID e request ID |
| Ranking sul motore | Ranking testuale e visuale con chiamate reali a `/embeddings`, modello `google/embeddinggemma-2`, task similarity e punteggi finiti; nessuna nuova importazione di OpenVINO/Transformers/PyTorch nel percorso |
| Frontend completo | Due miniature, apertura/chiusura dialog, persistenza dopo reload, secondo turno senza rigenerazione con entrambi gli allegati ripristinati; tre heartbeat HTTP e modifica del draft durante il compute |

Le identità disponibili restano vincolate con Literal e senza selettore per
l'unico modello di ogni operazione. Il tool espone tutti i parametri pubblici
implementati dall'API v1; non introduce opzioni assenti dal motore. Gli output
multipli condividono una richiesta, con esecuzione seriale della recipe nativa.

## Correzione della history

`snapshot_message` usa il codec cache già esistente per esportare gli allegati
con i pixel. È condiviso da MessageHistory, snapshot del nodo e history delle
skill chiuse. `message_dump()` conserva il contratto compatto precedente.
Il converter e il collector accettano sia il formato lossless della libreria
sia quello UI già salvato; il collector legge le URL senza decodificare i pixel
a ogni aggiornamento dello stream. Le anteprime includono un breve contesto
testuale che le identifica come risultato della generazione completata.

## Evidenze e riproduzione

[Manifest e risultati](verification.json) registrano richieste, hash dei PNG,
misure di responsività, log, sorgenti, binario e corrispondenza sorgenti/wheel/
installazione/lock. La [patch ExMekh](exmekh-lib.patch) include tutte le modifiche
alla dipendenza esterna rispetto al commit base indicato nel manifest, con
verifica reverse-apply. I log e le schermate integrali sono conservati in
`performance-results/native-suite-image-e2e-20261022/`.

Ambiente: Ubuntu 24.04 WSL, RTX 3090 sm_86, CUDA single GPU, componenti BF16
della configurazione nativa, Qwen3.8-27B UD-Q4_K_S con proiettore vision F16,
MTP disabilitato e contesto 8192. Una sola istanza grande:

```sh
./ds4-server-suite -m gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  --vision gguf/mmproj-Qwen3.8-27B-F16.gguf --cuda --no-mtp \
  --ctx 8192 --port 18091 --suite-config config/native-suite.example.json
# Attendere il log listening, poi da un altro terminale:
export DS4_SUITE_HTTP_URL=http://127.0.0.1:18091/v1
export DS4_SUITE_E2E_ARTIFACTS=$PWD/performance-results/native-suite-image-e2e-20261022/tools
agent/.venv/bin/python -m unittest tests.test_suite_image_tools_e2e -v
export DS4_SUITE_E2E_ARTIFACTS=$PWD/performance-results/native-suite-image-e2e-20261022/browser
cd agent
.venv/bin/pytest -n 0 --no-cov --run-live tests/test_native_image_generation_e2e.py -x
```

I test live sono opt-in: senza URL il test del tool viene saltato; il browser
richiede anche `--run-live`. Il runner usa un AgentDesign di prova compilato
dal percorso ordinario, con generazione del titolo ausiliario fissata. Non
sostituisce LLM, tool o API native. La prima fixture chiedeva impropriamente
al root di uscire con un tool e usava contesto 4096 insufficiente; i relativi
fallimenti sono conservati separatamente. La qualifica finale corregge quelle
istruzioni e verifica anche il secondo turno dopo il reload.

Un replay con trasferimenti dei pesi molto lenti ha impiegato 285 secondi per
il compute: superata l'attesa tool predefinita di 60 secondi, il modello ha
ricevuto `running` e ha richiesto una seconda generazione. Il test ha fallito
correttamente il vincolo di una sola richiesta; tracce e log sono conservati in
`browser-slow-default-wait/`. La prova finale richiede esplicitamente
`execution_mode=wait, wait_seconds=300`, controlli già supportati dal runtime.
Non qualifica la capacità del modello di gestire job oltre quella finestra o
di recuperare autonomamente un risultato dopo lo stato `running`.

La suite offline completa della libreria passa con 3624 test e 76 live esclusi;
le regressioni frontend selezionate passano con 162 test. Il manifest contiene
conteggi e durate dei gate live, schema, Ruff, Pyright, testi prodotto e documenti.
La wheel è ricostruita dai sorgenti e installata con lock aggiornato.

| Gate finale | Esito |
| --- | --- |
| Tool registrato e ranking nativi | 3 PASS, 316.155 s |
| Browser con modello e generazione reali | 1 PASS, 73.89 s |
| Libreria offline completa | 3624 PASS, 76 live esclusi |
| Frontend/history/restore/steering e browser preview | 162 PASS |
| Schema del servizio | 7 PASS |
| Documentazione | 10 PASS |
| Ruff e Pyright | PASS, zero errori e warning Pyright |

La continuazione immediata dopo il tool riusa il prefisso Responses e la KV
multimodale. Nel secondo turno dopo reload il server rileva un prefisso diverso
(`common=2948`, `token-mismatch`) ed esegue il prefill della richiesta ricostruita,
con cache degli embedding vision utilizzata. Il test dopo reload verifica
pixel, risposta e assenza di rigenerazione; non qualifica KV reuse attraverso
la serializzazione della cronologia UI. Il log completo è nel manifest.

Questa prova qualifica l'integrazione sul backend locale. I gate qualità
immagini T11, gli altri backend e T19–T22 mantengono il loro stato precedente.
