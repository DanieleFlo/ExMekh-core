# ExMekh Agent

L'agente del progetto e il frontend NiceGUI in `agent/`. Il package `exmekh_agent`
usa `exmekh_lib.core.models.client.Client` e `DeepAgent` per conversazioni, strumenti,
skill gerarchiche e subagenti. `ds4-server` gestisce inferenza e checkpoint KV;
ExMekh Agent conserva chat e configurazione dei progetti in JSON.

## Avvio in WSL / Linux

Da root, avviare il server in un terminale:

```sh
./agent/start-ds4-server.sh
```

In un altro terminale:

```sh
./agent/run.sh
```

Il launcher server usa Qwen3.8 e MTP con `--ctx 28000`, porta 8080.
Il frontend si apre su [http://127.0.0.1:8000](http://127.0.0.1:8000).
Il launcher installa le dipendenze dal lockfile quando necessario. Da `agent/`,
il comando equivalente è `uv run --frozen exmekh-agent`. La distribuzione si
chiama `exmekh-agent` e il namespace Python è `exmekh_agent`; `exmekh-lib`
rimane la CLI separata della libreria.
I launcher risolvono i percorsi dalla propria posizione e funzionano anche
quando invocati da un'altra directory. `run.sh` e `sync_exmekh_lib.sh` cercano `uv`
nel `PATH` o in `~/.local/bin/uv`; `UV_BIN=/percorso/uv` permette di sceglierlo.

## Configurazione e stato

Copiare `agent/.env.example` in `agent/.env` per personalizzare il provider.
Le impostazioni della UI sono `EXMEKH_AGENT_HOST`, `EXMEKH_AGENT_PORT` e
`EXMEKH_AGENT_TITLE`; sostituiscono i precedenti `FE_HOST`, `FE_PORT` e `FE_TITLE`.
Le chiavi del provider come `BASE_URL`, `API_KEY` e `DEFAULT_MODEL` restano uguali.
`BASE_URL=http://localhost:8080/v1`, `DEFAULT_MODEL=Qwen3.8-27B-UD-Q4_K_S`
e `API_TYPE=responses` collegano il client alle API agentic di DS4.
Il limite del client deve restare entro l'allocazione del server; il frontend
scopre ID e limiti dei modelli attraverso l'endpoint dei modelli.

I progetti usano DeepAgent e una directory di lavoro scelta dall'utente;
Quick Chats usano il client per conversazioni ordinarie. ExMekh Agent conserva chat in
`agent/data/chats/`, progetti in `agent/data/projects/` e gli allegati temporanei
sotto `tmp/` della directory di lavoro. Questi dati, `.env`, ambienti Python e
cache locali sono esclusi da Git.

## Modifica dei messaggi

Il pulsante con la matita compare accanto a quello di copia quando si passa
sopra un messaggio utente; su dispositivi touch rimane visibile. L'editor
permette di annullare oppure salvare con **Save & regenerate**. Il salvataggio
interrompe l'eventuale generazione, elimina il ramo successivo e avvia una
nuova risposta. Conserva gli allegati espliciti, mentre il testo lungo
trasformato automaticamente in file viene ricreato dalla versione modificata.

La modifica elimina anche gli input in coda e la history dei sottoagenti
successiva al punto di ripartenza. Azzera il contesto delle approvazioni dei
comandi, mantenendo la modalità di approvazione scelta. Se si modifica uno
steering, la risposta che lo contiene riparte mantenendo gli input precedenti.
Il ramo aggiornato viene salvato prima di avviare la nuova risposta e rimane
coerente dopo il reload della pagina.

ChatSession possiede le operazioni sul ramo della conversazione;
SubAgentConversation conserva uno snapshot per chiamata e gestisce il replay;
ChatSettings azzera le approvazioni obsolete. SessionManager salva il ramo
prima di applicarlo alla sessione in memoria. Gli upload testuali di Quick Chat
rimangono inline nelle richieste al provider anche dopo una modifica.

I test e2e usano il client mock con un provider deterministico attraverso il
ciclo ExMekh-Lib reale. Coprono le posizioni iniziale, intermedia e finale, la generazione
in corso, la riconnessione, l'annullamento, le approvazioni pendenti e gli allegati
testuali, immagine e PDF. Stop e accodamento sono limitati alla chat selezionata;
i tool mostrano lo stato di errore del modello anche nelle schede della chat.

## Confini dell'integrazione

ExMekh-Lib possiede i messaggi e il risultato dell'esecuzione dei tool. ExMekh Agent usa
`Message.set_metadata()` e `Message.get_metadata()` per gli snapshot locali:
questi dati restano separati da `message_dump()` e non vengono inviati al
provider. Le letture e le copie dei messaggi non condividono valori mutabili.
I nomi dei campi dei vecchi snapshot salvati restano compatibili.

Gli eventi ExMekh-Lib distinguono una fine riuscita da un errore di argomenti o di
esecuzione. ExMekh Agent conserva questa distinzione nella conversazione principale,
nei sotto-agenti e nella UI. Un risultato applicativo come `{"ok": false}`
resta un risultato completato: non viene interpretato come errore del runtime.

Il converter converte messaggi e snapshot; `ExecutionHistoryRestorer` ricostruisce
history e cache usando le API pubbliche ExMekh-Lib. Tool, skill e steering condividono
il parser degli header `RecordedToolOutput`. Le riletture conservano l'identità,
gli argomenti e i riferimenti dell'output originale, senza sostituirli con i
metadati della chiamata di lettura. Lo steering nelle skill conserva testo,
immagini e ID canonico anche dopo la compattazione e nei turni successivi.

## Dipendenza ExMekh-Lib

`agent/pyproject.toml` installa ExMekh-Lib dalla wheel versionata in `agent/wheels/`.
La `.python-version` fissa Python 3.13 come nell'ambiente sorgente.
`uv sync --frozen` ricrea l'ambiente anche senza la cartella locale `exmekh_lib/`.
Quest'ultima contiene una copia dei sorgenti ExMekh-Lib e dei suoi test, con una
repository Git autonoma e la cronologia dell'originale. Rimane esclusa dal
Git di DS4 tramite il `.gitignore` della root. Le modifiche locali a `exmekh_lib/`
non modificano la repository originale sul Desktop.

Per controllare la copia locale, dalla root DS4:

```sh
git -C exmekh_lib status
git -C exmekh_lib log -5 --oneline
```

Per aggiornare la wheel, dalla root in WSL:

```sh
./agent/sync_exmekh_lib.sh
```

Il launcher esegue `agent/src/exmekh_agent/sync.py`: ricostruisce ExMekh-Lib dai sorgenti locali,
copia la wheel in `agent/wheels/`, aggiorna il percorso in `pyproject.toml` e
il lockfile e reinstalla ExMekh-Lib. Gestisce anche una nuova versione della wheel e
rimuove quelle precedenti dopo il successo. Non richiede un build precedente
né un ambiente ExMekh Agent già installato. La pulizia delle wheel precedenti
viene eseguita solo dopo l’installazione riuscita.

Il namespace Python è `exmekh_lib`, mentre la distribuzione e la CLI sono
`exmekh-lib`. Il bridge ExMekh Agent si trova in `agent/src/exmekh_agent/exmekh_bridge/`.
Da `agent/` si possono invocare i comandi installati:

```sh
uv run sync-exmekh-lib
uv run --frozen --env-file .env exmekh-lib
```

All'avvio, i percorsi salvati delle capabilities standard sotto
`site-packages/qs/capabilities` vengono migrati ai valori predefiniti della
libreria installata. La migrazione aggiorna progetti e chat senza modificare
percorsi personalizzati, messaggi, date o approvazioni.

Le chat JSON e i metadati di ripristino conservano il formato esistente. Le nuove
cache della libreria usano `.exmekh_lib/`; le vecchie `.qs/` restano ignorate da
Git. Le variabili `EXMEKH_LIB_API`, `EXMEKH_LIB_MODEL` e `EXMEKH_LIB_TOKENIZER_CACHE_DIR` sono
quelle esposte dalla configurazione pubblica della libreria rinominata. Dopo una
modifica dei sorgenti occorre sincronizzare la wheel per aggiornare la configurazione
installata. La CLI carica `BASE_URL` e `API_KEY` dal file `.env` con il comando sopra; `EXMEKH_LIB_MODEL` e
`EXMEKH_LIB_API` permettono di scegliere modello e API indipendentemente dalla UI.

## Verifica

Da `agent/`:

```sh
uv run playwright install chromium
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

I test browser usano Edge quando installato, altrimenti Chromium di Playwright.
ExMekh-Lib e ExMekh Agent usano automaticamente tutte le CPU logiche disponibili al processo.
Da ciascuna directory, `uv run pytest -n 16` sceglie sedici worker;
`PYTEST_XDIST_AUTO_NUM_WORKERS=16 uv run pytest` imposta il conteggio di `auto`.
In WSL questo segue il limite configurato in `.wslconfig`; il numero di worker
puo superarlo, ma la capacita CPU rimane quella della VM.
I test ordinari includono i provider mock. I test con un modello reale
richiedono opt-in e server configurato:

```sh
uv run pytest --no-cov --run-live -m "live and not mock"
```

Le verifiche delle API e dei checkpoint server restano in
`tests/test_agentic_api.py`, `tests/test_agentic_checkpoint.c` e nella matrice
`tests/test_server_sampling_matrix.py`. Le misure datate dei fixture ritirati
non certificano il frontend corrente.
