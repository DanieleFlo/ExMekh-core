# Frontend Agent FE / QS

L'agente del progetto e il frontend NiceGUI in `agent/`. Il package `fe`
usa `qs.core.models.client.Client` e `DeepAgent` per conversazioni, strumenti,
skill gerarchiche e subagenti. `ds4-server` gestisce inferenza e checkpoint KV;
FE conserva chat e configurazione dei progetti in JSON.

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
Il launcher FE installa le dipendenze dal lockfile quando necessario.
I launcher risolvono i percorsi dalla propria posizione e funzionano anche
quando invocati da un'altra directory. `run.sh` e `sync_qs.sh` cercano `uv`
nel `PATH` o in `~/.local/bin/uv`; `UV_BIN=/percorso/uv` permette di sceglierlo.

## Configurazione e stato

Copiare `agent/.env.example` in `agent/.env` per personalizzare il provider.
`BASE_URL=http://localhost:8080/v1`, `DEFAULT_MODEL=Qwen3.8-27B-UD-Q4_K_S`
e `API_TYPE=responses` collegano il client alle API agentic di DS4.
Il limite del client deve restare entro l'allocazione del server; il frontend
scopre ID e limiti dei modelli attraverso l'endpoint dei modelli.

I progetti usano DeepAgent e una directory di lavoro scelta dall'utente;
Quick Chats usano il client per conversazioni ordinarie. FE conserva chat in
`agent/data/chats/`, progetti in `agent/data/projects/` e gli allegati temporanei
sotto `tmp/` della directory di lavoro. Questi dati, `.env`, ambienti Python e
cache locali sono esclusi da Git.

## Confini dell'integrazione

QS possiede i messaggi e il risultato dell'esecuzione dei tool. FE usa
`Message.set_metadata()` e `Message.get_metadata()` per gli snapshot locali:
questi dati restano separati da `message_dump()` e non vengono inviati al
provider. Le letture e le copie dei messaggi non condividono valori mutabili.
I nomi dei campi dei vecchi snapshot salvati restano compatibili.

Gli eventi QS distinguono una fine riuscita da un errore di argomenti o di
esecuzione. FE conserva questa distinzione nella conversazione principale,
nei sotto-agenti e nella UI. Un risultato applicativo come `{"ok": false}`
resta un risultato completato: non viene interpretato come errore del runtime.

## Dipendenza QS

`agent/pyproject.toml` installa QS dalla wheel versionata in `agent/wheels/`.
La `.python-version` fissa Python 3.13 come nell'ambiente sorgente.
`uv sync --frozen` ricrea l'ambiente anche senza la cartella locale `qs/`.
Quest'ultima contiene una copia dei sorgenti QS e dei suoi test, con una
repository Git autonoma e la cronologia dell'originale. Rimane esclusa dal
Git di DS4 tramite il `.gitignore` della root. Le modifiche locali a `qs/`
non modificano la repository originale sul Desktop.

Per controllare la copia locale, dalla root DS4:

```sh
git -C qs status
git -C qs log -5 --oneline
```

Per aggiornare la wheel, dalla root in WSL:

```sh
./agent/sync_qs.sh
```

Il launcher esegue `agent/src/fe/sync.py`: ricostruisce QS dai sorgenti locali,
copia la wheel in `agent/wheels/`, aggiorna il percorso in `pyproject.toml` e
il lockfile e reinstalla QS. Gestisce anche una nuova versione della wheel e
rimuove quelle precedenti dopo il successo. Non richiede un build precedente
ne un ambiente FE gia installato.

## Verifica

Da `agent/`:

```sh
uv run playwright install chromium
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

I test browser usano Edge quando installato, altrimenti Chromium di Playwright.
QS e FE usano automaticamente tutte le CPU logiche disponibili al processo.
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
