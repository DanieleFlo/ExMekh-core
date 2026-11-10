# Preparazione Qwen3.8 27B Quick Chat su Metal

29 ottobre 2026, baseline `8e8ac45a7b463f7d4d72d859234f794b815019ef`,
MacBook Pro M5 Pro 24 GiB. Stato **PREPARED; INFERENCE NOT_VERIFIED**.
L'utente ha scelto Quick Chat e comunicherà in seguito quando la RAM sarà
disponibile. Questa fase prepara ambiente, profilo e controlli; non chiude
il gate di inferenza T03/T04 o i flussi reali F01–F12.

[verification.json](verification.json) conserva sorgenti, hash, comandi,
output e risultati distinti. Nessun kernel, forward, guard di memoria o
modello è stato modificato. Il frontend usa il wheel locale bloccato nel
lockfile, senza sostituire ExMekh-Lib.

## Esiti

| Controllo | Esito |
| --- | --- |
| Ambiente frontend | `uv sync --frozen`, Python 3.13.13 arm64; dipendenze locali installate |
| Startup frontend | PASS: HTTP 200 in 5,27 s, porta temporanea 8017, pagina ExMekh renderizzata nel browser |
| Entry point finale | PASS: `agent/run.sh` con il preset e dipendenze offline, HTTP 200 in 5,32 s, poi SIGINT |
| Test frontend | PASS: 47 test su configurazione, context tracking, lifecycle impostazioni e istruzioni del renderer |
| Launcher | PASS: 11 test; il preset esplicito 4k attraversa i wrapper esistenti anche da un'altra directory |
| Protocollo interattivo | PASS: `make test-server-interactive`, fixture host senza inferenza |
| Rendering prompt | PASS: quattro richieste iniziali costruite con i nodi reali del frontend, catturate prima dell'inferenza; renderer Python/C byte-exact dopo la correzione |
| Smoke 27B, contesto 256 | SKIP_RESOURCE_LIMIT: 14,76 GiB necessari contro 5,40 GiB utilizzabili |
| Smoke 27B, contesto 4096 | SKIP_RESOURCE_LIMIT: 15,22 GiB necessari contro 5,65 GiB utilizzabili |

Il frontend temporaneo è stato arrestato con SIGINT e cleanup ordinario.
Non rimangono server modello avviati da questa verifica. Il browser ha
mostrato la UI: non è stata inviata una richiesta di chat al modello.

## Prompt completo e scelta di 4k

Il testo utente della fixture è «Reply with OK.». Le definizioni degli
strumenti e le istruzioni di sistema non vengono omesse dal conteggio.
Il probe C usa parser e renderer di `ds4_server.c` senza engine o forward;
la CLI `--dump-tokens --raw --prompt-file` usa il tokenizer del GGUF reale,
leggendo metadata/vocabolario senza eseguire inferenza CPU.

| Profilo | Strumenti | Token senza reasoning | Token con reasoning |
| --- | ---: | ---: | ---: |
| Quick Chat | 6 | 2497 | 2495 |
| Agente generico | 23 | 4460 | 4458 |

Il preset selezionato è **Quick Chat, contesto 4096, prefill 64, target-only,
Responses, reasoning none**. Per la fixture lascia 1599 token complessivi
per output e crescita della conversazione. Non è una garanzia su richieste
più lunghe, allegati o cicli con tool. Il profilo generico non entra in 4k.
Il precedente limite locale 2k resta una qualifica iniziale distinta; il
preset 4k è esplicito e non cambia il default 512/64 del launcher Metal.

Il contatore frontend aveva una frase aggiuntiva nel suffisso degli strumenti,
assente nelle macro native. La correzione elimina una sovrastima di 17 token
per queste fixture; non cambia il prompt inviato al modello dal server.
Il nuovo test confronta letteralmente prefisso/suffisso delle risorse con
`ds4_prompt_settings.h`. Un newline aggiunto durante la modifica è stato
rilevato dal test e rimosso, conservando il confine nativo originale.

Il server conserva il whitespace dei JSON degli strumenti. La prima prova
del diagnostico usava JSON indentato: non rappresentava il wire del client.
Il confronto finale usa il JSON compatto UTF-8 verificato nell'encoder httpx
installato; conserva separati conteggio iniziale, correzione e risultato finale.

## Avvio preparato

I comandi completi sono nella [guida frontend](../../../../agent/README.md),
sezione Quick Chat Metal. Da due terminali nella root del repository,
caricare prima lo stesso profilo:

```sh
set -a
source config/qwen27-metal-quick.env.example
set +a
```

Nel primo terminale avviare `./agent/start-ds4-server.sh --host 127.0.0.1`.
Nel secondo avviare `./agent/run.sh`, poi aprire `http://127.0.0.1:8000`
e scegliere **New Quick Chat**. Il profilo mantiene allineati server e UI
sui 4096 token; non riscrive un `.env` personale.

Il profilo richiede circa **17,22 GiB disponibili prima della riserva host**,
oltre a rispettare il working set Metal. Le stime vanno ricontrollate quando
sarà liberata la memoria, con il frontend e le altre applicazioni presenti.
Ridurre il contesto non elimina i 14,30 GiB dei pesi.

I due tentativi registrati sono terminati nel guard prima del caricamento,
in circa un secondo, con swap invariato nei rispettivi campioni iniziale/finale.
Non sono stati forzati oversubscription, cambio di quantizzazione o riduzioni
silenziose delle capacità. Restano da provare caricamento/decode, logits,
risposta Quick Chat reale, tool e checkpoint con inferenza, media e suite.

## Primo tentativo dopo la preparazione

Su richiesta dell'utente, nuovo smoke Metal target-only a contesto 4096,
prefill 64 e massimo quattro token. [live-attempt-01.json](live-attempt-01.json)
conserva comando, hash del binario, output e campioni. Esito
**SKIP_RESOURCE_LIMIT**: 15,22 GiB richiesti contro 8,78 GiB utilizzabili dopo
la riserva di 2 GiB. Rifiuto in 1,04 s prima dei pesi; swap iniziale/finale
zero, nessun arresto forzato. Non è stata prodotta inferenza.

Nello screenshot fornito dall'utente, Monitoraggio Attività mostra 24 GB
fisici, 14,04 GB utilizzati e 10,26 GB di file nella cache, con compressione
e swap a zero. La cache è memoria riutilizzabile, già in buona parte inclusa
nelle pagine inattive conteggiate dal guard; non è capacità aggiuntiva ai
24 GB. La somma RSS dei processi può contare pagine condivise più volte e
non è stata usata come misura della RAM fisica occupata.

Il controllo statico del loader conferma che il percorso Metal non esegue
il prefetch CPU di `model_open()`, mentre `qwen_metal_memory_admit()` precede
il riscaldamento e il binding dei pesi. Non emerge un caricamento anticipato
che giustifichi un bypass: il guard e la riserva restano invariati. Mancano
circa 6,44 GiB di margine rispetto all'ultima misura; il gate reale resta aperto.
