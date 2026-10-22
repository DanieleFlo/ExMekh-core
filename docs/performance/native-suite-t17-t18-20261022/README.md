# T17–T18: servizio HTTP e integrazione ExMekh

Qualifica locale del 22 ottobre 2026: Ubuntu 24.04 WSL, RTX 3090 sm_86,
CUDA single GPU, suite BF16 nativa, Qwen3.8-27B UD-Q4_K_S senza MTP,
contesto 2048. T17/T18 sono completati nei gate di servizio/client descritti
qui. I gate qualità image T11 e la consegna integrata T19–T22 restano aperti.

L'[audit E2E successivo](../native-suite-image-e2e-20261022/README.md) usa il
browser con LLM e generazione reali. Ha rilevato e corretto la perdita degli
allegati nell'export DeepAgent, non coperta dal precedente browser scripted,
e lo schema che rifiutava il prompt vuoto nell'editing image-only. Gli hash e
i risultati di questo report restano la fotografia storica della prima qualifica;
la nuova patch, la wheel aggiornata e le prove complete sono nel follow-up.

## Implementazione

`ds4-server-suite` aggiunge catalogo, Images generation/edit, Embeddings e Files
al worker del LLM. Il gestore ammette il lavoro e pubblica solo dopo il restore.
Eseguibili e launcher legacy mantengono percorso e dipendenze precedenti.
[Avvio, configurazione, esempi e limiti](../../architecture/native-suite-client.md).

ExMekh usa il trasporto esistente con `Client.native_suite()`: identità Literal,
nessuna scelta per il singolo modello disponibile, width/height interi.
`generate-image` e `embed-content` sono tool registrati, con riferimenti
ordinati, output multipli, media ID, seed/recipe e anteprime configurabili.
Il ranking usa il servizio embedding nativo, senza inferenza OpenVINO/SigLIP
nel client. La UI conserva gli allegati e usa miniature/dialog esistenti.

## Evidenze

[Manifest e verifica](verification.json) contengono hash di sorgenti, binario,
wheel, asset, log e riepiloghi di risposte committate. I log integrali locali
sono in `performance-results/native-suite-t17-t18-20261022/logs/`.
[Revisione del codice](../../architecture/native-suite-t17-t18-clean-code-review-20261022.md).

| Gate | Esito locale |
| --- | --- |
| Build CUDA e protocollo host | PASS; uint64, multipart/spooling, ricevute, TTL, geometria e deadline |
| Risorse/runtime/executor/numeric lifetime | PASS; lease annidati, cancellazione e fault/recovery host |
| Server legacy | PASS, `ds4_server_test` |
| Contratti, manifest, API, embedding preparation e processor | 44 PASS |
| SDK OpenAI reale | 6 PASS; discovery, quattro modalità, retry, edit, batch e disconnessione |
| ExMekh live | PASS; tool registrato, continuazione con cached token e anteprima reale True/scala 0.5 |
| Libreria offline completa | 3622 PASS, 76 live deselected |
| Frontend e browser | 96 PASS, 1 live preesistente skipped |
| History/collector/restore/steering | 112 PASS |
| Pyright/Ruff/testi prodotto | PASS; zero errori e warning Pyright |

Embedding 128 con norma L2 entro 1e-5; PAD escluso e float32 identici in base64.
Composizione ordinata testo/image/video/audio con un vettore per elemento.
Seed uint64 `18446744073709551614` preservato; retry simultanei con identici
risultato/request ID, conflitto 409 per corpo diverso. Download identico ai
byte PNG della risposta; opaque con alpha 255, n=2 con seed consecutivi,
editing image-only con due riferimenti validi. Ripresa dopo disconnessione
verificata mediante un nuovo compute.
Il replay finale comprende `images.generate` e `images.edit` dello SDK e il
rifiuto delle dimensioni in overflow prima di ogni compute.

Responses genera un vero tool call e riprende con `function_call_output`:
ALFA17 ricordato e `cached_tokens > 0`. Il Client ExMekh restituisce BETA18
con lo stesso controllo sul wire Responses. La conversazione usa risultato
senza pixel; una seconda generazione reale prova l'allegato True e rescale
256→128. Il browser usa un provider scripted e non qualifica il modello.

## Comandi riproducibili

Build/avvio nella [guida](../../architecture/native-suite-client.md), con gli
asset verificati dei task precedenti e una sola istanza grande. Dalla radice:

```sh
make ds4-server-suite tests/libds4_suite_cpu.so test-suite-http \
  test-resources test-executor test-numeric-lifetime tests/test_runtime ds4_server_test \
  SUITE_MEDIA_PREFIX=performance-results/native-suite-t13-t14-20261021/dependencies/installed \
  SUITE_CUDA_LIBDIR=agent/.venv/lib/python3.13/site-packages/nvidia/cu13/lib \
  CUDNN_HOME=agent/.venv/lib/python3.13/site-packages/nvidia/cudnn
./tests/test_runtime
./ds4_server_test
PYTHONPATH=performance-results/native-suite-t13-t14-20261021/development \
  agent/.venv/bin/python -m unittest tests.test_suite_contracts \
  tests.test_suite_baseline_manifests tests.test_suite_artifacts tests.test_suite_api \
  tests.test_embedding_gemma2 tests.test_gemma_media -v
DS4_SUITE_HTTP_URL=http://127.0.0.1:18091/v1 \
  PYTHONPATH=performance-results/native-suite-t13-t14-20261021/development \
  agent/.venv/bin/python -m unittest tests.test_suite_http_live -v
DS4_SUITE_HTTP_URL=http://127.0.0.1:18091/v1 \
  agent/.venv/bin/python -m unittest tests.test_suite_exmekh_live -v
```

PyAV serve solo per la fixture MP4 H264; in alternativa impostare
`DS4_SUITE_HTTP_VIDEO`. Il runtime C non avvia Python/Transformers/OpenVINO.

```sh
cd exmekh_lib
.venv/bin/pytest -n 0 --no-cov -m 'not live'
.venv/bin/ruff check src
.venv/bin/pyright
cd ../agent
.venv/bin/pytest -n 0 --no-cov tests/test_native_media_preview.py \
  tests/test_native_media_preview_e2e.py tests/test_image_input.py tests/test_images_e2e.py \
  tests/test_provider_context.py tests/test_clean_review_contracts.py \
  tests/test_turn_cancellation.py tests/test_settings_lifecycle.py
.venv/bin/pytest -n 0 --no-cov tests/test_collector.py tests/test_execution_restore.py \
  tests/test_chat_history_preservation.py tests/test_recorded_history_contracts.py \
  tests/test_raw_history.py tests/test_steering.py tests/test_steering_recovery.py
```

Dipendenza sorgente esterna: `exmekh_lib`, commit base
`9e8d696ebd9cb61b1458bde73db961ffc772cdae`. La [patch completa](exmekh-lib.patch)
include file nuovi e si applica dalla sua radice. Contratto in
`exmekh_lib/docs/22-native-suite.md`. Wheel ricostruita dai sorgenti, senza
modificarne direttamente il contenuto; lock e installazione aggiornati:

```sh
cd exmekh_lib
uv build --offline --wheel --out-dir ../performance-results/native-suite-t17-t18-20261022/wheel
cp ../performance-results/native-suite-t17-t18-20261022/wheel/exmekh_lib-0.1.0-py3-none-any.whl ../agent/wheels/
cd ../agent
uv lock --offline --upgrade-package exmekh-lib
uv sync --offline --locked --reinstall-package exmekh-lib
```

Il recorder `tools/record_suite_service_qualification.py --media-directory
/tmp/ds4-suite-<id>` conserva manifest e ricevute prima dell'arresto; non
sostituisce l'esecuzione dei test.

## Diagnosi e limiti

La prova iniziale con riferimento 64×96 richiedeva un resize Qwen3-VL;
la recipe HTTP preserva la geometria. La validazione pura ora lo rifiuta
con 400 prima del compute; image-only e riferimenti multipli 256×256 passano.
La fixture MPEG4 iniziale ha incontrato un codec non compilato ed è stata
corretta a H264. Il log dei due fallimenti è conservato separatamente;
nessuna soglia, corpus o golden numerico modificato.

Aggiornati inventari e riferimenti HDS per i nuovi tool. Il benchmark prompt
storico esclude esplicitamente le capability successive alla baseline,
preservando schemi e casi. L'SDK aggiunge `shutdown_date=None` ai modelli:
il gate usa `exclude_unset` per verificare i campi realmente ricevuti.

Non qualificati qui Metal/ROCm/TP/distribuito/SSD/batch server. Il percorso
legacy è compilato separatamente e coperto dalle regressioni host, senza
dichiarare nuova qualifica hardware. Fault/recovery provati nell'executor
host; nessuna iniezione di perdita device o errore restore HTTP su GPU.
Disconnessione provata live, deadline nel protocollo host.

PNG soltanto; riferimenti nella geometria nativa ammessa. Quota, connessioni
e TTL espliciti nella discovery. Replay idempotente nella vita del processo
per 600 secondi. Contatori copie del contesto numerico nativo, esclusi kernel
e restore legacy; tempi separati Gemma `null`. Latenze descrittive, senza
promuovere speedup o qualità image T11.
