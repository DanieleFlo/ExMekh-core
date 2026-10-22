# T00–T02: baseline, contratti e runtime CUDA

20 ottobre 2026. T00 e T01 completati; T02 implementato e verificato sul
percorso CUDA single GPU, con qualifica degli altri backend ancora
`NOT_VERIFIED`. La roadmap mantiene T02 `IN_PROGRESS` per non confondere
questa evidenza con una verifica di Metal, ROCm o TP/distribuito.

I kernel nativi Qwen Image/Gemma e i nuovi endpoint non fanno parte di questa
consegna. Python/Transformers/Diffusers sono usati solo dagli oracle di sviluppo.
Il server C conserva le proprie dipendenze e i modelli predefiniti.

## Evidenze congelate

Rettifica del 22 ottobre: il corpus Gemma r2 ereditava da PyTorch le riduzioni
interne BF16, mentre il profilo nativo richiede accumulo FP32. Il
[report T15/T16](../native-suite-t15-t16-20261022/README.md) conserva la diagnosi,
i confronti FAIL contro r2 e un riferimento FP32 separato sullo stesso corpus.
Gli artefatti storici e tutte le soglie di questo report rimangono invariati.

| Area | Evidenza |
| --- | --- |
| Sorgenti e binari originali | [sources.json](sources.json): HEAD, hash dei file C/backend, otto repository con SHA fisse e file consultati, binari e modelli locali |
| Artefatti ufficiali | [models.json](models.json): revisioni HF, hash LFS pubblicati distinti dagli SHA calcolati; pesi ufficiali scaricati e verificati localmente; tokenizer/processor/template inclusi |
| Tensor reali | [tensors.json](tensors.json): directory safetensors con nomi, dtype, shape e offset. Qwen: 1285 tensor in quattro file; Gemma: 1376 tensor. Tutti BF16 |
| Hardware | [hardware.json](hardware.json): RTX 3090 24 GiB, driver e toolkit 12.4, RAM WSL e memoria disponibile, campioni PCIe/SSD; probe pinned da 64 MiB e staging legacy osservato da 4×64 MiB; non un massimo fisico |
| Qwen27 originale/candidate | [baseline](qwen27-baseline.json), [candidate](qwen27-candidate.json): quattro casi, nove righe full-vocabulary e payload per caso; tutti i confronti bit-exact, tutti i logits finiti |
| Altro LLM locale | [flash.json](flash.json): Flash con profilo tiered del launcher, logits bit-exact rispetto al benchmark originale. [flash-payload.json](flash-payload.json): snapshot aggiuntivo dopo T02, etichettato senza attribuirlo al binario originale |
| Gemma oracle | [BF16](gemma-oracle.json) e [FP32](gemma-f32-oracle.json): testo, immagine, video, audio e interleaved; warm-up e cinque campioni per profilo; differenza massima fra ripetizioni 0 |
| Qwen oracle | [qwen-oracle.json](qwen-oracle.json): 512×512 RGBA, otto step, CFG 1, CPU offload; warm-up e cinque campioni; differenza massima fra ripetizioni 0 |
| Corpus e ranking | [retrieval.json](retrieval.json): 104 elementi, vettori e ranking a 128/256/512/768 dimensioni; split separati. Encoder sensibili ai media: 16/16 immagini diverse, 16/16 audio, 8/8 video |
| Rumore iniziale | [noise.json](noise.json): due rumori per ogni shape held-out, array conservati; [calibrazione](noise-calibration.json) bit-exact contro prepare_latents ufficiale |
| Soglie | [tolerances.json](tolerances.json): primitive, forward, SSIM/RGBA/alpha e regressione retrieval; congelate prima del porting dei kernel |
| Provenienza | [provenance.json](provenance.json): licenze e SHA dei file consultati; nessun kernel esterno importato da questi task |
| Verifica finale | [verification.json](verification.json): hash dei sorgenti/binari finali, comandi, log e matrice di qualifica |
| Array e payload voluminosi | [artifacts.json](artifacts.json): percorsi locali e SHA. I dati completi restano in performance-results, non nei sorgenti del server |

Il corpus versionato è `tests/fixtures/native-suite-v1/corpus.json`: 48 testi/codice,
16 immagini, 8 video come frame canonici con FPS, 16 audio PCM (8 voci
sintetizzate e 8 suoni/silenzi), 16 input interleaved. Contiene 24 recipe Qwen
held-out × due rumori, con editing singolo/multiplo e alpha. I media hanno
hash e appartenenza allo split verificati. I video canonici non qualificano
ancora un decoder MP4/WebM del server, che appartiene a T15/T17.

Le query enumerate sono 192, 32 per direzione/collegamento. Le query inverse
riutilizzano i media del corpus: i conteggi non sono un claim di 32 media
indipendenti per ogni direzione. Questo corpus misura regressioni sintetiche,
non la qualità del modello su benchmark reali. La valutazione image held-out
completa e l'accettazione delle implementazioni native appartengono a T11/T16/T21.

Il profilo Gemma dell'oracle è `bf16-eager-mean-f32-l2`: forward BF16,
pooling mask-aware e normalizzazione FP32, con prompt inclusi. I dump conservano
anche hidden state grezzi e vettori prima della normalizzazione. Non equivale
a una qualifica di ogni implementazione SentenceTransformers o di FP16.
Il profilo FP32 ha un proprio oracle; il [confronto fra dtype](gemma-cross-dtype.json)
è diagnostico, separato dai gate per una futura implementazione nativa. I cosine
minimi BF16/FP32 sono compresi fra 0,9998038 e 0,9999776; gli hidden state
mostrano differenze maggiori, conservate nel confronto senza promuoverle a PASS.

La pipeline Qwen di riferimento ha consumato un picco PyTorch di 17,59 GB;
Gemma BF16 2,32 GB e FP32 3,69 GB. Sono picchi dell'oracle, esclusi allocazioni driver e stato
del server: non costituiscono le stime di ammissione del futuro scheduler.
I tempi dell'oracle e le bande osservate non sono risultati prestazionali dei
kernel nativi DS4 né una promessa di latenza su altre macchine.

La [revisione clean code](../../architecture/native-suite-clean-code-review-20261020.md)
registra refactoring, comportamenti corretti e debito residuo. Gli output dei
probe sono ora validati per presenza, shape, payload e finitezza prima di
essere confrontati o promossi; un exit code 0 da solo non basta.

## Riproduzione

I tool rifiutano directory di output già usate. Non rigenerare golden o
tolleranze in seguito a un fallimento. Le revisioni effettive sono nei manifest,
non nei branch correnti di Hugging Face/GitHub.

```sh
make tools/probe_suite_baseline tests/test_runtime tests/test_runtime_cuda CUDA_ARCH=sm_86 -j2
make test-runtime test-suite-contracts
python3 -m unittest tests.test_product_text_configuration -v
# NumPy richiesto solo dai tool di sviluppo, come indicato in tests/requirements-suite.txt.
python -m unittest tests.test_suite_numeric_metrics -v
python3 tools/run_suite_baseline.py --output performance-results/new-candidate \
  --compare performance-results/native-suite-t00-20261020-r2
./tests/test_runtime_cuda gguf/Qwen3.8-27B-UD-Q4_K_S.gguf
```

Eseguire i processi modello in sequenza. La baseline originale completa è in
`performance-results/native-suite-t00-20261020`; il tentativo iniziale vision
con ctx 512 fallito è conservato, non promosso. La baseline valida usa ctx
2048 ed è in `native-suite-t00-20261020-r2`. I risultati finali CUDA sono in
`native-suite-review-final-qwen-20261020`; Flash ha un run separato
in `native-suite-review-final-flash-20261020`. Callback/snapshot MTP, alternanza
Qwen27/Flash/Qwen27, diagnostic vision ripetuto e ASan/UBSan hanno log dedicati
in `native-suite-review-20261020`, elencati in `verification.json`. Il test runtime verifica anche il lock OS prima e dopo
la chiusura dell'ultima reference, incluso un servizio senza modelli e il rifiuto del diagnostic vision standalone
mentre un runtime è attivo. La chiusura e riapertura attraversano anche
due varianti LLM reali, Qwen27 e Flash, preservando i tensori dei componenti.

Gli oracle richiedono il Python di sviluppo con PyTorch. Preparazione isolata:

```sh
python tools/prepare_suite_oracles.py --manifest <freeze>/baseline.json \
  --output performance-results/new-oracles --qwen
uv pip install --python <development-python> --target performance-results/new-oracles/deps \
  --no-deps tokenizers==0.23.1 accelerate==1.15.0 soxr==1.1.0
python tools/run_suite_oracle.py --prepared performance-results/new-oracles \
  --output performance-results/new-gemma --model gemma
python tools/run_suite_oracle.py --prepared performance-results/new-oracles \
  --output performance-results/new-gemma-f32 --model gemma --dtype float32
python tools/run_suite_oracle.py --prepared performance-results/new-oracles \
  --output performance-results/new-qwen --model qwen
python tools/score_suite_gemma_corpus.py --prepared performance-results/new-oracles \
  --corpus tests/fixtures/native-suite-v1/corpus.json --output performance-results/new-retrieval
```

L'ambiente misurato usa Python 3.13, torch 2.14.1+cu130, NumPy 2.5.3;
la validazione schema usa jsonschema 4.10.3. I sorgenti Transformers/Diffusers
sono importati dalle SHA archiviate, non installati sopra l'ambiente dell'agente.
`run_suite_oracle.py` accetta anche `--corpus`, `--recipe`, `--seed` e
`--latents` per riprodurre le recipe held-out e i rumori congelati.

## Limiti di completamento

T02 non abilita due GGUF legacy contemporanei. Il runtime può avviarsi senza
LLM; un engine può essere chiuso e ricollegato esplicitamente mentre altri
componenti restano vivi, conservando stream e cuBLAS. L'eviction dei soli pesi
con engine/sessioni vivi e il ripristino automatico della transazione sono T04/T05.
I nuovi handle componente e i loro tensori sono indipendenti dal caricatore legacy. Il lifecycle e l'inventario
dei globali sono descritti in `docs/architecture/native-suite-runtime-lifecycle.md`.

Metal, ROCm e TP/distribuito non hanno ricevuto un PASS locale. Il task resta
aperto per quella verifica e per eventuali adeguamenti richiesti dai relativi
risultati. Non sono stati eseguiti test distribuiti. Non sono ancora qualificati
modelli nativi image/embedding o i loro profili quantizzati.

I pesi Qwen Turbo della revisione verificata dichiarano `qwen-research`;
EmbeddingGemma 2 dichiara Apache-2.0. Unsloth Studio dichiara AGPL-3.0-only,
distinta dai kernel Unsloth Apache-2.0. Il registro conserva questa distinzione:
non è stato copiato codice Studio nel runtime DS4.
