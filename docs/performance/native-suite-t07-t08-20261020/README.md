# T07–T08: primitive native ed encoder Qwen Image

Qualifica locale: Linux/WSL, RTX 3090 `sm_86`, profilo BF16 ufficiale,
revisione `fc13ba1986fcb905854dc8de2f52eee395f49b5b`. Le API C sono utilizzabili
da `libds4-suite-cpu.a` e `libds4-suite-cuda.a`. La reference CPU copre le
primitive; il forward completo è qualificato su CUDA, una richiesta per lease.
DiT, VAE completo e integrazione HTTP restano nei task successivi.

## Implementazione

`ds4_numeric.h` espone buffer con ownership, view row-major e operazioni
BF16/FP32 con accumulazione FP32: GEMM, norm diretta/zero-centered/senza scala,
GELU esatta/tanh, SiLU e gating, RoPE split-half/adiacente, attention GQA/MQA,
convoluzioni, reorder, pooling indicizzato e nearest-exact. I confini di
casting sono espliciti; FP16 e TF32 non sono fallback. La conversione bounded
supporta F32/BF16/Q8_0/Q4_0; altri candidati quantizzati richiedono T20.

L'attention FP32 usa softmax online senza matrice dei punteggi. BF16 conserva
i rounding QK/probabilità dell'oracle con tile di 256 query, K/V impacchettati
e scratch lineare nelle chiavi. Bounds per query rappresentano padding,
causalità, blocchi, segmenti e finestre. Audio dispone anche di termine
relativo e softcap. Le convoluzioni usano l'API C cuDNN, algoritmi
deterministici, accumulazione FP32 e workspace ammesso fino a 64 MiB.
La reference CPU è indipendente.

`ds4_qwen_image_encoder.h` implementa il backbone dedicato di 36 layer,
vision di 27 layer, mRoPE `[24,20,20]`, merger e tre deepstack. Valida tutti
i 750 tensori prima del forward. I 68 gruppi separano i pesi usati da
embedding, norm finale e testa autoregressiva: gli embedding vengono raccolti
dallo storage mmap; norm finale e `lm_head` non sono eseguiti né caricati GPU.

Il tokenizer image ha regex e classi Unicode proprie, byte BPE e NFC della
versione congelata. Unicode 9 per NFC e le classi del motore regex
`tokenizers 0.23.1` sono dati dedicati: le classi legacy differiscono su 4.657
scalari. Template e sistema sono in `config/qwen-image-encoder.json`, compilati
offline nell'asset verificato. Il percorso C non avvia Python.

Il preprocessing usa bicubica antialiased separabile con intermedio uint8,
ties-to-even, ordine esatto delle patch e normalizzazione FP32. L'alpha è
composto sul bianco in una copia solo per vision; prompt e reference originali
rimangono intatti e nell'ordine di ingresso.

`ds4_qwen_image_prepare_padded` rende esplicito il padding a sinistra senza
troncamento: pad esclusi prima dei 14 token di sistema. Il checkpoint usa
posizioni fisiche arange nel testo puro e mRoPE sui token validi nel
multimodale; il codice conserva la distinzione. Il conditioning precede la
RMSNorm finale. Il controllo pre/post norm delle fixture ha distanza relativa
0,88–0,94 e distingue nettamente i due output.

## Lifetime e memoria

Bundle, contesto numerico e runtime vivono più a lungo dell'adapter. L'input
possiede prompt, token, posizioni, griglie e pixel copiati. Il risultato
possiede buffer BF16 e image mask: resta valido dopo il rilascio dei pesi e
la chiusura dell'adapter, finché vive il contesto numerico. Il chiamante
libera input e risultato con le rispettive API.

Il riesame del 21 ottobre rende opachi entrambi i proprietari: i client usano
`ds4_qwen_image_input_data` e `ds4_qwen_image_conditioning_data`, snapshot
borrowed che non espongono buffer posseduti o record del registro. Il modulo
`ds4_qwen_image_input` possiede la preparazione; `ds4_numeric_weights` adatta
il protocollo dei gruppi attraverso i metodi pubblici del core numerico.

I pesi restano residenti fra richieste; `release_weights` li rilascia dopo
una fence. Il workspace è della richiesta; i deepstack vivono fino all'ultimo
uso nel testo. I lease impediscono eviction da callback durante il forward.
La preparazione riserva memoria host prima delle allocazioni grandi; buffer
GPU, workspace cuBLAS/cuDNN e metadata posseduti hanno accounting.
Le allocazioni opache di driver/librerie restano nella misura device esterna.

Un errore fence/free conserva gli handle: `ds4_num_free` mantiene l'ownership
del chiamante; `ds4_num_dispose` trasferisce i temporanei al contesto fino a
`ds4_numeric_recover`/close riusciti. Non ripete il compute fallito. Nessuna
distruzione resetta CUDA o modifica cache e flag del percorso LLM.

Il registro autorizza ora il distruttore dello storage soltanto dopo l'evento
ereditato dalla prenotazione. Un'allocazione fallita conserva anche i propri
metadata pendenti fino al recupero. Le operazioni numeriche preservano la
coda di un output sovradimensionato, rifiutano overlap parziali e richiedono
`CAST` per cambiare dtype. La cancellazione viene ricontrollata dopo le
callback e prima di pubblicare il conditioning.

Le verifiche dopo il refactoring sono in `review-verification.json` e nella
[revisione di responsabilità](../../architecture/native-suite-clean-code-review-20261020.md).

## Build e riproduzione

Il profilo qualificato usa runtime CUDA 13, cuBLAS/cuBLASLt 13.1.1 e cuDNN
9.24.0, librerie native caricate dal processo C. Localmente sono installate
nelle directory NVIDIA dell'ambiente di sviluppo; i percorsi possono indicare
un normale SDK. I target LLM conservano le librerie CUDA 12. Sostituire le
dipendenze del binario LLM richiederà un gate d'integrazione proprio.

```sh
make CUDA_ARCH=sm_86 \
  CUDNN_HOME=agent/.venv/lib/python3.13/site-packages/nvidia/cudnn \
  SUITE_CUDA_LIBDIR=agent/.venv/lib/python3.13/site-packages/nvidia/cu13/lib \
  libds4-suite-cpu.a libds4-suite-cuda.a \
  tests/probe_qwen_image_encoder tests/libds4_numeric_cuda.so \
  tests/libds4_numeric_cpu.so tests/libds4_image_suite.so \
  tests/libds4_qwen_image_tokenizer.so test-numeric-lifetime test-qwen-image-input

PYTHONPATH=performance-results/native-suite-oracles-20261020/deps \
  agent/.venv/bin/python tools/compile_qwen_image_tokenizer.py \
  performance-results/native-suite-oracles-20261020/models/Qwen/Qwen-Image-2.1-Turbo/processor/tokenizer.json \
  performance-results/native-suite-t07-t08-20261020/tokenizer.ds4q

DS4_SUITE_NUMERIC_CUDA=1 agent/.venv/bin/python -m unittest tests.test_suite_numeric -v
```

L'oracle di sviluppo `tools/run_qwen_image_encoder_oracle.py` produce fixture
e dump completi. Il probe C riceve index, root, tokenizer e coppie
fixture/output nello stesso processo. `DS4_SUITE_ENCODER_CHECKS=1` attiva
diagnostica di budget, cancellazione, reload, lifetime e padding;
`DS4_SUITE_ENCODER_REPEATS=5` misura cinque forward senza trace dopo il warmup.
`tools/compare_qwen_image_encoder.py` legge le soglie T00 senza modificarle.
`--native-padding-reference` nell'oracle verifica `*-padded` alla stessa shape.

## Evidenze e limiti

`verification.json` conserva hash, risultati e misure; `provenance.json`
registra fonti e scelte. Tensori completi e log, inclusi gli esperimenti
respinti, restano in `performance-results/native-suite-t07-t08-20261020/`.

Un batch BF16 con due prompt non sostituisce automaticamente due forward
singoli: il controllo aggiuntivo ha misurato differenze oltre T00. Il confronto
fallito è conservato; non allarga le soglie e non promuove batching non
implementato. L'API accetta un input per forward. Batching/concorrenza
necessitano qualifica propria.

I benchmark hanno cinque ripetizioni, statistiche e CV. CV oltre il 5%
implica `NEED_MORE_DATA`; nessun guadagno di velocità è dichiarato. Il picco
del registro conta storage/workspace posseduti, non il totale fisico GPU.
Metal, ROCm e distributed sono `NOT_VERIFIED`. Gemma completo, generazione
DiT/VAE e quantizzazione restano nei task successivi; le primitive comuni
sono verificate indipendentemente.

## Risultati del gate

| Caso | Confini | Bit-exact | Mediana forward caldo | CV |
| --- | ---: | ---: | ---: | ---: |
| text | 38 | 38 | 62.64 ms | 1.3% |
| reference | 69 | 69 | 657.52 ms | 10.5% |
| multi | 69 | 69 | 1049.40 ms | 5.7% |
| reversed | 69 | 69 | 1084.47 ms | 4.5% |
| rgba | 69 | 69 | 658.86 ms | 4.3% |
| resized | 69 | 69 | 731.56 ms | 4.2% |
| unicode | 38 | 38 | 58.53 ms | 4.8% |
| empty | 38 | 38 | 50.43 ms | 1.6% |

Picco storage/workspace posseduto: 15,087,555,040 byte (14.051 GiB).
Padding a 64 token: token/posizioni e conditioning bit-exact nei due controlli.
Primitive: 165 confronti CPU e 165 CUDA; 14 workload di benchmark con output completi.
Regressioni Qwen27 testo/vision/MTP: quattro casi bit-exact rispetto a T00; CUDA SSD PASS.
ASan/UBSan: lifetime numerico e 11 test loader/converter PASS. Tokenizer/NFC, pixel/resize, runtime/resources/executor e build sm_86 PASS.
I tempi escludono apertura/hash, upload iniziale e preprocessing; misurano forward sincronizzati con pesi residenti. CV oltre 5% resta NEED_MORE_DATA.

Il riesame del 21 ottobre ripete il gate dopo il refactoring: 459 confini
bit-exact negli otto casi, padding bit-exact e 169 confronti numerici per
backend in 12 test CPU e 12 CUDA. Passano i nuovi test di ownership e degli
eventi pendenti, ASan/UBSan, contratti host/server, quattro regressioni LLM e
CUDA SSD. La suite completa dell'agente registra 1128 test passati, 44 saltati
e copertura del 100%; i test live opzionali saltati restano non verificati.
Hash, comandi e risultati sono in [review-verification.json](review-verification.json).
La tabella sopra conserva le misure del gate iniziale; nessuna accelerazione
è dichiarata dal riesame.
