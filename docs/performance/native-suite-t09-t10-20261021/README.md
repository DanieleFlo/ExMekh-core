# T09–T10: DiT Turbo e VAE Qwen Image 2.1

Adapter C Qwen Image 2.1 Turbo, profilo `bf16-native-v1`, revisione
`fc13ba1986fcb905854dc8de2f52eee395f49b5b`. La qualifica riguarda
Linux/WSL e RTX 3090 `sm_86`, un input per lease. La CPU è reference delle
primitive. Pipeline completa, HTTP e sospensione/ripristino LLM appartengono
a T11/T17/T21.

Il [riesame successivo](review-verification.json) registra il refactoring di
observer/risultato RGBA, la validazione dei piani e le correzioni di teardown.
Il manifest `verification.json` conserva la qualifica originale precedente;
sorgenti/binari dopo il riesame sono identificati nel nuovo manifest.

## DiT e scheduler

`ds4_qwen_image_dit.h` espone input opaco, forward, generazione e stima di
memoria. Il grafo single-stream ha 32 blocchi, dimensione 4096, 32 head da
128 e FFN 12288. L'inventario dei tensori BF16 è esaustivo: checkpoint,
profilo, shape, stride e componenti incompatibili sono rifiutati prima del
forward. I 33 gruppi separano proiezioni comuni e blocchi; il percorso
qualificato mantiene tutto il DiT residente. Lo streaming resta T19.

RoPE usa assi `[16,56,56]`, frequenze FP32 congelate e moltiplicazione complex
con ordine FMA esplicito. Gli slot reference sono espansi quattro volte;
i latenti sono unpacked a 64 canali. La maschera conserva testo causale e
blocchi immagine bidirezionali. Math SDPA usa intermedi FP32 e tile di 256
query: lo storage dei punteggi è lineare nelle chiavi, anche per le head
VAE da 768/1152. Nessuna matrice quadratica della sequenza completa.

L'input preparato copia conditioning e latenti reference. Possiede K/V
detached del solo prefisso per ciascun layer; nessuna view conserva il
forward completo. Cambiare input richiede nuova preparazione. Generazione
svuota la cache all'ingresso e all'uscita, anche su errore/cancellazione.
La chiusura dell'adapter è `BUSY` finché esistono suoi input.

La recipe usa CFG 1 e gli otto sigma ufficiali
`1,.978453,.95418,.926626,.89508,.845148,.704534,.414568`, poi zero terminale.
Il checkpoint disabilita dynamic shifting e ha shift fisso 1. Timestep,
noise iniziale, prodotto delta-sigma/velocity e aggiornamento Euler rispettano
i cast BF16 dell'oracle. L'output packed normalizzato è FP32 contenente i
valori BF16 finali. Rumore esplicito permette confronti a input identico.
Il seed nativo usa SplitMix64/Box–Muller versione 1, senza stato globale o
accesso all'RNG LLM; non promette la stessa sequenza di PyTorch. Le stime
separano pesi, prefix cache, primo passo, passi cached, host e dipendenze.

## VAE, alpha e tiling

`ds4_qwen_image_vae.h` implementa VAE Qwen 2.1 a 64 canali e compressione
spaziale 16. Encode usa la moda della posterior, senza sampling implicito,
poi mean/std BF16 dei 64 canali. L'immagine conserva i shortcut temporali
del primo frame. RGB aggiunge alpha opaco; RGBA mantiene tutti i canali,
senza compositing o resize impliciti. Orientamento top-to-bottom; decode
clampa RGBA a `[0,1]`.

Encoder e decoder hanno gruppi distinti rilasciabili/ricaricabili. Il planner
sceglie full decode o core tiled secondo budget esplicito: halo 14 pixel
latenti, overlap un pixel latente e blending lineare. L'attention globale
del mid-block usa l'intero reticolo prima dei tile. Il piano include output
host, blending, readback BF16 con halo e scratch; non riduce la risoluzione.

Il risultato RGBA è solo host e non trattiene VAE o buffer GPU. Il registro
risorse deve vivere fino a `ds4_qwen_image_rgba_free`. PNG conserva alpha
con il codec C esistente; scrittura/chiusura fallite sono propagate. CRC
incrementale evita una copia dell'intero chunk. Il probe salva dopo il
rilascio dei pesi e verifica anche l'errore su `/dev/full`.

## Verifica e riproduzione

`verification.json` contiene comandi, hash, confronti e misure;
`provenance.json` documenta fonti/licenze e decisioni. Dump completi, log e
tentativi respinti restano in `performance-results/native-suite-t09-t10-20261021/`.
Le soglie T00 sono invariate. Stati interni VAE e posterior grezza sono
diagnostici: i gate T10 riguardano latenti normalizzati e pixel RGBA.

Il confronto cache verifica forward full/cached con lo stesso latente a
ogni passo. Le traiettorie Euler indipendenti sono diagnostiche: rounding
diversi nelle GEMM full/target-only si accumulano. Il caso reversed supera
il limite relativo nell'ultimo passo della traiettoria uncached (0,011841);
la recipe cached ufficiale coincide con l'oracle. L'esecuzione interamente
uncached non è un profilo alternativo qualificato. Il fallimento originale
è conservato in `qualification-v2/dit-reversed-comparison.json`.

```sh
make CUDA_ARCH=sm_86 \
  CUDNN_HOME=agent/.venv/lib/python3.13/site-packages/nvidia/cudnn \
  SUITE_CUDA_LIBDIR=agent/.venv/lib/python3.13/site-packages/nvidia/cu13/lib \
  SUITE_PYTHON=agent/.venv/bin/python \
  libds4-suite-cpu.a libds4-suite-cuda.a \
  tests/probe_qwen_image_graph test-qwen-image-graph

agent/.venv/bin/python tools/run_qwen_image_graph_oracle.py \
  --prepared performance-results/native-suite-oracles-20261020 \
  --conditioning performance-results/native-suite-t07-t08-20261020/oracle-v2 \
  --component dit --output /tmp/qwen-dit-oracle
# Stesso comando con --component vae e nuova directory per le fixture VAE.

DS4_SUITE_GRAPH_REPEATS=5 tests/probe_qwen_image_graph \
  /path/to/bundle.ds4b /path/to/model-root dit \
  /tmp/qwen-dit-oracle/dit-text /tmp/qwen-native-text
agent/.venv/bin/python tools/compare_qwen_image_graph.py \
  /tmp/qwen-dit-oracle/dit-text /tmp/qwen-native-text \
  --output /tmp/qwen-comparison.json
```

Il probe accetta più coppie fixture/output nello stesso processo; usare
`vae` per encode/decode. Il bundle completo T06 è valido; gli indici per
componente di sviluppo conservano gli hash dei file originali e riducono
solo il tempo di verifica iniziale. `--check-uncached` nell'oracle confronta
anche il full forward sul medesimo input cached. Python/Diffusers sono
strumenti offline. Dipendenze native: CUDA 13, cuBLAS/cuBLASLt 13.1.1 e
cuDNN 9.24.0. I binari LLM mantengono il link CUDA precedente.

Misure calde sincronizzate: cinque campioni dopo warmup, fuori profiler;
JSON con min/p10/mediana/media/p90/max/deviazione/CV. CV oltre 5% significa
`NEED_MORE_DATA`; nessuno speedup dichiarato. Il picco del registro misura
memoria posseduta; `nvidia-smi` include driver, librerie e altri processi.
Tempi di processo includono verifica hash/caricamento e non sono misure
cold di inferenza isolate.

Target DiT qualificati 128×192/256×256, reference sintetiche 256×256 con
conditioning reale T08; VAE 64×96/256×768. Nessuna qualifica di qualità
end-to-end, risoluzioni superiori, batching, video, quantizzazione,
streaming dei pesi, Metal, ROCm o distributed.

## Risultati locali

Ogni caso DiT ha 288 confini oracle bit-exact (32 blocchi, joint, modulation,
velocity e latente per ognuno degli otto passi), più sette confronti
full/cached a latente identico entro T00. Il controllo aggiuntivo reversed
ha sette velocity full bit-exact con il nuovo dump ufficiale uncached.

| DiT, otto passi caldi | Mediana s | CV % |
| --- | ---: | ---: |
| Testo, 128×192 | 1,6394 | 5,58 |
| Reference, 256×256 | 2,9531 | 1,67 |
| Multi-reference | 3,4964 | 1,31 |
| RGBA conditioning | 3,0159 | 1,67 |
| Reference in ordine invertito | 3,6370 | 1,88 |

Il tempo del testo è `NEED_MORE_DATA`. VAE 256×768: mediana encode 0,1390 s
(CV 4,79%), decode completo 0,3870 s (CV 4,59%). I casi VAE piccoli con CV
oltre 5% restano `NEED_MORE_DATA`; tutti i campioni sono nel JSON.

VAE wide: latente normalizzato RMSE 0,006679 e relativo 0,006456; pixel
RMSE 0,001018, SSIM RGB 0,999675, alpha MAE 0,000428. Tiled/full: RMSE
0,000890, SSIM 0,999719, alpha MAE 0,000328 e alpha sui bordi 0,000330.
Tutti rispettano T00. Le feature interne e la posterior grezza wide
superano alcuni ceiling diagnostici; i rispettivi errori restano nel JSON.

Picco numerico posseduto del processo DiT: 14.728.585.216 byte, con
14.230.249.472 byte di pesi e prefix cache massima 285.212.672 byte.
VAE: 896.741.056 byte. Dopo ogni caso entrambi tornano agli 8.519.680 byte
del workspace numerico iniziale; dopo close il registro è vuoto. Picchi
globali campionati: DiT 15.888 MiB, VAE 2.447 MiB.

Passano primitive CPU/CUDA (14 test ciascuno), grafo/RNG/mask (3), bundle
(11), lifecycle numerico anche ASan/UBSan, input/risorse/runtime/executor,
codec immagini e build dei quattro binari legacy. La regressione T08
confronta nuovamente testo, reference e multi-reference dopo la selezione
dei descrittori cuDNN immagine.
