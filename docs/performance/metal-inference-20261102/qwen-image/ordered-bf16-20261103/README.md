# I-P01 — Proiezioni BF16 ordinate su Metal

Flash Q2 e Qwen27 hanno già chiuso i cicli locali della roadmap. Questa prova
apre il caso Qwen Image richiesto, con un solo modello caricato. Baseline
`f2fed3e`, Qwen-Image-2.1-Turbo BF16, revisione pesi
`fc13ba1986fcb905854dc8de2f52eee395f49b5b`, 256×256, otto passi, CFG1, seed
20261020, M5 Pro 24 GiB, macOS 26.6.2. Cap device comune 7,5 GiB per confrontare
la stessa residenza, sotto la protezione host di 2 GiB. Binario, bundle e fixture
congelati in `build/metal-perf-image-20261103/baseline-sha256.txt`.

## S1 — Evidenza e fonti prima del codice

Due campioni host di 8 s a 10 ms. Il primo attraversa l'encoder e l'inizio
del DiT, con 215/657 campioni in `nm_gemm` e 402 in `memmove`. Il secondo
cade nel DiT, con 469/655 campioni in `nm_gemm` e 170 in `memmove`. Le tre
proiezioni MLP (`ds4_qwen_image_dit.c:block`) pesano 373 campioni, in attesa
di Metal. Il campionamento attribuisce lo stack host e include la latenza di
submit e completamento. I tempi delle fasi si misurano fuori dal profiler.

Fonti rilette il 3 novembre 2026 ai pin locali. Nessun codice importato.

- [Diffusers](https://github.com/huggingface/diffusers/tree/aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e)
  `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
  `src/diffusers/models/transformers/transformer_qwenimage21.py` (proiezioni
  separate in `QwenImage21SwiGLUFeedForward.forward`, QKV e maschere nel blocco,
  prefix cache in `QwenImage21AttnProcessor`),
  `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py:__call__`
  (otto passi della recipe locale, cast per VAE, KV con causal_condition),
  `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`
  (`decode`, Conv3d causali). Il prefisso è già riusato nel nostro DiT e
  le convoluzioni VAE sono un'operazione distinta.
- [Transformers](https://github.com/huggingface/transformers/tree/d4fcbccd0966ace9505b9cbd3041d28ad822d8a7)
  `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
  `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
  (`Qwen3VLTextMLP.forward`, `Qwen3VLTextAttention.forward`, proiezioni
  e normalizzazioni separate, maschera e MRoPE conservati).
- [ATen](https://github.com/pytorch/pytorch/tree/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e)
  `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`,
  `aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp`,
  `dot_with_fp32_arith_main_loop_no_bfdot` e coda. I 32 flussi FP32
  indipendenti permettono di distribuire il lavoro tra lane mantenendo ogni
  FMA e l'albero delle somme. La coda, il bias e l'uscita BF16 fissano
  altre frontiere da conservare.
- [Strata](https://github.com/Niko1221/Strata/tree/46c152a1310f77cd2f0ce2c7cd213962df14fe83)
  `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/kernels/dense_mm.metal:dq_f16`,
  `metal/src/mtl_rt.mm:Context::has_family_apple/supports_metal4`,
  `metal/tests/dense_mm_parity.cpp:run_all`. Il wrapper di prova prepara
  tile e verifica la geometria e la precisione degli input arrotondati a FP16.
  Trasferibile il riuso degli operandi, incompatibili conversione e MMA
  con l'ordine ATen. La prova è sulle proiezioni dense dei pack LLM.
- [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp/tree/dde879bc4f954b809892fd0d660515f42536394a)
  `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/kernels/mul_mm.metal:kernel_mul_mm` (tile e MMA),
  `ggml-metal-ops.cpp:ggml_metal_op_mul_mat` (routing GEMM e geometria),
  `ggml-metal.cpp:ggml_backend_metal_device_supports_op`,
  `tests/test-backend-ops.cpp:test_mul_mat`. Riutilizzo dei carichi utile,
  confronto con tolleranze e ordine MMA diverso. Il pin copre primitive e
  Qwen3-VL, senza grafo DiT 2.1. Il grafo è presente in
  [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/tree/c6dfa9def12fd0cb36ac7149ba30a75792bedbf0)
  `c6dfa9def12fd0cb36ac7149ba30a75792bedbf0`,
  `src/model/diffusion/qwen_image_2_1.hpp` (layout causale e cache prefisso).
- Il profilo Hirai elenca una repo pubblica. `apply_l25l26.py` alla revisione
  `ad5ca05d0e6433a0e1b4b54fb81320e29d7b95f3` moltiplica due scalari di
  Gemma 4 31B per accuratezza. Mancano sorgenti del motore YASAKANI e una specifica
  verificabile di .mgtm. I 380,6 token/s dichiarati riguardano RTX 5090 e
  quella rappresentazione. La scelta di formati per tensore può ridurre
  byte su SSD per Qwen27, ma richiede una campagna separata di qualità su
  nuovi pesi. Flash Q2 usa già quantizzazione mista. Per Image, il profilo
  BF16 e le frontiere della recipe escludono una patch dei pesi in I-P01.
  [Profilo](https://github.com/morphicode-jp),
  [script](https://github.com/morphicode-jp/f32-patch-gemma/blob/ad5ca05d0e6433a0e1b4b54fb81320e29d7b95f3/apply_l25l26.py),
  [annuncio del 31 ottobre](https://x.com/yasakani_dev/status/2116372088420536611).
- [Ricerca numerica locale](../../../metal-apple-silicon-20261031/t06-t07/upstream-numerics.md),
  `metal/numeric_ordered.metal:nm_bf16_linear,nm_bf16_attention_dot`,
  `ds4_numeric_metal.m:nm_gemm`, `ds4_qwen_image_dit.c:block`. L'attention
  distribuisce già gli stessi 32 parziali con shuffle espliciti. I replay
  precedenti hanno rifiutato SIMD matrix BF16 e MPS per i tie CPU.

Scelta: tile di quattro righe per due colonne, con un SIMD-group per tile
e riuso di entrambi gli operandi. Un lane per ciascuna catena FMA, albero
16/8/4 più le due coppie finali. Si conserva anche l'addizione della coda
zero per K multiplo di 32. Si usa il percorso scalare per K con coda, N
dispari, proiezioni piccole e layout trasposti. Le alternative di prefetch
aumentano le pagine live e richiedono un intervento indipendente sul cache
owner. MMA e riduzioni non ordinate richiederebbero cambiare il gate.

Gate isolato: stessa matrice e bias su baseline, candidato e CPU,
incluse dimensioni reali, riga di coda e shape di fallback. Gate completo:
fixture testo, RGB e RGBA congelate e confronto con oracoli indipendenti
già generati, previa verifica dell'hash di input/rumore. Misure appaiate e
nuovo profilo solo dopo il gate isolato.

## S2 — Replay della primitiva

Otto forme BF16 con input deterministici, incluse tre forme DiT da 256
righe, una coda di 1 riga con bias e tre fallback per K non allineato, N
dispari e pochi token. Entrambi i binari coincidono con PyTorch CPU 2.14.1
su tutti gli elementi e hanno gli stessi SHA256 di output. Riletto il main
loop e l'epilogo ATen dopo la parità. Il tile conserva ogni accumulatore
della variante scalare e il punto in cui il bias entra nella somma.

Il primo screening caricava entrambe le librerie nello stesso processo,
generando un avviso di classe Objective-C duplicata. È conservato come
diagnostica, ma il replay accettato usa un processo per variante. Tre
warmup e sette chiamate sincrone con gli operandi già caricati.

| Righe × K × N | Baseline, mediana ms | Tile, mediana ms |
| --- | ---: | ---: |
| 256 × 4096 × 4096 | 20,055 | 9,754 |
| 256 × 4096 × 12288 | 116,650 | 30,378 |
| 256 × 12288 × 4096 | 112,257 | 31,983 |

I casi piccoli, inclusi i fallback, variano tra processi. Il loro tempo
serve per il controllo funzionale, non per attribuire un guadagno. I
campioni completi sono in `isolated-before.json` e `isolated-after.json`
nella directory di lavoro.

## Piano di conferma

Il primo testo candidato attraversa due stop di macOS (19:10:43–19:26:55 e
19:27:40–19:38 circa). Il run supera il gate indipendente e tutti gli undici
artefatti baseline coincidono byte per byte, ma i tempi sono esclusi dalla
conferma. Gli stop sono conservati nel log pmset. Il timer del runtime usa
CLOCK_MONOTONIC, mentre le sospensioni e il completamento GPU possono
produrre finestre distorte. Nessun tempo viene corretto sottraendo lo stop.

La conferma usa caffeinate per la sola durata delle prove e un monitor host
esterno. Il piano, fissato prima del primo blocco non tracciato, prevede una
coppia di warmup (`text-before-2`/`text-after-2`), poi cinque coppie
alternate (`text-*-3..7`). Due casi editing separati controllano numerica
RGB e RGBA. Il profilo candidato RGB rimane diagnostico e separato dalle
coppie testo. Confrontare le tre fasi e il tempo dell'intera operazione,
oltre al tempo del processo. Tutti i campioni e gli esiti vengono conservati.

## S3 — Dispatch e profilo dopo il kernel

Riletti gli stessi wrapper di Strata e ggml del confronto S1. La geometria
GEMM di Strata ammortizza la dequantizzazione su chunk di prompt, mentre
qui i pesi sono già BF16. Le MMA di ggml accumulano matrici e hanno criteri
di dispatch propri. Entrambi suggeriscono riuso dei carichi tra output,
ma il loro accumulo resta incompatibile con il gate CPU di questo caso.
Il nostro dispatch seleziona il tile per almeno 64 righe, K/N almeno 1024,
K multiplo di 32, N pari e layout non trasposto. Il backend richiede già
SIMD32 e almeno 256 thread per threadgroup per tutte le pipeline. Il
calcolo del numero di tile usa uint64 e controlla il limite della griglia
prima di moltiplicare per 32. Le righe parziali rileggono solo righe valide
e scrivono soltanto gli output esistenti.

Il campione RGB candidato ha 236/711 stack nelle proiezioni e 434 in
memmove. Il campione baseline S1 era testo, quindi questi conteggi indicano
il prossimo hotspot senza costituire una percentuale di accelerazione
appaiata. Un ulteriore profilo candidato testo, a cap comune, è previsto
dopo le cinque coppie senza profiler.

La coppia di warmup misura 204,708 → 125,629 s sull'operazione. Il DiT passa
da 136,434 a 57,099 s, encoder 14,059 → 14,286 e VAE 54,215 → 54,243 s.
Stessi 83.620.429.176 byte logici H2D, stessi 159 caricamenti DiT e stesso
limite pesi DiT di 7.115.124.736 byte. I tempi di warmup sono separati
dalla distribuzione di conferma.

Un caso operativo finale usa il budget automatico del backend e la riserva
host di 2 GiB. È separato dal confronto a cap comune. La policy Image
sceglie tra piani di cache dei pesi già esistenti. Le misure di questo caso
quantificano il comportamento di quell'ingresso normale con la RAM del
momento, senza attribuire al kernel un eventuale cambio di cache.

## S4 — Conferma sul modello completo

Riletti Strata `dense_mm.metal` e `dense_mm_parity.cpp` allo stesso pin,
oltre a ggml `test-backend-ops.cpp:test_mul_mat`. Il tiling e il riuso dei
carichi restano trasferibili, mentre conversioni FP16, MMA e test con
tolleranza richiedono criteri diversi dal nostro replay ordinato. I-P01
usa le 32 catene FP32, l'albero originale e il round BF16 dell'epilogo.
I confronti con questi runtime sono statici. Le misure seguenti confrontano
esclusivamente i nostri due binari sullo stesso Mac.

Una coppia di warmup, poi cinque coppie alternate senza profiler.
Tutti i 110 artefatti delle dieci generazioni sono bit-exact contro la
baseline nativa. Stesse fixture, limite, working set pianificato,
caricamenti e byte H2D. Il monitor osserva almeno 4,801 GiB host
disponibili e zero crescita dello swap.

| Fase | Mediana baseline s | Mediana I-P01 s | Delta geometrico appaiato, bootstrap95% |
| --- | ---: | ---: | --- |
| Encoder | 14,606 | 14,116 | −0,80% [−2,35; +0,53] |
| DiT | 138,127 | 61,629 | −55,33% [−57,32; −53,24] |
| VAE | 54,596 | 54,198 | −0,22% [−0,53; +0,004] |
| Operazione completa | 207,581 | 129,852 | −36,90% [−38,46; −35,29] |
| Processo completo | 220,863 | 143,110 | −34,60% [−36,11; −32,99] |

CV dell'operazione 2,90% / 1,81%, DiT 3,99% / 3,63%. Medie, p10/p90,
deviazioni e tutti i campioni sono in `performance.json`. Il processo
include avvio e scrittura dei dump. I tempi della pipeline includono
calcolo, caricamenti e gestione nativa. La prova usa 256×256 e una recipe
di otto passi, CFG1 e seed20261020. Le metriche descrivono questo subset.

Il riesame successivo introduce il fallback di capability e porta il
dispatch dentro il proprietario delle pipeline. La sorgente dello shader
resta identica a I-P01. Le verifiche e i controlli del binario riesaminato
sono nel [riesame dell'API Metal](../../../../architecture/metal-image-ordered-projection-clean-code-review-20261103.md).

## Hotspot residui

Il nuovo campione testo DiT contiene 216 stack nelle proiezioni e 429 nella
copia dei pesi. Il campione VAE contiene 415 stack in `nm_conv` sui 418
dell'owner decode. Sono finestre distinte, conservate in `profiles.json`;
la quota host include l'attesa del lavoro GPU e non si somma al tempo
delle fasi. Il VAE pesa circa 54 s sui 130 s dell'operazione del candidato.

Le direzioni successive sono il kernel Conv BF16 ordinato del VAE e il
trasporto dei gruppi di pesi DiT. Entrambe richiedono un audit dedicato e
un confronto sul tempo della richiesta, inclusi buffer, caricamenti e
confini BF16. La policy Image seleziona piani discreti di residenza. Un
eventuale budget continuo va studiato sul relativo owner e sul picco
della pipeline, con la riserva di 2 GiB. Questi interventi restano nel backlog.

Il caso operativo con budget automatico sul binario riesaminato passa in
129,570 s, con stessi output e con margine host minimo 5,307 GiB e swap
invariato. La RAM del momento ammette lo stesso livello discreto della
prova a cap fisso, con cache DiT di 7.115.124.736 byte. Questo risultato è
registrato separatamente in `automatic-budget.json`.

## Integrazione e riesame

Il riesame completa tre nuove pipeline indipendenti testo/RGB/RGBA,
con altri 35 artefatti nativi bit-exact. Otto forme per quattro varianti
coincidono con CPU, incluso il tile indisponibile nella fixture di
capability. Gemma multimodale passa 480/480 frontiere e i vettori.
Numeric conserva due asserzioni sin/cos preesistenti, riprodotte anche
sulla baseline. La regressione del nuovo tile passa.

Il primo run testo riesaminato era più lento nel DiT. Due coppie dirette
alternate misurano operazione 127,536/127,558 s e 127,308/125,002 s
(I-P01/riesame), con delta +0,017% e −1,812%, tutti gli output esatti.
Il rallentamento isolato non si ripete. Il rapporto conserva i limiti
delle due coppie, senza attribuire un guadagno al refactoring.

Il kernel e la regressione sono nel commit `9634c83`. La selezione nel
proprietario delle pipeline e il fallback di capability sono nel commit
`689c131`. Dati, fonti e hash dei binari conservano separatamente I-P01 e
il binario finale riesaminato.
