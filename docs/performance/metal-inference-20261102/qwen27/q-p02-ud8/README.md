# Q-P02: riuso delle unità UD su otto righe prefill

Stato finale: **REJECT del dispatch generalizzato**. Scheda S1 redatta
prima del codice il 2 novembre 2026.
B0/Bi: `089255a1d3b6162e59c61a6bede9ece5e672330f`, binario e shader
congelati in `build/metal-perf-qwen27-20261102/`. Nessun cambiamento aritmetico.

## Differenza osservata e ipotesi

Trace Q-P01 separata, 64 input + quattro decode: prefill 6056,5 ms wall e
3726,5 ms span GPU; decode 2893,8–3013,1 ms wall e 468,2–514,6 ms GPU.
Le proiezioni `ds4_qwen_ud_project_23` dominano gli encoder aggregati;
seguono Q5_K, Q3_K e IQ3_S. L’aggregato non è una misura per singolo dispatch.
La baseline senza profiler misura 9,62–10,06 token/s prefill e 0,32–0,34 decode.

Il kernel dequantizza un’unità per lane e la riusa su quattro righe. Ipotesi:
riusarla su otto righe riduce dequantizzazione/caricamento dei pesi per token
in prefill, con lo stesso accumulo FP32 per ogni output. I registri aggiuntivi
possono annullare il beneficio: in quel caso REJECT, senza sweep arbitrario.
Nessun aumento di chunk, nuova conversione FP16/BF16 o riassociazione di K.
Dispatch candidato solo per `rows >= 8`; decode e piccoli batch restano 4.

## Fonti rilette e confronto pertinente

Revisioni complete, URL, hash e data sono in [sources.json](sources.json).
Tutte le comparazioni esterne sono statiche; nessun benchmark upstream eseguito.

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `src/transformers/models/qwen3_5/{modeling,configuration}_qwen3_5.py`
  e `tests/models/qwen3_5/test_modeling_qwen3_5.py`: MLP gate/up/SiLU/down,
  proiezioni dense, residui, norm FP32 con cast dopo la scala, softmax FP32 e
  stato GDN. La famiglia GGUF `qwen35`, 64 layer/5120/17408, è verificata nel
  manifest Qwen3.8 locale; il nome commerciale non identifica un nuovo kernel.
  Il BF16 del riferimento HF non giustifica arrotondare gli operandi FP32
  del backend quantizzato. Non si cambia nessuna maschera o frontiera di cast.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/kernels/gemv.metal`: `gemv`/`NTOK` riusano i pesi su 1–8 righe;
  `metal/tests/gemv_parity.cpp` verifica formati e finestre contro double.
  `dense_mm.metal` e `dense_mm_parity.cpp` usano attivazioni/pesi FP16 e
  matmul MPP: l’arrotondamento e l’ordine tensoriale non sono trasferibili.
  Al pin, le istanziazioni dense non comprendono IQ3_XXS/IQ2_S, nonostante i
  decoder condivisi: copertura formati parziale rispetto al GGUF reale.
  `metal/src/mtl_rt.mm` controlla capability e max thread pipeline;
  `metal/src/gpu/runtime.cpp`/`engine/iq_host.cpp` non provano il supporto
  completo Qwen27 Metal. `include/strata/qwen35/engine.hpp` espone un engine
  CUDA `qwen35moe` (Qwen3.6-35B-A3B), non questo target denso: quindi non si attribuisce a Strata un full-model Metal Qwen27 verificato.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/models/qwen35.cpp`: 64 layer, attention/GDN e FFN denso corretti;
  `ggml/src/ggml-metal/ggml-metal-ops.cpp`, `ggml_metal_op_mul_mat` seleziona
  matvec a più righe e GEMM secondo shape e capability SIMD. `kernels/mul_mm.metal`
  riusa tile dequantizzati con MPP e cooperative tensors. Il riuso è pertinente;
  quel percorso non conserva automaticamente ordine delle somme e rounding
  del kernel locale. I test backend ggml già acquisiti vengono riletti per
  `test_mul_mat` prima dell’integrazione, non usati come oracle eseguito qui.
- Ricerca locale: `qwen38-compatibility.md`, `qwen_architecture.md`, report
  Metal T00–T07, `upstream-numerics.md`, indice research/llama-cpp e studio
  `qwen27-weight-streaming-20261008.md`. I tile FP16 già scartati per altre
  frontiere BF16 e il drift cambiando chunk escludono quelle scorciatoie.
  Gli slot CUDA/pinned/PCIe non spiegano il costo del prefill Metal shared.
  `metal/qwen35_quants.metal`, `ds4_qwen35_metal.inc` e test correnti forniscono
  decoder, riduzione SIMD32, capability check e oracle scalare indipendente.

Scelta: template 4/8 righe, mantenendo il kernel 4 originario e tutti gli indici
K/lane, FMA, scale e `simd_sum`. Alternative: MPP (rounding diverso), tuning
SSD (prioritario per decode ma non rimuove la quota GPU prefill), tile più grandi
(registri e superficie di prova), incremento chunk (cambia numerica).
Nessun codice esterno nuovo copiato; attribuzione MIT Strata già presente.
Il config HF del target `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
`config.json`, consultato via web, conferma `Qwen3_5ForConditionalGeneration`,
BF16, 64 layer e le dimensioni sopra. URL primario:
https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json.
Lo shader B0 è fissato esplicitamente tramite l’hook diagnostico già esistente
`DS4_METAL_QWEN_UD_SOURCE`: il solo binario non congela shader caricati a runtime.

## Gate e misure fissati prima dell’implementazione

Primitive: tutti i formati supportati, code output e righe 1/3/4/7/8/9/16/17;
confronto double indipendente con soglia esistente `2e-4 + 2e-5*abs(ref)`;
dump A/B identici su input isolati e geometrie reali. Modello completo:
probe con prompt diverso dal benchmark, prefill 128 (due chunk), nove vettori
full-vocabulary, payload/stato, checkpoint e park/reload; B0 contro candidate
bit-exact. Replay MTP e fixture grafo se toccati dal dispatch multi-riga.

Direzione candidate: due sweep 64/128 + otto teacher-forced, uguali a B0.
Se utile: warm-up separato e cinque coppie A/B alternate 64/128 + quattro
teacher-forced, congelati prima delle misure promozionali (workload distinto
da B0 direzionale a otto output). Timeout 120 s/processo, cache 2 GiB invariata.
Promozione: prefill riproducibilmente positivo, correttezza esatta e nessuna
regressione decode spiegabile dal codice; rumore segnalato esplicitamente.
Una regressione misurata riapre l’audit prima di nuovi cambiamenti.


## S2: prima parity isolata

I 270 casi isolati B0/candidate sono **bit-exact**: nove formati UD,
K=256/5120/17408, output 17 (coda), righe 1/3/4/7/8/9/16/17/63/64;
352512 byte di output. B0 usa il suo shader congelato e oggetto Metal salvato,
la candidate il nuovo wrapper e shader. Riletti i contratti NTOK Strata,
il template locale e gli indici di ogni riga: la condivisione delle unità
non aggiunge riduzioni fra token. Nessuna nuova discrepanza numerica.
Il gate full-model B0 indipendente è PASS (128 input, 48,80 s totali con
checkpoint/park/reload e otto decode); il suo tempo non è una misura
promozionale della richiesta.


### Nuova differenza del test esteso: Q8 a 17 righe

La prima esecuzione dell’oracle esteso fallisce a Q8_0, K=256, N=17,
T=17, soglia scalare `2e-4 + 2e-5*abs(ref)`. L’identico test compilato
contro oggetto/shader B0 fallisce nello stesso punto. Evidenze conservate in
`scalar-quant-gdn/` e `scalar-expanded-b0/`; non sono PASS.

Indagine specifica prima di proseguire: `ds4_metal.m:ds4_gpu_matmul_quant_impl_tensor`
manda Q8 oltre la finestra matvec alla `kernel_mul_mm_q8_0_f32`;
`metal/dense.metal` la istanzia con operandi `half`/`simdgroup_half8x8`.
Strata `dense_mm.metal:act_f16,dq_f16,gemm_f16` e relativo test esplicitano
il riferimento con operandi arrotondati FP16; ggml `mul_mm.metal`/wrapper e
`tests/test-backend-ops.cpp:test_mul_mat` usano aritmetica/tolleranze diverse.
Transformers `Qwen3_5MLP` resta il riferimento semantico, non un oracle per la
quantizzazione GGUF. La ricerca locale T06–T07 conferma che i percorsi half
non preservano automaticamente il riferimento FP32. Queste fonti sono state
rilette per questa differenza, senza assumere che l’audit UD la spiegasse.

Scelta: nessuna correzione al kernel/dispatch Q8 in Q-P02. Conservare i casi
originali 1/3/5 e relative soglie per Q4_0/Q8_0/Q4_K; aggiungere le frontiere
4/7/8/9/16/17 solo ai nove formati UD effettivamente cambiati. È una
limitazione esplicita della copertura, non una qualifica dei nuovi casi Q8.
Un intervento su Q8 richiederà un esperimento distinto con proprio oracle.


## S3: verifica prima delle misure appaiate

Oracle scalare UD con code e frontiere 8/9/16/17 PASS, stessi criteri;
fixture hybrid resident/streamed e feature checkpoint/cancel/MTP PASS.
Il full-model candidate passa: **9/9 vettori da 248320 logits e payload
prefill da 176630352 byte bit-exact** con B0. Token, contatore finale del reader dopo reload
(119969660928 byte logici) e storage finale (1302325248) identici. Il counter
si azzera al reload e non rappresenta tutto l’I/O del processo. I byte letti sono
logici, non traffico SSD fisico.

Rilette per S3 le geometrie/NTOK del kernel Strata e la selezione small-batch
vs GEMM di ggml: nessuna ragione di cambiare K, chunk o precisione. Si misura
una sola variante a otto righe. Il primo sweep candidate mostra un segnale
piccolo, ancora insufficiente per una decisione; nessun veto percentuale.

Per evitare l’arrotondamento TPS a due decimali del CSV originale, il solo
harness locale delle coppie usa sei decimali. È una copia di `ds4_bench.c`
con il formato `fprintf` finale cambiato; nessuna nuova lettura GPU, logging
per-token, sincronizzazione o finestra di timing. I due eseguibili vengono
collegati allo stesso harness e ai rispettivi oggetti Metal B0/candidate;
lo shader B0 resta fissato tramite `DS4_METAL_QWEN_UD_SOURCE`. Le misure
iniziali originali sono conservate e restano distinte. Il tempo richiesta
sintetico è prefill + eval di quattro token; non è TTFT HTTP né latenza ExMekh.


## Esito della direzione: REJECT per il dispatch generalizzato

Microbenchmark sincronizzato, quattro geometrie reali, un warm-up e dieci
campioni per forma, un processo B0 poi candidate; nessun profiler. Mediane:

| Formato / K→N / 64 righe | B0 ms | Tile8 ms | Throughput locale |
| --- | ---: | ---: | ---: |
| IQ4_XS 5120→17408 | 16,587 | 17,292 | −4,08% |
| IQ4_XS 17408→5120 | 20,212 | 19,575 | +3,25% |
| IQ4_XS 5120→10240 | 9,685 | 10,087 | −3,99% |
| Q5_K 5120→17408 | 13,435 | 16,022 | −16,15% |

Sono confronti direzionali; nessuna promozione o misura di Strata/ggml.
Gli input sono già FP32, senza conversione o workspace aggiunto; il timer
include dispatch e sincronizzazione, non l’allocazione una tantum della fixture.

Nuova trace: prefill 5917,1 ms wall, 3780,6 ms GPU; B0 6056,5/3726,5.
Decode 2384,1–2658,4 ms wall, 360,4–431,5 GPU. La riduzione decode senza
cambio al suo dispatch mostra condizioni host/device diverse: non attribuirla
al tile prefill. I due sweep candidate misurano prefill 9,97–10,11 TPS e
decode 0,32–0,35, ma il segnale piccolo non è qualificato. Campioni grezzi
conservati; le coppie finali pianificate non sono state lanciate per questa
candidate, fermata allo screening.

Nuova indagine sulla regressione prima di scegliere altro: rilette NTOK/NR e
riduzioni di `gemv.metal`, `dense_mm_parity.cpp`, wrapper ggml small-batch,
il template locale e i dati per forma. Otto accumulatori aumentano pressione
sui registri e riducono numero di threadgroup; è un’ipotesi causale, non un
contatore misurato. Non trasferire automaticamente un tile migliore su K lungo
alle altre proiezioni. Una specializzazione IQ4_XS down resta una possibile
componente piccola da confermare, non viene scartata perché inferiore al 2%.

Decisione: **REJECT del dispatch generalizzato**, ripristino del kernel e
wrapper B0, patch e binari sperimentali conservati negli artefatti grezzi.
La regressione su tre forme e lo span GPU non migliorato motivano l’esito;
non è un veto ai piccoli guadagni. L’estensione dei test UD resta utile e
mantiene tutti i casi originari degli altri formati. Nessuno speedup promosso.
S4: integrazione annullata, API/kernel di produzione tornati a B0; correttezza
della candidate resta documentata e distinta dall’esito prestazionale.
