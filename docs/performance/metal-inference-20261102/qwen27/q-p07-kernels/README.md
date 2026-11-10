# Q-P07: riscrittura delle proiezioni UD Metal

3 novembre2026, S1 prima del codice. Mandato esplicito: riscrivere anche i
kernel per cercare1TPS, senza commit e senza allargare criteri.
B0 continuazione `fc8fb63`, Bi Q-P05/P06, snapshot prima di modificare shader.

## Audit specifico kernel/dispatch

- HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py:Qwen3_5MLP`, GDN,
  attention, RMSNorm: le proiezioni sono dense con attivazioni del percorso
  qualificato; non cambiare cast/norm/SwiGLU o accumuli per nascondere drift.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/kernels/gemv.metal:gemv` usa NTOK costante e geometri LPR/NR/SG/KS;
  `metal/tests/gemv_parity.cpp` confronta contro double e misura le forme.
  `metal/src/mtl_rt.mm` e `metal/src/gpu/runtime.cpp` possiedono capability,
  binding e buffer; `metal/kernels/dense_mm.metal` separa il multi-riga.
  Trasferibile: specializzare numero di righe a compile-time e riusare decode.
  Incompatibili: s16/altre rappresentazioni dell'attivazione, alberi di somma
  diversi e assenza di IQ3_XXS/IQ2_S nel dense_mm al pin; il modello27B
  Metal completo non è coperto. Nessun codice copiato automaticamente.
- ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/kernels/mul_mm.metal`,
  `ggml-metal-ops.cpp` (`mul_mat/get_rows`), `ggml-metal-device.cpp` e
  `tests/test-backend-ops.cpp:test_mul_mat`: tile e dispatch per forma,
  accumulation/capability; i percorsi half/tensor non sono equivalenti per
  costruzione ai nostri FP32. Un'eventuale GEMM richiede gate proprio.
- Locale: `metal/qwen35_quants.metal:project/deq`, wrapper UD,
  oracoli indipendenti e Q-P02. Il kernel attuale mantiene4 accumulatori e
  quattro riduzioni SIMD anche in decode con una sola riga. Il tile8 di
  Q-P02 aveva regressioni prefill su tre forme: non riutilizzarlo come default.
  Riaperti `ds4_cuda.cu:qwen35_q8_1_r8_quantize_kernel`, dot Q4/Q5 e dispatch
  UD, `qwen36-performance-ledger.md:341–389`, `qwen38-compatibility.md`,
  guida MTP e contratto Metal.

## Esperimenti pianificati

A: riscrivere project con numero di righe compile-time, kernel decode1 e
prefill4. Stessa assegnazione lane/unità, FMA seriali e simd_sum per ciascuna
riga; il decode elimina accumulatori/riduzioni inattivi. Non è un aumento
di tile indiscriminato. Gate bit-exact su input isolati poi full-model.
Micro per9 UD e forme effettive, T1/3/64, incluso head; GPU e wall separati.
Il wrapper seleziona internamente per rows, senza flag prodotto.

B: studiare Q8_1+R8 su una vera attivazione catturata dopo prefill (il trace
prima del prefill cambia workflow). Il CUDA conserva `x ≈ d0*q0+d1*q1`,
scale half e due accumulatori DP4A. I vecchi errori full-model piccoli ma
non nulli non dimostrano il gate Metal, né il guadagno sul Mac. Prima
simulare il costo numerico della rappresentazione; se incompatibile con
le soglie frozen, documentare il limite e preferire riscritture FP32.
Questo screening non sarà chiamato benchmark GPU o kernel già qualificato.

Alternative ulteriori: cooperazione nel decode IQ/codebook, tile per forma
e GEMM FP32/simdgroup. Scegliere dai nuovi micro/profili, non da percentuali
CUDA riportate. Ogni mismatch o regressione richiede nuovo audit specifico.

S2 A:270 confronti isolati bit-exact, oracoli scalari/GDN PASS e nove logits
completi/payload128 bit-exact. Primo micro (sei campioni dopo un warmup,
sequenza B0/candidate, non promozionale) mostra piccoli segnali misti:
IQ4_XS up+2.76%, down+4.23%, qkv−2.29%, Q5_K up−2.72%, Q6_K head+6.74%,
Q3_K qkv+3.45%, IQ3_S up+2.02%. Nessun vantaggio2x attribuibile a questa
specializzazione. Prefill conserva il tile4. Misura full e profilo separati.

S1 successivo, prima della riscrittura decode16: riletti ancora
Strata `gemv.metal:lane_sum/LPR`, `gemv_parity.cpp:Cfg/cfg_ok`, wrapper/runtime,
ggml `mul_mm`/dispatch/capability, HF denso e riduzioni UD locali. Il kernel
32-lane per output ripete staging/tabella per soli4 output a threadgroup.
Provare16 lane per output e8 output/threadgroup128: stessa dequant e FP32,
ordine delle somme diverso e dichiarato. Il risultato deve passare prima
l'oracolo originale `2e-4+2e-5*abs(ref)` e il confronto completo entro il
criterio graph esistente `2e-5+5e-5*abs(ref)`, tutti valori finiti e token
esatti. Non assumere bit-exact né modificare i criteri per accettare drift.
Se non passa, riaprire audit della discrepanza prima di correggere/ritirare.
L'alternativa Q8_1+R8 resta una rappresentazione approssimata distinta.

## Screening Q8_1+R8, senza kernel GPU nuovo

Acquisita una vera `attn_norm`5120 a posizione128/layer0 dopo load del payload
prefill (nessun replay scalare del prompt). Simulazione host bounded, 17 righe
originali per cinque matrici GGUF (IQ4_XS/Q5_K/IQ2_XS/IQ2_S/Q6_K), oracle
indipendente `ud_value`, accumulo FP64 per isolare l'errore di rappresentazione:
MAE attivazione6.0201e−6, max3.70026e−4. Tutti85 dot rientrano nella soglia
isolata originale2e−4+2e−5|ref|; un dot qkv supera la soglia graph applicata
localmente2e−5+5e−5|ref| (maxqkv2.5791e−5). Questo non è il risultato di
un kernel Metal R8 né un gate full-model: non prova il fallimento o successo
end-to-end, e non include l'ulteriore rounding degli accumulatori CUDA/Metal.
Conferma che R8 non è una sostituzione esatta gratuita. Nessun'approssimazione
attivata in produzione e nessuna soglia cambiata. Dati/script/sorgenti negli
artefatti; il port richiede una campagna numerica specifica prima di promozione.

S2 decode16: oracoli originali e gate128 PASS. Nove full-vocabulary rientrano
nella soglia graph congelata, max delta2.74181366e−5 su un vettore; token
greedy13/5044/4434/279/1534/11316/13/561 identici. Il cambiamento è roundoff
dell'ordine di somma, non bit-exact. Nessuna soglia adattata. Micro wall
sequenziali: benefici IQ4_XS~6.7–12.2%, IQ2_XS13.7%, Q6_Khead6.8%; forti
regressioni Q3_K/Q5_K/IQ4_NL e variazioni molto grandi in altre forme.
Riaperti dequant per formato, confrontoLPR e test Strata: geometrie diverse
sono ammesse, non si attribuisce una causa hardware senza contatori dedicati.
La misura singola full decode0.48/0.51 non promuove un dispatch generalizzato.
Un eventuale dispatch per formato richiede controllo appaiato e gate del bundle.

Riesame del segnale micro prima di un dispatch per formato: la timeline del
modello (stessi conteggi, quattro decode più head prefill) ha Q5_K206.3ms
contro218.9ms decode1, mentre Q3_K111.0 contro100.0 e IQ4_XS442.9 contro513.9.
I forti delta di alcuni micro wall non sono un effetto kernel dimostrato:
conservati, senza escludere formati sulla loro sola base. Totali di encoder
non vanno sommati al wall, né le differenze fra trace considerate promozione.
Il bundle finale dovrà essere valutato con misure appaiate fuori dal profiler.

La nuova qualificaQ-P09 depth4 ha trovato almeno un logit ordinary/spec oltre
soglia, nonostante i16 token del benchmark esatti. Q-P07b non è quindi
promovibile dal solo PASS target-only/depth2: stato **NUMERICAL_INVESTIGATION**.
Audit riaperto nel reportQ-P09, controllo32vs16 a input identici prima di
ulteriori cambi. Non si amplia la soglia e non si nasconde il fallimento.
Alternativa da studiare dopo il controllo: due output perSIMD32, stessa
assegnazione unit/lane e riduzione originale, per riusare le attivazioni senza
cambiare l'associazione delle somme (configLPR32/NR2 già presente inStrata).

Esito successivo definitivo diQ-P07b: **REJECT**, differenza MTP4 riprodotta
su input uguali con32 PASS/16 FAIL. Il guadagno micro non compensa il gate
numerico fallito. Il kernel16 è rimosso dalla produzione; snapshot/risultati
restano negli artefatti. Q-P12 valuta una coppia di output conservandoSIMD32.
