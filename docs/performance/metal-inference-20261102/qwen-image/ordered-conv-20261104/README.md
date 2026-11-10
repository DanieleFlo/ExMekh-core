# I-P02 — Convoluzioni BF16 ordinate del VAE

Baseline `4fef55f`, M5 Pro 24 GiB, Qwen Image 2.1 Turbo BF16 a 256×256,
otto passi, CFG1 e seed20261020. Un modello alla volta, cap comune7.5GiB,
riserva host2GiB, fixture e gate indipendenti ereditati da I-P01.
Nessun commit autorizzato per questa continuazione del4novembre.

## S1 — Profilo e confronto prima del codice

Il profilo successivo a I-P01 conserva415/418 stack VAE in `nm_conv`.
Il tempo decode VAE è54.198s su129.852s dell'operazione. Sono stack host
comprensivi di attesa GPU, non una misura separata del solo shader.
Il kernel generico ricalcola tre assi e divisioni64bit per ogni prodotto,
e mantiene quattro somme indicizzate dinamicamente per output.

Fonti rilette il4novembre; i confronti sono statici e nessun codice viene
importato. Hash dei file e revisioni nel manifesto `sources.json`.

- Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
  `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`:
  `QwenImage21CausalConv3d` deriva da Conv2d, rimuove il singolo frame,
  applica padding esplicito e ripristina il frame. Residual/resample/decoder
  usano1×1 e3×3. Non esiste contesto temporale in questa specializzazione.
  `transformer_qwenimage21.py` e `pipeline_qwenimage21.py` mantengono il
  confine BF16 tra DiT e VAE e la recipe. Nessun cambiamento di recipe.
- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
  `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`: il patch embedder
  Conv3d dell'encoder resta un'operazione distinta, fuori dal dispatch2D.
- ATen `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`,
  `aten/src/ATen/native/ConvolutionMM2d.cpp` lowering e bias,
  `cpu/BlasKernel.cpp:sum,gemm_notrans_`: quattro stream di prodotti FP32,
  coda aggiunta al primo e riduzione strettamente `((s0+s1)+s2)+s3`.
  Il bias entra prima del round BF16. Non trasferibile la dot32 usata
  dalle proiezioni lineari, né una MMA con diverso ordine.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/kernels/dense_mm.metal`, `metal/src/mtl_rt.mm:require_fit`,
  `metal/tests/dense_mm_parity.cpp:run_all`: nessuna Conv2d/VAE in questo
  sottoinsieme Metal. Trasferibili riuso degli operandi e verifica limiti
  pipeline. FP16 e MMA non rispettano gli operandi BF16 e l'ordine richiesto.
- llama.cpp/ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/kernels/conv.metal:kernel_conv_2d`,
  `ggml-metal-ops.cpp:ggml_metal_op_conv_2d`,
  `ggml-metal-device.cpp:ggml_metal_library_get_pipeline_conv_2d`,
  `tests/test-backend-ops.cpp:test_conv_2d,test_im2col`: direct2D elimina
  il lowering e calcola i bordi spaziali. Il wrapper accetta input/output
  FP32 e pesi FP16/FP32; non copre la nostra Conv BF16 ordinata. Test con
  NMSE5e-4, incompatibile come oracle esatto. Im2col è alternativa con
  workspace esplicito, rifiutata qui per il picco di memoria.
- Ricerca locale: `docs/research/INDEX.md`,
  `docs/performance/metal-apple-silicon-20261031/t06-t07/upstream-numerics.md`
  (Conv2d/Conv3d e differenze linear/conv), `metal/numeric_ordered.metal`,
  `ds4_numeric_metal.m:nm_conv`, `ds4_qwen_image_vae.c:va_conv`.
  CUDA/cuDNN conserva un cast prima del bias e non è oracle CPU Apple.

Ipotesi: specializzare1×1/3×3 2D e assegnare i quattro stream a quattro
lane contigue; due punti spaziali condividono il peso. Ogni stream conserva
la successione FMA originaria, inclusi i bordi saltati. Riduzione esplicita
in ordine0,1,2,3 e bias finale. Input canali/gruppo multipli di4, profondità1,
kernel e profondità output1; altre forme restano sul kernel generico.
Pipeline opzionali, capability e limite griglia verificati dal proprietario.
Prima replay esatto delle forme reali, gruppi, batch, padding, stride,
dilatazione, code spaziali, bias e sottoviste; poi gate indipendente
text/RGB/RGBA e warmup+5coppie prima di promuovere le prestazioni.

## Verifica YASAKANI

Il4novembre il [profilo pubblico](https://github.com/morphicode-jp) elenca
ancora una repository, [f32-patch-gemma](https://github.com/morphicode-jp/f32-patch-gemma).
Nessun motore né specifica.mgtm pubblica verificati. La patch modifica pesi
Gemma e non è applicabile a questo intervento. Nessun benchmark RTX5090
viene usato come misura o obiettivo Metal.

## S2 — Replay iniziale

Otto forme isolate, incluso il primo decoder64→1152, il residual1152,
i blocchi384/192 e l'uscita96→4. Tutti gli elementi sono bit-exact contro
Conv2d PyTorch CPU e gli hash coincidono col kernel generico precedente.
Anche batch/gruppi, bias assente, stride/dilatazione, padding asimmetrico,
coda spaziale e sottoviste passano. Riletto `BlasKernel.cpp:sum` dopo la
parità: ogni lane possiede un solo stream e la riduzione resta sequenziale.

Le tre chiamate misurate dopo un warmup indicano318.315→59.248ms per
1152→1152 a16×16, e276.682→52.216ms per192→96 a128×128.
È screening, non qualifica della richiesta. Le due forme minuscole8→12
mostrano costi sub-millisecondo variabili tra processi; un confronto mirato
è necessario prima di decidere un eventuale fallback per la piccola riduzione.
Il primo gate completo candidato viene eseguito senza cambiare il dispatch.

## S3 — Primo gate e dispatch

Testo/RGB/RGBA passano il gate indipendente congelato CPU encoder/DiT e
MPS VAE. Nel testo,11artefatti nativi (conditioning,8latenti,RGBA,PNG)
coincidono byte per byte con I-P01. Primo warmup129.742→81.449s,
VAE54.103→10.427s, stesso cap/residenza/caricamenti e83.620GB logiciH2D.
Questi tempi singoli restano separati dalla conferma a cinque coppie.

Riletti wrapper capability e test Strata/ggml: le due pipeline Conv sono
opzionali, controllate dal backend, con fallback generico per3D, kernel
rettangolari, canali/gruppo non multipli4, griglie e capability non ammesse.
I parametri della convoluzione restano nell'adapter C; la selezione e gli
handle Metal appartengono a DS4MetalNumeric. Nessuna staging im2col.

Il campione post-kernel, denominato inizialmente dit-rgba, cade in realtà
nel VAE:666stack nell'owner decode,656in Conv, nessuno in GEMM o memmove.
Il nome grezzo è conservato, il significato è corretto nel report. Include
attesa GPU: non dimostra che656campioni corrispondano al solo calcolo shader.
Il gate RGBA profilato è escluso dalle coppie testo senza profiler.

Il primo test host bundle era limitato dalla sandbox all'apertura del server
HTTP loopback. Ripetuto con accesso locale, passa12/12; build, test Image e
Numeric CPU46 passano (13skip Metal/CUDA). Planner cache e lifetime passano.
Le modifiche I-P03/I-P04 vengono congelate separatamente e non entrano nei
binari prima/dopo della conferma Conv.

## S4 — Cinque coppie e limite encoder

Warmup0, poi cinque coppie alternate1..5. Tutti110artefatti sono bit-exact,
stessi83.620GB logiciH2D,159caricamenti DiT, cap7.5GiB e picco del piano.
VAE mediana54.666→10.511s, delta geometrico appaiato−80.768%
IC95[−80.787,−80.750]. Operazione129.566→84.811s, delta−34.918%
[−38.565,−31.055]. Processo142.343→97.695s, delta−31.471%.
Metriche per fase, CV, raw e memoria nel `performance.json`.

L'encoder misura14.088→14.397s, delta appaiato+3.144%
[+0.108,+5.989]; il DiT−1.264% con intervallo[−10.834,+9.334].
Il delta encoder non viene nascosto né attribuito automaticamente al rumore.
Riaperto il confronto prima di una possibile modifica: Transformers al pin
sopra mantiene text attention/MLP e patch Conv3d separati; Strata dense_mm
mantiene il proprio dispatch FP16; ggml mul_mm ha capability/formati diversi.
Nel percorso testo locale non viene chiamata Conv; il patch media è3D e resta
sul generico. Proiezioni, shader e copie dell'encoder sono identici nei due
snapshotI-P02, con lo stesso numero di caricamenti. Non emerge un nuovo
percorso aritmetico da correggere. Nessun kernel o dispatch viene cambiato
per spiegare questo delta; la scelta KEEP riguarda il guadagno della richiesta
completa e conserva il limite di attribuzione della fase encoder.

Le due forme minuscole sono state ripetute con10warmup e40chiamate:
mediane1×1 0.244→0.232ms e3×3 0.274→0.185ms. Tutti i bit coincidono;
non si ripete il rallentamento del primo screening. Nessuna nuova guardia
nel dispatch. Il replay delle otto forme con entrambe le pipeline Conv
forzatamente assenti è bit-exact contro CPU e contro il tile; il fallback
è una fixture separata, senza opzioni permanenti nel prodotto. Il diagnostico
pubblico usa il budget Numeric con riserva2GiB e storage limitato1GiB.

## Riesame del wrapper prima della qualifica finale

Il callbackC Conv nuovo chiamava direttamente il metodo del backend, uscendo
così dal pool autorelease che `nm_run` forniva prima. Nessun mismatch o leak
misurato viene dedotto da questa ispezione, ma i command buffer autoreleased
devono avere lo stesso confine di vita delle altre operazioni sincrone.
Prima della correzione riletti Strata `mtl_rt.mm:Context::run` al pin46c152a
(pool attorno alla creazione/completamento del command buffer), wrapper ggml
C++ che delegano ad API C e non forniscono qui un oracle ARC, oltre a
`nm_run`, `nm_gemm` e `nm_copy` locali. HF/Diffusers e il kernel sono invariati:
il pool riguarda solo gli oggetti temporanei dell'host e non l'ordine numerico.
Scelta: ripristinare il pool al confine del callbackC Conv, mantenendo il
dispatch nel proprietario. Le prove già raccolte restano associate al loro
binario; la qualifica finale ricompila entrambi i controlli cache con lo stesso
wrapper corretto e ripete i gate. Nessun cambiamento dei criteri numerici.
