# Riesame HF e ggml del 1 novembre 2026

Questo confronto precede qualsiasi modifica ai kernel numerici della suite.
I riferimenti nuovi non sostituiscono le revisioni degli oracle congelati.
Il manifest `sources.json` dell'audit registra URL, dimensione e SHA256 di ogni
file consultato; gli originali scaricati restano in `build/metal-qualification/hf-audit-20261101/`.

| Progetto | Revisione consultata | Ambito effettivo |
| --- | --- | --- |
| [Transformers](https://github.com/huggingface/transformers/tree/ca8c871ede62b521a75a2ddfc5e1968ac9c8e097) | `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097` | EmbeddingGemma 2, Gemma4, Qwen3-VL e dispatch delle implementazioni attention |
| [Diffusers](https://github.com/huggingface/diffusers/tree/86ef420b17363c28e36ebad6b58555e9aa8feda2) | `86ef420b17363c28e36ebad6b58555e9aa8feda2` | Pipeline Qwen Image 2.1, DiT e VAE |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/dde879bc4f954b809892fd0d660515f42536394a) | `dde879bc4f954b809892fd0d660515f42536394a` | Qwen3-VL/Gemma4 e primitive ggml Metal; il tree consultato non contiene il DiT Qwen Image |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/tree/c6dfa9def12fd0cb36ac7149ba30a75792bedbf0) | `c6dfa9def12fd0cb36ac7149ba30a75792bedbf0` | Integrazione e grafo Qwen Image 2.1, conversione Qwen3-VL |

## Semantica e ottimizzazioni già presenti

Il file EmbeddingGemma 2 attuale coincide byte per byte con quello della
revisione Transformers dell'oracle (`d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`).
Anche pipeline e DiT Qwen Image coincidono con Diffusers fissato nell'oracle
(`aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`). Le differenze Qwen3-VL rilevate
sono annotazioni e selezione dei logits autoregressivi, senza una correzione
numerica dell'encoder. Aggiornare le dipendenze non risolve i gate aperti.

Gemma calcola RMSNorm in FP32 e moltiplica il peso prima di tornare a BF16.
Qwen3-VL arrotonda invece il valore normalizzato a BF16 prima della
moltiplicazione per il peso BF16. Il parametro pubblico `round_normalized`
della nostra primitiva conserva questa differenza. Anche gli arrotondamenti
di score e probabilità attention sono rappresentati esplicitamente.
Resta da misurare la differenza tra divisione per radice e moltiplicazione
per reciproco della radice, oltre all'ordine delle riduzioni.

Il dispatch HF offre SDPA, Flash e Flex attention. Il percorso SDPA può
evitare copie delle teste KV con GQA sulle versioni MPS compatibili.
Questo è uno spunto di scheduling; cambiare il backend eager dell'oracle
non costituisce una correzione del nostro risultato.

Diffusers acquisisce il conditioning Qwen prima della RMSNorm finale.
Il nostro encoder fa già lo stesso. Mantiene inoltre padding/mask,
composizione RGBA su bianco per l'encoder e alpha separato per il VAE.
La cache DiT dei prefissi a timestep zero è già presente nel percorso nativo:
non serve ricalcolare questi prefissi a ogni passo. Il confronto con
stable-diffusion.cpp conferma che encoder, DiT e VAE sono fasi distinte,
con dtypes e trattamento alpha da verificare separatamente.

## Esperimenti scelti e criteri di integrazione

Il [GEMM ggml Metal](https://github.com/ggml-org/llama.cpp/blob/dde879bc4f954b809892fd0d660515f42536394a/ggml/src/ggml-metal/kernels/mul_mm.metal)
ha una specializzazione BF16 con operandi SIMD-group BF16 e accumulo FP32.
Questo evita l'espansione degli operandi in FP32 usata dal nostro GEMM MPS.
La semplice sostituzione del tipo in `MPSMatrixMultiplication` è stata
provata e rifiutata dal framework; non dimostra un limite del dispositivo.

Il primo esperimento è quindi un kernel autonomo, scritto per il probe,
con tile 32×32×32, quattro SIMD-group, operandi BF16 e accumulo FP32.
Si confrontano prima input piccoli con code e poi operandi reali congelati
di Gemma PLE e della proiezione Q del primo layer Qwen. Il confronto include
il nostro MPS e `torch.nn.functional.linear` CPU BF16, senza caricare un
secondo modello. I tempi rilevati durante altri test GPU non sono benchmark.
Nessun codice upstream è importato nel prodotto da questo esperimento.

Il percorso generale rimane necessario sui dispositivi privi delle capacità
richieste. Un'eventuale integrazione deve ridurre il working set, rispettare
gli arrotondamenti e ripetere primitive e gate reali indipendenti. Un minor
numero di differenze locali non chiude automaticamente i gate Gemma o
conditioning Qwen Image. Le soglie congelate restano invariate.

Riesame Strata S1: `metal/kernels/dense_mm.metal` alla revisione già fissata
`46c152a1310f77cd2f0ce2c7cd213962df14fe83` usa conversione FP16 e
`mpp::tensor_ops::matmul2d`, tile 64/128 e quattro SIMD-group. Quel percorso
è adatto alle proiezioni quantizzate considerate da Strata, ma la conversione
BF16→FP16 non preserva esattamente i nostri operandi. Il probe mantiene BF16,
usa tile più piccoli per le poche righe degli input congelati e non riusa
quel codice. `gemv.metal` e la licenza MIT sono registrati nello stesso audit.
S2–S4 del percorso prodotto restano subordinati ai risultati dell'esperimento.

## Risultati dei probe isolati

Il BF16 SIMD-group compila ed esegue sul M5 Pro. Tutti i risultati BF16 del
probe coincidono con il percorso MPS corrente: caso costante 2×8×2,
code 35×39×37, caso casuale 8×512×96 e due proiezioni reali.
La proiezione Gemma 11×512×12288 mantiene **19/135168** differenze contro
CPU; Qwen Q0 34×4096×4096 mantiene **58/139264**. Non è una soluzione del
gate numerico. Il conteggio dei buffer GPU del probe è 13,13/34,39 MB,
contro picchi del ledger MPS 38,59/102,33 MB, con gli stessi operandi.
Il probe include un output FP32; sono conteggi di storage delle operazioni,
non misure di riduzione del footprint complessivo del processo.

RMSNorm è stata confrontata su quattro layer Gemma, quattro layer Qwen e
due QNorm Qwen, sempre con input CPU identici. Moltiplicare per reciproco,
`rsqrt` o `pow` elimina una differenza Gemma al layer zero. `rsqrt`/`pow`
eliminano cinque differenze QNorm Qwen al layer zero, ma ne introducono una
al layer sei; gli altri casi esaminati restano esatti. Un cambio globale
non dimostra quindi la compatibilità bit per bit richiesta dal gate.

Un ulteriore probe confronta accumulo sequenziale, inverso, quattro/otto
accumulatori FP32 e somma compensata. La somma compensata mantiene
8 differenze Gemma e 18 Qwen contro CPU. Gli ordini più semplici danno
risultati diversi a seconda della proiezione. Per i punti divergenti sono
salvati anche dot product FP64 degli stessi operandi. Queste misure spiegano
la sensibilità ai tie BF16, senza giustificare una scelta di kernel basata
sul minor errore di due soli esempi.

L'ispezione successiva di [PyTorch alla revisione dell'oracle](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp)
identifica il percorso BF16 CPU dedicato. Il probe scritto per riprodurre
quell'ordine di accumulo FP32 a 32 somme indipendenti restituisce **zero
differenze su entrambe le proiezioni reali**, senza eseguire il calcolo su CPU.
La riduzione a cascata del `SumKernel.cpp` della stessa revisione, seguita
da `pow`, riproduce inoltre tutti i dieci casi RMSNorm isolati esaminati.
Questi risultati localizzano operazioni concrete, ma non dimostrano ancora
la compatibilità dell'intero forward.

Decisione S2: confrontare due eseguibili sperimentali sul forward Gemma e
sull'encoder Qwen completo, aggiungendo una terza variante con riduzione
RMS a cascata. Le esecuzioni attendono la conclusione della coda di modelli
esistente. I kernel del prodotto, gli oracle e le soglie restano invariati
durante la verifica. I probe sono separati dai gate; nessun miglioramento
di precisione o velocità del modello è ancora rivendicato. L'opportunità BF16
diretta per ridurre staging richiede una successiva integrazione con capability,
admission/lifetime e riesame S3–S4 su tutte le forme e sui modelli completi.

## Nuove differenze: riesami ripetuti

La variante GEMM/RMS completa produce i primi tre layer Qwen esatti; il quarto
diverge in due elementi attention su 139264, mentre le altre primitive del
layer sono esatte sugli stessi input. Il riesame dell'eager HF e di ATen
`BlasKernel.cpp` distingue il prodotto QK da PV: QK usa 32 somme FP32 e PV
quattro somme indipendenti con riduzione finale ordinata. La nuova variante
isolata che conserva questi ordini rende esatto anche il replay dell'attention
del quarto layer. Il forward completo di questa variante resta da eseguire.

Per questo riesame sono stati riletti anche i report locali
`native-suite-t11-t12-20261021` (softmax lunga e RMS dei batch corti),
`native-suite-t13-t14-20261021` (geometria GEMM e pooling) e i kernel attention
in `ds4_numeric_cuda.cu`. Documentano casi analoghi di tie BF16 amplificati.
La gerarchia CUDA e l'attention sparsa QSA Strata non possono essere trasferite
assumendo lo stesso ordine della CPU Apple: costituiscono riferimenti per la
diagnosi, con geometria e hardware differenti. Il forward resta l'arbitro.

Gemma completo mantiene 4/81 frontiere esatte. Il replay PLE delle tre stringhe
localizza la prima differenza residua nella RMSNorm: linear e scalatura sono
esatti, mentre la norm differisce in 1/135168, 0/110592 e 2/110592 elementi.
Il calcolo CPU manuale coincide con tutti e tre i dump PLE indipendenti.
Prima di un'altra modifica viene quindi esaminata anche la specializzazione
di `pow(-0.5)` in `PowKernel.cpp` della revisione PyTorch fissata, oltre alla
semantica HF e alle riduzioni già studiate in ggml e nei report locali.

Il riesame di `PowKernel.cpp` mostra che PyTorch specializza `pow(-0.5)`
chiamando `rsqrt_kernel`: il nome dell'operazione Python non implica una
chiamata generica a pow. Un ulteriore controllo dell'SDK Apple distingue
`MTLMathModeSafe` da `mathFloatingPointFunctions`, il cui default documentato
in `MTLLibrary.h` è `MTLMathFloatingPointFunctionsFast`. La nostra configurazione
imposta il primo ma non il secondo. Prima di modificare il prodotto si
confrontano quindi librerie isolate con funzioni FP32 precise, usando sia pow
sia reciproco di sqrt sulla medesima riduzione verificata. Riferimento:
[documentazione Apple](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathfloatingpointfunctions).

Il replay PLE con funzioni precise e reciproco di sqrt elimina tutte le
differenze sulle tre stringhe: linear, scalatura, norm e confronto con i dump
CPU sono bit-exact. La sola modalità precise con pow generica conserva
alcuni tie, coerentemente con la specializzazione ATen consultata. La
combinazione con QK/PV ordinati supera trenta verifiche ridotte di primitive,
inclusi bias, code, norm in-place e cleanup. Il forward completo costituisce
ancora un gate distinto.

Il forward con precisione globale migliora il conditioning Qwen da RMSE
0,452584 a **0,179743**, ancora FAIL rispetto a 0,02/0,01. Introduce però
una differenza già al layer uno. Il nuovo replay a input identico localizza
un singolo elemento attention diverso (max 1,5259e-5); proiezioni, norm,
RoPE e SiLU sono esatte. Il controllo delle funzioni Apple e della softmax
HF/ATen suggerisce quindi di isolare la funzione precisa richiesta dalla
RMSNorm, preservando l'altra matematica per un confronto separato.
Si prepara una variante con il solo `precise::sqrt` nella norm, prima del
successivo test di primitive e forward. Il forward Gemma con precisione
globale ha 6/81 frontiere esatte: tutti gli embedding e PLE, ma nessun blocco
decoder completo. Anche questa nuova differenza richiede un proprio replay.

Il riesame successivo della funzione `sum()` in `BlasKernel.cpp` individua
un errore nel prototipo PV: la CPU accumula i blocchi completi in quattro
somme, poi aggiunge tutti gli elementi residui alla prima somma. Il prototipo
usava ancora indice modulo quattro anche nella coda. Per 34 chiavi la seconda
chiave residua finiva quindi nella seconda somma, anziché nella prima.
Si corregge questa gestione delle code in una nuova variante isolata,
preservando le precedenti prove fallite e ripetendo prima i replay.

Dopo questa correzione Qwen coincide fino al layer sette. Il replay del
layer otto isola una probabilità softmax diversa e 32 elementi attention;
tutte le altre primitive restano esatte. HF usa softmax FP32; ATen applica
massimo globale, quattro somme vettoriali e moltiplicazione per il reciproco
della somma. Il percorso Metal usa invece normalizzatori online a blocchi
di otto e divisione. Anche ggml `softmax.metal` moltiplica per il reciproco,
ma la sua riduzione SIMD non coincide automaticamente con quella CPU.
Il caso locale CUDA di softmax lunga documenta già la necessità di conservare
la gerarchia del riferimento. La nuova variante va verificata sulle
probabilità isolate prima di ripetere il forward.

Il primo blocco Gemma eseguito dalla classe HF fissata coincide con il dump
CPU. Su input identici, linear, norm e attention sono esatti; differiscono
solo GELU MLP (35/22528) e GELU PLE (1/5632). `Gelu.h` calcola prima il cubo,
poi lo moltiplica per kappa. Il nostro kernel moltiplica prima kappa e poi
tre volte x. ggml `unary.metal` usa esplicitamente `precise::tanh` nella GELU
e GEGLU; la sua formula fattorizzata non conserva necessariamente l'ordine
ATen. Si confrontano pertanto due varianti isolate: ordine ATen con tanh
ordinaria e ordine ATen con tanh precisa, sui dump di attivazione salvati.
Le precedenti primitive CUDA della suite restano un confronto semantico;
il risultato sul dispositivo Apple deve essere misurato autonomamente.

Il solo riordino del cubo non elimina i 35 errori GELU; con tanh precisa
ne restano 28. È stata quindi verificata la revisione SLEEF effettivamente
inclusa da PyTorch (`5a1d179df9cf652951b59010a2d2075372d67f68`, Boost 1.0).
Il suo sorgente richiede esplicitamente contrazione FP disabilitata; la
suite CUDA già usa `--fmad=false`. Le tre tanh problematiche confrontate
con libm in doppia precisione coincidono con la CPU se si conserva lo stesso
argomento FP32. Prima di importare un'altra approssimazione matematica si
verifica quindi una variante Metal con contrazione implicita disabilitata,
mantenendo le FMA esplicite del dot product verificato.

La contrazione disabilitata conserva 28 differenze GELU MLP e una PLE.
Si separano quindi argomento FP32 e tanh FP32 con due chiamate pubbliche
al backend sperimentale: questo distingue un errore di raggruppamento
da quello della funzione trascendente, prima di scegliere una portabilità
del sottoinsieme SLEEF con licenza e attribuzione conservate.

Il replay separato conferma argomenti GELU FP32 identici in tutti i
28160 elementi; `precise::tanh` differisce invece di al più 1,1921e-7,
sufficiente a spiegare tutte le 28+1 differenze BF16. Si prepara perciò
un adattamento sperimentale del solo percorso SLEEF tanh/exp a coppie FP32,
con FMA esplicite e contrazione implicita disabilitata. Sono conservati
copyright, revisione e licenza Boost 1.0; nessuna nuova dipendenza del prodotto.

Per softmax FP32 è stata verificata anche `functional_base.h` del wheel
PyTorch fissato: quattro stream di somme e riduzione NEON `(s0+s2)+(s1+s3)`.
Il candidato attention usa tre passate limitate (massimo, denominatore,
PV), senza matrice quadratica, e moltiplica per il reciproco. È limitato
al percorso BF16 con arrotondamenti QK/probabilità e senza chiavi relative.
Gli altri percorsi restano separati. Il costo della terza passata QK dovrà
essere misurato se il gate indipendente supera le soglie esistenti.

Il replay con tanh SLEEF effettivamente selezionata elimina tutte le
differenze del blocco Gemma: GELU MLP 0/22528 e GELU PLE 0/5632;
le altre primitive e il confronto HF con il dump restano esatti.
Il candidato softmax a tre passate conserva invece 41 differenze attention
al layer otto. Prima di un altro forward viene quindi riesaminata `xexpf`
della stessa SLEEF: riduzione ln2 con FMA, polinomio FP32 e scalatura in due
passi. Si confrontano una softmax autonoma con lo stesso ordine CPU e la
variante attention usando questa exp, mantenendo distinti i risultati falliti.

Con exp SLEEF il replay Qwen layer otto diventa esatto: softmax di tutti
i 32 head, attention 0/139264 e tutte le altre primitive. Si combinano
quindi i soli cambi già verificati in una nuova libreria isolata, prima
dei forward completi e degli stessi gate indipendenti.

Una verifica aggiuntiva copre tutti i 65280 valori BF16 finiti per GELU e
SiLU. Restano 508 differenze per operazione, tutte nel dimezzamento dei
valori subnormali/minimi FP32; nessuna differenza sui valori del modello.
La [specifica Metal Apple](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
consente il flush-to-zero dei subnormali durante l'aritmetica. HF/ATen
mantiene invece questi valori nel prodotto per 0,5. Il riuso Strata tramite
FP16 non conserva tale intervallo, e ggml non garantisce l'equivalenza CPU
per queste conversioni. Il candidato limita quindi un percorso intero
BF16 al solo intervallo in cui entrambe le funzioni arrotondano esattamente
a x/2, conservando segno e ties-to-even; gli input normali restano nel kernel
matematico già verificato. Il sweep completo deve essere ripetuto.

La specifica Apple consultata è quella del 4 giugno 2026, §8.1 e §8.5,
pagine 368 e 375–376; URL, SHA256 e testo estratto sono archiviati.
Con il percorso intero limitato, il sweep restituisce **zero differenze
su tutti i 65280 valori BF16 finiti**, sia per GELU tanh sia per SiLU.
Passano inoltre trenta verifiche di matrici/norm/alias/cleanup del candidato.

Il replay audio contiene 1/1122304 e 6/70656 differenze nelle due convoluzioni,
4815/1122304 nella prima LayerNorm; la seconda norm e la proiezione lineare
sono esatte. Il modulo HF CPU coincide con il dump indipendente. Riesaminati
`layer_norm_kernel.cpp` e `moments_utils.h` della revisione fissata: Welford
vettoriale BF16 a due metà FP32, chunk di 16 e albero di momenti; affine
`(x*rstd - rstd*mean)*gamma + beta`. ggml `norm.metal` usa due passate e
riduzioni SIMD, quindi non conserva quell'ordine. L'inventario Strata alla
revisione fissata non offre un kernel Metal norm dedicato confrontabile.
Il caso CUDA locale `ds4_numeric_cuda.cu:500` riproduce invece Welford della
CPU CUDA: il metodo è pertinente, ma geometria e affine differiscono dal
percorso CPU Apple. Si prepara una variante separata con la sequenza ATen
CPU BF16, conservando il candidato precedente e il percorso FP32 del prodotto.

La nuova LayerNorm conserva un solo tie nella prima norm e ne introduce uno
nella seconda. Si riesaminano quindi distintamente `AddMoments`,
`AddMomentsVec` e affine di ATen e le FMA esplicite del precedente CUDA.
Un replay diagnostico espone statistiche FP32 attraverso la normale API norm
della sola libreria sperimentale, per confrontarle con `native_layer_norm`
CPU in modalità parametri FP32 prima di cambiare ancora il calcolo.

Il candidato combinato supera entrambi i forward indipendenti: Gemma testo
**81/81 frontiere esatte**, Qwen tutti i 36 layer e conditioning bit-exact
(RMSE e relative L2 zero), con token/posizioni/maschere identici. Viene
integrato il percorso BF16, conservando MPS e i kernel FP32 precedenti e
limitando la contrazione FP alle nuove funzioni. Il primo test memoria
imponeva un aumento del picco anche senza staging: ora verifica il limite
superiore sia in BF16 sia in FP32, con oracle del dtype pubblico, senza
cambiare tolleranze. La suite numerica comprende 37 test, tre riservati CUDA.
Il forward media sperimentale migliora da 6 a **123/480 frontiere esatte**,
con vettori finali PASS; il gate media resta FAIL e richiede altri replay.

Le statistiche Welford residue differiscono in 66/8768 e 654/2208 medie
FP32, e in tre/cinque reciproci della deviazione. La trasformazione affine
CPU manuale non fusa coincide con tutti i risultati BF16; fondere quella
trasformazione introduce invece 113 e una differenza. Il residuo è quindi
nelle statistiche. Il riesame separa le FMA esplicite dei vettori dalle
espressioni scalari `AddMoments`, dove il compilatore CPU può contrarre
l'espressione. Una variante isolata confronta queste due FMA scalari,
mantenendo non fusa la trasformazione affine già verificata.

Le FMA scalari eliminano tutti i residui delle due norm audio, sia nelle
statistiche FP32 sia nell'output BF16. Il percorso viene integrato con guardie
su dtype, allineamento e profondità dell'albero; il backend possiede anche
il dispatch dei kernel attention e le relative risorse threadgroup. Il
confronto esatto con PyTorch comprende bias e operazioni in-place alle
larghezze 32/128/1152/4608. La suite aggiornata passa **38 test**, con tre
skip riservati CUDA.

Il nuovo target numerico locale è l'oracle CPU indipendente. I precedenti
run MPS rimangono separati: CPU e MPS possono divergere già eseguendo lo
stesso sorgente upstream BF16, quindi l'esattezza contro CPU non implica
il superamento della vecchia comparazione MPS. Si misura direttamente anche
questa differenza fra oracle, senza allargare alcuna tolleranza, poi si eseguono
nuovi encoder/DiT/VAE CPU per generazione, reference RGB e reference RGBA.

L'unfold della prima convoluzione audio seguito dal GEMM BF16 CPU/native
è esatto; nella seconda, CPU linear e CPU Conv2d differiscono già in quattro
elementi, mentre native linear coincide con CPU linear. Il riesame di
`ConvolutionMM2d.cpp` e `BlasKernel.cpp` spiega perché: la convoluzione usa
GEMM non trasposto con quattro somme, la proiezione lineare usa la dot NEON
a 32 somme. Il backend CUDA locale applica la propria geometria cuDNN e
conserva un cast prima del bias: quel percorso non è una prova dell'ordine
CPU Apple. Si confrontano quindi soltanto convoluzioni BF16 isolate con
quattro somme, FMA esplicita o moltiplicazione/somma separate. Nessun
workspace im2col quadratico viene aggiunto al prodotto.

Entrambe le varianti a quattro somme rendono esatte le due convoluzioni
audio: i prodotti BF16 sono rappresentabili in FP32, quindi la contrazione
non cambia il risultato in questi casi. Si integra la variante con FMA
esplicite soltanto per Conv2d BF16; FP32 e Conv3d mantengono i loro percorsi
finché non vengono qualificati separatamente. Le due norm e la proiezione
audio restano esatte sul replay a input identici.

Il video a due frame isola 28 valori attention diversi, con patch, norm,
linear e GELU esatte. Il replay per head mostra softmax esatta ma diversi
tie QK. Riesaminati `_matmul_impl`, `addmm_impl_cpu_` e `bmm_out_or_baddbmm_`
in `LinearAlgebra.cpp` e il dispatch `CPUBlas.cpp` della revisione fissata:
il reshape dei batch multipli materializza Q/K nei layout contigui, cambiando
il ramo BLAS da `gemm_transa_` (dot NEON a 32 somme) a `gemm_notrans_` (quattro
somme). Il riferimento ggml SIMD non conserva automaticamente tale geometria;
la ricerca CUDA locale sui 1072 token aveva già mostrato questo effetto del
layout. Si prepara un candidato attention con QK a quattro somme soltanto
per batch multipli con più head, da verificare su tutti i frame identici.

Il candidato QK con geometria batch corretta elimina tutte le 28 differenze
del video: 19 replay di primitive e il blocco HF completo sono esatti sui
due frame originali. Si integra quella selezione nel kernel attention BF16,
preservando il ramo a 32 somme dei batch singoli e i percorsi FP32/relativi.

Il replay del primo blocco audio coincide con il dump HF indipendente e
localizza le frontiere FP32: QK bloccato, relative QK e PV nativi sono esatti;
`tanh` differisce di al più 1,1921e-7 in 5330/13824 valori e la softmax di
2,3842e-7 in 4097/13824. Il riesame delle stesse primitive SLEEF e di
`layer_norm_kernel.cpp` conferma che i dtype FP32 audio richiedono la
funzione e la riduzione CPU già verificate sul percorso BF16, conservando
però output FP32. Si prepara una libreria isolata con tanh e softmax FP32
compatibili; DiT usa tanh, quindi un'eventuale integrazione richiede anche
il replay completo immagini, oltre al gate audio. Gli altri due residui
delle primitive audio restano separati e saranno esaminati a parte.

Il replay audio FP32 con SLEEF diventa esatto per tanh, softmax e tutti i
prodotti bloccati. La norm channel-major del backend è già esatta: il primo
probe la trasformava in una riga contigua, cambiando la geometria del
riferimento. Corretto il solo diagnostico perché usi `channel_rms` e la API
pubblica per matrici trasposte. L'unico residuo resta `linear_end`, 11/70656
valori BF16. Il riesame di `addmm_impl_cpu_` e `gemm_transab_` identifica
nuovamente quattro somme per l'input channel-major; MPS conserva un ordine
differente. Si aggiunge pertanto un prototipo dedicato a quel layout,
senza accedere alla memoria privata del chiamante o modificarne il formato.

Il prototipo con la proiezione trasposta a quattro somme rende esatti
tutti i replay del primo blocco audio, compreso il blocco HF rispetto al dump.
Si integrano quel layout e tanh/softmax FP32 già esatte. I kernel ordinati
sono raccolti in `metal/numeric_ordered.metal`; il proprietario backend
seleziona i layout tramite il contratto pubblico, senza modificare i buffer
del modello. I subnormali della tanh FP32 vengono conservati con un controllo
sui bit prima dell'aritmetica. DiT e media richiedono nuovi forward dopo
questa modifica; il vecchio risultato immagini non viene promosso.

Il primo blocco vision Qwen sulla reference RGB originale isola 60 differenze
attention su 294912 elementi (max 0,000488281), con tutte le altre primitive
esatte. Il riesame della dot BF16 NEON in
ReducedPrecisionFloatGemvFastPathKernel.cpp conferma la riduzione a 32 somme
dei blocchi completi, poi una coda distinta a otto elementi e somme a coppie.
Il ramo Metal a 72 canali non usava ancora questo percorso. Il prototipo
conserva separatamente blocchi, coda vettoriale e coda scalare; l'ordine
batch a quattro somme del video resta distinto. Gli shader ggml e la ricerca
CUDA locale sono riferimenti di geometria, senza equivalenza automatica.

La nuova generazione completa supera il gate locale con il profilo reference
CPU encoder/DiT e MPS VAE: conditioning bit-exact, RMSE massimo degli otto
latenti 0,0045133, SSIM RGB 0,999323 e alpha MAE 1,8179e-5, entro le soglie
congelate. PNG ispezionato: quadrato rosso e cerchio blu sul fondo bianco.
La VAE CPU scalare resta NOT_VERIFIED (interruzione ordinata dopo 1111 s
misurati); stack, identità del processo e cleanup senza SIGKILL sono salvati.

Il replay della coda a 72 canali rende esatta l'attention Qwen sulla reference.
Aggiunto il confronto autonomo della patch Conv3d: 69/294912 differenze,
max 0,00390625. Riesaminati `ConvolutionMM3d.cpp` e `gemm_notrans_` della
revisione fissata: i tre assi diventano la stessa riduzione a quattro somme,
con il bias nell'epilogo GEMM. Il caso CUDA locale mantiene un cast cuDNN
prima del bias; quel dettaglio non va trasferito al backend CPU scelto.
Si verifica ora il prototipo Conv a quattro somme anche sul patch embedder
3D reale, prima di estendere la selezione del prodotto oltre Conv2d.

Il replay della patch Conv3d a quattro somme elimina tutte le 69 differenze;
si estende quindi il relativo dispatch BF16 ai tre assi. La dot attention
a blocchi/coda viene raccolta in un helper che possiede entrambi gli ordini
CPU, usando quattro lane per il caso batch e preservando le somme. Il ramo
a 72 canali è ammesso solo dopo il replay reale esatto; i test e i forward
completi restano richiesti sulla combinazione integrata.

Il replay audio al layer sei conserva una sola differenza in channel RMSNorm;
tutte le altre primitive e probabilità sono esatte. Riesaminati la norm HF,
la specializzazione ATen pow(-0,5)→rsqrt, vectorized_outer_sum/row_sum
e le alternative ggml/locali: questo percorso usa ancora sqrt Metal veloce
e divisione, mentre la norm per righe usa già reciproco di sqrt precisa.
Prima di cambiare l'ordine di riduzione si confronta un candidato limitato
a RMS channel-major BF16 con lo stesso reciproco preciso; L2 e FP32 restano
separati. Se il replay non diventa esatto, la somma esterna CPU verrà isolata.

Il solo reciproco preciso conserva il tie channel RMS al layer sei. Si
riesamina quindi vectorized_outer_sum: row_sum usa quattro stream e una
cascata a quattro livelli, poi aggiunge in ordine le quattro somme. La
riduzione generica GPU su 256 lane non conserva quell'ordine. Il candidato
successivo è limitato a channel RMS BF16, mantiene il layout fisico originale
e usa quattro lane per esprimere la stessa cascata; L2/FP32 restano invariati.

Il riesame riga per riga precisa una distinzione del percorso outer:
i gruppi di 32 posizioni non ridotte usano una cascata per posizione su
tutti i canali; soltanto i gruppi residui di quattro e le posizioni scalari
usano row_sum con quattro stream. Il candidato riproduce questa selezione
in base alla geometria completa, senza scegliere i singoli valori numerici.

Con la geometria outer completa, il replay del layer sei è interamente
esatto, comprese RMS channel-major e probabilità FP32. Si integra il kernel
dedicato mantenendo invariati L2 e FP32 e si verifica anche con larghezze
non multiple di quattro e code temporali di una/quattro posizioni.


Il forward multimodale completo successivo passa tutte le 480/480 frontiere
BF16/probabilità e i vettori finali, con gli stessi dump CPU e criteri congelati.
Questo risultato sostituisce il precedente 125/480 soltanto nel sottoinsieme locale.

Il nuovo conditioning RGBA resta FAIL (RMSE 0,767220), pur passando tutti
i latenti e i pixel. Il trace completo RGB localizza la prima differenza nel
blocco vision zero, mentre pixel/input IDs e patch Conv3d sono esatti.
Il riesame riparte da Transformers ca8c871 (vision_utils.py interpolazione,
Qwen3VLVisionRotaryEmbedding), PyTorch d0f2af8 (SumKernel outer e vec128_float_neon)
e SLEEF 5a1d179 (xsinf/xcosf u35, non u1). I riferimenti Strata 46c152a
units.metal e llama.cpp dde879b mtmd/models/qwen3vl.cpp conservano architettura
RoPE e interpolazione ma non l'ordine CPU/transcendentali; non sono un oracle
bit-exact. Riesaminate anche le precedenti diagnosi RoPE CUDA/Metal del repository.

Due cause indipendenti misurate: l'espressione C dell'embedding posizionale
contrae le moltiplicazioni/somme (66524 FP32,15 BF16,10 input vision differenti).
La stessa espressione con contract(off) è interamente esatta. Le frequenze
angolari host coincidono tutte con CPU; sin/cos Metal differiscono di pochi ULP,
producendo 9/5 differenze BF16 dopo RoPE su Q/K. Anche libm host non coincide
con le funzioni SLEEF CPU e non costituisce una soluzione. Si prepara un prototipo
Metal dei polinomi SLEEF u35 nell'intervallo documentato |x|<125, mantenendo
il percorso Metal generale fuori da quell'intervallo, e si verifica prima
sulle frequenze identiche. Nessun nuovo calcolo del forward su CPU.


Il prototipo sin/cos u35 è bit-exact su 8484 valori per funzione: tutte le
frequenze vision originali, intervallo denso con segno, zero con segno e
subnormali. Si integrano i polinomi nel backend numerico con FMA esplicite
solo dove le usa SLEEF; fuori |x|<125 resta la funzione Metal generale.
L'encoder C viene compilato con -ffp-contract=off per conservare le operazioni
del processor nell'interpolazione, senza calcolo dei layer su CPU. Ripetuti
encoder RGB/RGBA completi e regressioni numeriche; anche l'audio Gemma usa
sin/cos, quindi il suo gate completo deve essere ricontrollato dopo l'integrazione.


Dopo interpolazione e trigonometria, tutti i 27 blocchi vision RGB sono
bit-exact. Restano i merger deepstack: il replay indipendente dei quattro
moduli verifica norm e proiezioni esatte, mentre GELU erf differisce
(12860/379/3715 valori nei tre deepstack, zero nel merger finale, max1,5259e-5).
Riesaminati Qwen3VLVisionPatchMerger (nn.GELU senza tanh), Gelu.h e
vec128_float_neon.h alla revisione d0f2af8: questa erf NEON usa una specifica
approssimazione con FMA e vexpq_f32_u20, non xerff SLEEF. Il precedente
approssimante erfc Metal conserva code non nulle dove il riferimento
calcola 1+erf==0, quindi vicinanza matematica non garantisce i tie BF16.
Gli shader ggml erf_approx e Strata units restano approssimazioni diverse;
il precedente audit GELU tanh non copriva questo caso. Si prepara un candidato
BF16 erf limitato alla sua API, mantenendo il percorso generico FP32.
La nuova verifica Gemma completa conserva 480/480 frontiere esatte dopo sin/cos.


Il candidato erf NEON rende esatte tutte le attivazioni dei quattro merger.
Lo sweep BF16 finito, ordinato e rimescolato, isola 508 differenze soltanto
sui subnormali: è la stessa FTZ già documentata dal manuale MSL e dal precedente
sweep GELU tanh/SiLU. Si riusa il percorso intero RNE per x/2 anche per GELU erf
nell'identico intervallo minuscolo; il primo risultato fallito resta separato.
La selezione exp_u20/SLEEF per valore non introduce differenze BF16 nello sweep
ordinato né in quello rimescolato, né nelle quattro geometrie reali.


GELU erf con il percorso tiny condiviso passa bit-exact tutti i 65280 BF16
finiti sia in ordine di storage sia rimescolati, oltre a 1179648 attivazioni
reali dei quattro merger. Si integra quel ramo soltanto per input/output BF16;
GELU FP32 conserva il percorso precedente. Attribuzione ATen/Arm e licenza
PyTorch esistente sono collegate nel sorgente; la revisione e lo header
ufficiale sono salvati nel manifest dell'audit.


Il replay linguistico RGB layer zero conferma esatti tutti i calcoli nativi,
compresi cos/sin BF16, RoPE, attention e MLP. Il confronto del trace text-0
rivelava però una frontiera diversa: il forward hook HF osserva il decoder
prima di _deepstack_process, che clona il tensore nel modello contenitore;
l'observer C era dopo l'aggiunta. Si allinea l'observer all'uscita del decoder
senza cambiare il calcolo e si corregge il diagnostico: il replay del layer
successivo aggiunge la deepstack precedente al suo input. Conditioning
finale, oracle indipendenti e criteri numerici restano invariati.

## Riesame finale delle famiglie integrate

I tentativi e i riferimenti precedenti restano cronologici: le frasi che
lasciano un esperimento aperto descrivono quel punto dell'indagine. La tabella
riassume il percorso prodotto finale e distingue i quattro riesami. Le
revisioni sono quelle fissate sopra; nessun oracle è stato aggiornato per
assorbire una differenza e nessuna soglia è stata allargata.

| Famiglia | S1: fonte e scelta | S2: primo confronto identico | S3: tuning e misure | S4: integrazione |
| --- | --- | --- | --- | --- |
| GEMM BF16 e layout trasposti | ATen CPU/NEON: ordini 32 elementi o quattro stream secondo il layout. Strata FP16 e ggml SIMD-group BF16 non conservano i tie CPU; candidate SIMD non integrate. | Replay PLE/proiezioni sugli stessi operandi CPU; code e layout verificati prima dei forward. | Baseline BF16 nelle tre geometrie originali: 47,12/9,21/35,92 ms, due campioni. Nessun confronto di velocità contro Strata. | Gemma testo/media e Qwen encoder completi bit-exact; workspace esplicito e ammissione verificati. |
| RMSNorm, LayerNorm e channel RMS | ATen riduzioni cascade/Welford, scala e round distinti Gemma/Qwen; Strata/ggml usano riduzioni diverse. | Stessi input scalari/vettoriali, geometrie outer/channel e code audio; le varianti con drift sono state rifiutate. | Verificati costo/storage bounded e full forward; la baseline misura separatamente RMS BF16 e norm generiche FP32. Non attribuisce questi tempi alla LayerNorm BF16. | Tutte le frontiere norm nei gate finali 81/480/69 sono esatte; parametri di layout posseduti dall'API numerica. |
| Attention, softmax, tanh/exp | ATen seleziona dot stream differenti per batch/teste/layout; SLEEF fissa exp/tanh. Le primitive Strata/ggml sono confronti statici, senza equivalente di tutti questi confini BF16. | QK/probabilità/PV isolati, D72, B2 video, maschere e FP32 audio; niente matrice NxN temporanea del kernel BF16. | Baseline window BF16 e MQA FP32, due campioni; ledger e workspace bounded. Ordine mantenuto anche quando più costoso di MPS. | Probabilità FP32 audio e tutti i blocchi media/Qwen esatti; numerica image completa ripetuta dopo tanh. |
| Conv2D/Conv3D BF16 | ATen ConvolutionMM/BLAS: quattro accumuli e bias FP32 finale. Le ricette generiche ggml non sono un oracle di arrotondamento. | Replay sugli stessi patch/audio input, confronto unfold/linear/conv; Conv3D e dimensioni di bordo verificate. | Patch Conv3D della geometria reale: mediana 65,47 ms. Nessun nuovo lowering im2col o conversione lossy. | Patch vision e stem audio esatti; media 480/480, pipeline image completa PASS. F32 esistente separato. |
| GELU erf/tanh, SiLU e valori tiny | ATen NEON/Arm exp_u20 e SLEEF; approssimanti erf ggml/Strata incompatibili con i tie. Manuale MSL per FTZ, licenze complete conservate. | Tutti i 65280 BF16 finiti ordinati/rimescolati e 1179648 attivazioni merger reali esatti; tiny RNE condiviso. | Nessuno speedup rivendicato. La baseline GELU generica non sostituisce la qualifica dell'erf BF16; gli sweep includono anche gli input subnormali. | Quattro merger, Gemma e conditioning finali esatti; ramo erf limitato a BF16, percorso FP32 conservato. |
| Sin/cos, RoPE e interpolazione host | HF Qwen3-VL, SLEEF u35, ATen e precedenti ricerche locali; Strata/ggml hanno semantica RoPE ma trascendentali differenti. | 8484 angoli per funzione, signed zero/subnormali; interpolazione senza contrazione FMA, patch input e ogni blocco vision esatti. | Baseline RoPE BF16 e input identici; nessuna misura Strata alle stesse condizioni. Polinomi limitati all'intervallo verificato. | RGB/RGBA 69/69 ciascuno e Gemma media 480/480 dopo il cambiamento condiviso; encoder/DiT/VAE completi PASS. |

La baseline locale completa contiene 14 primitive, un warmup e due campioni,
con operandi/output NPZ e SHA della libreria. Tutte rispettano i limiti
preesistenti, 11 sono bit-exact. Si tratta di `BOUNDED_SAMPLE`, non di una
qualifica statistica. I tempi di fase dei forward reali e i picchi posseduti
sono nei rispettivi report; non sono confronti con altri Mac o con CUDA.
La copertura Strata del modello image e degli ordini CPU BF16 richiesti è
insufficiente per un benchmark equivalente: il confronto S3 resta statico,
come consentito dal piano, e non giustifica alcuno speedup dichiarato.
