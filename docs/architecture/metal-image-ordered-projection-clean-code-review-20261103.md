# Riesame delle proiezioni BF16 Metal

Riesame del diff I-P01 e dei consumer contigui richiesto il 3 novembre.
Baseline repository `f2fed3e6dbbb8473adae1733a9ececd7cb0d8eb8`.
I-P01 rimane congelato nei binari `probe-after` e `numeric-after.so`;
le cinque coppie documentate conservano la loro attribuzione.

## Responsabilità e problema trovato

Letti `ds4_numeric.c` (validazione linear/matmul, slicing, overlap, lifetime),
`ds4_numeric_internal.h`, `ds4_numeric_weights.c`,
`ds4_numeric_metal.m` (costruttore, binding, runKernel, nm_gemm),
`ds4_numeric_metal_args.h`, `metal/numeric_ordered.metal`,
`tests/suite_numeric_api.py` e `tests/test_suite_numeric.py`, oltre ai
consumer encoder, DiT, VAE e cache dei pesi Qwen Image.

L'API numerica pubblica valida shape, dtype, intervalli e alias prima del
backend. Il backend possiede buffer, pipeline, capacità e completamento.
Il tile I-P01 era stato aggiunto alle pipeline obbligatorie. Un fallimento
del solo tile, o un suo limite inferiore a 256 thread, poteva quindi
rendere indisponibile tutto il backend anche con la proiezione scalare
operativa. È un problema di capability riscontrato dal codice. Il M5 Pro
in prova supporta entrambe le pipeline.

Il costruttore deve poter omettere la sola pipeline ottimizzata. La scelta
tra tile e scalare deve appartenere a `DS4MetalNumeric`, proprietaria
delle pipeline. Il callback C `nm_gemm` delega l'operazione BF16 al metodo
del backend e conserva conversioni e MPS per le altre geometrie.

## Audit prima del refactoring del dispatch

Riletti i pin della [scheda I-P01](../performance/metal-inference-20261102/qwen-image/ordered-bf16-20261103/README.md).

- ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/ggml-metal-ops.cpp:ggml_metal_op_mul_mat`
  seleziona MMA soltanto con `has_simdgroup_mm` e conserva altri percorsi.
  `ggml-metal.cpp:ggml_backend_metal_device_supports_op` delega la
  disponibilità al device. Trasferibile la distinzione tra operazione
  disponibile e singolo algoritmo. Il nostro fallback deve mantenere
  la riduzione FP32 ordinata già qualificata.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/mtl_rt.mm:Context::pipeline/require_fit`, legge capacità
  effettiva di ogni pipeline ed evita dispatch oltre il limite, con
  un hook di test sul massimo dei thread. `metal/tests/dense_mm_parity.cpp`
  verifica la geometria delle proiezioni. Le sue MMA con input FP16
  conservano le incompatibilità numeriche registrate in I-P01.
- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
  `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`, e Diffusers
  `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
  `src/diffusers/models/transformers/transformer_qwenimage21.py`,
  `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`,
  `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`.
  Restano pertinenti proiezioni BF16, cache causale, otto passi della
  recipe locale e VAE distinto. La scelta di capability appartiene al
  backend e conserva dtype, operazioni e confini di arrotondamento.
- Locale `DS4MetalNumeric:initWithResources/runKernel`, API
  `ds4_num_linear/ds4_num_matmul_batch_layout`, gate I-P01 e riesami
  numerici T06–T07. Niente accesso nuovo allo storage privato del tensore.

Scelta minima prima del codice. Rendere opzionale il tile e racchiudere
parametri e selezione della proiezione in un metodo privato del backend.
Le pipeline scalari restano necessarie per tutte le forme di fallback.
Nessuna estrazione di utility generiche o configurazione del prodotto.

## Verifiche previste

Replay BF16 su input identici, incluso il tile con una riga parziale e
bias, contro I-P01 e CPU. Build diagnostica con la capability del tile
rifiutata nel costruttore, usando l'API pubblica per verificare il fallback.
Poi pipeline indipendenti testo/RGB/RGBA sul binario riesaminato, test
Numeric e regressione Gemma condivisa, oltre alle build di prodotto.
Le misure I-P01 vengono mantenute distinte dalla verifica del riesame.

## Nuova differenza di tempo e controllo

Il primo testo riesaminato passa tutti i gate ma misura 137,131 s, DiT
68,735 s, contro 129,852 / 61,629 s mediani I-P01. Encoder e VAE restano
14,074 / 54,322 s, con stessi byte e caricamenti. È un confronto iniziale
non appaiato. Viene conservato e motiva due coppie di controllo dirette
I-P01/riesame, con ordine alternato e cap identico, dopo i gate sul modello completo.

Riaperto l'audit per questa differenza prima di cambiare codice.
Transformers e Diffusers ai pin sopra mantengono le stesse proiezioni BF16,
maschere, cache e recipe. ATen `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`,
`ReducedPrecisionFloatGemvFastPathKernel.cpp`, main loop no_bfdot e
riduzione, conferma le catene già riprodotte. La sorgente Metal è identica
tra I-P01 e riesame. Strata `46c152a...`, `mtl_rt.mm:require_fit`,
richiede il limite prima del dispatch. ggml `dde879...`,
`ggml-metal-ops.cpp:ggml_metal_op_mul_mat`, sceglie il percorso per shape e
capability. Il riesame invia gli stessi argomenti e griglia; aggiunge la
query interna della pipeline opzionale. I micro sulle tre forme reali
misurano I-P01/riesame 9,754/9,869, 30,378/30,062 e 31,983/31,876 ms.
Questi dati non attribuiscono i sette secondi del DiT a una causa.
Riletti anche l'owner dei comandi e i risultati locali Q-P24, dove il
micro non spiegava il modello completo. Il controllo appaiato verifica questa differenza
con kernel, dispatch e margini congelati.

## Risultati del riesame

- Build di prodotto, server suite e archivio Metal PASS.
- Test di input/output Image, grafo ridotto e timestep, bundle e factory
  acceleratore PASS dopo il refactoring. Il primo test bundle nel sandbox
  non poteva aprire localhost. La ripetizione autorizzata passa tutti i
  dodici casi e conserva il log del tentativo iniziale.
- Otto forme isolate per quattro varianti (baseline scalare, I-P01,
  riesame e fallback forzato) coincidono bit per bit con CPU e tra loro.
  Il fallback viene provocato nel costruttore di una build diagnostica
  separata e verificato tramite l'API numerica pubblica.
- Gemma multimodale condiviso conserva tutte le 480 frontiere esatte e
  passa il controllo indipendente dei vettori a dimensione 128.
- Numeric Metal esegue 43 test. Passano 39 test, tre sono saltati. Il caso
  `test_vision_trigonometric_boundaries` conserva due asserzioni fallite,
  op13 e op14, riprodotte anche dal binario baseline in un processo
  distinto. La nuova regressione BF16 con bias e riga parziale passa.
  Le soglie restano congelate. Il dettaglio è in `review-regressions.json`
  nel report I-P01.

La revisione mantiene il contratto pubblico di validazione e ownership.
I consumer C passano viste e dimensioni al metodo del backend. La
disponibilità delle pipeline rimane privata a DS4MetalNumeric. Le verifiche qui
raccolte coprono il backend Metal locale.

I tre gate completi testo/RGB/RGBA del binario riesaminato passano con le
soglie indipendenti congelate. Tutti i 35 artefatti nativi coincidono byte
per byte con la baseline. Il caso con budget automatico passa e conserva
almeno 5,307 GiB host disponibili, con swap invariato.

Il controllo diretto allo stesso cap misura I-P01/riesame
127,536/127,558 s nella coppia A e 127,308/125,002 s nella coppia B.
Delta dell'operazione +0,017% e −1,812%, output tutti bit-exact. Il
rallentamento iniziale non si ripete. Le due coppie controllano questa
differenza e non stimano un guadagno del refactoring. Sono conservati sia
il primo campione lento sia i controlli in `review-controls.json`.

La conferma prestazionale di I-P01 resta quella dei cinque blocchi
alternati contro la baseline precedente al tile. Mediana dell'operazione
207,581→129,852 s e delta appaiato −36,90% [−38,46;−35,29]. I numeri sono
attribuiti ai binari congelati, mentre gate, micro e controlli qui
descritti qualificano l'integrazione del riesame.

## Commit di integrazione

`9634c83` contiene il kernel e la regressione BF16. `689c131` contiene il
metodo proprietario della proiezione e il fallback di capability.
Le verifiche del riesame qualificano le due modifiche insieme.

## Secondo riesame dopo b2d94a8

Riletti i tre commit integrati, il validatore pubblico linear/matmul, il
binding Metal, le viste, la gestione dei pesi, il lifecycle numerico e i
consumer encoder/DiT/VAE. Il riesame conferma responsabilità e fallback
dell'implementazione integrata. La lacuna trovata riguarda la copertura
di viste con offset e capacità residua, comuni nei buffer riutilizzati.

La fixture Numeric ora espone `slice`, che delega a `ds4_num_slice`.
Il chiamante usa offset e numero di elementi. ABI, dimensione dei tipi e
indirizzo della sottovista restano gestiti dal binding e dall'API nativa.
Gli argomenti Python vengono validati prima della conversione uint64,
così valori negativi o fuori range non diventano offset validi tramite
wrap. La vista rimane borrowed e il Numeric proprietario mantiene il
buffer originale. I test verificano che la sottovista non allochi storage
e che i limiti siano rifiutati coerentemente.

La nuova regressione Metal usa una matrice BF16 con 65 righe, bias e
1030 colonne, offset non nulli per input/pesi/bias/output, output nella
capacità inutilizzata del buffer di input e sentinelle. Confronta tutti
i bit con l'oracolo CPU, verifica la conservazione della coda e rifiuta
l'overlap attivo prima del dispatch. I casi adiacenti di RMS e alias usano
la stessa API di slicing al posto dei calcoli manuali degli indirizzi.

Verifiche dopo le modifiche. La suite CPU esegue 45 casi, con 33 pass e 12 skip.
La suite Metal esegue 45 test, con 41 pass, tre skip e il solo test
trigonometrico già noto, fallito nei due subtest op13/op14. La nuova
regressione e la fixture delle sottoviste passano. Build di prodotto e
test di recovery/ownership del lifecycle numerico PASS.

Sorgenti C/Objective-C/Metal ed eseguibili di inferenza coincidono con
quelli qualificati nel primo riesame. Questo intervento cambia fixture,
copertura e documentazione. I gate completi e le misure precedenti
mantengono la loro attribuzione; il secondo riesame non introduce una
nuova misura prestazionale.

Evidenze del secondo riesame in [followup-review.json](../performance/metal-inference-20261102/qwen-image/ordered-bf16-20261103/followup-review.json).

Il secondo riesame è integrato in `e42497b` (API delle sottoviste) e
`9e38b1b` (copertura del tile e consumer adiacenti della fixture).
