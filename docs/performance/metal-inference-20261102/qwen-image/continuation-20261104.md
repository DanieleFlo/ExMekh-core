# Qwen Image: kernel VAE e RAM dinamica, 4 novembre 2026

Il ciclo mantiene le convoluzioni BF16 ordinate e la cache a budget continuo,
conservando i gate numerici. Nessun commit. Modello Qwen-Image-2.1-Turbo BF16,
revisione `fc13ba1986fcb905854dc8de2f52eee395f49b5b`, M5 Pro24GiB,
256×256,8passi,CFG1,seed20261020. I risultati riguardano questo subset locale.

## Risultato della release

Cinque coppie contro Conv scalare/cache discreta, stesso sampler finale,
pool autorelease, fixture e ceiling8.5GiB. VAE53.960→10.406s;
operazione156.165→115.846s, delta appaiato−25.214%
IC95[−26.848,−23.665]. Tutti110artefatti identici, memoria host minima4.398GiB
e crescita swap0. Il processo passa169.374→128.624s.
Il dettaglio separato delle fasi è nel [report finale della cache](continuous-cache-20261104/README.md).

Il DiT misura+4.835% in queste coppie: il piano aggiornato usa4.747–6.213GB
rispetto ai7.115GB iniziali del controllo, con più ricaricamenti. La release
non presume che più byte inizialmente stimati siano ancora spendibili.
Il guadagno misurato riguarda il bundle, non ogni fase singolarmente.

## Implementazione

- [I-P02](ordered-conv-20261104/README.md): Conv2D BF161×1/3×3, quattro lane
  per i quattro stream FP32 del CPU BLAS, due punti spaziali condividono il peso.
  FMA, riduzione sequenziale, bias e round BF16 preservati. Capability opzionale,
  fallback generico, selezione nel proprietario Metal e pool al confine C.
- [I-P04](continuous-cache-20261104/README.md): il planner amplia in byte la
  variante ammessa usando l'API pubblica dell'executor. Conserva picchi,
  checkpoint/restore, vittime e limiti espliciti. Numeric aggiorna la capacità
  senza esporre gli handle Metal; la quota viene ricalcolata per richiesta.
- Il conteggio Numeric esclude la parte di file cache protetta da XNU prima
  della compressione/swap. Il floor viene letto dal sistema; la riserva resta
  2GiB. Il probe automatico usa il ceiling hardware, senza congelare il primo
  campione di disponibilità. Tre automatici finali passano con swap growth0;
  la quota DiT osservata cambia2.323→6.231→5.977GB con la capacità sicura.
- [I-P03](weight-transport-20261104/README.md): copia parallela dei pesi ritirata.
  Il micro caldo migliorava ma peggiorava l'encoder completo; nessun worker,
  buffer extra o flag di quell'esperimento rimane nel prodotto.

## Verifica e fonti

Otto forme isolate esatte contro PyTorch CPU, sottoviste/sentinelle,
batch/gruppi/stride/dilatazione e fallback senza pipeline opzionali PASS.
La release supera testo/RGB/RGBA contro gli oracle congelati CPU encoder/DiT
e MPS VAE, senza cambiare criteri.57artefatti finali esatti nei cinque gate,
oltre ai110nelle coppie. Gemma multimodale480/480 e vettori128PASS.
Build prodotto e test CPU PASS; Numeric Metal conserva soltanto le due
asserzioni sin/cos preesistenti, riprodotte sul controllo. Test policy memoria,
planner, risorse, executor/restore, lifetime, bundle e Image PASS.

Il riesame finale legge il totale dei pesi dal campo memoria del gruppo,
anziché dal costo: entrambi erano inizializzati agli stessi byte. Build e test
mirati ripassano; un ulteriore gate automatico sul prodotto finale passa in
115.458s, con11artefatti identici e crescita swap0. Non è una nuova misura
appaiata. `final-artifacts.json` distingue gli hash dei binari misurati da
quelli dopo questo refactoring; `reviewed-gate.json` conserva l'ultimo gate.

Il profilo successivo del prodotto finale conferma il residuo nel trasporto:
427/666campioni principali nella copia dei pesi e217nelle proiezioni/attesa
GPU. Anche questa richiesta passa il gate,11artefatti identici e swap growth0;
resta esclusa dalle misure appaiate (`final-profile.json`).

Gli audit prima delle implementazioni e dopo ogni differenza sono nei report,
con pin/hash di Diffusers, Transformers, ATen, Strata, ggml e XNU. Il profilo
[GitHub di YASAKANI](https://github.com/morphicode-jp) è stato ricontrollato:
nel profilo indicato resta pubblica soltanto `f32-patch-gemma`, una patch dei
pesi Gemma per accuratezza. Non è stato trovato codice pubblico del motore
da trasferire; il benchmark RTX5090 non è usato come misura Metal.

## Riesame prima dell'integrazione

La validazione pubblica calcolava l'output della convoluzione con padding a
64 bit. I wrapper CPU e Metal lo ricalcolavano a 32 bit. Nel replay Metal con
larghezza 1, padding sinistro `UINT32_MAX`, padding destro 1 e stride
`UINT32_MAX`, la validazione ammetteva due output e il wrapper ne scriveva
uno. Il risultato osservato era `[0, -1]` con la sentinella al secondo posto.

Riletti Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`
(`src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`,
`QwenImage21CausalConv3d`) e Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`
(`src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`, patch embedder).
Il primo applica padding 2D esplicito e il secondo usa una Conv3d distinta.
Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`
(`metal/src/mtl_rt.mm`, `metal/kernels/dense_mm.metal`,
`metal/tests/dense_mm_parity.cpp`) fornisce il controllo delle capacità ma
non una convoluzione equivalente. ggml `dde879bc4f954b809892fd0d660515f42536394a`
(`ggml/src/ggml-metal/kernels/conv.metal`, `ggml-metal-ops.cpp`,
`ggml-metal-device.cpp`, `tests/test-backend-ops.cpp`) calcola i confini
spaziali per il proprio contratto FP16/FP32. Il [report I-P02](ordered-conv-20261104/README.md)
fissa gli ordini BF16 e i fallback locali. Il wrapper cuDNN già rifiuta
dimensioni/stride fuori dal suo dominio `int`.

La geometria CPU e Metal riusa il calcolo per asse di Numeric. Metal conserva
il limite di griglia e prepara il proprio ABI dentro il metodo proprietario.
I kernel conservano il medesimo ordine.
Il planner usa una mappa fase/componente unica e riutilizza il campione
all'ingresso del tentativo di ammissione. Il fallback può ricostruire
l'executor al tentativo successivo se una ricostruzione fallisce.

Build, test del planner anche con ASan/UBSan, Numeric CPU (47 casi, 13 skip),
fondazioni Metal residenti e streamed, fixture Flash SSD, bundle e HTTP
locale superati. Numeric Metal supera il replay con padding largo e le
sottoviste BF16. Le due asserzioni sin/cos restano riproducibili sul controllo.

La prova Metal del cap termina con `NOMEM` sulla piccola allocazione di
recupero. Il gate Gemma e i due probe Image, controllo congelato e prodotto
riesaminato, si fermano a `ds4_bundle_open` prima dell'inferenza. Il campione
contestuale di XNU mostra 391227 pagine file contro un minimo protetto di
458611 e circa 0,93 GiB fra libere e speculative. Dopo l'attesa il limite
superiore della capacità recuperabile è 1,69 GiB, sotto la riserva di 2 GiB.
I gate completi del riesame sono `NOT_VERIFIED` per risorse. Le misure e i
gate del ciclo I-P02/I-P04 restano attribuiti ai binari congelati. Log, hash
e verifiche del riesame sono in `integration-review.json`.

## Residui del piano

Restano profilazione del trasporto per gruppo/prefetch bounded, misure su
risoluzioni/recipe più grandi e la distinta campagna prestazionale Gemma.
Il rifiuto della copia parallela non dimostra l'inutilità di ogni tecnica SSD.
Qwen27 e Flash mantengono i cicli locali precedenti; nessun nuovo speedup LLM,
CUDA, distribuito o F08 è dedotto da queste prove Image.
