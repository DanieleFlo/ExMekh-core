# Qwen27: budget adattivo e streaming per layer — 3 novembre 2026

Questa pagina conserva la campagna precedente all’integrazione. Il
[riesame e i commit successivi](../../../../architecture/metal-qwen27-adaptive-cache-clean-code-review-20261103.md)
registrano le correzioni del lifecycle e i gate del codice integrato.


La cache Qwen27 Metal ora cresce e si riduce durante l’uso, usa il residuo
misurato dopo grafo/restore/runtime e mantiene2GiB di margine. Il ring denso
a due slot resta attivo;8worker è stato provato e ritirato. Nessun commit.

Su64input/16output/8192allocati, tre richieste operative della versione finale Q-P25 con RAM dinamica:

| Modalità | Prefill mediano | Decode mediano |
| --- | --- | --- |
| Target-only |16.962359TPS|1.572404TPS|
| MTP4 |15.801759TPS|4.093100TPS|

I gate completi Qwen/Flash della versione qualificata sono bit-exact ai frozen. La suite Numeric
ha un limite preesistente su due assert sin/cos, riprodotto identico con
il sampler originale; dettagli e nessuna soglia allargata sotto. I confronti
a cap fisso e gli outlier sono distinti da queste misure operative.

B0: `ab62874723c837ade442c192b90f0fba68bd2f73`, bundle P18 più riesame Clean Code. Nessun commit autorizzato in questa continuazione. Artefatti grezzi: `build/metal-perf-qwen27-20261102/phase2-20261103/`; snapshot/hash `initial.json`.

## Q-P20 S1 — confronto prima dell’implementazione

Problema osservato: il default fisso 2 GiB non segue il budget della macchina. Il tentativo di profilo fresco a1536MiB/8192/64 è stato rifiutato: cache+grafo+runtime2.92GiB, budget2.46GiB, host4.46GiB, pressure1. Il primo tentativo aveva un’opzione CLI errata (`--teacher-forcing`): NOT_RUN, conservato separatamente. Il secondo è un rifiuto admission, non una misura TPS. Il profilo P18 precedente allo stesso algoritmo resta evidenza storica di I/O dominante, non una nuova misura.

Fonti rilette per questo intervento:

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`, `src/transformers/models/qwen3_5/modeling_qwen3_5.py:Qwen3_5DecoderLayer.forward`: norm, mixer, residual, norm, MLP e residual sono ordinati; cache e stato devono restare identici. Il budget non modifica operazioni, maschere, dtype o rounding. Nessuna competenza Diffusers coinvolta finché Image resta in coda.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`, `metal/src/engine/stream.cpp:make_arena`, `arena_budget.h`, `mtl_rt.mm`: riservare prima gli slot minimi e poi aggiungere dati residenti nel residuo; recheck della capacità. Le classi di esperti e il margine percentuale/target M5 Max non sono trasferibili al27B denso/24GiB con riserva utente2GiB. Kernel `gemv.metal`/`dense_mm.metal` e test già nel manifest precedente non richiedono variazioni per il solo budget.
- llama.cpp/ggml `dde879bc4f954b809892fd0d660515f42536394a`, `ggml/src/ggml-metal/ggml-metal-device.cpp`, `ggml-metal.cpp`, `src/llama-mmap.cpp`, `src/models/qwen35.cpp`, `tests/test-backend-ops.cpp`: storage host/device condiviso, sincronizzazione prima del rilascio e limiti buffer; non importare stime CUDA VRAM né kernel con diverso contratto di quantizzazione.
- antirez/ds4 `890898cfb437ed256be58ebcf7c58a6dfa8f2f53` (HEAD remoto ricontrollato), `ds4_ssd.c:ds4_ssd_auto_cache_plan`, `docs/SSD_STREAMING.md`: costo fisso prima della cache e cap sul modello utile. Percentuale80%, minimo forzato di un esperto e assunzione tutti i pesi non-routed residenti non si applicano al27B denso su questo Mac.
- Locale: `ds4_metal_memory.h`, `qwen_metal_memory_admit`, `qwen35_metal_stream_retain`, Q-P06/P13/P17/P18/P19, audit CUDA27B `docs/experiments/qwen27-4bis-20261008/documentation/qwen27-weight-streaming-20261008.md` e riesame kernel/streaming20261103. Q-P19 ha già dimostrato che ridurre la sola riserva embedding non basta: non viene riattivato.

Scelta: capacità incrementale = min(host reclaimable misurato, recommended−allocated)−2GiB; sottrarre grafo, sessioni, runtime, MTP e vision prima di scegliere la cache. Cache=min(residuo, modello utile, eventuale limite esplicito). Il valore scelto è un limite lazy, non un’allocazione di padding. Pressione critica/telemetria assente/cap insufficiente continuano a rifiutare. I launcher Metal senza cap esplicito delegano all’admission; CUDA conserva i default. Il secondo cap percentuale generico non deve sovrascrivere il budget Qwen Metal già calcolato. Alternativa rifiutata: aumentare ciecamente2→4GiB, usare RAM fisica totale o ridurre la riserva.

Gate previsti: aritmetica CPU con underflow/limiti/cap espliciti/pressione; dry-run launcher; fixture Metal residenti/stream/lifecycle; full-model128 target/MTP2/MTP4 e Flash seriale se ammessi, confronti bit-exact contro i payload frozen. Criteri primitivi2e-4+2e-5|ref|, graph2e-5+5e-5|ref| invariati; per il solo trasporto si richiede bit-exact. Misure operative auto e prestazioni a cap comune restano distinte.

## Q-P21 — ricognizione, non ancora implementato

Nuova fonte primaria: tkhr-sait/ds4 branch `ssd-streaming-brushup`, pin remoto `9d8c1a8f1189dbe6403ca93097b551f8cc1e2f9c`, `ssd-streaming-memo/README.md`. Descrive doppio buffer per parità, riuso delle slab e rilascio prima della generazione; sono idee di lifetime da verificare nel codice, non speedup trasferibili. F_NOCACHE/page-alignment, no-mmap e MoE DS4 hanno assunzioni diverse. Non si importeranno claim dal memo senza codice/test e misure locali. Il pack massimo reale Qwen è259360768byte: head+due pack1561665536byte, quindi serve capacità sufficiente anche per binding temporanei. Un ramo per capacità può mantenere il percorso attuale quando non entra.

## Q-P20 S2 — gate

Build all/memory policy PASS;41 launcher+qualifier tests PASS dopo ripetizione con localhost autorizzato (prima40, un bind bloccato dalla sandbox). Fondamenta Metal residenti/stream/lifecycle PASS. Full Qwen128/8192 target con cache automatica2.32GiB PASS:10 artefatti logits/stato bit-exact rispetto al riesame frozen. La capacità è variabile, non un limite fisso su questo Mac. `tests/probe_qwen_metal_stream.c` usa ora l’admission automatica di default; il cap diagnostico esplicito resta disponibile. MTP e Flash ancora da ripetere.

## Q-P21 S1 — pack densi e doppio buffer prima del codice

Profilo B0 finalmente ammesso,1536MiB/8192/64,64input+4greedy target: prefill14.472991TPS, decode0.548800TPS;2481CB e4878encoder, somma span GPU5295.909ms. Profiler separato, nessuna percentuale promossa. Conferma il residuo fuori GPU e i molti confini di comando; il profilo MTP4 P18 aveva già53.54GB logici letti per16output.

Rilette specificamente HF Qwen3_5DecoderLayer.forward (pin/path sopra), ggml Metal asyncset/get/synchronize e rilascio dopo fence (`ggml-metal.cpp:ggml_backend_metal_free` e interfacce eventi), Strata stream.cpp size-class/min_slots/make_arena e i wrapper gemv/dense, oltre a Q-P13/P17/P18/P19 locali. Riletto ora il codice del nuovo fork tkhr-sait/ds4 `9d8c1a8f1189dbe6403ca93097b551f8cc1e2f9c`, `ds4_metal.m:ds4_gpu_stream_nommap_routed_issue/ensure/free`, cache slab e residency set per parità: aspettare lettura prima di pubblicare, drenare il consumatore prima di riscrivere, evitare buffer orfani. Il fork si basa sul drain per layer di MoE/Metal4; questo progetto usa batch asincroni su una sola queue, quindi servono fence dell’ultimo CB consumatore per slot. Non copiare volatile/dispatch_group né assumere che fine encode significhi fine GPU. Non modificare F_NOCACHE/FD/mmap in questo esperimento. La fonte CUDA27B locale riserva già due pack e supporto MTP; pinned/H2D/eventi CUDA non sono API Metal.

Scelta: il core fornisce head, embedding e intervalli completi per layer; il backend valida metadata, riserva due pack massimi e binding generici prima di trattenere head/layer nel residuo, e carica i pack lazy con il reader esistente. Il thread host legge il layer successivo mentre il CB del precedente gira, senza un nuovo worker di orchestrazione. Due buffer di capacità fissa, nessun resize/per-layer-free; prima di riscrivere si attende solo l’ultimo consumatore dello slot. Le attese parziali non dichiarano completata tutta la queue e non contano due volte gli span. Cap insufficiente mantiene la cache precedente. La riserva embedding per UD è96righe più piccoli pesi fuori layer, perché la nuova pipeline ha già riservato due pack; formati generici conservano la tabella completa. Questo è un piano diverso da Q-P19, da misurare a cap comune prima della promozione. Ordine FP32, dequantizzazione, KV, masks e payload invariati.

Gate: stessi input su grafi ridotti in batch e token singolo, slot riutilizzati oltre due layer, cap ridotto/fallback, errori EOF ripetuti/retry e close; full128 target/MTP2/MTP4 e Flash indipendenti bit-exact. Screening/profilo candidato poi cinque coppie contro P20 allo stesso cap, se admission consente. Se peggiora, nuova indagine upstream prima di cambiare dispatch o kernel.

Dettaglio dispatch S1: il submit prima del read è già nel percorso P08. Nel ring si evita di committare un batch vuoto: il controllo `g_batch_has_work` distingue lavoro codificato da una semplice creazione del CB; il flush del batch con il consumatore avviene prima del wait parziale/lettura. Nessun lavoro viene saltato, e le fence finali restano invariate.

## Nuova discrepanza Q-P20 — picco del restore MTP

Il ring supera i quattro gate ridotti (grafi/feature, Metal4 e fallback) e full target128 bit-exact. Con cache auto4.70/4.52GiB, MTP2/MTP4 avanzano correttamente fino al restore, poi la creazione del grafo candidato fallisce per memoria. `lifecycle_checkpoint_restore` mantiene il grafo live e chiama `ds4_session_create` per validare il candidato prima dello swap: è comportamento atomico intenzionale. L’admission automatica riservava solo le sessioni ordinarie. Il cap fisso precedente lasciava incidentalmente spazio e nascondeva il costo. Non è drift numerico e non si modifica il lifecycle per liberare prematuramente lo stato valido.

Riaperto l’audit per questo nuovo caso, prima della correzione: Transformers `ca8c871...`, Qwen3_5DecoderLayer.forward/past_key_values, mantiene stato del modello ma non implementa il nostro restore transazionale su disco; non trasferire il suo lifetime al candidato C. Strata `46c152a...`, stream.cpp:make_arena/arena_budget.h, conta minimi e working set prima delle residenze; non copre il restore Qwen27 locale. ggml `dde879...`, ggml-metal.cpp:ggml_backend_metal_free/synchronize, drena prima del rilascio e non autorizza a riscrivere stato in flight. Locali `ds4_lifecycle.inc:lifecycle_checkpoint_restore`, `ds4.c:qwen_graph_mtp_memory_estimate`, enable_mtp e report startup; audit CUDA27B20261008 già distingue snapshot/scratch/picchi. Alternative: distruggere prima il live (rompe atomicità), usare la riserva2GiB per il restore (vietato), cap fisso di nuovo (spreca RAM in altri contesti). Scelta: riservare un grafo candidato completo target+MTP oltre alle sessioni dichiarate, pesi ausiliari contati una sola volta. La cache usa tutto il residuo utile dopo questo picco.

Il primo run Flash full ha PASS interno ma il confronto frozen era su input diversi:64 token richiesti nel nuovo runner contro35 del riferimento senza padding. Il confronto viene marcato NOT_COMPARABLE e ripetuto senza padding, senza modificare kernel/soglie. I file e il fallimento dell’assert restano nei grezzi.

## Q-P21 S2 — full gate dopo la correzione del picco

Target128, MTP2, MTP4 e Flash35/8192/64 PASS;10/12/12/10 artefatti rispettivamente bit-exact contro i frozen del riesame, incluse le due verifiche MTP e i payload. FullMTP2/4 riproducono gli8 ID greedy e le massime differenze interne preesistenti1.1920929e-5/7.15255737e-6 entro le soglie congelate; il confronto con il medesimo percorso frozen è bit-exact. Cache automatica effettiva: target4.56GiB, MTP2~2.6GiB, MTP4~1.98GiB; valori operativi variabili, non un confronto TPS. Il restore atomico ora rientra. Nessuna misura precedente FAIL o NOT_COMPARABLE è stata cancellata.

## Q-P21 S3 — screening e controllo dei campioni

Profili a1536MiB: target14.472991→16.888972TPS prefill e0.548800→0.904232decode; CB2481→326, encoder4878 identici. MTP4 prefill13.282586→16.039063 e decode1.931332→3.218323, CB2580→425, encoder5997 identici; byte logici53.542→53.628GB, cioè il guadagno non deriva dal saltare pesi. Sono profili separati, non i campioni di promozione. Screening operativo auto: target1.138716TPS/prefill16.905121 con cache4655500800byte; MTP4 2.790572/prefill15.558171 con cache2583376512byte. Cache e momento diversi: non dedurre che più cache peggiori o migliori dal confronto non appaiato.

Il primo blocco di coppie è stato interrotto al warmup MTP4: il candidato ha rispettato l’admission e ridotto il cap a1537094272byte (1.43GiB) perché la RAM disponibile era scesa. Lo script ha escluso il campione non comparabile, non lo ha promosso come1536MiB. Ripetere a cap comune e conservare ogni tentativo; attendere il rilascio dopo il processo è ammesso, ridurre il margine o escludere il picco restore non lo è.

Riesame statico prima dell’ultima build: un binding abbandonato prima di creare il CB può lasciare un pin di argomento senza consumatore. Come gli entry pin già esistenti, il confine layer deve cancellarlo anche con `g_batch_cb=nil`; l’ultimo vero CB resta conservato. Le fence DS4/tkhr e ggml sopra distinguono encode/completamento e non richiedono attesa per un comando mai codificato. Aggiungere un test di output insufficiente prima del dispatch e successivo riuso dell’altro layer sullo stesso slot. Non cambia la matematica o il percorso valido misurato.

## Q-P23 S1 — budget elastico durante l’uso (richiesta esplicita successiva)

L’utente precisa che l’uso della RAM deve essere dinamico, non solo all’avvio. Prima del codice, riaperti Strata `46c152a...` `metal/src/engine/window.cpp` (copia pinned `continuation-20261103/strata-window.cpp`), grow/recheck/shrink dell’arena, e `stream.cpp:make_arena`: budget rivalutato, minimi prenotati e nessuna crescita cieca. Nel suo scheduler i layer/esperti residenti e gli outstanding prefetch impongono fence; qui i pesi sono densi e i due slot hanno consumatori espliciti. ggml `dde879...` `ggml-metal.cpp` sincronizza prima di liberare, modello Qwen35 immutato; HF `ca8c871...` forward e cache restano identici. tkhr `9d8c1a8...` `ds4_metal.m` issue/ensure/free e slab riusate separano metadata/storage e drenano GPU/read prima del riuso; non copiare la sua assunzione per-layer end_commands. Locali `ds4_resources.h`, owner delle sessioni, `lifecycle_checkpoint_restore` trasferisce già l’owner del candidato dopo lo swap; Q-P17/P18 richiedono quota distinta da live e retry senza buffer orfani.

Scelta: nuovo calcolo del cap prima di ogni forward e prima delle allocazioni di sessione/restore, usando campioni host/Metal correnti. Accreditare una sola volta i byte già posseduti dalla cache e sottrarre soltanto allocazioni future: il registro pubblico delle risorse permette di riconoscere grafi live delle sessioni e pesi ausiliari residenti senza leggere storage privato dei tensori. Mantenere sempre almeno un grafo candidato completo più runtime/vision, anche con sessioni oltre l’hint. Cap espliciti restano tetti, non disattivano la protezione. Critico/telemetria assente non autorizzano crescita. Se il budget scende, il backend ricalcola regioni e drena prima di liberare; se sale, aggiunge regioni lazy conservando i buffer comuni e il head caldo. Le variazioni che non cambiano regioni/slot aggiornano solo il limite: niente reload per rumore di pochi MiB. Gli intervalli del modello restano copiati anche quando due pack non entrano, così una crescita successiva può attivarli; il fallback mantiene letture esplicite e non torna mai a mmap dell’intero modello, nemmeno al cap zero.

Non inserire mutazioni pesanti in un predicato chiamato require_resident: il punto comune diventa prepare_execution, che dichiara la preparazione del budget e conserva le verifiche di residenza. Ingressi normali/spec/prefill usano lo stesso punto. Nessuna distruzione del KV per adattare i pesi. Test: sequenze deterministiche di memoria simulata nell’aritmetica pura, resize reale piccolo↔grande con input identici e accounting, pressioni basse senza fallback mmap, restore; poi full gate indipendente e misura dei costi dinamici. Le precedenti misure statiche P21 restano attribuite a quel binario.

## Nuova discrepanza Q-P23 — snapshot XNU dopo release

P23 isolati PASS, MTP2 e Flash full bit-exact; target/MTP4 falliscono al resume. Il probe diagnostico usa solo API pubbliche e mostra: target prima del release owned9158942232byte/Metal10418831360, dopo release owned0/Metal1235632128; il campione host resta esattamente3667902464byte in tutti e sei i punti ravvicinati. Quindi i buffer sono stati liberati, ma la VM sample non riflette ancora il rilascio. Non correggere aumentando artificialmente la telemetria o spendendo il margine.

Nuovo audit prima della correzione: Apple XNU HEAD verificato `f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`, `osfmk/kern/host.c:rate_limit_host_statistics` e `vm_stats`: per processi non platform il risultato può essere cached dopo2–10richieste in una finestra1secondo. `bsd/vm/vm_unix.c:2565–2574` espone i contatori UINT read-only `vm.page_free_count`, `vm.page_speculative_count`, `vm.page_pageable_external_count`; il free_count Mach comprende già speculative, mentre il sysctl free è il contatore base. I tre sysctl sono stati verificati disponibili sul Mac. Sorgenti raw/hash nei grezzi. Strata `46c152a...` window.cpp:avail_now usa anch’esso host_statistics64, quindi non risolve automaticamente questo caso e il suo timeout/recheck non va copiato. ggml `dde879...` Metal synchronize/free e tkhr `9d8c1a8...` free/slab confermano che il release deve seguire il drain, già osservato dal calo currentAllocatedSize. HF `ca8c871...` state/cache non coinvolge il campionamento OS, nessun dtype/operation order cambia. Locali ds4_metal_device.h e resource tests: telemetria distinta dalla policy; il ledger non deve fingere nuova RAM fisica.

Scelta: campionare free e speculative dai sysctl read-only correnti (sommandoli una sola volta), mantenere il limite conservativo inactive e limitarlo ulteriormente al contatore pageable_external corrente; non usare external_page_count cached per accreditare file cache scomparsa. Pressione e currentAllocatedSize restano misurati come prima. Se i contatori richiesti non sono leggibili la query fallisce chiusa. Nessun processo/helper esterno o attesa fissa di un secondo nel forward; nessun incremento sintetico dei byte liberi. È un aggiornamento del sampler condiviso, quindi verificare anche fixture Numeric Metal oltre a owner storico, e non attribuirgli un nuovo speedup Image.

## Q-P23 S2 — budget elastico e sampler corretto

Tutti i full gate ripetuti PASS: target128/MTP2/MTP4/Flash35,10/12/12/10artefatti bit-exact contro il riferimento frozen. Il percorso operativo rivaluta cache prima di ogni forward e delle allocazioni sessione; cresce e riduce le regioni senza modificare KV e logits. Nei quattro gate il monitor esterno resta sopra2GiB e non osserva crescita swap; dettagli `dynamic-memory-gates.json`. Il target tocca2.33GiB disponibili (footprint massimo10.97GiB), MTP2 3.04GiB (footprint11.67GiB). Sono campioni del monitor, non una garanzia contro allocazioni concorrenti istantanee di altri processi. Nessun margine o criterio è stato ridotto.

Gate isolati: resize16→28→4→0→16MiB, batch attivo con resume, head comune conservato senza read aggiuntivo, argomento abbandonato prima del dispatch, metadata fuori range/overlap, EOF parziale ripetuto/retry, stato/logits identici; feature Metal4 e fallback PASS. Cap zero mantiene lo streaming esplicito e rifiuta il forward, senza ricadere in mmap completo.

## Regressioni finali e limite Numeric separato

`make all`, fondamenti Metal (residenti/stream/lifecycle),50host, feature Metal4/fallback, Flash primitive/grafo, CPU-only compilation, reader/queue ASan+UBSan PASS. Il sampler è condiviso con Numeric, quindi eseguita anche la sua suite. Il primo tentativo con python3 globale è NOT_RUN per NumPy/Torch assenti; ripetuto con l’ambiente del repository `agent/.venv/bin/python`:42test,38PASS,3skip e2subtest FAIL nella singola funzione trigonometric_boundaries (seno/coseno).

Prima di modificare qualunque kernel per questa nuova discrepanza sono stati riletti l’audit T06/T07 `upstream-numerics.md`, i pin HF/ggml/Strata citati sopra e i kernel `metal/numeric.metal`/`numeric_ordered.metal`: nessun kernel Numeric è cambiato in questa campagna. Ricostruito il backend con entrambi gli header sampler originali da `ab628747...`; sugli identici input le uscite prima/dopo sono bit-exact. Entrambi falliscono gli stessi2375valori sin e2595cos rispetto a Torch2.14.1 (`5c4886908584029761b579af026dcfb627c84070`), maxabs1.1920928955e-7/5.9604644775e-8. L’audit storico fissava PyTorch `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; non si attribuisce il cambiamento dell’oracle come causa definitiva senza ulteriore indagine. È un limite preesistente riprodotto, non un PASS globale della suite e non una regressione del sampler. Nessuna soglia allargata, nessun kernel vision/Image modificato. Il primo diagnostico aveva import path errato ed è NOT_RUN; retry valido conservato.

## Q-P24 S1 — profondità I/O sui pack (micro separato, prima del codice)

Il nuovo profilo ha gli stessi4878/5997encoder ma molti meno CB; resta~53.6GB di letture per4pass target/16output MTP4. Direzione successiva bounded: confrontare2/4/8worker sulle effettive dimensioni32MiB, layer259360768byte e head1042944000byte, senza alterare soglia32MiB/chunk8MiB o allocare staging di produzione. Gli audit soglieQ-P14 eQ-P16 sono stati riletti: rigettavano8/64MiB con4worker, non sono una misura sul numero di worker per pack grande. Strata `46c152a...` stream.cpp:nio_/split_/read_part/io_main usa6worker e split configurabile, ma esperti, segmenti/page alignment e macchina diversa. tkhr `9d8c1a8...` ds4_metal.m:routed_pread_split/fill_submit usa pezzi di almeno64MiB e defaultsplit1; più richieste non sono automaticamente meglio. ggml `dde879...` llama-mmap.cpp rawread gestisce limiti/EINTR senza una soglia universale di concorrenza. HF `ca8c871...` Qwen3_5 forward nessun cambiamento delle operazioni. Reader locale supporta già1/2/4/8worker con drain/error/counters testati; non serve una nuova API o flag permanente.

Gate previsto: un warmup poi10blocchi ruotati fra2/4/8, read+wait su buffer riusati; memcmp completo fuori timer e byte counters. Nessun processo di inferenza contemporaneo. Solo se il micro sostiene un guadagno, sperimentare backend separato e ripetere gate completo/profilo/coppie prima di promuovere. Altrimenti conservare4 e registrare il plateau locale di questa direzione; niente regola di scarto2%.

## Q-P24 S2 — micro e prossimo confronto

33triplette,1warmup+10misure per taglia,memcmp e counters PASS.8vs4worker: tempo geometrico−26.13% su32MiB,−25.42% sul layer259360768byte,−31.76% sul head1042944000byte;2worker peggiorano+40.45/+32.50/+52.79%. Intervalli appaiati interamente separati da zero (`p24-micro-statistics.json`). Su letture ripetute di un sottoinsieme caldo con offset GGUF non page-aligned, F_NOCACHE non dimostra da solo assenza di file cache:20ms/GiB non è throughput fisicoSSD rivendicato. Serve il full-model con lo stesso cap e, separatamente, cache dinamica.

Prima del candidato riletto il confronto S1 e il codice reader open/run_batch e configure Qwen35: il numero di worker va scelto dall’owner prima di avviare la coda. Esporre un metodo limitato di configurazione del reader, senza ispezionarne fd/coda dal backend. Qwen35 lo usa dopo aver validato il piano e prima di qualsiasi lettura; Flash conserva4worker. Nessuna nuova coda o staging, nessun flag utente, semantica degli errori invariata. Se l’effetto sul modello non è sostenuto, ritirare la configurazione e il relativo metodo invece di lasciare API sperimentali inutili.

## Q-P20/P21/P23 — risultato consolidato prima di Q-P24

Warmup+5coppie alternate,1536MiB identici,8192allocati/64occupati: target64+4, MTP4 64+16. Tutti i dump di ciascuna coppia bit-exact. Nessun campione lento escluso.

| Caso | Prefill mediano prima→dopo | Decode mediano prima→dopo | Somma prefill+decode prima→dopo |
| --- | --- | --- | --- |
| Target |14.602224→17.053274TPS|0.603471→0.915662TPS|11.026228→8.118018s|
| MTP4 |14.046636→15.844459TPS|2.034222→3.281572TPS|12.487359→8.941444s|

Delta geometrici appaiati (bootstrap95%,5coppie): target prefill+16.57%[15.66,17.46],decode+33.98%[3.57,68.29],tempo−20.72%[−30.76,−7.88]; MTP4+14.26%[11.27,17.33],+61.22%[59.23,63.22],−28.59%[−29.36,−27.63]. Target decode CV10.55/23.17% e request7.30/15.97%: instabile, anche una coppia decode−11.28%/tempo+1.36%. Non rivendicare costanza del target dal suo mediano. MTP4 decodeCV1.01/0.71%,request0.96/0.81%: guadagno stabile sul caso misurato. Dati/p10/p90/media/deviazione in `ring-paired-statistics.json`. La somma delle due finestre di inferenza esclude apertura/tokenizzazione/dump e non è latenza HTTP end-to-end; elapsed del processo è conservato separatamente nei grezzi.

Tre richieste operative64+16 senza cap cache, tutte incluse: target decode1.569052–1.723219TPS (mediana1.591937),prefill16.886223–17.048136 (17.007282),somma13.048043–13.987312s; MTP4 4.110904–4.385514 (4.352134),prefill15.471176–15.702508 (15.671490),somma7.724157–7.975937s. RAM cambia durante il run: questi non sono delta appaiati rispetto al cap fisso. Tutti16 ID e prefill bit-exact al frozen, MTP4 anche vettore finale identico. Margine minimo monitorato2.51GiB nei6run, swap invariato. `dynamic-operational-statistics.json` registra aggiustamenti del cap. L’obiettivo1TPS è superato su tutte e tre le richieste target-only misurate, senza MTP.

Profilo finale separato:326CB/4878encoder target e425/5997MTP4, stessi conteggi aritmetici del precedente;~53.628GB logici per4pass target/16outputMTP4. Span GPU4526.77/5145.95ms. Il kernel non salta righe né pesi; byte leggermente maggiori di B0. Il8worker successivo è una direzione distinta, non ancora incluso in questi risultati.

## Q-P24 S3 — correttezza PASS, otto worker non promossi

Tutti4isolati e full target/MTP2/MTP4/Flash bit-exact PASS. Warmup+5coppie4vs8 allo stesso1536MiB: MTP4 decode delta geometrico−3.623%[−7.078,−1.575], peggiora in5/5coppie; request+2.928%[−0.380,5.521]. Target decode+9.629%[−1.985,35.977] dominato da un singolo rallentamento della baseline, con4/5coppie negative e prefill−1.834%[−3.665,−0.091]; request non sostenuta. Tutti i campioni conservati, anche il calo prefill tardivo su entrambe le versioni. `pmset -g therm` non registra warning: non prova assenza di contesa e non viene usato per inventare una causa. Nessuna esclusione o veto per guadagno inferiore al2%.

Riaperto l’audit per la nuova discrepanza prestazionale prima del rollback: tkhr `9d8c1a8...` ds4_metal.m:routed_pread_split/fill_submit tiene segmenti>=64MiB e split1default;8sul nostro layer259MB li riduce a~32MB. Strata `46c152a...` stream.cpp usa6thread/esperti e capacità/shape differenti, non una raccomandazione universale8. ggml `dde879...` read_raw_unsafe/synchronize non dimostra un vantaggio di profondità per le letture fredde di tutto il modello. HF `ca8c871...` Qwen3_5 state e kernel invariati; non c’è differenza numerica da correggere. Locale Q-P14/Q-P16 già rigettava il parallelismo/soglie derivato solo da micro; qui la rilettura ripetuta calda del solo head/layer non rappresenta il working set full. Queste sono incompatibilità osservabili, non una spiegazione misurata di ogni stallo. Profilo candidato separato prima di ritirarlo. Decisione: conservare4worker, eliminare metodo di configurazione e test/API sperimentali. Nessun kernel/dispatch numerico modificato per inseguire il micro. Plateau locale di questa direzione: budget/ring restano attivi, più worker non aiuta il caso reale.

## Chiusura e limiti

Il profilo8worker ha gli stessi326/425CB e4878/5997encoder e gli stessi byte
logici53.628GB: il micro non giustifica una modifica di default. Rimossi il
metodo configure_workers e i suoi test sperimentali; reader e test tornano
esattamente aHEAD. Il codice core/Metal torna esattamente allo snapshot P23
qualificato; anche ds4.o e ds4_metal.o sono byte-identici. Il relink cambia
l’eseguibile: la chiusura registra hash nuovi e ripete comunque i gate.

Risultato locale: ring/cache elastica KEEP;8worker REJECT. Questo è un plateau
per il tuning del numero di worker sui pack di questo caso, non un tetto
teorico di Metal. La domanda1TPS è soddisfatta nei tre run target-only16output
misurati; non promettere la stessa velocità con meno RAM o diverso prompt.
HTTP/ExMekh end-to-end, prompt effettivi quasi8k, altri domini/accettazione MTP,
Image/DiT/VAE ed EmbeddingGemma non sono nuovi gate di questa campagna.
Il sampler condiviso ha ricevuto il controllo Numeric descritto sopra,
non una qualifica prestazionale Image. Flash è solo regressione seriale.

Prossime campagne richiedono nuovi profili per modello: Image usa il proprio
owner, Flash potrebbe beneficiare di residenza fra layer/PLE selezionata,
ma nessuna di queste idee è promossa per analogia con il27B. Conservare
2GiB, pressione/swap guard e audit per ogni nuovo intervento.

## Q-P25 S1 — rilascio fisico delle regioni scartate

La chiusura finale4worker ha ripetuto target/MTP2/Flash PASS ma MTP4 ha rifiutato il resume due volte: cache disponibile0.94/0.87GiB, inferiore al binding head~0.971GiB. I guard non sono scattati: l’admission ha conservato il checkpoint e rifiutato. Non cancellare questi run. Il diagnostico con telemetria fresca mostra un’altra differenza rispetto al primo casoXNU: owned8.123GB→0 e currentAllocatedSize10.785→1.303GB dopo release, ma host4.295→6.531GB; molta memoria liberata da Metal non diventa immediatamente free nel campione. Con più chiamate diagnostiche il resume torna ammesso a1.68GiB. Questo è compatibile con pool/ritardo di restituzione del driver, non prova di memoria persa o contesa esterna. Nessuna stima sintetica viene aggiunta alla telemetria.

Nuovo audit prima dell’intervento: tkhr-sait/ds4 `9d8c1a8f1189dbe6403ca93097b551f8cc1e2f9c`, ds4_metal.m:stream_nommap_routed_free: per buffer già drenati e non appartenenti alle slab riusate chiama setPurgeableState:MTLPurgeableStateEmpty prima di nil, per restituire pagine al sistema invece di lasciarle al pool. SDKMetal26.5 `Metal.framework/Headers/MTLResource.h:setPurgeableState` documenta discard sincrono e accessi successivi indefiniti. Strata `46c152a...` mtl_rt.mm:zc_map mostra memoria no-copy/file-backed, non è un contratto di rilascio per questi buffer owned; stream.cpp/window.cpp riservano margine e drain. ggml `dde879...` ggml-metal.cpp:free/synchronize drena prima di distruggere storage; nessun trasferimento automatico del suo mapping ai buffer owned locali. HF `ca8c871...` Qwen3_5.forward/state non cambia; non modificare dtype, kernel o checkpoint per questo problema. Locale: ds4_gpu_tensor_free conserva la correttezza di alias/eventi generici, quindi non aggiungere purge indiscriminato a quella API.

Scelta: solo l’owner streaming scarta esplicitamente le sue regioni retained e i due slot quando close/resize li elimina, dopo la fence globale già esistente. Non scartare regioni trasferite/comuni, slot riusati, graph/KV, mmap o transienti Flash per layer. Questo recupera memoria realmente scartabile senza contare RAM fittizia o cambiare il margine. Metodo privato unico; nessuna nuova API pubblica. Gate ridotti resize/EOF/readiness, full128 target/MTP2/MTP4/Flash con release/resume; diagnostica host dopo release; riprofilare e ripetere richieste auto prima di attribuire altri TPS. Le misure4worker precedenti restano attribuite allo snapshot antecedente a questo fix del rilascio.

## Q-P25 S2 — versione finale

Fondamenta e full target/MTP2/MTP4/Flash PASS,10/12/12/10 artefatti bit-exact;
MTP4 qui include i soli campioni pubblici di memoria del diagnostico. Le
ripetizioni senza strumentazione aggiuntiva sono registrate separatamente.
Profilo finale a1536MiB: target17.209376prefill/0.919857decodeTPS,
MTP4 15.804116/3.402909;326/425CB e4878/5997encoder, stessi byte
logici del ring precedente. Non è una nuova percentuale di speedup per il
solo purge. Il confronto appaiato principale resta attribuito aQ-P23;
a cap fisso il discard avviene alla chiusura fuori dalle finestre misurate.

Tre richieste operative finali64+16, senza cap cache: target
1.487178–1.607101TPS (mediana1.572404),prefill16.958409–17.119843
(16.962359),somma13.694167–14.532570s. MTP4 4.085220–4.310006
(4.093100),prefill12.963832–15.832649 (15.801759),somma7.762474–8.853370s.
Tutti6run hanno16 ID e prefill esatti al frozen; MTP4 anche vettore finale.
Margine minimo monitorato2.46GiB e swap invariato; cache varia realmente
nei forward, aggiustamenti in `p25-dynamic-operational-statistics.json`.
MTP4 ha un prefill lento conservato (CV11.08% e request7.12%), quindi
nessuna garanzia di latenza costante. DecodeCV3.97%target/3.06%MTP4.
Le differenze daQ-P23 non sono un A/B isolato del discard: RAM e momento
diversi; non promuovere né rigettare il fix di ownership da quei delta.

Chiusura: anche due ripetizioni del probe normale senza campioni diagnostici
aggiuntivi PASS (22.366/22.361s di processo, non TPS),12artefatti ciascuna
bit-exact. Feature residenti/streamed, Metal4/fallback, ripetute PASS.
`make all` aggiornato e `git diff --check` pulito. Nessun commit creato.
Limite Numeric sin/cos preesistente resta aperto e distinto da questi PASS.
