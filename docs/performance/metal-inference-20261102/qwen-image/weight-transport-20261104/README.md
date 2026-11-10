# I-P03 — Trasporto dei pesi Image su memoria unificata

## S1 — Confronto prima del codice

Baseline I-P02 congelata in `build/metal-perf-image-20261104/transport/`.
Il DiT è identico a I-P01: stesso ordine,159caricamenti e69.192GB logici
per richiesta. La prima coppia I-P02 misura61.582/57.706s nel DiT.
Il profilo I-P01 testo attribuisce429stack alla copia e216alle proiezioni;
quel codice non è cambiato in I-P02. Il profilo post-I-P02 disponibile è
invece VAE e non viene etichettato DiT. Il confronto trasporto resta quindi
un'ipotesi da misurare e riprofilare, non uno speedup acquisito.

Fonti rilette il4novembre prima del codice, stessi pin verificati da I-P02:

- Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
  `models/transformers/transformer_qwenimage21.py:forward` e
  `pipelines/qwenimage21/pipeline_qwenimage21.py:__call__`: blocchi rivisitati
  per ogni passo, pesi immutabili e prefixKV. Nessun cambiamento di ordine,
  precisione o recipe è necessario per copiare regioni disgiunte in parallelo.
- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
  `models/qwen3_vl/modeling_qwen3_vl.py`: pesi BF16 dell'encoder immutabili,
  stesso requisito di pubblicazione completa prima del consumer.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp:load_dense,read_part,read` e `stream.h`:
  quattro thread per tensori densi, split di miss in regioni disgiunte,
  join prima dell'uso. I pack expert e la trasformazione FP16 non sono
  applicabili al bundle BF16 Image. `mtl_rt.mm:require_fit` e
  `metal/tests/dense_mm_parity.cpp` confermano un percorso diverso dalle Conv.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/llama-mmap.cpp:llama_mmap::impl`: mapping lazy e advice per intervalli.
  `ggml-metal-ops.cpp` e capability del dispatch2D restano riferimenti statici;
  non implementano il nostro staging Numeric. La page cache va lasciata
  al sistema, senza mlock dell'intero file.
- DS4 originale `890898cfb437ed256be58ebcf7c58a6dfa8f2f53`, `ds4.c`,
  precedente auditQ-P15: separa static resident da miss SSD. Il reader
  selected/CUDA non appartiene a Image e non viene trapiantato.
- Locale: `docs/research/INDEX.md`, auditQ-P15, `ds4_file_reader.h:read`
  (quattro letture da32MiB), `ds4_bundle.c:weight_group_load,hint_worker`,
  `ds4_numeric.c:ds4_num_buffer_stage`, `ds4_numeric_metal.m:copyTo`.
  Metal usa già una sola copia mmap→shared; non c'è un secondo staging
  da eliminare. Il backend possiede buffer/range, il bundle solo i pesi.

Ipotesi: per pesi da almeno32MiB, quattro copie CPU su intervalli disgiunti
possono sovrapporre page fault/trasferimenti nella stessa allocazione shared.
Usare il callback `weight_copy` già esposto dal backend Numeric, con join
sincrono prima del ritorno e pubblicazione del gruppo invariata. GCD riusa
il pool OS: nessun buffer o thread persistente aggiuntivo, nessun kernel,
nessun cambiamento a copie di attivazioni o ai backend CPU/CUDA.
Copie piccole e sorgente/destinazione sovrapposte conservano memmove.
Alternative: pread richiede un nuovo contratto bundle/descriptor; prefetch
per gruppo richiede un owner futuro e memoria bounded; usare staging separato
raddoppierebbe i byte copiati su memoria unificata. Confrontare prima copia
isolata dal bundle reale e poi la richiesta a cap/residenza identici.

## S2 — Copie esatte e discrepanza sul modello

Quattro dimensioni31byte,32MiB−1,32MiB+5,128MiB, sorgente nel bundle reale,
offset sorgente/destinazione7byte, sentinelle complete e memmove sovrapposta:
tutti i byte coincidono, inclusa la coda non divisibile4. Nel micro caldo,
128MiB passa dalla mediana2.117a1.188ms,32MiB+5da0.506a0.342ms.

Il modello completo conserva tutti gli output e i contatori ma misura
encoder18.338s contro14.397s della medianaI-P02, DiT57.613s, totale86.477s
contro84.811s. Una seconda richiesta con cache continua ha ancora encoder
18.758s. Non si attribuisce al micro caldo un guadagno della richiesta.

Nuovo audit prima di ritirare il candidato: Strata `stream.cpp:load_dense`
ripartisce tensori diversi fra thread, mentre `read_part` segmenta esperti
in pread e pack propri. Non è una prova per quattro fault concorrenti del
nostro grande embedding mmap. `llama-mmap.cpp` mantiene advice lazy, senza
un contratto di vantaggio per questo split. Le note localiQ-P24 avevano già
osservato un micro caldo vincente con8worker e un full-model peggiore.
HF conserva la stessa semantica dei pesi, nessun problema numerico da
correggere. Qui il micro ricopia128MiB residenti; l'encoder percorre13.892GB
mappati, competendo con cache e SSD. L'incompatibilità del working set è
osservata; il motivo preciso di ogni fault/stallo non è stato misurato.

Scelta: ritirare la copia parallela generalizzata, poi misurare l'ablation
seriale mantenendo kernel e cache continui. Nessun flag semantico o
orchestrazione per modello dentro Numeric. Un nuovo prefetch per gruppo
richiede un esperimento distinto, non viene dedotto da questi risultati.

L'ablation con copia seriale ripristina encoder13.118s e totale84.639s,
con la stessa cache continua, stessi output e numerica. Il candidato I-P03
resta escluso dal codice finale. Le copie restano singole mmap→shared;
nessun worker GCD, flag o campo di scheduling aggiunto al backend di rilascio.
