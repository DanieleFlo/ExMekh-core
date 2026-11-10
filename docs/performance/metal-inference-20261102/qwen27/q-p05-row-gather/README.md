# Q-P05: riga UD nel decode, 3 novembre 2026

S1 prima del codice. Bi/B0 di questa continuazione: `fc8fb63`, checkout pulito.
Obiettivo utente: cercare almeno 1 token/s senza alterare inferenza; nessun commit.
Artefatti: `build/metal-perf-qwen27-20261102/continuation-20261103/`.

Baseline nuova 8192/64/cache2GiB, target-only, stessi pesi e workload storico:
64/128 (64 nuovi) +4 teacher forced: prefill9.97/11.05 TPS, decode0.46/0.45.
Trace separata64+4: prefill11.49, decode0.48 TPS (tempi sotto profiler, non promozione).
Profilo storico e trace attuale concordano: grandi vuoti host tra buffer GPU.
Nel wrapper `qwen35_ud_tensor`, anche il gather scalare richiede tutte le
`vocab * row_bytes` della tabella; il generico quant seleziona già una riga.
Lettura evitabile nota staticamente, da confermare con contatore isolato.

## Confronto specifico prima dell'intervento

Fonti rilette ai pin verificati del precedente manifest, senza trattare il
precedente audit come sostituto del presente confronto:

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py`: embedding tramite
  `embed_tokens`, `Qwen3_5MLP`, `Qwen3_5RMSNorm`, decoder e GDN. Lookup
  indipendente per riga; ordine/cast/riduzioni delle altre operazioni invariati.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp:load_dense/read_part/io_main`, `stream.h`,
  `metal/kernels/gemv.metal`, `metal/src/mtl_rt.mm`,
  `metal/tests/arena_budget_test.cpp`, `gemv_parity.cpp`: carica tutti i densi
  residenti, separa trasporto e aritmetica. Non prova il nostro gather UD SSD;
  manca copertura completa Qwen27 Metal. Nessun kernel/assunzione d'arena copiato.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/models/qwen35.cpp:load_arch_tensors` (embedding/head distinti o tied),
  `ggml/src/ggml-metal/ggml-metal-ops.cpp:ggml_metal_op_get_rows`, capability
  in `ggml-metal-device.cpp`, mapping shared in `ggml-metal.cpp`,
  `src/llama-mmap.cpp:read_raw_unsafe/llama_mmap::impl`, `tests/test-backend-ops.cpp`.
  Lookup indicizzato quantizzato e buffer no-copy sono pertinenti; la residenza
  mmap non è un percorso SSD bounded equivalente, né oracle degli errori I/O.
- Locale: `ds4_metal.m:ds4_gpu_embed_token_quant_tensor` legge già una riga
  nei formati generici; UD usa lo stesso gather per singolo e batch.
  Riaperti kernel UD, oracoli scalari, owner/file reader e probe completo,
  report Clean Code/Q-P03, ricerca `guides/problem-to-source.md`,
  `experiments/qwen27-4bis-20261008/documentation/qwen27-weight-streaming-20261008.md`.
  Il pack/ring CUDA offre idee per un intervento successivo, senza trasferire PCIe.

Ricerca nuova: [Apple LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language)
propone sparsità e raggruppamento: non si adotta sparsità approssimata del target.
[ssd-llm cache.rs](https://github.com/quantumnic/ssd-llm/blob/3e8c8b616e16552d1b840e2741a82f152155ef47/src/model/cache.rs)
letto al pin `3e8c8b616e16552d1b840e2741a82f152155ef47`: LRU/pinning utile per
il prossimo audit residenza; budget che può essere superato se tutto pinned e
prefetch non contabilizzato sono incompatibili con il nostro contratto.
Nessun confronto prestazionale con librerie esterne eseguito.

## Ipotesi e piano congelato

Nel solo lookup scalare UD validare prima l'intera tabella originale, poi
associare la riga `token * row_bytes` e passare token0/vocab1 allo stesso kernel.
Batch, proiezioni, shader, precisione e byte della riga invariati. Funziona anche
residente, senza nuovo flag. Alternativa batch compact rimandata: gestione ID
GPU e snapshot introduce più modifiche del necessario per questa ipotesi.

Gate isolato: nove formati UD, primo/medio/ultimo token, oracle scalare e
resident/SSD esatti; contatore legge una sola riga, errori e recovery. Poi
probe Qwen128 indipendente, nove logits e payload identici a Bi. Misure prima
brevi, poi warmup+5 coppie alternate per un bundle mantenuto; profile separato.
Soglie congelate; nessuna inferenza CPU grande, un modello/processo per volta.

S1 supplemento prima del batch: la parity scalare è PASS e il gate128 è
bit-exact; per evitare che il prefill continui a imporre la tabella intera
come working set, estendere il lookup SSD multi-riga con gli stessi lookup
scalari, uno per ID già prodotto, leggendo gli ID tramite `tensor_read`
(che conserva il batch). Validare tutti gli ID prima di cambiare l'output,
poi usare view d'output e identico kernel/riga. Nessuno shader nuovo.
Riletti gli stessi get_rows ggml, gather UD e contratto di read/view locale:
le righe sono indipendenti e non alterano ordine/riduzioni. Resident batch
conserva il dispatch originale; il costo dei dispatch SSD sarà misurato.
Gate aggiuntivo: batch con ID ripetuti/non ordinati e cap di una riga;
uguaglianza scalare e invalid ID senza output parziale.

## S2/S3: gate e screening

Scalare e batch: oracoli originali PASS, nove logits completi/payload128
bit-exact con Bi, replay/checkpoint/park/reload PASS. Nuove prove isolano
primo/medio/ultimo token per12 formati e2 larghezze; nei9 UD anche batch
ripetuto/non ordinato, cap di una riga, invalid ID senza output parziale.

Screening singolo: prefill11.50/11.31, decode0.49/0.40 TPS. Il secondo punto
è più lento del B0 singolo: dato conservato, nessuna attribuzione a carico host
senza prova. Riaperti wrapper/loop gather e get_rows ggml: kernel e byte delle
righe invariati, nuovi dispatch confinati al prefill/multi-riga, decode scalare
stesso kernel; guard pressione1, nessuna crescita swap. Trace separata successiva
64+4: prefill11.83/decode0.50; decode ancora circa2s/token con vuoti host.
La variabilità impedisce promozione isolata: **PENDING_BUNDLE**, riesaminare
su coppie alternate e workload completo. Nessun aggiustamento dei criteri.
