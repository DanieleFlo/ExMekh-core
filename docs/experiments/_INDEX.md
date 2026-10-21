# Indice: experiments

Indice generato; non modificare a mano.

## [Qwen 27B — esperimento 4 BIS archiviato](qwen27-4bis-20261008/README.md)

1.

- righe 8–33: **Idee provate** — 1.
- righe 34–63: **Risultati e motivi del ritiro** — La macchina locale aveva RTX 3090, WSL e PCIe Gen4 x2, storicamente circa
- righe 64–85: **Contenuto e recupero** — da engine, CUDA, header, frontend, build e strumenti condivisi.

## [4 BIS — Modalità di contesto esteso per Qwen3.8 27B](qwen27-4bis-20261008/documentation/qwen27-kv-capacity-4bis-20261006.md)

`qwen35`: 64 layer target, attenzione completa ogni quattro layer, 4 KV head,

- righe 9–38: **Vincoli verificati nel checkout** — `qwen35`: 64 layer target, attenzione completa ogni quattro layer, 4 KV head,
- righe 39–64: **Percorsi implementati** — 1.
- righe 65–98: **Parametri e lifetime** — ridotta silenziosamente.
- righe 99–165: **Esperimenti e decisione** — `tests/test_qwen35_kv_capacity` confronta off, grow e hybrid-attention
- righe 166–356: **Speculazione nel contesto esteso — aggiornamento 7 ottobre** — Richiesta: scegliere il drafter affidabile più veloce sul GGUF/3090 locale e
  - righe 197–230: **Backbone DSpark e qualificazione numerica (7 ottobre)** — Il backbone DSpark è ora implementato, riusando embedding, GEMM, RMSNorm,
  - righe 231–315: **NextN e DSpark a 64K effettivi (7 ottobre)** — I confronti v41–v43 generano 32 token dopo 65.536 token reali, mantenendo
  - righe 316–356: **Perché NextN può aggiungere tempo al prefill** — La profondità di proposta appartiene al decode.
- righe 357–406: **Riesame Strata del 7 ottobre e prova 128K** — Il checkout Strata esaminato è `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`.

## [Qwen 27B: KV residente e streaming dei pesi, 8 ottobre 2026](qwen27-4bis-20261008/documentation/qwen27-weight-streaming-20261008.md)

`--qwen-weight-streaming on` richiede Qwen denso 27B, CUDA e una sola GPU.

- righe 8–51: **Percorso sperimentale** — `--qwen-weight-streaming on` richiede Qwen denso 27B, CUDA e una sola GPU.
- righe 52–77: **Provenienza e qualificazione** — Da Strata si riprendono ordine di consumo noto, residenti separati da
