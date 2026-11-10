# Metal Apple Silicon: implementazione e prove di base

Stato **IN_PROGRESS**, 29 ottobre 2026. Baseline
`837894c2af5aaf3382dfc0a0750689baee73363e`, MacBook Pro M5 Pro, 24 GiB,
macOS 26.6.2, SDK 26.5. La matrice completa resta nel
[piano](../../roadmaps/metal-apple-silicon-support-20261028.md).
[qualification.json](qualification.json) conserva metriche, hash ed esiti.
Questi hash descrivono il primo snapshot del porting. Il successivo
[riesame](../../architecture/metal-clean-code-review-20261029.md) e il suo
[registro](review-verification.json) identificano separatamente sorgenti e
prove dopo il refactoring; non sovrascrivono i risultati iniziali.
La [continuazione](continuation/README.md) aggiunge replay isolato della prima
divergenza BF16 Gemma, editing con reference RGBA reale, riserva host dei
probe e confronto statico Qwen Image con Transformers/llama.cpp/Diffusers.
Il suo registro mantiene separati i nuovi esiti e i gate ancora aperti.
Il [riesame dei diagnostici](../../architecture/metal-diagnostics-clean-code-review-20261029.md)
e [cleanup-review.json](continuation/cleanup-review.json) documentano la
successiva estrazione dei binding, le correzioni di ownership e i test.
Le [prove senza GGUF](offline-foundations/README.md) aggiungono la build CPU
isolata e il lifecycle Qwen27/Flash con recovery e ledger verificati entro
un cap device di 1 GiB; non qualificano l'inferenza dei due LLM.
La [preparazione Quick Chat](quick-chat/README.md) aggiunge frontend macOS,
profilo esplicito 4k e confronto del prompt con il server. Lo smoke del 27B
resta rifiutato per RAM insufficiente; nessuna inferenza reale è qualificata.
La [verifica T00–T03](t00-t03/README.md) aggiunge
inventario e hash del Qwen27 locale, budget Metal ordinario, pressione host,
formati Q4_0/Q8_0/Q4_K e replay del grafo ibrido ridotto attraverso chunk.
Il gate reale T03 resta aperto fino allo streaming T06-B.
Il [riesame delle fondamenta](../../architecture/metal-foundations-clean-code-review-20261029.md)
e il [registro successivo](foundation-cleanup/README.md) separano campionamento
e policy di memoria, incapsulano runner/report e verificano i contratti pubblici
di lifecycle e frontend dopo il refactoring.
Le prove usano Metal reale; gli oracle PyTorch CPU sono limitati al piccolo
EmbeddingGemma e a fixture numeriche. Nessun LLM grande viene eseguito su CPU.

## Risultati

| Prova | Esito e limite |
| --- | --- |
| Build | Eseguibili ordinari e `ds4-server-suite` compilati su Darwin; controllo `DS4_NO_GPU` in oggetto separato |
| Primitive native | 27 test, tre skip specifici CUDA; tolleranze T00 conservate, inclusi BF16, attenzione lunga, convoluzioni e picco dello scratch |
| Qwen27 | Nove tipi UD, embedding e proiezioni verificati; GDN contro riferimento scalare; mRoPE/QK e attention contro float64. Inferenza 27B non eseguita: servono 14,76 GiB incrementali, l'ultima ammissione trova 6,57 GiB dopo il margine di 2 GiB |
| Flash SSD | Cache Metal esplicita per layer, cap predefinito 2 GiB; esperti selezionati compattati, `pread` bounded e normali kernel Flash. Fixture Q4_K, IQ2_XXS/Q2_K, MXFP4, Q8 e dominio di 512 esperti; cap ed errore di lettura verificati. Modello reale ancora in download |
| Flash lifecycle | Stato/cache con shape completa di Flash su buffer Metal reali, senza forward: payload bit-exact dopo restore e park/release/resume; checkpoint corrotto e memoria insufficiente devono conservare la sessione viva. Lo streaming usa scratch privato; il batching residente upstream conserva la sua arena |
| Gemma testo | Modello reale, 128/256/512/768 dimensioni, ragged batch, due ripetizioni identiche, cancellazione/riprova e rifiuto sotto budget |
| Gemma media | Modello reale su immagine, audio, video e interleaved; vettori unitari confrontati con oracolo manuale e API ufficiale. Cancellazione/riprova e budget insufficiente verificati |
| Ranking | Due query e quattro documenti, ordine completo identico all'oracolo, positivo al primo posto per entrambe. È una prova ridotta del modello; il flusso di ricerca ExMekh completo resta aperto |
| Qwen Image BF16 | Generazione e modifica RGB 256×256, otto passi, seed 20261029; PNG RGBA ispezionati. Picchi posseduti 3,43 e 3,70 GiB, entrambi entro piano; zero storage numerico vivo dopo il caso |

![Generazione: quadrato rosso](generation.png)
![Editing: quadrato blu](edit.png)

Generazione: 160,27 s (encoder 13,45 s, DiT 94,16 s, VAE 52,66 s).
Editing: 158,08 s (VAE encoder 9,78 s, conditioning 14,86 s, DiT 81,86 s,
VAE decoder 51,57 s). Il processo è arrivato a circa 6,1 GiB RSS in un
campione: le pagine mmap e il driver sono distinti dallo storage numerico
posseduto. Non sono benchmark di velocità qualificati.

## Gate ancora aperti

I vettori Gemma passano le soglie congelate dell'output finale; i dump BF16
interni **non sono tutti bit-exact** rispetto all'oracolo CPU (2/70 testo,
4/400 media). Il gate rigido delle frontiere interne resta FAIL: non è stato
allargato. Questi risultati non chiudono la qualifica completa F04/F05.

Qwen Image usa il profilo pubblico `bf16-native-v1`, con streaming dei gruppi
di encoder/DiT/VAE. Il profilo DiT Q4 ha un manifest valido, ma gli adapter
di esecuzione lo rifiutano già nella baseline: non viene dichiarato supportato.
Restano confronto numerico dei latenti/RGBA con un oracle image indipendente,
reference RGBA e gate alpha/qualità. La disponibilità di un PNG corretto
verifica il percorso completo di base, non tutti i task visuali.

Restano le prove reali Qwen27/Flash di testo, MTP, vision, alternanza con la
suite e F07–F11 su server/frontend ExMekh. Il lifecycle Flash è verificato
con descrittori sintetici e storage vero, non con logits di inferenza.
CUDA non disponibile (confermato dall'utente); regressioni CUDA/ROCm,
distribuite e altri chip Apple non eseguite. La discovery Metal mantiene
`qualified_backends: []` fino alla qualifica richiesta.

## Riesami ripetuti Strata

Riferimento MIT fissato a
[`46c152a1310f77cd2f0ce2c7cd213962df14fe83`](https://github.com/Niko1221/Strata/tree/46c152a1310f77cd2f0ce2c7cd213962df14fe83/metal).
I file consultati sono elencati e identificati da SHA-256 nel JSON.

| Famiglia | S1: layout/progetto | S2: prima parity | S3: tuning/riesame | S4: integrazione |
| --- | --- | --- | --- | --- |
| UD/GEMV | `quant.metal`, `gemv.metal`, `ref_quant.cpp`: decoder e codebook riusabili | Fixture scalari per ogni formato, singolo e più token | Codebook in threadgroup, quattro token/quattro righe per riuso; tail 1/3/5 token e 256/512 colonne verificati | Binding mmap, bounds e lifetime nel runtime legacy; licenza in `third_party/strata/` |
| GDN 27B | `gdn.metal`/`gdn_prompt.metal`: tenere distinta la semantica del 27B | Oracolo indipendente di conv, stato e output | Quattro shard, 512 thread, limite esplicito del threadgroup come in Strata; le riduzioni conservano tolleranze FP32 | Prefill sequenziale e snapshot verificati. Il 27B conserva stato head-major, quattro slot conv e gate SiLU; Strata usa layout interleaved, tre history slot e gate sigmoid |
| Suite BF16 | `dense_mm.metal`: non trasferire la conversione F16 al modello BF16 | Primitive contro PyTorch senza cambiare storage/rounding | MPS FP32 con espansione BF16 lossless delle sole matrici correnti; scratch stimato e misurato. Tanh saturata evita NaN osservati sul device per input positivi grandi | Gemma reale e pipeline image con cap di gruppi; il gate BF16 interno ancora fallisce |
| Flash I/O | `engine/arena_budget.h` e binding GEMV: separare byte residenti e mmap | Compattazione selezioni senza cambiare i kernel numerici | Cap in byte, fence per layer; primo percorso sincrono, senza rivendicare overlap I/O/calcolo | Errori di lettura e cap verificati; checkpoint con buffer reali. Parity del modello completo e ottimizzazione del riuso restano aperte |

S3 è un confronto statico delle scelte, accompagnato da misure del solo fork.
Non è stato eseguito un benchmark Strata con le stesse shape/dtype: nessuno
speedup rispetto a Strata è rivendicato.

## Riproduzione essenziale

```sh
make -j2 all ds4-server-suite
make test-qwen35-metal test-qwen4-metal-stream test-flash-metal-lifecycle
make test-suite-numeric-metal SUITE_PYTHON=build/metal-venv/bin/python
```

`ds4-server-suite` richiede le dipendenze media locali; il builder esistente
supporta ora Darwin arm64 senza installazioni globali. Le prove reali usano
i manifest con revisioni/hash fissati in `config/suite-bundles/`. Comandi,
input/output e log compatti sono conservati nel JSON e nelle sue evidenze.

Il comando seguente conserva la diagnostica storica a contesto 256. La qualifica
locale corrente segue invece T06-A/T06-B nella
[roadmap](../../roadmaps/metal-apple-silicon-support-20261028.md), con capacità
8192/prefill 64; questo comando non chiude quel gate:

```sh
./ds4 --metal -m gguf/Qwen3.8-Flash-Next-Q2.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 2GB \
  --ctx 256 --prefill-chunk 64 --no-mtp --raw --temp 0 -p '1+1=' -n 4
```

Non avviare questo comando insieme a un altro LLM grande. Il controllo della
RAM resta attivo; `--ctx 256` riduce KV/scratch, non i pesi del 27B.
