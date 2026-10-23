# Indice tecnico di DwarfStar

Questo file è la mappa di ingresso al progetto. Deve essere letto insieme ad
`AGENT.md` e le roadmap pertinenti in `docs/roadmaps/` prima di iniziare un task.
Va aggiornato quando un file rilevante viene aggiunto, rimosso, rinominato o cambia responsabilità.

DwarfStar non è un runner GGUF generico. È un motore specializzato che riconosce
un insieme ristretto di famiglie e forme di modello, ne valida esplicitamente i
metadata e costruisce percorsi di inferenza dedicati. La semantica del modello e
lo scheduling principale vivono in C; Objective-C è usato solo per il runtime
Metal, mentre CUDA e HIP contengono le implementazioni GPU specifiche.

Catalogo, loader e downloader accettano solo Qwen3.8 per la famiglia GGUF
`qwen35`. Kernel e strumenti numerici condivisi restano disponibili;
documenti e fixture dei modelli ritirati conservano evidenze storiche.
Gli identificativi della shape e delle cache Qwen3.8 restano invariati.

## Integrazione upstream del 1 ottobre 2026

Il branch dedicato integra `antirez/ds4@0aaea5a` conservando ExMekh Agent come unico
agente del prodotto e distinguendo Qwen27 `qwen35` da Flash Next/Mini `qwen4exp`.
Il [report di merge](docs/architecture/ds4-upstream-merge-20261001.md) contiene
provenienza, matrice PASS/FAIL/NOT_VERIFIED, baseline e limiti hardware.
Il [riepilogo JSON](docs/performance/ds4-upstream-merge-20261001.json) conserva
misure, hash e gate in forma verificabile.
Il [report vision Qwen27](docs/architecture/qwen27-vision-checkpoint-20261001.md)
descrive encoder condiviso, mRoPE interleaved e checkpoint completi durante
l’alternanza vision/testo. Le [evidenze vision compatte](docs/performance/qwen27-vision-checkpoint-20261001.json)
registrano correttezza, memoria, prestazioni e prove applicative rimandate. Il matching mmproj si scarica con
`download_model.sh qwen38-27b-vision`; il modello selezionato resta Qwen27.
Le [guide upstream](docs/upstream/_INDEX.md) sono incluse nella wiki indicizzata.

| Componente importato | Responsabilità |
| --- | --- |
| `ds4_qwen4_cuda.cuh`, `ds4_qwen4_vision.h`, `ds4_qwen4_unicode.inc` | Kernel, vision e tokenizer Flash Next/Mini, distinti da Qwen27; cache CUDA VRAM/RAM/SSD opt-in, routing condiviso mid/down, copie a gruppi con eventi e doppio staging, slab RAM bounded e ranking esatto; budget dal contesto/GGUF e metadata dei checkpoint |
| `ds4_read_queue.h`, `tests/test_read_queue.c` | Coda host bounded con worker persistenti e drain; Flash CUDA anticipa solo miss selezionati dal router in slot RAM riservati, con descriptor separati e gestione errori |
| `ds4_qwen4_expert_cache.h`, `tests/test_qwen4_expert_cache.c` | Policy C di sostituzione e storia esatta degli ultimi 32/64/128 token; test di pinning e cambi di distribuzione |
| `tests/qwen4_expert_policy_replay.py` | Confronto diagnostico delle policy su selezioni reali del router, senza cambiare l’inferenza |
| `ds4_deepseek41_gpu.h`, `ds4_deepseek41_cuda.cuh` | API GPU e kernel DeepSeek 4.1 |
| `ds4_image.c/h`, `third_party/iris/` | Decodifica immagini e input multimodali |
| `ds4_deepseek4_vision_gpu.cuh`, `ds4_glm53_vision_gpu.cuh` | Encoder vision per famiglie DeepSeek/GLM |
| `ds4_prompt_prefix.c/h` | Identità e riuso di prefissi testuali/multimodali |
| `ds4_engram.c/h`, `ds4_linux_memory.h` | N-gram/Engram, letture disco e budget memoria |
| `ds4_gpu_tp.h`, `ds4_tp.c/h`, `ds4_ssd.c/h` | Trasporto GPU/TP e streaming ereditato |
| `ds4_eval_cases.c/h`, `gguf-tools/*41*`, `gguf-tools/*53*` | Valutazione, conversione e quantizzazione delle nuove famiglie |

Il payload ABI della cache disco è 3: le cache precedenti al merge vengono
rifiutate. I checkpoint skill/subagent mantengono i contratti del fork e non
introducono un secondo agente frontend.

## Ordine della spiegazione dettagliata

Questi sono i capitoli da seguire per comprendere il motore dall'esterno verso
l'interno. Ogni capitolo presuppone quelli precedenti.

1. **Confini del sistema e build**
   - Quali eseguibili vengono prodotti.
   - Come vengono scelti backend, modello e modalità di esecuzione.
   - Separazione fra policy applicativa, semantica del modello e kernel.

2. **API pubblica: engine, token e sessione**
   - Ruolo di `ds4_engine`, `ds4_session` e `ds4_tokens`.
   - Vita di un engine e di una sessione.
   - Contratto di `sync`, `eval`, logits, sampling e snapshot.

3. **Apertura e lettura del GGUF**
   - Apertura del file, lock, `mmap`, header, metadata e directory dei tensor.
   - Perché i pesi restano mmap-backed.
   - Ricerca, shape, tipo, offset e allineamento dei tensor.

4. **Riconoscimento e validazione del modello**
   - Selezione della famiglia tramite `general.architecture`.
   - Shape supportate e parametri globali.
   - Validazione semantica di attention, RoPE, MoE, quantizzazione e MTP.
   - Rifiuto anticipato di modelli incompatibili.

5. **Binding e disposizione dei pesi**
   - Collegamento fra nomi GGUF e strutture interne dei layer.
   - Pesi residenti, no-copy e pesi streamed da SSD.
   - Placement su una o più GPU.

6. **Tokenizer e protocollo di chat**
   - Caricamento del vocabolario e dei merge.
   - Token speciali, encoding/decoding e template per famiglia.
   - Costruzione del prompt completo e modalità thinking.

7. **Creazione della sessione**
   - Allocazione di tensori persistenti, scratch, logits e KV.
   - Context capacity e stato iniziale.
   - Differenze fra CPU reference, Metal, CUDA, ROCm e distribuito.

8. **Sincronizzazione del prompt e riuso del prefisso**
   - Confronto fra token richiesti e timeline già viva.
   - Riuso, estensione, rollback o ricostruzione della KV.
   - Scelta fra prefill e decode.

9. **Prefill, passo per passo**
   - Embedding e stato residuo.
   - Elaborazione layer per layer.
   - Norm, proiezioni attention, RoPE, aggiornamento KV e attention.
   - FFN/MoE, shared expert, routed expert e combinazione del residuo.
   - Output norm, output head e logits del token successivo.
   - Chunking dei prompt lunghi.

10. **Decode autoregressivo e sampling**
    - Lettura dei logits correnti.
    - Argmax o filtri temperature/top-k/top-p/min-p.
    - Inserimento del token scelto, aggiornamento KV e nuovo forward pass.
    - EOS, stop sequence e streaming del testo.

11. **Attention e KV cache**
    - Layout delle cache raw e compresse per DeepSeek/GLM.
    - Posizioni, RoPE, finestre, indexer e selezione delle righe.
    - Capacità, lifetime, memoria per token e vincoli di correttezza.

12. **Mixture of Experts**
    - Router, scelta top-k, pesi di routing e normalizzazione.
    - Shared expert e routed expert.
    - Cache degli esperti, hot set e sovrapposizione I/O/calcolo.

13. **Backend GPU**
    - API tensoriale condivisa.
    - Runtime e command lifetime Metal.
    - CUDA, cuBLAS, kernel custom e multi-GPU.
    - ROCm/HIP, hipBLASLt e differenze di piattaforma.

14. **Batching e più sessioni**
    - Proprietà della KV per sessione.
    - Batching di decode e batch misti prefill/decode.
    - Fallback ordinati e oracle full-vocabulary.

15. **SSD streaming dei pesi**
    - Quali pesi restano residenti.
    - Lettura degli esperti e delle layer mancanti.
    - Cache, preload, cold mode e sincronizzazione con i comandi GPU.

16. **Snapshot e KV cache su disco**
    - Serializzazione dello stato di una sessione.
    - Envelope, payload ABI, fingerprint e compatibilità.
    - Politiche differenti di CLI e server.

17. **Inferenza distribuita e tensor parallel**
    - Suddivisione del modello fra worker.
    - Trasporto di attivazioni e logits.
    - Ownership distribuita della KV.
    - TCP/RDMA e ricomposizione degli snapshot.

18. **MTP, DSpark e speculative decoding**
    - Support model, proposta e verifica.
    - Accettazione completa o parziale.
    - Rollback della KV e identità con il decode ordinario.

19. **Server**
    - Parsing OpenAI/Responses/Anthropic.
    - Coda dei job, worker residenti e coordinatore del batching.
    - SSE, tool call, prompt rendering e disk KV policy.

20. **ExMekh Agent**
    - Transcript, sessioni, strumenti e ciclo agentico.
    - Progetti, Quick Chats, approvazione dei comandi e steering.
    - Persistenza JSON del frontend e checkpoint KV gestiti dal server.

21. **Correttezza e osservabilità**
    - Dump di token, logits e top-logprob.
    - Oracle ufficiali, llama.cpp e golden locali.
    - NLL/perplexity, test brevi e long-context.
    - Bit-exact interno e tolleranze cross-engine.

22. **Prestazioni e diagnosi**
    - TTFT, prefill e decode token/s.
    - Profiler dei singoli stadi.
    - Memoria, cache hit, throughput aggregato e regressioni.

## Mappa dei file principali

### Istruzioni, obiettivi e documentazione

- `docs/architecture/native-suite-clean-code-review-20261020.md` — revisione
  di responsabilità, incapsulamento, errori e debito residuo; regressioni
  di lifecycle T00–T04, invalidazione di sessioni parcheggiate, rilascio CUDA,
  build e validazione degli artefatti numerici.
  Il riesame T05/T06 copre anche ammissione nel registro, scope fra thread,
  cancellazione prima del commit e stato dei gruppi dopo un rilascio fallito.
  Il riesame T09/T10 separa observer e risultati RGBA, valida i piani VAE,
  distingue accounting host sincrono da eventi GPU e conserva i close falliti.
- `docs/architecture/native-suite-runtime-lifecycle.md` — inventario dello stato
  globale, confini del bridge legacy, teardown selettivo e matrice di qualifica.
- `docs/architecture/native-suite-v1-contract.md`, `schemas/ds4-suite-*.schema.json`
  — contratti C/HTTP e catalogo congelati; implementazione nel target CUDA
  esplicito `ds4-server-suite`, con limiti dichiarati nella discovery.
- `ds4_server_suite_types.h`, `ds4_server_suite.inc`, `ds4_server_suite_http.inc`,
  `ds4_server_suite_operations.inc`, `ds4_suite_text.h` — registro dei due bundle,
  HTTP Images/Embeddings/Files, multipart spooled, media/receipt su disco,
  validazione, cancellazione, osservabilità e transazioni sul worker LLM esistente.
  Pubblicazione dopo restore; recovery conserva handle e checkpoint fino al recupero.
  I ledger estesi sono opt-in (`DS4_SUITE_RESOURCE_METRICS=1`) per conservare
  risposte tool compatte nei contesti LLM brevi.
- `config/native-suite.example.json`, `docs/architecture/native-suite-client.md` —
  configurazione locale seriale CUDA ed esempi con un solo client/base URL;
  helper e tool usano width/height Literal indipendenti (32..2048, passo 32),
  validazione locale del limite batch e nessuna scelta per modello unico.
- `tests/test_suite_http.c`, `tests/test_suite_http_live.py`,
  `tests/test_suite_exmekh_live.py` — protocollo host, SDK reale sulle quattro
  modalità, retry/editing/disconnessione e conversazione ExMekh con KV riusata.
- `tests/probe_suite_phase_server.c`, `tools/qualify_suite_disconnects.py` —
  server di sola qualifica con barriera compilata alle frontiere image;
  disconnessioni socket reali, ripresa del LLM con cache, owner bounded e
  nessun output parziale. Il binario ordinario non include la barriera.
- `tools/record_suite_service_qualification.py`,
  `docs/performance/native-suite-t17-t18-20261022/` — evidenze, hash dei sorgenti,
  binary/wheel, patch riproducibile della dipendenza esterna e limiti T17/T18.
- `docs/architecture/native-suite-t17-t18-clean-code-review-20261022.md` —
  revisione dei confini HTTP/executor, lifetime, validazione e integrazione client.
- `docs/performance/native-suite-t00-20261020/` — source/model/tensor manifest,
  baseline dei LLM, oracle multimodali, soglie e registro provenienza/licenze.
- `ds4_resources.c/.h` — registro C di storage e view, ownership, refcount,
  eventi, prenotazioni atomiche, budget device/host/pinned/checkpoint e trimming
  tramite destructor del proprietario; contabilità posseduta e disponibilità
  globale distinte. Le query memoria sessione usano i totali del registro.
  Il registro verifica anche il reclaim proiettato con la stessa politica
  della prenotazione reale e protegge i lease finché esistono scope attivi.
- `ds4_executor.c/.h` — planner dell'intera operazione, visite ripetute, budget,
  costi e isteresi; executor validate/reserve/quiesce/save/execute/restore/commit,
  cancellazione, journal e recupero senza replay. Le prenotazioni scoped/nidificate
  riusano capacità fra fasi. Il runtime serializza anche transazioni fra modelli.
  Le varianti possono conservare nuovi gruppi nella cache dell'adapter dopo
  successo senza vittime; errori e ripristino rilasciano i nuovi gruppi.
- `ds4_bundle.c/.h` — directory binaria verificata SHA-256, mmap dei file GGUF e
  safetensors originali, inventario esaustivo e binding dell'adapter; caricamento
  e rilascio di gruppi mediante callback del backend, senza espandere i pesi.
- `ds4_numeric.c/.h`, `ds4_numeric_internal.h`, `ds4_numeric_cuda.cu`,
  `ds4_numeric_cudnn.cu` — API numerica C della suite BF16/FP32: buffer e
  accounting, GEMM cuBLAS, norm, attivazioni, RoPE, attention online/tiled,
  convoluzioni cuDNN e layout/audio/VAE. Reference CPU indipendente;
  librerie statiche suite separate dal link dei binari LLM legacy.
  Softmax lunga FP32/BF16 conserva l'ordine di riduzione e allineamento
  dell'oracle eager; RMS usa la gerarchia della mean anche nei batch corti.
  La L2 channel-major conserva l'ordine della riduzione prima del transpose.
  L'attention BF16 usa query complete entro 128 MiB di score e tile da 256
  oltre il cap; gli adapter riservano l'extra score/result proporzionato.
  Le convoluzioni materializzano il padding e dimensionano il workspace sulle
  attivazioni effettive, con margine di 64 MiB e accounting nel piano adapter.
  Test a 2048–14461 chiavi e attribuzioni in `third_party/pytorch/`.
- `ds4_numeric_weights.c/.h` — bridge dei gruppi bundle verso il contesto
  numerico tramite API pubbliche; disponibilità asincrona opt-in senza cambiare
  il layout ABI v1 del backend. Il core dei buffer non conosce i modelli.
- `ds4_weight_cache.c/.h`, `tests/test_weight_cache.c` — cache seriale generica
  con cap in byte, gruppi mantenuti in base alle visite, eviction dopo l'ultimo
  uso e contatori di payload trasferito; condivisa da encoder, DiT, VAE e Gemma.
- `ds4_transfer.c/.h`, `tests/test_transfer.c` — coda mmap/host bounded con due
  slot, worker persistente e protocollo DMA/riuso; accounting di entrambi gli
  staging, drain recuperabile e trace diagnostica. Binding CUDA nel contesto
  numerico, separato dal runtime legacy.
  Le liberazioni CUDA native aggiornano il campione globale del registro
  condiviso, anche prima della prima allocazione transitoria del LLM.
- `tools/qualify_suite_release.py`, `tests/test_suite_release.py` — stress
  HTTP di almeno 50 alternanze dopo warm-up, tool registrati e continuazioni
  Responses, quattro modalità/interleaving, hash, tempi e owner dopo cleanup.
- `tools/evaluate_suite_profiles.py`, `config/native-suite-candidates.json` —
  registro e gate dei candidati numerici; nessuna promozione con dati mancanti,
  CV instabile, regressioni o profili senza binding nativo.
- `tools/package_native_suite.py` — pacchetto nativo con indici/asset, codec,
  dipendenze dichiarate e albero sorgenti per build isolata; rifiuta il rilascio
  quando un gate obbligatorio non è PASS. Staging temporaneo e pubblicazione
  atomica evitano directory parziali; i pacchetti di sviluppo sono etichettati.
- `docs/performance/native-image-quality-20261023/` — diagnosi sulle immagini
  originali, correzione del doppio arrotondamento BF16 del timestep, confronto
  di latenti/RGBA/alte frequenze con Diffusers e geometria Literal esaustiva;
  `source-audit.md/.json` conservano fonti Hugging Face, pipeline completa fino
  a 2048×1536 e prove VAE/precisione/griglie; `unsloth-audit.md/.json`
  registrano parità della ricetta, ottimizzazioni isolate, nuove generazioni
  native e precisione del denoiser. Default ExMekh 2048×2048, dimensioni
  minori esplicite rispettate. `clean-code-review.md` registra il riesame dei
  confini: configurazione dichiarativa, validazione posseduta dal client e dai
  modelli della risposta, preflight dei file e commit della dipendenza.
  Diagnosi percettiva degli originali ancora aperta.
- `docs/architecture/native-suite-release.md`,
  `docs/performance/native-suite-t19-t22-20261022/` — ricette, qualifica,
  candidati, provenance e limiti residui T19–T22.
- `ds4_qwen_image_encoder.c/.h`, `ds4_qwen_image_tokenizer.c/.h`,
  `config/qwen-image-encoder.json` — conditioning Qwen3-VL dedicato, tokenizer
  image, padding esplicito, vision/deepstack e output pre-norm finale.
  Q/K/V vengono separati con tre reorder GPU per layer; le immagini mantengono
  attention separate conformi alla geometria dell'oracle eager.
  `ds4_bundle` consente gruppi selettivi senza caricare la testa autoregressiva.
- `ds4_qwen_image_input.c/.h`, `ds4_qwen_image_input_internal.h` — proprietario
  opaco dell'input preparato, compositing vision, token/posizioni/padding e
  snapshot pubblico di sola lettura; l'encoder orchestra soltanto il forward.
- `ds4_qwen_image_dit.c/.h`, `ds4_qwen_image_dit_constants.h` — DiT Qwen 2.1
  single-stream a 32 blocchi, RoPE complex, scheduler Turbo a otto passi e RNG
  indipendente; input immutabile e prefix K/V per layer posseduti dalla richiesta,
  gruppi residenti e stime distinte per primo passo/cache/workspace.
  La prefix cache BF16 può risiedere su device o host con trasferimenti per
  layer, secondo il piano di memoria, senza cambiare il grafo numerico.
- `ds4_qwen_image_vae.c/.h`, `ds4_qwen_image_vae_stats.h` — VAE Qwen 2.1 a
  64 canali, encode RGB/RGBA e decode completo/tiled con attention globale a
  bassa risoluzione, piano di memoria e risultato host con salvataggio PNG.
  Encoder e decoder sono gruppi ricaricabili separati.
- `ds4_qwen_image_rgba.c/.h`, `ds4_qwen_image_rgba_internal.h` — proprietario
  opaco del risultato host, factory privata, stima dello storage, view e PNG;
  il VAE usa i metodi del risultato senza conoscerne la rappresentazione.
- `ds4_qwen_image_observer.c/.h` — contratto comune di trace/cancellazione
  per encoder, DiT e VAE; fence e controllo dopo callback in un solo modulo.
- `ds4_qwen_image_graph.h`, `ds4_qwen_image_attention.c/.h` — helper privati
  per binding esaustivo/lifecycle degli adapter e math SDPA FP32. DiT usa tile
  di 256 query; la singola head VAE conserva la GEMM completa quando lo score
  entra nel limite di 512 MiB, altrimenti usa gli stessi tile bounded.
- `ds4_unicode_nfc.c/.h`, `third_party/unicode/`,
  `tools/generate_suite_nfc.py`, `tools/generate_suite_unicode.py` — NFC e
  classi Unicode delle dipendenze congelate, con dati/provenienza dedicati.
  `ds4_image_resize.inc` — resize uint8 antialiased del processor image,
  distinto dalla preprocessing vision legacy; notice in `third_party/pytorch/`.
- `tools/compile_qwen_image_tokenizer.py`,
  `tools/run_qwen_image_encoder_oracle.py`, `tools/compare_qwen_image_encoder.py`,
  `tools/diagnose_qwen_image_encoder.py`, `tools/diagnose_qwen_image_vision.py`,
  `tools/inventory_suite_primitives.py`, `tools/bench_suite_numeric.py` —
  conversione, inventario, oracle, diagnosi e benchmark offline T07/T08.
- `tests/probe_qwen_image_encoder.c`, `tests/test_suite_numeric.py`,
  `tests/test_numeric_lifetime.c`, `tests/test_qwen_image_input.c`, `tests/test_qwen_image_tokenizer.py`,
  `tests/test_qwen_image_preprocess.py` — gate completi di primitive,
  conditioning, padding, pixel, tokenizer, budget, cancellazione e recovery.
- `docs/performance/native-suite-t07-t08-20261020/` — qualifica CUDA sm_86,
  confronti con oracle, hash, fonti, misure e limiti del batching.
  `review-verification.json` registra il riesame del 21 ottobre: API opache,
  rilascio dopo gli eventi, cancellazione finale, aliasing e regressioni.
- `tools/compile_qwen_image_constants.py`, `tools/run_qwen_image_graph_oracle.py`,
  `tools/compare_qwen_image_graph.py`, `tools/diagnose_qwen_image_dit.py` — dati
  numerici congelati, oracle Diffusers e diagnosi offline DiT/VAE T09/T10.
  Gli oracle conservano il cast BF16 del timestep prima della divisione per
  1000 della pipeline ufficiale; `tests/test_qwen_image_timestep.c` protegge
  lo stesso ordine nel forward nativo (regressione del 23 ottobre).
- `tests/probe_qwen_image_graph.c`, `tests/test_qwen_image_graph.py`,
  `docs/performance/native-suite-t09-t10-20261021/` — gate di ogni blocco/passo,
  cache/full recompute, RNG, VAE/alpha/tile, budget e rilascio; report locale
  RTX 3090 con hash, fonti, misure e limiti, distinto dalla pipeline T11.
  La modalità `decode` verifica il VAE su latenti preservati, con controllo
  del ritorno delle allocazioni possedute al livello iniziale per ogni caso.
  `third_party/diffusers/` conserva licenza e attribuzioni dei port Qwen 2.1.
  `tests/test_qwen_image_output.c`, `tests/test_qwen_image_graph_comparison.py`
  e `review-verification.json` coprono lifetime host/PNG, eventi, input corrotti,
  inventario dei confini completo e riesame successivo al port.
- `ds4_qwen_image_pipeline.c/.h`, `ds4_qwen_image_components.h` — pipeline
  seriale text-to-image/edit con input copiati, piano completo prima della GPU,
  residency dei componenti e prefix cache host/device, observer e report;
  pubblicazione RGBA subordinata al ripristino dell'executor comune.
  Copia degli input, pianificazione e metadati sono responsabilità separate;
  le stime VAE e resize sono interrogate nei rispettivi moduli. Dopo il run,
  la query del piano restituisce quello eseguito senza ricalcolarlo.
  Il bound DiT dipende dalla sequenza e dal segmento query più grande,
  consentendo cache K/V residenti quando il budget effettivo lo permette.
- `ds4_qwen_image_reference.c/.h`, `tests/test_qwen_image_reference.py`,
  `third_party/pillow/` — recipe di resize reference ad area esplicita,
  Lanczos/alpha uint8 conforme a Pillow 12.3, stima del proprio workspace,
  test pixel e provenienza.
- `ds4_embedding_gemma2.c/.h`, `ds4_gemma_tokenizer.c/.h` — backbone testuale
  BF16 EmbeddingGemma 2, tokenizer BPE C, PLE da proiezione, attenzione locale
  bidirezionale/full, task, batch ragged isolati, mean/L2 e Matryoshka;
  `third_party/transformers/` conserva licenza e attribuzioni.
- `tools/compile_embedding_gemma2.py` — compilazione offline di tokenizer e
  configurazione per layer con hash delle sorgenti e checksum del payload.
- `ds4_media.c/.h`, `ds4_media_internal.h`, `ds4_media_av.c` — originali e
  media decodificati con ownership e limiti, decoder immagini condiviso,
  libav C in memoria, piano di campionamento comune a raw/container,
  frame/VFR/segmenti espliciti, traccia audio e soxr HQ.
- `ds4_gemma_media.c/.h`, `ds4_host_memory.h` — processor host deterministico,
  patch/posizioni, feature audio e maschere, piano token/memoria e cache LRU
  limitata; buffer host sincroni registrati senza ereditare eventi GPU.
- `ds4_gemma_vision.c/.h`, `ds4_gemma_visual.c/.h`,
  `ds4_embedding_gemma2_internal.h` — torre vision Gemma con frame raggruppati,
  pooling/proiezione, input ordinati e batch ragged isolati; il backbone
  possiede validazione e scatter dei soft token prima della PLE; il builder
  visual misura lo spazio effettivo dei prompt. Sin/cos delle fasi RoPE
  FP32 vengono valutati sul backend numerico anche per le sequenze lunghe.
- `ds4_gemma_audio.c/.h` — torre audio BF16 nativa: subsampling, Conformer,
  attenzione relativa a blocchi, depthwise causale e proiezione dei token validi.
- `ds4_gemma_multimodal.h` — composizione C v2
  tipizzata a quattro modalità nel wrapper visual condiviso; segmenti espliciti,
  identità completa, query di residenza esposte dai rispettivi encoder,
  raccolta unica delle stime e piano di co-residenza o rilascio
  selettivo delle torri prima del backbone. Le primitive numeriche riusano GEMM
  a layout esplicito anche BF16 e RMS channel-major con scala.
- `tools/compile_gemma_media.py`, `tools/build_suite_media_dependencies.py` —
  asset processor vincolato al bundle e build isolata FFmpeg/soxr con hash;
  `third_party/ffmpeg/`, `third_party/soxr/` conservano licenze e provenienza.
- `tools/qualify_gemma_visual.py`, `tools/run_gemma_visual_oracle.py`,
  `tools/suite_visual_cases.py`, `tests/probe_gemma_visual.c`,
  `tests/test_gemma_media.py`, `tests/suite_media_api.py` — fixture, oracle e
  gate pixel/feature/hidden/vettori/retrieval, budget, cache e lifetime T13/T14.
- `tools/qualify_gemma_multimodal.py`, `tests/fixtures/native-suite-audio-v2/` —
  qualifica sequenziale T15/T16, clip/lingue/silenzio, quattro modalità,
  retrieval e trace dei pesi; waveform sintetiche con provenienza e SHA.
- `docs/performance/native-suite-t15-t16-20261022/`, `third_party/numpy/` —
  gate C/CUDA audio/composizione, diagnosi della baseline T00 con riduzioni BF16,
  nuovo riferimento FP32 separato, inventario e provenienza del rounding FFT.
- `docs/architecture/native-suite-t15-t16-clean-code-review-20261022.md` —
  riesame di responsabilità e API, cleanup unico delle feature e workspace,
  layout audio esplicito, output d'errore e aliasing delle primitive;
  `review-verification.json` nel report T15/T16 conserva test e replay esatti.
- `docs/performance/native-suite-t13-t14-20261021/` — qualifica locale media,
  vision/video e regressione testuale; limiti, dipendenze, risultati e sviluppi
  successivi. Audio e composizione sono qualificati nel report T15/T16;
  routing HTTP è ora nel target esplicito `ds4-server-suite` (T17).
- `tests/test_suite_api.py` — output di ownership azzerati su input invalidi,
  comprese le uscite indipendenti dei tokenizer; nessun modello richiesto.
- `docs/architecture/native-suite-t11-t12-clean-code-review-20261021.md` —
  revisione di responsabilità, API opache, errori e regressione del refactoring.
- `docs/architecture/native-suite-t13-t14-clean-code-review-20261022.md` —
  revisione di decoder/processor/vision/backbone, admission coerente,
  validazione delegata al proprietario dei token e regressione numerica.
- `tools/qualify_embedding_gemma2.py`, `tools/run_embedding_gemma2_oracle.py`,
  `tools/qualify_qwen_image_pipeline.py`, `tools/run_qwen_image_pipeline_oracle.py`,
  `tests/probe_native_suite.c`, `tests/probe_qwen_image_pipeline.c`,
  `tests/test_embedding_gemma2.py` — oracle offline e gate T11/T12, dump completi,
  ranking/dimensioni, ordini reference, ammissione e guasti/cancellazione.
  `tests/test_suite_oracle_attention.py` verifica i mask del tiling per query
  dell'oracle math, con asse delle chiavi intero e nessuna approssimazione.
- `docs/performance/native-suite-t11-t12-20261021/` — qualifica locale dei due
  adapter, comandi, hash, provenienza e limiti; il servizio T17 è qualificato separatamente.
- `tools/suite_bundle.py`, `tools/prepare_suite_bundles.py`,
  `config/suite-bundles/`, `schemas/ds4-suite-bundle-v1.schema.json` — compiler
  offline e conversione deterministica, profili con revisioni/hash fissati,
  download esplicito con resume/checksum e discovery; nessuna inferenza Python.
- `ds4_cuda_accounting.cuh` — accounting host delle allocazioni CUDA e pinned;
  i tensori condividono lo storage tramite view, le copie D2D mantengono il
  budget fino alla fence. Nessun nuovo kernel o percorso numerico.
- `ds4_checkpoint.c/.h`, `ds4_sha256.h` — envelope lifecycle versionato con
  identità/configurazione e checksum SHA-256; riuso dello SHA delle immagini,
  validazione completa prima del restore e payload legacy invariato.
- `ds4_lifecycle.inc`, `ds4_lifecycle_text.h` — adapter interno all'engine per
  checkpoint, parcheggio/ripresa, query memoria, trimming e rilascio dei soli
  pesi Qwen27 CUDA; mantiene identità e callback e valida un grafo candidato.
  Espone anche le vittime all'executor tramite un adapter opaco, mantenendo
  stime e snapshot nella responsabilità dell'engine.
- `tests/test_resources.c`, `tests/test_lifecycle_cuda.c` — gate host di lifetime,
  alias, overflow, concorrenza e file corrotti; gate reale di continuazione,
  OOM/SSD, MTP e stato multimodale. Qualifica nel report T03–T04.
- `docs/performance/native-suite-t03-t04-20261020/` — comandi, inventario degli
  adapter, risultati e limiti della qualifica delle nuove operazioni lifecycle.
- `tests/test_executor.c`, `tests/test_executor_cuda.c`, `tests/test_bundle.c`,
  `tests/test_suite_bundle.py` — matrice budget/fault/recovery, continuazione
  Qwen27 reale e gate del loader/converter/downloader; eseguiti anche con sanitizer.
- `docs/performance/native-suite-t05-t06-20261020/` — evidenze T05/T06, inventari
  reali Qwen/Gemma, memoria di apertura, regressioni e istruzioni delle nuove API.
- `tests/fixtures/native-suite-v1/`, `tests/test_suite_*.py`, `tests/test_runtime*.c`
  — corpus sintetico congelato e test dei contratti, gate numerici e lifecycle.
- `tools/freeze_multimodal_baseline.py`, `tools/export_suite_baseline.py`,
  `tools/prepare_suite_oracles.py`, `tools/run_suite_*.py`,
  `tools/score_suite_gemma_corpus.py`, `tools/freeze_suite_noise.py`,
  `tools/build_suite_corpus.py`, `tools/create_suite_speech.ps1`,
  `tools/suite_numeric_metrics.py`, `tools/suite_artifacts.py`,
  `tools/probe_suite_baseline.c` — strumenti
  di sviluppo T00, separati dall'inferenza di produzione.
- `docs/architecture/native-multimodal-suite-analysis-20261020.md` — analisi
  della suite nativa Qwen-Image-2.1-Turbo / EmbeddingGemma 2: audit del runtime,
  fonti Unsloth/llama.cpp e altri motori, semantica numerica, residenza dinamica
  e contratti client; proposta, non funzionalità già implementata.
- `docs/roadmaps/native-multimodal-suite-implementation-20261020.md` — piano
  operativo in 23 task con sottotask, dipendenze, invarianti e gate numerici,
  multimodali, di memoria/ripristino e prestazionali; T00/T01 e gate locali T05/T06
  completati, T02 CUDA verificato; qualifiche residue T02/T03/T04 esplicite.
- `AGENT.md` — vincoli architetturali, qualità, sicurezza e test obbligatori per
  chi modifica il progetto.
- `docs/roadmaps/` — roadmap ordinate, obiettivi di inferenza e gate di correttezza.
- `README.md` — manuale utente e panoramica delle funzionalità, inclusi backend,
  streaming, distribuito, server, agent, KV e strumenti diagnostici.
- `MODEL_CARD.md` — descrizione dei meccanismi specifici dei modelli DeepSeek,
  in particolare attention compressa, indexer e KV.
- `QA_BEFORE_RELEASES.md` — matrice manuale e automatica che blocca una release
  quando un percorso non è stato verificato.
- `CONTRIBUTING.md` — regole di sviluppo, build, test e contributi.
- `STRIXHALO.md` — note operative per build ed esecuzione ROCm su Strix Halo.
- `PROJECT_INDEX.md` — questo indice e ordine della spiegazione tecnica.
- `.gitattributes` — fine riga LF per i launcher WSL e i fixture Qwen verificati
  tramite checksum, inclusi corpus/manifest/schemi della suite nativa; gestione
  dei diff per le continuazioni del modello e
  conservazione dei byte originali dei report e della patch 27B archiviati.
- `docs/INDEX.md` — indice madre della wiki tecnica, con tassonomia, regole di
  costruzione e accesso agli indici semantici di ogni macro-argomento.
- `docs/performance/qwen36-drift-hypotheses.md` — registro vivo delle ipotesi sul drift
  Qwen3.6, con evidenze, esperimenti ordinati e gate dei 32 token.
- `docs/performance/qwen36-numerics-lab.md` — banco differenziale per distinguere errori
  semantici, roundoff CPU/CUDA e policy aritmetiche Q4_K/Q8_K; include l'audit
  del percorso Ollama/llama.cpp senza eseguire Ollama.
- `docs/performance/qwen36-performance-experiments-2026-08-05.md` — registro riproducibile
  dei test prestazionali Qwen CUDA, inclusi candidati scartati, confronto locale
  con LM Studio, gate 500/15 e verifiche numeriche associate.
- `docs/performance/qwen36-performance-ledger.md` — registro canonico e compatto delle
  ottimizzazioni Qwen CUDA mantenute, scartate o ancora da confermare; unifica
  le decisioni disperse nei documenti datati e conserva il ciclo sperimentale.
- `docs/performance/qwen-inference-audit-2026-09-09.md` — verifica GPU degli
  interventi dell'audit, motivi di rigetto, correzione del carry MTP e
  dequantizzazione cooperativa IQ4_XS, con misure e regressioni.
- `docs/performance/qwen38-qualification-2026-09-17.md` — confronto numerico
  dell'intero corpus Qwen3.8 con llama.cpp e verifica dei rami FP16 preesistenti.
- `docs/research/cuda/` — componente della knowledge base dedicato a toolchain,
  profiling, SASS/resource usage, primitive Ampere `sm_86` e mappa delle
  prossime ipotesi CUDA falsificabili per DS4.
- `docs/performance/qwen36-lmstudio-decoding-analysis-2026-08-06.md` — guida didattica al
  decoding LM Studio/llama.cpp su RTX 3090: MMVQ Q8_1, caratteristiche GPU,
  implementazioni DS4 riuscite o scartate, risultati dei gate e piano per
  replicare il percorso veloce senza quantizzare la KV cache.
- `docs/performance/qwen36-cuda-optimization-progression.md` — checklist ordinata delle
  ottimizzazioni CUDA Qwen3.6 Q4_K_S ancora da sviluppare: ricerca upstream e
  letteratura, gate numerici/prestazionali, budget RTX 3090 da 24 GB, policy di
  rollback e confronto CPU obbligatoriamente finale.
- `docs/architecture/qwen36-mtp-design.md` — progetto di supporto Qwen3.6 MTP: audit dei
  repository LM Studio/Unsloth/llama.cpp e del GGUF NextN, semantica del kernel,
  allineamento token-hidden, modello prestazionale, verifier target microbatch,
  rollback Gated DeltaNet, scheduling adattivo, fallback e gate di test per il
  target CUDA Q4_K_S/Q4_K_M.
- `docs/architecture/clean-code-review-20261004.md` — revisione delle modifiche Flash e
  dei confini del frontend: responsabilità, contratti tool/provider, verifiche e
  limiti dell'audit; distingue qualifica prestazionale e refactoring successivo.
- `docs/architecture/clean-code-review-20261012.md` — revisione del server XML,
  ownership degli snapshot, API del documento e policy di sampling unica;
  esiti e limiti sono in `docs/performance/interactive-edit-clean-review-20261012.json`.
- `docs/architecture/qwen38-compatibility.md` — audit pinning del target
  Qwen3.8-27B-UD-Q4_K_S e del relativo MTP: shape, inventario quantizzato,
  integrazione CLI/server e formati CUDA ancora necessari.
- `docs/agentic/agent-api.md` — contratto dell'estensione Responses `agentic`: registry
  statico, capability tool/skill disgiunte, checkpoint SSD gerarchici e
  semantica di `return` senza full prefill del parent.
- `docs/agentic/agentic-checkpoint-validation-2026-08-07.md` — gate CUDA Q4_K_S dei
  task 10–11: equivalenza bit-exact, casi lunghi 10k, rollback MTP, metriche
  SSD/GPU e copertura della suite HTTP agentica.
- `docs/roadmaps/constrained-json-decoding-plan.md` — checklist e criteri di completamento
  per simulatore DSML, JSON Schema, masking dei logits e output strutturati.
- `docs/roadmaps/interactive-edit-file-server-plan.md` — piano server per tool call
  interattive XML v2: autorizzazioni per elemento, chiusura finale senza code,
  checkpoint delle chiamate parziali, retry/replay, masking di old_text da
  inizio riga, vitalità dei prefissi, sampling locale e gate MTP;
  le attività client restano in `exmekh_lib/todo.md`.
- `docs/agentic/interactive-calls.md` — contratto `edit_file.v3` sul trasporto
  interattivo v2: corpi letterali e due marcatori con ID server a tre cifre;
  creazione e cancellazione sono tool separati, senza inspect/read.
  `docs/performance/interactive-edit-xml-v2-qualification-20261011.json` registra
  lo storico con code del 11 ottobre, prefill reale e confronto maschera/oracolo.
  La qualifica del 9 ottobre conserva soltanto lo storico v1.
- `docs/performance/interactive-edit-xml-plain-smoke-20261012.json` — verifiche
  mirate del formato predefinito senza code/prefill, framing finale e lookahead;
  esiti target/MTP, fallimenti di contenuto e wire sintetico separato.
- `docs/performance/interactive-edit-xml-v2-full-tests-20261011.md` — esiti
  della raccolta completa e delle prove live target/MTP/batch, con responsabilità
  server/client, fallimenti conservati nel JSON e wire prima/dopo la correzione
  dei prefissi XML che ammettevano escape nativi senza una prosecuzione valida.
- `docs/roadmaps/todo-agentic-server-performance.md` — diario misurabile degli interventi
  su sampling vincolato DSML/JSON, telemetria server e piccoli suffissi, senza
  modifiche ai kernel CUDA.
- `docs/performance/agent_performance_contract.md` — gate operativo per gli esperimenti CUDA:
  baseline, ipotesi falsificabile, correttezza, statistica e verdetto.
- `docs/roadmaps/todo-harness.md` — roadmap completa del GPU kernel engineering harness.
- `docs/roadmaps/qwen38-implementation-roadmap.md` — registro vivo
  dell'integrazione Qwen3.8 UD-Q4_K_S: kernel GGML mancanti, parita' CLI/server,
  MTP, checkpoint SSD, doppi gate Qwen3.6/Qwen3.8, decisioni e risultati.
- `docs/performance/qwen38-flash-3090-residency-20261003.md` — diagnosi Flash Next Q2, esperimenti CUDA di residency/copie byte-identiche, fonti e gate non promossi sulla RTX 3090.
- `docs/roadmaps/qwen-flash-performance-action-plan-20261004.md` — audit statico Flash/27B, colli di bottiglia I/O e PCIe, confronto runtime esterni e piano progressivo per performance, checkpoint e portabilità.
- `docs/performance/qwen38-flash-pipeline-20261004.md` — implementazione Flash isolata, routing compatto, staging per gruppi, slab RAM e ranking esatto; qualifica locale e limiti hardware/scenario.
- `docs/performance/qwen38-flash-prefill-latency-20261004.md` e il relativo JSON —
  profilazione del prompt completo dell'agente, misura reale del link PCIe x2,
  costo del tetto RAM a 16 GiB e qualifica del budget automatico disponibile.
- `docs/performance/qwen38-flash-prefill-async-20261004.md` e il relativo JSON —
  letture anticipate bounded degli esperti Flash CUDA, gate bit-exact e checkpoint,
  misure cold/warm/HTTP e rigetto dei chunk più grandi per drift numerico.
- `docs/research/guides/flash-prefill-offload-20261004.md` e il manifest fonti —
  confronto di letteratura e sei runtime MoE con il codice Flash: riuso pesi,
  letture asincrone, cache distinta per fase e gate numerici; proposte non qualificate.
- `docs/performance/qwen38-flash-prefill-replay-20261004.md` — diagnosi del doppio
  prefill ChatML, confine condiviso RAM/SSD, conteggio model-backed e limiti della
  diagnosi prestazionale sul prompt reale lungo.
- `docs/performance/qwen38-flash-pipeline-20261004.json` — campioni, statistiche, comandi, provenienza e gate numerici/applicativi della qualifica Flash/Qwen27 locale.
- `docs/hardware/` — inventario della macchina RTX 3090 e note di riferimento
  su GA102, compute capability 8.6, memoria, numerica CUDA e determinismo.

- `docs/roadmaps/inference-idle-time-roadmap.md` — roadmap e registro degli esperimenti di sovrapposizione CPU/GPU e riduzione delle attese Qwen3.8.
- `performance/idle-time-qwen38-2026-09-17.json` — evidenze compatte dei 22 esperimenti su tempi morti Qwen3.8, identita dei binari/output, dispersione e avvertenze di provenienza.

### Build ed eseguibili

- `Makefile` — selezione piattaforma, compilazione C/Objective-C/CUDA/ROCm,
  gate rapidi e confronto model-backed `perf-direction-ab` in un comando,
  linking, target dei binari e suite di test; espone anche la parita' CUDA
  isolata del backend MMQ vendorizzato tramite `test-mmq-parity`.
- `download_model.sh` — download dei modelli e support model conosciuti dal
  progetto: DeepSeek Flash 0731/DSpark, PRO 0813, GLM e gli artefatti Qwen3.8
  fissati con revisione e checksum.
- `start-ds4-server.sh`, `agent/start-ds4-server.sh`,
  `agent/start-ds4-server-flash.sh`, `agent/run.sh` —
  launcher eseguibili nativi per server CUDA e frontend ExMekh Agent dalla copia WSL2
  `/home/daniele/ds4`; il launcher principale mantiene Qwen27 come default e aggiunge il profilo Flash esplicito; i due launcher in `agent/` selezionano rispettivamente Qwen27B e Flash Next, con proiettore corretto e `--print-command`, MTP attivo per entrambi e contesto Flash 64K. Il launcher agent Qwen27 avvia `ds4-server-suite`, contesto 26000 e bundle scelti da `DS4_SUITE_CONFIG`, per immagini/embedding sulla stessa URL. `config/native-suite.env.example` documenta i percorsi delle dipendenze native nel file locale ignorato `config/native-suite.env`; il progetto non modifica proxy o firewall Windows.
- `ds4_cli.c` — eseguibile `ds4`: parsing opzioni, modalità one-shot e REPL,
  prompt, generazione, sampling, perplexity e dump diagnostici. `--mtp` senza
  argomento seleziona il sidecar Qwen auditato; `--mtp FILE` e l'alias
  `--mtp-model FILE` scelgono un support model esplicito, incluso DSpark.
  CLI e server condividono la semantica dei flag booleani MTP.
- `ds4_server.c` — eseguibile `ds4-server`: API HTTP compatibili, code, worker,
  vincoli dei tool nella sintassi nativa Flash condivisi con il replay MTP,
  confine di sistema condiviso fra cache residente e checkpoint SSD per evitare
  rewind e doppio prefill Flash su prompt ChatML lunghi,
  session batching, streaming, tool-call mapping e policy della KV su disco;
  compila i riferimenti locali finiti `#/$defs/nome` degli output Pydantic,
  mantenendo i vincoli congiunti; riferimenti esterni/ricorsivi falliscono prima
  dell’inferenza. Parser, limiti di espansione e classifier JSONSchemaBench
  hanno regressioni dedicate; il client Flash viene verificato in streaming;
  il riuso dei checkpoint visibili Qwen include Flash anche con tool dichiarati
  e reasoning omesso o reinviato, preservando il KV campionato;
  espone per Qwen ID API canonici senza l'estensione `.gguf`, mantenendo gli ID
  storici come alias di compatibilità;
  per Responses gestisce inoltre il namespace `agentic`, il masking DSML dei
  nomi e le frame gerarchiche delle skill. Per Qwen3.8 UD-Q4_K_S abilita
  automaticamente una cache durevole con budget condiviso fra sessioni,
  prefissi di sistema e checkpoint delle skill; le history Responses vengono
  canonicalizzate senza reasoning storico prima della persistenza. La
  canonicalizzazione accetta la riscrittura dell'ultimo token BPE Qwen solo se
  il KV live conserva l'intero prompt byte per byte; le continuazioni agentiche
  riconoscono il tail tool anche dopo lo scope di sistema sintetico.
  Possiede il writer comune del framing nativo e la policy dei batch OLD;
  cambio di fase, query di capacità ed esito della verifica sono operazioni
  distinte. I contatori dei draft da snapshot sono separati da quelli MTP.
- `ds4_prompt_settings.h` — template del protocollo di Qwen/Qwen4, DeepSeek/DeepSeek 4.1, GLM e output strutturato; il server non aggiunge istruzioni agentiche o reminder ai messaggi del client.
- `ds4_server_text.h` — diagnostica, errori API e formati dei log del server; prompt e testi agentici restano nei rispettivi config.
- `ds4_agentic_settings.h` — messaggi di validazione dello scope agentico e diagnostica dei checkpoint; le istruzioni operative delle skill appartengono al client.
- `docs/architecture/server-agent-guidance-removal-20261014.md` — rimozione delle istruzioni agentiche aggiunte dal server e criteri dei test live per distinguere bug da limiti del modello.
- `ds4_interactive_checkpoint.h` — possiede cattura, staging, budget e pubblicazione dei rollback delle pause; un rollback pubblicato è protetto da richieste ausiliarie e finalizzazioni ripetute fino al ripristino sul worker.
- `ds4_interactive_settings.h` — proprietario unico di limiti, versione wire, marcatori edit e messaggi del protocollo interattivo; le descrizioni dei tool provengono dal client.
- `ds4_interactive_config.h`, `ds4_interactive_grammar.h`,
  `ds4_interactive_runtime.h` — protocollo Responses interattivo v2: registro
  dei comportamenti, policy del documento, pause interne alla stringa con KV,
  ripresa della stessa chiamata, ricevute idempotenti e interruzioni/replay.
  Il runtime possiede lifecycle, restore dello stato generato, budget e wire;
  il dispatcher usa queste operazioni senza gestire gli internals della continuazione.
- `docs/architecture/steering-cache-fix-20261014.md`, `docs/performance/steering-cache-fix-20261014.json` — correzione della continuazione Responses dopo steering e del salvataggio prima delle richieste ausiliarie; checkpoint distinti per resume e rollback delle pause edit, con prove live e native.
- `ds4_interactive_edit.h` — comportamento `edit_file.v3`: snapshot per edit,
  adattatore JSON e policy di sampling unica; valida il nome del campo compilato. Configurazione e stato sono
  opachi per il protocollo e liberati dai callback del comportamento.
- `ds4_edit_document.h` — riconoscitore di `exmekh_edit` e `exmekh_new_text`,
  corpi letterali, ID casuali a tre cifre unici nella chiamata, ownership dei
  percorsi e snapshot condivisi; checkpoint incrementali ai confini UTF-8.
  Le prove del masking sono read-only. I draft del testo vecchio propongono
  byte letterali dello snapshot, poi protetti dal trasporto nativo.
- `ds4_edit_constraint.h` — vincolo dinamico sui byte logici da inizio riga,
  unicità globale e indice suffix array/LCP dello snapshot, esposto tramite API
  di matching e candidati senza accesso agli indici interni.
  La frontiera delle selezioni letterali conserva i candidati dello snapshot e
  confronta soltanto i nuovi byte logici; avanza dopo l'accettazione del token.
  Il contratto wire è in `docs/agentic/interactive-calls.md`.
- `agent/` — sorgenti, test e configurazione del frontend NiceGUI ExMekh Agent;
  `agent/src/exmekh_agent/exmekh_bridge/` integra Client e DeepAgent, `agent/src/exmekh_agent/state/`
  persiste chat e progetti e `agent/src/exmekh_agent/ui/` gestisce l'interfaccia.
  La distribuzione e la CLI web sono `exmekh-agent`; le variabili UI sono
  `EXMEKH_AGENT_HOST`, `EXMEKH_AGENT_PORT` e `EXMEKH_AGENT_TITLE`.
  Il converter usa l'API pubblica dei metadati Message, senza esporre snapshot
  al provider; l'adapter dei tool conserva lo stato di errore emesso da ExMekh-Lib per
  root, conversazioni figlie e UI.
- `exmekh_lib/src/exmekh_lib/core/models/native_suite.py`, `native_media_tools.py`,
  `config/native_suite.py` e capability `generate-image` —
  dipendenza sorgente esterna: API tipizzate con Literal per le identità,
  trasporto OpenAI condiviso, dimensioni/count Literal enumerati nel tool,
  rifiuto locale della geometria prima degli upload, riferimenti ordinati e
  allegati configurabili.
  Gli embedding restano un servizio interno della ricerca web, senza capability
  autonoma. Il provider possiede il trasporto; il fallback standalone ne gestisce
  la chiusura. Il ranking usa cache per chiamata e API del backend condiviso.
  `core/web_navigation/ranking.py` usa il servizio embedding nativo al posto
  dell'inferenza client OpenVINO/SigLIP. Contratto in `exmekh_lib/docs/22-native-suite.md`;
  patch dal commit base e hash sono conservati nel report T17/T18.
- `agent/tests/test_native_media_preview.py`, `test_native_media_preview_e2e.py`,
  `run_native_media_preview_agent.py` — conservazione degli allegati tool,
  miniature assistant, dialog e persistenza dopo reload; nessun encoder implicito UI.
- `exmekh_lib/src/exmekh_lib/core/models/message_snapshot.py` — export lossless
  degli allegati tool tramite il codec cache esistente, condiviso da history,
  snapshot del nodo e skill chiuse; il dump compatto dei messaggi resta invariato.
  Il frontend ripristina sia questi snapshot sia gli allegati UI preesistenti,
  tramite il codec della libreria che possiede anche l’estrazione delle URL.
- `agent/tests/test_native_image_generation_e2e.py`,
  `run_native_image_generation_agent.py`, `tests/test_suite_image_tools_e2e.py` —
  prove opt-in con LLM, tool registrato, generazione GPU e ranking nativo reali;
  output multipli, preview, responsività, reload e secondo turno con pixel ripristinati.
  `tools/record_native_image_e2e.py` registra l'audit di `Funzionalità extra` in
  `docs/performance/native-suite-image-e2e-20261022/`, con patch esterna e hash wheel.
- `docs/performance/native-suite-clean-review-20261022/` — revisione successiva:
  ownership HTTP, API pubbliche, codec allegati uniforme, ranking interno senza
  tool embedding, timestamp file stabili e chiusura incrementale dei modelli.
- `agent/src/exmekh_agent/config/application.py`, `agent/src/exmekh_agent/exmekh_bridge/providers/provider_health.py`,
  `agent/src/exmekh_agent/ui/settings/settings_drawer.py` — discovery del modello attivo e del
  contesto allocato: il probe legge e valida i metadati, AppConfig sceglie il
  modello in memoria e la UI aggiorna i controlli dopo riavvii Flash/27B, senza
  salvare preferenze durante i health check.
  `agent/tests/test_model_discovery_e2e.py` e `agent/tests/run_model_discovery_agent.py`
  verificano barra modello/contesto, tooltip e modello delle richieste nella stessa pagina.
- `agent/src/exmekh_agent/exmekh_bridge/history/converter.py` — serializzazione delle skill, conversione della history e
  deduplicazione del prompt corrente tramite le API pubbliche Message;
  `agent_runner.py` coordina l'esecuzione e mantiene gli import di compatibilità.
- `exmekh_lib/src/exmekh_lib/capabilities/hds/quick.md` e relativo catalogo —
  HDS Quick con ricerca/lettura web, ispezione/lettura file e generazione immagini;
  le Quick Chat usano lo stesso ciclo DeepAgent e la stessa persistenza dei progetti,
  selezionando solo queste capacità. `ToolContext.call_id` identifica anche gli hook
  prima/dopo gli override degli argomenti.
- `agent/src/exmekh_agent/exmekh_bridge/tools/tool_feedback.py` — feedback per
  invocazione e per turno: stato delle approvazioni, originali PNG indipendenti
  dagli allegati inviati al modello e attesa cancellabile delle generazioni in
  background. La preferenza globale `include_generated_images_in_context`
  controlla `include_image_in_result` tramite hook, anche nei grafi personalizzati.
  `ChatMessage` espone immagini e galleria per tool; `chat_view.py` mantiene i box
  nella sequenza delle attività e conserva le immagini delle skill richiuse.
  `ui/shared/image_preview.py` gestisce navigazione, copia dei pixel e download.
  Regressioni in `agent/tests/test_feature_updates.py`,
  `agent/tests/test_feature_updates_e2e.py` e
  `exmekh_lib/tests/core/agents/test_quick_hds.py`.
- `agent/src/exmekh_agent/exmekh_bridge/agents/turn_request.py` — input tipizzato del turno, passato come valore unico al worker; il runner conserva il punto di ingresso pubblico compatibile.
- `agent/src/exmekh_agent/config/execution.py`, `agent/src/exmekh_agent/config/resources/execution.json` — policy dei titoli: sampling, budget, normalizzazione e fallback tramite metodi del tipo di configurazione.
- `agent/src/exmekh_agent/exmekh_bridge/history/recorded_history.py` — query pubbliche sulle trascrizioni, sugli output e sulle immagini; la correlazione degli ID legacy è condivisa e non modifica la cronologia. `history/tool_attachments.py` legge i riferimenti alle immagini senza decodificarle. ChatMessage espone raccolta e galleria, senza interpretazione duplicata nel collector o nella UI.
- `tools/check_product_text.py`, `tests/test_product_text_configuration.py` — gate contro prosa, messaggi, descrizioni, log e tooltip inline nel server di produzione e nei sorgenti Python di agente/libreria, incluse le capability. È eseguito da `make test-product-text` e dalla suite `make test`.
- `docs/performance/text-configuration-20261014.json` — migrazione aggiuntiva dei testi, confronto dei token C preprocessati CPU/GPU e verifiche di server, agente e gate.
- `docs/architecture/clean-code-live-review-20261014.md`, `docs/performance/clean-code-live-review-20261014.json` — revisione successiva con modifiche anche alla libreria, wheel verificata, copertura completa, test UI e qualifica CUDA live; include la regressione cancel/auxiliary/steering e il suo fix.
- `docs/performance/clean-code-review-20261014.json` — esito del refactoring e dell'audit trasversale, test non live, limiti della verifica e aree ancora da semplificare.
- `agent/src/exmekh_agent/exmekh_bridge/providers/provider_context.py` — osservazione uniforme
  delle richieste normali e strutturate, isolamento dei contesti figli e
  conservazione dei token totali dichiarati dal provider. Budget e UI usano
  la stessa stima completa, con schema strutturato e override del thinking.
  Come il wrapper di cancellazione, delega la serializzazione dei tool all'API
  pubblica del provider concreto.
- `agent/src/exmekh_agent/exmekh_bridge/providers/ds4_context.py`, `ds4_prompt.py` —
  stima del prompt letterale Qwen e checkpoint dei return skill: le query non
  aggiungono istruzioni operative rimosse dal server né modificano lo stato.
  I checkpoint verificano il prefisso del chiamante, inclusi i tool, e vengono
  consumati al ritorno anche quando la risposta non include i conteggi.
  Eliminate le risorse obsolete `ds4_scope_guidance` e `ds4_return_guidance`.
  `tests/test_qwen38_compatibility.py` verifica validazione e sampling agentico
  senza richiedere il vecchio simbolo dell'iniezione server.
- `agent/src/exmekh_agent/exmekh_bridge/agents/auto_compaction.py` — soglia globale
  predefinita all'80% del limite scoperto per endpoint e modello; usa i totali
  del provider e la stima della prossima richiesta, con riarmo dopo la riduzione.
  Un errore di compattazione ferma l'inferenza conservando la history e permette
  un nuovo tentativo al turno successivo, senza disabilitare la policy.
  Nuovi messaggi riarmano il trigger anche se il risultato precedente resta
  sopra soglia; il confronto usa contenuti, distinguendo modifiche da semplici
  ricaricamenti. `RequestBudget` controlla la capienza prima dell'invio.
- `agent/src/exmekh_agent/exmekh_bridge/history/compaction.py` — `HistoryArchive`
  possiede contesto, porzioni sostituite e confini del turno per root e figli;
  espone checkpoint indipendenti. Replay, conteggio e Raw Current includono
  anche i messaggi successivi al checkpoint; Graph e Raw Full espandono gli originali.
  Raw offre uno switch fra archivio completo e contesto corrente.
  `agent/tests/test_auto_compaction.py`, `test_auto_compaction_e2e.py` e
  `agent/tests/compaction_probe.py` verificano il client reale con inferenza mock,
  steering, nesting, ripresa, recupero degli output e UI desktop/mobile.
  `RecordedCompaction` conserva per turno identità, blocchi sostituiti e confine
  dell'evento applicato, distinti dai marcatori inseriti nel contesto del modello.
  `agent/src/exmekh_agent/ui/graph/graph_data.py` colloca un punto `compaction`
  per operazione, anche quando sostituisce più blocchi, con legenda e dettaglio
  dedicati; conserva la posizione rispetto a tool, steering, skill e sottoagenti.
  Le chat precedenti mostrano i marcatori disponibili senza inventare esecuzioni
  o timestamp. Regressioni in `agent/tests/test_graph_compaction.py` e qualifica
  in `docs/performance/graph-compaction-events-20261016.json`.
- `docs/performance/automatic-history-compaction-20261016.json` — esiti delle
  suite complete frontend/libreria, copertura, hash della wheel e screenshot
  della compattazione automatica; distingue i test mock dalla qualifica live.
- `tests/test_compaction_scoring_cache.py` — regressione live dello scoring con
  JSON Schema vincolato: riuso della history Qwen con thinking none/low/high/xhigh,
  ripresa dal checkpoint originale e fallback dopo modifiche alla history.
  Include il replay con immagini e marcatori locali alla richiesta; il server
  costruisce i prompt fuori dal lock usando copie di checkpoint compatibili.
  Lo schema resta nel decoder; le istruzioni di formato sono in coda al prompt.
- `tests/test_server_compaction_replay.py` — regressione live Responses contro
  un server già attivo: confronta lo stesso scoring con e senza binding del
  tool nella KV; il replay completo deve rispettare il proprio prompt.
  I test nativi coprono anche history sostituite, riuso del prefisso visibile
  e continuazioni che inviano soltanto gli output dei tool.
  I checkpoint skill conservano il prefisso del chiamante; un return con history
  riscritta usa il replay aggiornato e consuma il vecchio checkpoint dopo il sync.
  `exmekh_lib/tests/live/test_history_compaction_agent_live.py` verifica anche
  i token effettivi del server dopo entrambi i ritorni da skill annidate.
- `docs/performance/automatic-compaction-replay-fix-20261016.json` — correlazione
  fra chat e log del server, riproduzione prima/dopo, copertura e verifiche della
  correzione del riuso tramite ID durante lo scoring della compattazione.
- `exmekh_lib/src/exmekh_lib/core/history_compaction/inference_policy.py` — contratto
  pubblico della policy prima dell'inferenza e dopo la sostituzione. `CompactionResult`
  possiede la proiezione dei messaggi trattenuti; `MessageHistory.discard_latest`
  rimuove soltanto le occorrenze rifiutate, preservando input già accettati.
- `exmekh_lib/src/exmekh_lib/core/history_compaction/scoring.py` — scoring con
  history completa e catalogo temporaneo in JSON compatto, per lasciare spazio
  allo schema senza ripetere i tag XML per ogni campo. Risultati e archivio
  conservano il renderer XML; `docs/performance/compaction-catalog-overhead-20261016.json`
  documenta la riproduzione del limite e la verifica sulla chat originale copiata.
- `exmekh_lib/src/exmekh_lib/core/models/request_budget.py`,
  `exmekh_lib/src/exmekh_lib/core/history_compaction/context.py` — budget del prompt
  con spazio riservato alla generazione ed esportazione adattiva degli output
  nella copia usata dallo scoring. Scoring e riassunti si suddividono quando
  necessario; il runtime ricontrolla la history dopo compattazione e steering.
  `agent/src/exmekh_agent/exmekh_bridge/resources/prompts/ds4_structured_prefix.md`
  permette di contare anche l'istruzione e lo schema delle chiamate strutturate.
  `agent/tests/test_compaction_budget_live.py` verifica con il server esistente
  una history iniziale oltre capienza e la successiva ripresa senza tool reali.
  Gli esiti sono in `docs/performance/adaptive-compaction-budget-20261016.json`.
- `docs/architecture/compaction-budget-clean-review-20261016.md`,
  `docs/performance/compaction-budget-clean-review-20261016.json` — revisione
  delle responsabilità del budget, della preparazione degli output e degli
  adattatori; regressione degli allegati duplicati, suite complete e prova live.
  `RequestBudget.split_text()` verifica i frammenti come richieste complete;
  `ToolOutputPreparer.shorten()` e `OutputCandidate.with_presentation()`
  possiedono riduzione e aggiornamento della rappresentazione.
- `agent/src/exmekh_agent/exmekh_bridge/agents/cancellation.py` — inoltra richieste
  normali/strutturate tramite `ProviderRequest.with_event_handler()`, conservando
  opzioni e allegati senza espanderli di nuovo; interrompe chiamate e stream
  dei turni cancellati.
- `exmekh_lib/src/exmekh_lib/core/history_compaction/progress.py` — snapshot
  transitori, conteggio degli output e stima dei tempi basata sui soli riassunti
  effettivi. Il runtime emette `DeepAgentCompactionProgressEvent`; il collector
  conserva lo stato per la riconnessione, escluso dalla serializzazione dei messaggi.
  `agent/src/exmekh_agent/ui/chat/compaction_progress.py` visualizza la barra
  nell'area della risposta e la rimuove alla conclusione dell'operazione o del turno.
  Test in `exmekh_lib/tests/core/history_compaction/test_progress.py`,
  `agent/tests/test_compaction_progress.py` e `test_auto_compaction_e2e.py`.
  Qualifica e provenienza in `docs/performance/compaction-progress-retention-20261016.json`.
- `exmekh_lib/src/exmekh_lib/core/history_compaction/policy.py` — normalizzazione
  uniforme dei voti dopo tutti i lotti: quota configurata del 10% con minimo due,
  preservando tutti i pari merito al confine. Valori in `config/history_compaction.py`.
- `agent/src/exmekh_agent/ui/shared/neural_loader.py`,
  `agent/src/exmekh_agent/ui/resources/js/neural_loader.js` — indicatore canvas
  per client, con caricamento asincrono, pausa e cleanup dell'animazione.
  `agent/src/exmekh_agent/exmekh_bridge/providers/loader_tokens.py` campiona
  un pool bounded tramite l'API pubblica del tokenizer; test unitari e Playwright
  in `agent/tests/test_neural_loader.py` e `test_neural_loader_e2e.py`.
- `exmekh_lib/src/exmekh_lib/core/tokenizers/sequential_tokenizer.py` — possiede
  vocabolario filtrato e ID sequenziali; `get_id_vocabulary` espone token immutabili
  senza creare sessioni né consumare ID.
- `agent/src/exmekh_agent/exmekh_bridge/capabilities/agent_design.py` — costruisce
  gerarchie generiche e personalizzate tramite le API Node; il runner riceve
  la gerarchia senza conoscere figli bundled o layout delle capability.
- `agent/src/exmekh_agent/exmekh_bridge/agents/collector.py`,
  `agent/src/exmekh_agent/state/active_turns.py` — il raccoglitore possiede
  accumulo e steering; il registro gestisce i task attraverso la sua API pubblica.
- `agent/src/exmekh_agent/exmekh_bridge/models/messages.py`,
  `agent/src/exmekh_agent/exmekh_bridge/agents/event_handlers.py` — ChatMessage gestisce
  contenuto e segmenti reasoning; gli handler adattano una volta i callback legacy
  e pubblicano lo stesso scope per Quick Chat e DeepAgent.
- `agent/tests/test_clean_review_contracts.py` — regressioni sui confini pubblici,
  metadati invalidi, callback reasoning e osservazione delle richieste strutturate.
- `agent/src/exmekh_agent/exmekh_bridge/agents/execution_restore.py` — ricostruisce history e cache
  tramite le API pubbliche ExMekh-Lib, mantenendo identità e metadati degli output
  riletti e lo scope dei messaggi di steering dopo la compattazione; esclude dalle
  esecuzioni le chiamate annullate prima del dispatch conservate nella raw history.
- `agent/src/exmekh_agent/exmekh_bridge/history/recorded_output.py` — lettore dei risultati persistiti
  tramite l'API pubblica `XmlHeaderContent` della libreria; conserva campi base64,
  corpo esatto e riferimenti, con fallback legacy;
  riconosce le ricevute di annullamento prima del dispatch, che il grafo mostra
  senza inventare un evento di esecuzione.
- `agent/src/exmekh_agent/exmekh_bridge/models/messages.py` e
  `agent/src/exmekh_agent/ui/chat/chat_view.py` — confronto pubblico dei dettagli
  di esecuzione, controlli stabili durante il reasoning e stato di apertura
  condiviso per tool, skill e pannelli reasoning.
- `agent/src/exmekh_agent/exmekh_bridge/history/steering_history.py` — seleziona gli ID di steering
  del livello corrente, escludendo quelli delle skill discendenti.
- `agent/src/exmekh_agent/exmekh_bridge/history/message_edit.py` — prepara il testo modificato,
  conservando gli upload espliciti e rigenerando i file del testo incollato.
  Il controller ferma e attende il turno precedente prima di salvare il ramo
  aggiornato. ChatSession gestisce il ramo, ChatSettings le approvazioni e
  SubAgentConversation il replay e gli snapshot delle chiamate; SessionManager
  salva prima di pubblicare le modifiche in memoria. StagedAttachment ripristina
  gli upload tramite TempStorage; Quick Chat conserva i testi inline.
- `agent/tests/test_message_editing.py`, `agent/tests/test_message_editing_e2e.py`
  — regressioni su modifica a inizio/metà/fine chat, streaming, riconnessione,
  annullamento, approvazioni pendenti, testi/immagini/PDF e persistenza in
  Quick Chat e progetto. `agent/tests/run_message_edit_agent.py` usa un provider
  deterministico con il client mock e il ciclo di esecuzione ExMekh-Lib reale.
- `agent/src/exmekh_agent/exmekh_bridge/history/image_input.py` — decodifica immagini, orientamento,
  rendering PDF locale (prime 10 pagine), parti native ExMekh-Lib e serializzazione
  PNG della cronologia; i payload base64 non vengono contati come testo.
- `agent/src/exmekh_agent/ui/shared/image_preview.py` — miniature e anteprima modale condivise
  fra compositore e chat; gli allegati visivi non hanno un secondo chip file.
- `agent/tests/test_image_input.py`, `agent/tests/test_images_e2e.py`,
  `agent/tests/test_images_live.py` — regressioni multimodali, browser e prove
  opt-in sui due endpoint DS4, inclusi PDF, steering e replay.
- `agent/wheels/` — wheel ExMekh-Lib versionata, installabile con `uv sync --frozen`.
- `agent/tests/parallel_execution.py` — test ExMekh Agent per modulo su CPU logiche, con
  fase esclusiva per test seriali e provider reali dopo il teardown degli altri worker.
- `exmekh_lib/tests/conftest.py` — worker ExMekh-Lib sulle CPU logiche disponibili, opt-in dei
  test live e raggruppamento dei test seriali con `--dist=loadgroup`.
  Le regressioni distinguono errori di esecuzione da risultati applicativi
  negativi e verificano le due piattaforme di cancellazione dei processi.
- `exmekh_lib/old_agent/` — archivio storico Agent Wiki: launcher presenti ad agosto 2026, wheel `agent-wiki` 0.1.0 versionata a settembre e sorgenti estratti integralmente; provenienza e hash in `provenance.json`, guida in `README.md`.
- `exmekh_lib/` — repository Git locale dei sorgenti ExMekh-Lib, con cronologia propria,
  esclusa dal Git di DS4.
- `exmekh_lib/src/exmekh_lib/core/history_compaction/` — compattazione esplicita e isolata per
  nodo; valutazione, riassunti/presentazione e proiezione sono separati. Lo schema
  contiene un campo obbligatorio per candidato. Scope attivi, user/system e
  riferimenti agli output mantengono il proprio contratto.
  Il runtime ammette anche una policy per nodo prima dell'inferenza; l'evento
  `DeepAgentHistoryCompactionEvent` conserva le porzioni sostituite per i frontend.
- `exmekh_lib/src/exmekh_lib/core/models/xml_header.py` — renderer e lettore pubblico
  degli header; metadati decodificati e corpo esatto condivisi con il frontend.
- `exmekh_lib/src/exmekh_lib/core/cache/output_file_store.py` — letture di snapshot
  autorizzate dal proprietario, con verifica dell'identità prima e dopo l'accesso.
- `docs/architecture/history-compaction-clean-review-20261016.md` — responsabilità,
  regressioni di snapshot/XML/streaming e qualifica della wheel aggiornata.
- `exmekh_lib/src/exmekh_lib/core/models/edit_document.py` — parser letterale `edit_file.v3` con soli marcatori `exmekh_edit`/`exmekh_new_text`, ID e corpi senza decoding XML; nessun accesso al filesystem.
- `exmekh_lib/src/exmekh_lib/core/file_inspection/edit_selection.py`, `editing.py`,
  `interactive.py` e `edit_filesystem.py` — selezioni letterali e conflitti sullo snapshot
  iniziale, autorizzazioni per call ID/indice, raggruppamento alias, verifica dopo hook e
  commit per file. L'edit preparato possiede preflight e commit, l'approvazione verifica
  le evidenze; il runtime annulla anche cambi di scope durante una pausa.
  Filesystem/evidenze/permessi restano del client; masking/KV del server.
  `FileDeletionTarget` possiede validazione, identità e rimozione file/alberi;
  i batch del tool mantengono ordine e fallimenti indipendenti e rifiutano
  link simbolici espliciti senza eliminarne i bersagli.
- `exmekh_lib/src/exmekh_lib/core/file_inspection/lines.py` e
  `exmekh_lib/tests/core/file_inspection/test_lines.py` — API comune per righe fisiche,
  offset di lettura e confini CR/LF, inclusa la fine del file.
- `exmekh_lib/src/exmekh_lib/core/models/tool.py` — il tool possiede la copia della
  politica di esecuzione; i compositori usano la sua API per mantenere contratto
  interattivo e obbligo di esecuzione sincrona negli override.
- `exmekh_lib/src/exmekh_lib/config/interactive.py` — versione, tipo di snapshot ed endpoint condivisi tra negoziazione, validazione e trasporto del client.
- `exmekh_lib/src/exmekh_lib/core/file_inspection/edit_filesystem.py` — `FileWriteTarget` raggruppa risorsa, path canonico e snapshot, esponendo `validate()` e `commit()`; locking e pubblicazione atomica restano sotto questa responsabilità. Prefisso e permessi temporanei vengono da `config/file_editing.py`.
- `exmekh_lib/tests/core/file_inspection/test_edit_write_boundaries.py` e `tests/core/tool_calls/test_output_failure_boundaries.py` — nuovi casi di errore: scritture concorrenti, commit parziali, cancellazione, metadati e output non pubblicabili.
- `exmekh_lib/src/exmekh_lib/core/providers/interactive_contract.py` e
  `interactive_transport.py` — autorizzazioni wire immutabili per chiamata e
  classificazione uniforme degli errori JSON/encoding di HTTP e SSE.
- `exmekh_lib/tests/core/file_inspection/test_literal_file_tools.py`,
  `test_xml_edit_document.py` e `tests/live/test_file_editing_live.py` —
  regressioni client sui tre tool, ID omessi, corpi letterali, alias e rifiuto
  dell'intera chiamata; verifiche live dei byte e del routing del modello.
- `docs/architecture/client-xml-edit-v2-20261012.md` — qualifica client XML v2 e
  aggiornamento della wheel predefinita; esiti live distinti dai test deterministici.
- `docs/architecture/client-xml-clean-review-20261012.md` — revisione delle
  responsabilità dei tool, refactoring del client e nuove regressioni di cancellazione,
  con qualifica deterministica e live della wheel predefinita.
- `docs/architecture/file-tools-literal-v3-20261013.md` — migrazione client/server
  ai tre tool separati, wheel installata, verifiche deterministiche e limiti
  ancora osservati nelle prove agentiche con Qwen27/CUDA.
- `docs/architecture/file-tool-simplification-research-20261013.md` — ricerca sui
  tool di file di Codex, Claude Code/API, Aider, Pi, oh-my-pi e Qwen Code;
  confini DeepSeek/llguidance e proposta write/edit/delete con riuso del masking,
  rimozione di XML annidato, selezioni lunghe e confronto sperimentale.
- `agent/src/exmekh_agent/sync_config.py` — politica, percorsi e cataloghi della build del wheel, accessibili senza dipendenze applicative; `sync.py` separa build, installazione e orchestrazione. Il launcher carica i sorgenti in un interprete isolato anche senza un frontend già installato.
- `agent/src/exmekh_agent/package_paths.py` — percorsi condivisi del pacchetto,
  usati da configurazione e bootstrap; `utils/resources.py` possiede anche
  `catalog_message`, evitando lettori e percorsi duplicati nel sync.
- `agent/src/exmekh_agent/state/session_manager.py` — persistenza atomica dei record,
  configurazioni pubblicate dopo il salvataggio e viste con selezione indipendente
  per pagina; le viste condividono conversazioni, progetti e store.
- `agent/src/exmekh_agent/exmekh_bridge/models/timeline.py` — ordine di tool, skill
  e conferme dello steering, con identità delle invocazioni ripetute; chat e grafo
  consumano la stessa API anche senza raw history.
- `agent/src/exmekh_agent/ui/navigation/chat_routes.py` — URL leggibili e bookmark
  risolti per identità tramite il repository fornito dalla pagina.
- `agent/tests/test_clean_code_corrections_3.py` e `test_corrections_3_e2e.py` —
  regressioni su persistenza fallita, decisioni obsolete, ordinamento dello steering,
  schede indipendenti, bookmark e cronologia del browser.
- `agent/sync_exmekh_lib.sh` e `agent/src/exmekh_agent/sync.py` — ricostruiscono ExMekh-Lib dai sorgenti
  locali, aggiornano wheel e lockfile e reinstallano la dipendenza nel frontend.
  La distribuzione e la CLI sono `exmekh-lib`, il namespace Python è
  `exmekh_lib` e il comando installato di sync è `sync-exmekh-lib`. Dopo
  l’installazione riuscita il sync rimuove anche le vecchie wheel `exmekh_lib`.
- `docs/agentic/frontend.md` — avvio, configurazione, dipendenze e test di ExMekh Agent.
- `ds4_bench.c` — benchmark di prefill/decode a diverse frontiere di contesto;
  ripete gli sweep mantenendo il modello residente e usa snapshot o replay fuori
  dalle finestre cronometrate.
- `tools/perf_harness.py` — CLI senza dipendenze che orchestra probe hardware,
  suite ripetute di `ds4-bench`, confronti baseline/candidate in JSON, audit
  Q8_1 che separano i byte consumati da MMVQ dal metadata `ds.y` e confronti
  full-vocabulary di una riga fra run Qwen etichettati; il gate delle suite
  oracle conta sequenze complete e argmax greedy/teacher-forced; `doctor`
  segnala separatamente client, benchmark e server obsoleti. La suite completa
  2K–30K controlla floor e forma della curva; le suite MTP coprono 0–22K, la
  ricerca del crossover split-K e della profondità V(3)/V(2). `server-curve`
  esegue queste matrici via HTTP usando i token effettivi e il tempo di decode
  del server.
  Il comparatore richiede campioni, warm-up e identità compatibili prima di
  promuovere; il calcolo statico dei costi supporta UD e separa NextN dal target.
- `tools/bench_qwen_flash_agent.py` — riproduce prefill e decode dal runner
  ExMekh reale avviando `agent/start-ds4-server-flash.sh`, con stato isolato,
  fronti riusati, log per richiesta e memoria GPU; confronta configurazioni
  del launcher a sampling fisso senza caricare conversazioni utente.
- `tools/qualify_qwen_flash_launcher.py`, `tests/qwen_flash_quality_probe.c` —
  confronto full-vocabulary e loss teacher-forced tramite sessioni di produzione
  a capacità/chunk espliciti, incluso un prompt con gli schemi tool dell'agente.
  La qualifica numerica resta distinta dai benchmark e dai gate di stato/vision.
- `tests/test_flash_launcher.py` — esamina i comandi reali dei launcher senza
  modelli: profilo Flash standard/legacy, override, errori di configurazione
  e isolamento del launcher 27B, selezione suite e conservazione di contesto/MTP/visione.
- `tests/test_suite_embedding_tools_e2e.py` — accettazione live della wheel ExMekh:
  quattro modalità, batch/interleaving, task/dimensioni, audio dei video,
  ranking reale, cache, errori e ritorno al LLM. `make test-suite-agent-live`
  unisce queste prove ai tool immagini (inclusa lettura dei pixel generati
  con history nuova e paesaggio 1024×768) e al browser reale dell'agente.
- `tests/test_lifecycle_cuda.c` — verifica anche ownership dei buffer visivi
  ricreati dal payload e tre park/restore entro la capienza device dichiarata;
  `ds4_session_load_payload` attribuisce ogni allocazione lazy alla sessione.
  Il servizio libera i pesi Gemma a fine richiesta per lasciare spazio ai futuri
  transitori del LLM; la conservazione fra richieste richiede un bound completo.
- `tests/fixtures/native-suite-e2e/`, `tools/build_suite_e2e_video.py` — MP4 H264/AAC
  sintetico con checksum e provenienza; il generatore offline usa PyAV, i test
  live consumano la fixture versionata senza dipendenze codec Python.
- `docs/performance/native-suite-agent-live-20261022.md` e relativo JSON —
  riproduzione del 404 dell'agente, correzioni di restore/headroom e accettazione
  di 12 test live consecutivi con Qwen27, MTP, visione e contesto 26000.
- `docs/performance/native-suite-refactor-review-20261022.md` e relativo JSON —
  revisione T19–T22: API della cache proprietaria di budget/preload/rilascio,
  lifetime dello staging, shutdown fallibile e packaging atomico; regressioni
  negative, sanitizer, suite frontend completa e nuova accettazione live.
- `tools/qwen_flash_launcher.py` — legge i default dal dry-run del launcher;
  benchmark e qualifica non mantengono copie della configurazione Flash.
- `tools/qwen_flash_metrics.py`, `tests/test_qwen_flash_tools.py` — confronto
  teacher-forced condiviso, validazione dei dump e regressioni su lifecycle
  del server, isolamento dei provider e completezza delle coppie A/B.
- `ds4_qwen4_runtime.h` — vocabolario tipizzato dei workload Flash e metadata
  pubblici; `ds4_qwen4_tuning.cuh` possiede scope, selezione e reset delle preferenze.
- `ds4_qwen4_cpu.h`, `ds4_qwen4_cpu_internal.h` — API delle proiezioni CPU
  separata da dequantizzazione, tabelle e SIMD privati dell'implementazione.
- `docs/architecture/clean-code-review-20261008.md` — revisione dei confini
  Flash, benchmark e stato dell'agente; verifiche e limiti dei refactoring.
- `tools/import_jsonschemabench_subset.py` e
  `performance/jsonschemabench-subset.json` — sparse-fetch riproducibile e
  pinned dell'intero corpus esterno, classificazione fail-closed delle 9.558
  schema e tier `smoke`/`safety`, senza adattare gli schema unsupported.
- `tests/test_server_sampling_matrix.py` e
  `tests/run_server_sampling_matrix.ps1` — matrice server Qwen3.8 live per
  temperatura zero/nonzero, thinking on/off e target-only/MTP su testo libero,
  SEARCH agentico opzionale e tool obbligatorio; verifica contatori interni,
  semantica deterministica e contratto pubblico dei casi campionati.
- `docs/performance/qwen38-agent-search-static-2026-08-27.md` — baseline,
  candidate, gate full-vocabulary, matrice sampling/thinking/MTP, fonti primarie
  e risultato della partizione DSML `SEARCH`.
- `performance/constrained-workloads.json` — workload canonici DSML/JSON,
  inclusa la fixture nullable free-string che copre lo schema reale
  `anyOf: [string, null]` dell'agente.
- `tools/perf-qwen-validate.sh`, `tools/perf-qwen-r8.sh`,
  `tools/perf-qwen-long-context.sh` — workflow eseguibili richiamati
  dall'harness per build/correctness, relink congiunto di `ds4`, `ds4-bench` e
  `ds4-server`, A/B F32-vs-R8 e ciclo profile/direction/slow sui contesti
  8K–16K, compreso il confronto GQA scalare/fuso. R8 fuso, split-K32 da 96 token e
  GQA2 e i percorsi MTP conservano le primitive condivise Qwen CUDA. Le
  misure Qwen3.6 restano evidenze storiche, non gate per il runtime corrente.
- `performance/workloads.yaml` — workload canonici direction/quick/standard/slow/exhaustive
  e curva completa 2K–30K a passo 2K, incluse la conferma R8 breve e quella
  long-context a finestre da 64 token, separati per fase, batch, context e
  prefill chunk.
- `performance/README.md` — comandi brevi e confini verificati del harness.
- `performance/references.md` — fonti primarie, implementazioni di confronto e
  letteratura HTML, con criteri espliciti per separare evidenza e fuffa.
- `docs/research/INDEX.md` — ingresso alla knowledge base locale: paper HTML
  archiviati e hashati, schede delle piattaforme, fork RTX 3090, guide tematiche
  e indici gerarchici con intervalli di riga.
- `docs/research/forks/syv-ai-qwen38-27b-rtx3090.md` — analisi storica del
  serving Qwen3.8 su RTX 3090, con separazione fra ottimizzazioni pure, drift
  numerico, trade-off di qualità e failure di correttezza.
- `docs/research/papers/manifest.json`, `corpus-lock.json` — selezione motivata
  dei paper e provenance esatta degli artefatti scaricati.
- `docs/research/sources-lock.json` — commit fissati e grado di evidenza dei
  runtime, fork e raccolte operative analizzati.
- `tools/fetch_research_corpus.py` — scarica arXiv HTML, usa ar5iv come fallback
  dichiarato e genera il testo semantico locale.
- `tools/build_research_indexes.py` — rigenera catalogo e indici con range di
  riga calcolati dal contenuto effettivo.
- `tools/build_docs_indexes.py` — rigenera l'indice madre, gli indici tematici
  e il catalogo machine-readable dell'intera wiki.
- `tests/test_research_corpus.py` — verifica completezza, SHA, range, indici e
  limite dei link esterni nelle schede repository.
- `tests/test_docs_wiki.py` — blocca link interni rotti, range falsi e documenti
  Markdown orfani o lasciati nella radice di `docs`.
- `ds4_eval.c` — harness di valutazione end-to-end su piccoli set di domande,
  con rendering, inferenza reale ed estrazione/valutazione delle risposte.
- `ds4_help.c`, `ds4_help.h` — testo e rendering condiviso dell'help dei binari.
- `linenoise.c`, `linenoise.h` — editor di linea usato da CLI e agent.

### API, modello e orchestrazione centrale

- `ds4_runtime.c/.h` — descrittori immutabili, scope serializzate, attivazione,
  fence e ownership del runtime condiviso; bridge del caricatore legacy in `ds4.c`.
- `ds4.h` — API pubblica stretta: descrittore canonico di famiglia/target,
  opzioni engine, token, sessioni, sync/eval, logits, sampling filtrato,
  batching, snapshot, checkpoint ricorrenti Qwen per skill e callback.
- `ds4.c` — nucleo verticale del motore: parser GGUF mmap-backed, metadata,
  shape dei modelli, tensor binding, tokenizer, CPU reference, costruzione e
  scheduling del grafo, sessioni, prefill/decode, KV, logits, MTP e payload
  degli snapshot. Include il percorso specializzato Qwen `qwen35` text-only:
  validazione del Qwen3.8 27B UD-Q4_K_S, tokenizer/template Qwen e sessione ibrida
  full-attention/Gated DeltaNet CUDA e prefill layer-major a chunk. Espone il
  salvataggio/ripristino session-scoped su SSD dello stato ricorrente Qwen per
  le frontiere delle skill; per Qwen3.8 UD-Q4_K_S il payload generale portabile
  comprende inoltre timeline, logits, stato GDN, KV full-attention e stato MTP,
  e rifiuta il restore quando payload e runtime non concordano sulla modalita'
  MTP.
  Il batching CUDA nativo basato sul raw-KV DeepSeek e' separato dal grafo
  ibrido Qwen: Qwen usa il fallback multi-sessione serializzato e corretto finche'
  non esiste un encoder batch dedicato, evitando accessi al raw-KV non allocato.
  Il fork Qwen in memoria resta escluso. Lo scheduler distingue esplicitamente
  decode, prefill, verifica MTP, draft, replay e catch-up; il profiler Qwen
  opzionale emette tempi strutturati attention/FFN per fase e layer, consumati
  dal comando profile-network del performance harness.
- `ds4_qwen.h` — vocabolario interno unico delle fasi di esecuzione Qwen,
  contesto stage/layer passato alle proiezioni e dipendenze fra hidden,
  logits e readback CPU, condivisi fra scheduler e backend.
- `ds4_gpu.h` — interfaccia tensoriale condivisa fra il grafo C e i backend
  accelerati; descrive tensori opachi, operazioni, attention e batching,
  incluse le due primitive interne Qwen per full-attention gated e Gated
  DeltaNet.
- `ds4_env.h` — parsing booleano condiviso dei flag Qwen CUDA/MTP, con default
  per valori vuoti e riconoscimento di `0`, `false`, `off` e `no`.
- `ds4_gpu_args.c`, `ds4_gpu_args.h` — parsing condiviso di dispositivi, budget
  VRAM e configurazione multi-GPU.
- `ds4_layer_pack.c`, `ds4_layer_pack.h` — algoritmo puro C per assegnare
  intervalli contigui di layer ai tier/GPU rispettando i budget.

### Backend Metal

- `ds4_metal.m` — creazione del device, queue e library Metal; viste no-copy
  dei pesi mmap, command buffer, tensori persistenti, scratch e wrapper dei
  kernel.
- `metal/norm.metal` — RMS normalization e operazioni di normalizzazione.
- `metal/dense.metal` — moltiplicazioni dense e proiezioni.
- `metal/flash_attn.metal` — nuclei di attention accelerata.
- `metal/dsv4_kv.metal` — aggiornamento e lettura delle strutture KV specifiche.
- `metal/dsv4_hc.metal` — operazioni hyper-connection DeepSeek.
- `metal/moe.metal` — routing e calcolo Mixture-of-Experts.
- `metal/dsv4_rope.metal` — trasformazioni RoPE.
- `metal/dsv4_misc.metal` — operazioni specializzate non collocate negli altri
  moduli.
- `metal/softmax.metal`, `metal/argsort.metal`, `metal/sum_rows.metal` —
  riduzioni, ranking e normalizzazione usati da attention/router.
- `metal/get_rows.metal`, `metal/set_rows.metal`, `metal/cpy.metal`,
  `metal/concat.metal`, `metal/repeat.metal`, `metal/unary.metal`,
  `metal/glu.metal`, `metal/bin.metal` — primitive tensoriali di supporto.

### Backend CUDA

- `ds4_cuda.cu` — runtime CUDA principale: allocazione/copie, cuBLAS, kernel
  custom, quantizzazione, attention, MoE, KV, batching e supporto multi-GPU;
  attiva la cache tiered Flash tramite una capability dell’engine, isolando le variabili Flash dagli allocator degli altri modelli; contiene inoltre embedding/matvec Q4_K/Q5_K/Q6_K, RoPE parziale, attention gated
  e aggiornamento ricorrente Qwen task 6, più i kernel multi-riga causali e i
  GEMM di prefill Qwen. Il decode Q8_1-R8 predefinito rappresenta l'attivazione
  con due residui Q8_1 e riusa MMVQ Q4/Q5/Q6, decodificando una sola volta i
  pesi condivisi dai due accumulatori; il rollback F32 è
  `DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8=1`. Il
  percorso MTP Qwen aggiunge embedding/matvec Q4_0,
  range device distinti per target e sidecar, kernel warp-8 Q4_0 del drafter e
  Q4_K/Q5_K/Q6_K microbatch del verifier. I Qwen single-GPU che entrano in VRAM
  sono copiati per default sul device con letture dirette e quattro buffer pinned
  asincroni da 64 MiB sovrapposti ai trasferimenti H2D;
  `DS4_CUDA_MODEL_COPY_CHUNK_MB` regola il chunk diagnostico,
  `DS4_CUDA_LEGACY_MODEL_COPY=1` ripristina la copia pageable singola e
  `DS4_CUDA_NO_MODEL_COPY=1` conserva il percorso host-map diagnostico. I matmul
  Q8_0, inclusi quelli usati dal support model Qwen MTP,
  possono usare il tier MMQ upstream; i kernel del decode condividono lo stream
  coordinato necessario alla cattura e al replay dei grafi CUDA.
- `cuda/mmq/` — port locale e documentato dei matmul quantizzati CUDA upstream:
  ABI C sottile, repack/allineamento, dense e MoE Q8/Q2/Q4/IQ2/MXFP4, con
  provenienza in `VENDOR.md` e test standalone nella sottodirectory `test/`.
- `ds4_iq2_tables_cuda.inc` — tabelle costanti per dequantizzazione/calcolo IQ2
  nel backend CUDA.

### Backend ROCm

- `ds4_rocm.cu` — unità di build HIP/ROCm e integrazione del backend.
- `ds4_rocm.h` — API e tipi specifici ROCm esposti al resto del runtime.
- `ds4_rocm_compat.cu` — livello di compatibilità fra API comuni e backend
  ROCm.
- `ds4_rocm_unavailable.cu` — stub diagnostico per build senza ROCm.
- `rocm/ds4_rocm_runtime.cuh` — device, stream, memoria e lifecycle runtime.
- `rocm/ds4_rocm_common.cuh` — tipi, macro e utility condivise.
- `rocm/ds4_rocm_hipblaslt.cuh`, `rocm/ds4_rocm_matmul.cuh` — integrazione
  hipBLASLt e moltiplicazioni.
- `rocm/ds4_rocm_attention.cuh`,
  `rocm/ds4_rocm_attention_launch.cuh` — kernel e launcher attention.
- `rocm/ds4_rocm_norm_rope.cuh` — norm e RoPE.
- `rocm/ds4_rocm_moe.cuh`, `rocm/ds4_rocm_moe_launch.cuh`,
  `rocm/ds4_rocm_router.cuh`, `rocm/ds4_rocm_shared_expert.cuh` — router,
  shared/routed experts e launcher MoE.
- `rocm/ds4_rocm_compressor.cuh`, `rocm/ds4_rocm_indexer.cuh`,
  `rocm/ds4_rocm_hc.cuh` — attention compressa, selezione e hyper-connection.
- `rocm/ds4_rocm_fp8_kv.cuh`, `rocm/ds4_rocm_fp8_kv_launch.cuh` — cache KV FP8.
- `rocm/ds4_rocm_q8.cuh` — primitive per tensori Q8.
- `rocm/ds4_rocm_embedding_launch.cuh`,
  `rocm/ds4_rocm_output.cuh`,
  `rocm/ds4_rocm_hc_output_launch.cuh`,
  `rocm/ds4_rocm_misc_launch.cuh` — embedding, output head e launcher
  specializzati.
- `rocm/ds4_rocm_glm.cuh` — percorsi specifici GLM.
- `rocm/ds4_rocm_current_api_compat.cuh` — adattamento alla versione corrente
  dell'API GPU condivisa.

### SSD, KV persistente e stato

- `ds4_ssd.c`, `ds4_ssd.h` — parsing dei budget SSD, mapping/allocazioni e
  primitive comuni usate dallo streaming.
- `ds4_kvstore.c`, `ds4_kvstore.h` — formato durevole dei checkpoint KV,
  validazione dell'envelope e del payload ABI, lookup, budget/eviction e
  gestione dei file, inclusa l'eliminazione delle entry che falliscono il
  restore affinche' possano essere rigenerate; per Qwen rende durevoli soltanto
  gli anchor `agent-system`, limitati a 10 con LRU stretto. `ds4_server.c` conserva invece
  una sola history HDS temporanea per slot (`.dsh`, cambio system prompt con
  chiamata child pendente) e checkpoint deep-skill temporanei per `call_id`
  (`.dsk`); entrambi vengono consumati al ritorno e non sopravvivono al server.
- `ds4_streaming_hotlist.inc` — hotlist degli esperti DeepSeek da precaricare.
- `ds4_streaming_hotlist_glm52.inc` — hotlist equivalente per GLM 5.2.

### Multi-GPU e distribuito

- `ds4_gpu_mgpu.h` — strutture e API del runtime multi-tier/multi-GPU.
- `ds4_distributed.c`, `ds4_distributed.h` — coordinatore e worker di pipeline
  distribuita, protocollo, routing dei layer, KV distribuita e snapshot.
- `ds4_tp.c`, `ds4_tp.h` — tensor parallel fra Mac, protocollo lockstep e
  trasporto delle attivazioni via TCP o RDMA.
- `ds4_layer_pack.c`, `ds4_layer_pack.h` — placement contiguo dei layer,
  condiviso anche con i percorsi locali multi-GPU.

### Supporto applicativo

- `rax.c`, `rax.h`, `rax_malloc.h` — radix tree usato per lookup e strutture
  indicizzate del server/cache.
- `dir-steering/` — strumenti e fixture per costruire e applicare direction
  vectors di steering.

### Quantizzazione e confronto con altri motori

- `gguf-tools/deepseek4-quantize.c` — conversione/quantizzazione dei tensor nei
  formati supportati dal progetto.
- `gguf-tools/quants.c`, `gguf-tools/quants.h` — implementazioni CPU e
  definizioni dei formati quantizzati.
- `gguf-tools/quality-testing/data/qwen36-27b/` and
  `data/qwen36-27b-mtp/` — Qwen3.6 manifests and fixtures following the other
  sets under `data/`; shared validators and generators stay at the
  `quality-testing/` root.
- `gguf-tools/mixed/` — strumenti per comporre GGUF con precisioni differenti
  fra gruppi di layer/tensor.
- `gguf-tools/imatrix/` — costruzione dataset e strumenti per importance
  matrix.
- `gguf-tools/quality-testing/score_official.c` — scorer DS4 teacher-forced:
  logits locali, NLL, token e bytes decodificati e confronto con
  logprob/top-k ufficiali.
- `gguf-tools/quality-testing/score_llama.cpp` — scorer oracle equivalente
  basato su llama.cpp.
- `gguf-tools/quality-testing/compare_scores.py` — confronto e aggregazione dei
  report di qualità.
- `gguf-tools/quality-testing/collect_official.py` — acquisizione controllata
  delle continuazioni e top-logprob da API esterne.
- `gguf-tools/quality-testing/inspect_qwen36_gguf.py` — ispettore offline degli
  header Qwen3.6: verifica artefatti, metadata, tokenizer/template e directory
  tensor, producendo gli snapshot condivisi in `data/qwen36-metadata/`.
- `gguf-tools/quality-testing/generate_qwen36_oracle.py` — generatore in
  staging per llama.cpp, Transformers e vLLM; conserva path locali fuori dai
  manifest e separa i logits greedy e teacher-forced con checksum.
- `gguf-tools/quality-testing/verify_qwen36_run.py` — valida inventario,
  copertura, lunghezza e determinismo fra due candidati oracle senza
  promuoverli a golden.
- `gguf-tools/quality-testing/generate_qwen36_score.py` — prepara un run DS4 o
  llama.cpp sulla sequenza canonica di token di un oracle, lasciando agli
  scorer nativi inferenza e dump float32 e registrando checksum e provenienza
  nello stesso layout del task 2.
- `gguf-tools/quality-testing/run_qwen36_context_matrix.py` — espande il profilo
  16K in run sequenziali per frontiera, chunk e ripetizione, controllando la
  VRAM libera e il gate prestazionale prima di avviare ogni processo modello.
- `gguf-tools/quality-testing/qwen36_speed_gate.py` — valida il report breve di
  prefill/decode e blocca i run long-context sotto 500/15 token/s o con
  provenienza non confrontabile.
- `gguf-tools/quality-testing/compare_qwen36_trace.py` — confronta i dump
  float32 per layer/stadio prodotti da DS4 e llama.cpp sullo stesso prefisso e
  localizza la prima divergenza senza definire nuove soglie semantiche.
- `gguf-tools/quality-testing/diagnose_qwen36_numerics.py` — triangola le trace
  DS4, llama.cpp CPU e CUDA, canonicalizza lo stato GDN e classifica le
  deviazioni rispetto all'inviluppo numerico dei backend.
- `gguf-tools/quality-testing/compare_qwen36_equivalence.py` — harness unico
  DS4-vs-llama.cpp e DS4-vs-DS4: valida inventari, calcola metriche
  full-vocabulary per posizione, applica gate versionati inclusa la suite
  corta del task 4 e produce report auditabili `PASS`, `FAIL`, `NOT_VERIFIED`
  o `ERROR`.

### Test rilevanti per comprendere la correttezza

- `tests/ds4_test.c` — runner principale di unit test e test model-backed;
  include equivalenza Qwen MTP/target-only e fixture deterministiche per
  accettazione completa, rifiuto e rollback parziale.
- `tests/test_constrained_json_api.py` — suite HTTP live e model-backed per
  masking DSML/JSON Schema: matrice senza/con history sotto 20k token, prompt
  di injection ostili, duplicati, Unicode/escape e validazione degli output;
  usa un'allocazione CUDA Qwen di 22.593 token per restare sotto il limite VRAM
  osservato oltre la frontiera effettiva di 22K.
- `agent/tests/test_flash_structured_live.py` — matrice opt-in sul client ExMekh
  installato, con Flash come default e modello esplicito selezionabile tramite
  `DS4_FLASH_STRUCTURED_MODEL`: Chat/Responses, streaming, thinking, temperatura,
  schema Pydantic annidato/nullable, Unicode e rifiuto degli output troncati;
  eseguibile contro server target-only e MTP, senza dati di chat utente.
- `tests/test_server_sampling_matrix.py` — fixture model-free incluse in
  `make test` e gate live Chat/Responses per il prodotto cartesiano
  temperatura × thinking × MTP, con separazione fra identità semantica e
  contratto pubblico quando Responses non dispone di seed.
- `tests/test_perf_harness.py` e `tests/test_jsonschemabench_subset.py` — test
  dell'harness constrained, delle metriche di fase e dell'import esterno pinned.
- `tests/test-vectors/` — prompt corti/lunghi, vettori ufficiali e golden locali
  per token e distribuzione.
- `tests/test_engine_correctness.c` — confronto della continuazione greedy fra
  configurazioni CUDA.
- `tests/test_metal_session_batch.c` — oracle full-vocabulary del batching
  Metal contro sessioni indipendenti.
- `tests/test_cuda_session_batch.c` — oracle del batching CUDA.
- `tests/test_cuda_mixed_batch.c` — oracle bit-exact per batch misti
  prefill/decode CUDA.
- `tests/cuda_long_context_smoke.c` — controlli CUDA mirati alle strutture
  long-context e regressione sintetica del matmul Q6_K contro la
  dequantizzazione CPU.
- `tests/glm_long_context_smoke.sh` — integrazione GLM su prompt lunghi.
- `tests/test_server_batching.py` — concorrenza e batching osservati tramite
  API server, con coppie deterministiche a lunghezze corte, medie e lunghe sia
  non-streaming sia streaming.
- `tests/test_agentic_api.py` — suite HTTP sottile e concatenabile per API
  Responses agentiche: compatibilità standard, capability dinamiche, masking
  avversariale, distinzione tool/skill, nesting/ricorsione, restore e rifiuto
  atomico dei checkpoint mancanti, corrotti o troncati; i casi che verificano
  un nome preciso usano `tool_choice=required` con un unico nome consentito,
  separando il protocollo dalla propensione del modello a chiamare un tool.
- `docs/architecture/steering-cache-reproduction-20261014.md` e
  `docs/performance/steering-cache-reproduction-20261014.json` — misure delle
  perdite di riuso dopo steering e richieste ausiliarie, controlli positivi,
  riproduzione SSE lunga e isolamento senza comportamento edit interattivo.
- `tests/test_steering_cache_live.py` — riproduzione HTTP opt-in del prefill della
  storia dopo steering e richieste ausiliarie; controlli con tool completati,
  cancellazione di una pausa edit e contesti separati, misurati tramite i
  contatori cached/prefill del server. Non esegue tool sul filesystem.
- `docs/architecture/steering-completed-response-20261015.md` e relativo JSON
  in `docs/performance/` — correlazione fra chiamate generate ed esecuzioni nella
  chat originale; correzione client delle risposte completate ma superate dallo
  steering, conservate come evidenza con output cancellati senza dispatch.
- `docs/architecture/clean-code-client-review-20261015.md` e relativo JSON —
  audit delle responsabilità client/frontend, target di cancellazione, API delle
  evidenze di risposta, stati dei tool e bootstrap condiviso, con test e hash.
- `exmekh_lib/tests/live/test_steering_history_live.py` — due steering consecutivi
  tramite client/server reali, JSON e SSE; contatori wire della cache e nessuna
  esecuzione dei tool sintetici.
- `tests/test_server_interactive.c` — fixture host per vincoli testuali e
  oracolo indipendente, XML/entità/CDATA, chiusure letterali e finali, sintassi native,
  frontiere di sampling, prefissi ]/]] coerenti con lo snapshot, retry, batch,
  rifiuto e scadenza delle continuazioni.
- `tests/bench_interactive_edit.c` — benchmark ripetibile del costo del parser per
  old_text/new_text, testo semplice, HTML/CDATA, Unicode ed entità; warm-up e cinque
  campioni per dimensione, con parser completo conservato come confronto.
- `tests/bench_interactive_edit_live.py` — benchmark client contro un server già
  attivo: HTML, Unicode e selezioni letterali, controllo esatto dei byte e
  conservazione di messaggi/scambi per distinguere intenti falliti da prestazioni.
  Il report è in `docs/performance/interactive-edit-speed-20261012.md`.
  Include selezioni lunghe plain text e HTML/Unicode in CDATA per i batch 32/64.
  Con --server-log separa decoding ed end-to-end, escludendo i token ripristinati;
  `tests/test_interactive_edit_benchmark.py` protegge questi contatori.
  La qualifica dei draft è in `docs/performance/old-text-snapshot-chunks-20261012.md`.
- `agent/tests/mock_client.py`, `agent/tests/test_mock_client.py` — provider di
  test e contesti di installazione che ripristinano lo stato precedente anche
  dopo errori o nesting, evitando contaminazioni dei test di steering.
- `tests/test_interactive_api.py` — integrazione HTTP opt-in con modello reale:
  creazione, modifica multilinea, restore dopo riuso dello slot, SSE,
  call ID, ricevute, rifiuto/replay e annullamento; vecchio code rifiutato,
  ultima chiusura, CDATA e input vision, senza prefill per edit.
- `tests/run_server_qualification.py`, `tests/test_server_qualification.py` —
  esecuzione delle suite Python e report JSON con errori, sottocasi e skip;
  impedisce di perdere le anomalie live nel riepilogo di qualifica.
- `tests/test_exmekh_client_live.py` — accettazione opt-in con `uv run exmekh-lib`: esegue
  il prompt originale due volte sullo stesso server riavviando solo il client;
  verifica tool, skill e ritorno alla profondità zero, poi interrompe solo il client ExMekh-Lib
  e ripete sul server acceso; attende la risposta server dopo il return.
  `DS4_EXMEKH_LIB_MIN_ROOT_RETURNS=2` estende la prova a due cicli per client.
  `DS4_EXMEKH_LIB_REQUIRE_COMPLETION=1` abilita anche
  il gate separato che richiede una risposta finale spontanea.
- `tests/test_qwen35_mrope.c` — oracle float64 per mRoPE [11,11,10],
  identità del percorso testo e KV MTP; target CUDA `test-qwen35-mrope`.
- `tests/test_server_agentic_payload.c` — checkpoint completi skill/subagent
  per Qwen27 e Flash Next: logits/token bit-exact, sessione ricreata, file
  corrotti e nesting; opzionalmente immagini e contesto testuale popolato.
- `tests/exmekh_vision_live.py` — gate opt-in attraverso il client ExMekh-Lib installato
  per immagini e continuazioni sulle API Chat Completions e Responses.
- `tests/test_qwen35_vision_api.py` — gate live opt-in CUDA per immagini singole
  e multiple, thinking, JSON, tool, riuso KV, SSE e API Anthropic.
- `tests/test_qwen_mtp_callback.c` — regressione CUDA/MTP oltre 2K token:
  il callback può declinare una riga draft o bonus del verifier V(2); verifica
  ripristino bit-exact dell'intero payload e del generatore casuale, senza errore
  CUDA. Eseguire con target GGUF e sidecar MTP come argomenti.
- `tests/test_qwen_snapshot_drafts.c` — gate CUDA dei draft esterni 32/64:
  logits full-vocabulary e cache confrontati con decode seriale, mismatch,
  cancellazione/callback invalidi e RNG, fine contesto e rewind/sync,
  benchmark con cinque campioni.
  Il sidecar MTP è opzionale; il motore possiede rollback e catch-up.
- `docs/architecture/snapshot-draft-clean-review-20261012.md` — revisione dei
  confini documento/protocollo/scheduler/motore/client, regressioni dei restart,
  escaping comune, flag MTP e metriche distinte per origine dei draft.
- `tests/test_qwen_memory_lifetime.c` — audit live CUDA/Linux: RAM anonima,
  RSS e VRAM globale su 32 finestre OLD e quattro sessioni; fault injection
  delle allocazioni, rilascio dopo declino e confronto bit-exact del fallback.
- `tests/test_server_memory_lifetime.c` — regressione senza modello sul
  ridimensionamento delle copie host dei checkpoint e sul framing binario.
- `docs/architecture/inference-memory-audit-20261013.md` — durata delle
  allocazioni durante inferenza, cache riutilizzabili, correzioni RAM/VRAM e
  limiti della verifica; evidenze JSON in `docs/performance/`.
- `tests/test_server_resident_frontier.c` — regressione model-backed Qwen27/Flash della cache
  di sistema residente e dei checkpoint skill: conta i token del prefill cold
  per respingere rewind o doppie elaborazioni; dopo reset o riscrittura della
  history, verifica logits bit-exact su tutto il vocabolario, rifiuto atomico di
  file corrotti e rollback compatto per singola risposta. Sistema e skill
  conservano anche le righe KV full-attention, rispettivamente in RAM e su SSD.
  I test server coprono inoltre il return da un batch di skill sorelle, senza
  eliminare i checkpoint delle chiamate ancora pendenti, e verificano che lo
  scope corrente raggiunga la coda tool delle continuazioni live e dei return.
- `tests/test_agentic_checkpoint.c` — gate model-backed CUDA Q4_K_S per
  checkpoint/return: casi lunghi 10k, confronto full-vocabulary bit-exact,
  isolamento sessioni, cancellazione, nesting, context boundary, rollback MTP
  e ripristino del payload generale fra oggetti sessione distinti.
- `tests/run_agentic_checkpoint.sh` — runner riproducibile del gate precedente
  nelle varianti target-only e target con sidecar MTP, con report JSON e log;
  `DS4_AGENTIC_CHECKPOINT_BIN` permette di provare una build isolata senza
  sovrascrivere l'eseguibile locale preesistente.
- `tests/ds4_test.c --mtp-verify-depth` — verifica model-backed di accettazione,
  rollback e identità autoregressiva MTP; centralizza il contesto predefinito
  Qwen a 22.593 token, ancora sovrascrivibile con `DS4_TEST_MTP_CTX`.
- `tests/test_qwen38_compatibility.py` — artefatti e caricamento Qwen3.8, rifiuto
  di target/sidecar Qwen3.6 anche rinominati, downloader e help dei binari.
- `tests/test_qwen36_fixtures.py`, `tests/test_qwen36_equivalence.py`,
  `tests/test_qwen36_numerics.py` — schema, staging, checksum, metriche/gate e
  classificazione sintetica dell'inviluppo numerico Qwen3.6.
- `tests/qwen_numerics_probe.c` — oracle CPU indipendente per GDN, full
  attention Qwen e matvec Q4_K/Q5_K/Q6_K contro i kernel CUDA DS4, incluse le
  politiche Q8_1 e Q8_1-R8; eseguibile con `make qwen-numerics CUDA_ARCH=sm_86`.
- `tests/qwen_mtp_catchup_probe.c` — regressione CUDA con modello reale per
  hidden e coppia token/hidden della cache MTP, senza tracing né output head
  sui token intermedi; verifica anche il passaggio dal prefill a chunk ai
  token successivi, output separati e indipendenza dal contesto legacy CUDA.
  Accetta target GGUF e sidecar MTP come argomenti.
- `tests/qwen_iq4_dequant_probe.cu` — parità bit-exact delle due varianti
  cooperative IQ4_XS rispetto al decoder scalare F32/F16, su scale half
  finite, nibble estremi e code parziali dei blocchi CUDA.
- `cuda/mmq/test/test_mmq_parity.cu` — parita' sintetica CPU/CUDA delle primitive
  dense, MoE, pair e vector MMQ; eseguibile con
  `make test-mmq-parity CUDA_ARCH=sm_86`.
- `tests/test_sampling.c` — comportamento deterministico e filtri del sampling.
- `tests/test_gpu_args.c`, `tests/test_gpu_args_cli.sh` — rifiuto anticipato delle
  modalità di constraint sconosciute, parsing e propagazione
  della configurazione GPU.
- `tests/test_layer_pack.c` — invarianti del placement contiguo dei layer.
- `tests/test_gpu_model_cache.c`, `tests/test_gpu_lookup_cache_strict.c` —
  lifetime e correttezza delle cache di pesi/tensor GPU.
- `tests/test_engine_mgpu_placement.c`,
  `tests/test_engine_mgpu_runtime.c`,
  `tests/test_engine_mgpu_refusal.c` — placement, equivalenza e rifiuto sicuro
  delle configurazioni multi-GPU.
- `tests/dspark_acceptance_fixture.sh` — correttezza e acceptance del percorso
  speculativo DSpark.

### Verifica del percorso Qwen task 6

I gate locali minimi sono `make cpu`, `make cuda CUDA_ARCH=sm_86`,
`make ds4_test` e `./ds4_test --qwen35-layer-pattern`. Il confronto numerico
sul GGUF reale parte da `generate_qwen36_score.py` con lo scorer
`gguf-tools/quality-testing/score_official`, quindi usa
`compare_qwen36_equivalence.py` prima con `--suite short` e poi senza filtro.
`verify_qwen36_run.py` controlla separatamente inventario, 32 passi greedy,
32 posizioni teacher-forced e determinismo dei nove oracle llama.cpp. Un
comando non eseguibile per toolchain, backend, memoria o modello mancante va
registrato `NOT VERIFIED`; i test distribuiti richiedono autorizzazione
separata.

## File intenzionalmente non elencati uno per uno

Non fanno parte della mappa principale gli output di build, i dati voluminosi
delle fixture, i singoli prompt/continuation, i CSV e grafici di benchmark e le
risorse puramente grafiche. Vanno consultati attraverso il modulo o README che
li possiede quando il capitolo corrispondente lo richiede.
