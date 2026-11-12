# Prefill di Qwen27 in Core gestito · 12 novembre 2026

## Diagnosi prima della correzione

Il log dell'11 novembre mostra `RESPPROTO TOOLS`, un prompt da 4.456 token e incrementi da 64 token con media 29–30 token/s. Il profilo locale salvato è Qwen3.8-27B UD-Q4_K_S in VRAM, contesto 22.593, visione e immagini. Il profilo manuale CUDA lascia al motore la scelta del chunk. Il supervisore gestito, introdotto da `b13850b05308f5548efbb9d0824c80698e215597`, impone invece `--prefill-chunk 64` anche a CUDA residente. In `ds4.c`, `ds4_effective_prefill_chunk` rispetta l'override e il ramo `DS4_MODEL_FAMILY_QWEN35` di `ds4_session_create` usa 2.048 in assenza dell'override, al massimo quanto il contesto. Il valore di help generico per gli altri modelli non descrive questo ramo.

| Fonte esaminata | Pertinenza |
| --- | --- |
| Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`, `src/transformers/models/qwen3_5/modeling_qwen3_5.py`, SHA-256 locale `63b50baa3ad02a94ea04ebf5e24a9a2da09cc879f165123a82943dff1a2dc825` | `Qwen3_5GatedDeltaNet` passa lo stato ricorrente alle operazioni chunk/recurrent, calcola il gate con conversione FP32 e mantiene maschere separate per attenzione causale e ricorrente. La dimensione della partizione non autorizza modifiche alle precisioni, alle maschere o all'ordine delle dipendenze. |
| llama.cpp `1a064ab0921238c1daa397d6f4a900ef33884de2`, `src/models/qwen35.cpp`, `src/llama-context.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/mmq.cu`, `tests/test-backend-ops.cpp` | Copre il 27B denso da 64 layer e alterna GDN e attenzione. `n_ubatch` limita la partizione della valutazione, mentre CUDA seleziona MMQ o altri prodotti secondo tipo, geometria e capability. I test coprono GDN F32 e quantizzazioni come Q4_K. I default e l'aritmetica ggml non sono un sostituto delle soglie DS4. |
| Strata `6674a0065fb96bacde33e3eb10f91a1df86f95f2`, `src/prefill/prefill.cpp`, `src/prefill/gemm.cu`, `src/prefill/moe_mmq.cu`, `tests/cuda/prefill_mmq_kquant_test.cpp` | Il wrapper GEMM e il prefill ammortizzano preparazione dei pesi su più righe con condizioni di forma, backend e formato. I test MMQ citati coprono gli esperti Q4_K/Q5_K/Q6_K, non qualificano il 27B denso locale. Le policy per esperti, streaming e HIP non sono trasferibili a questa regressione. |
| ExMekh `bb30e2a7f4c45398380aa0a2c396f89ef5d64c5f`, `tools/managed_server.py`, `start-ds4-server.sh`, `ds4.c`, `ds4_server.c` | Il limite è imposto dal launcher. La callback del server mostra le partizioni e la rotta di gestione legge lo snapshot di Core. Nessuna operazione Qwen Image viene eseguita durante questo benchmark testuale. |

Precedenti pertinenti: [ledger Qwen](qwen36-performance-ledger.md) per il default residente 2.048 e la pressione a 4.096, [qualifica Qwen3.8](qwen38-qualification-2026-09-17.md) per i criteri numerici e [suite nativa](native-suite-agent-live-20261022.md) per le opzioni CUDA, visione e MTP. [Lo studio Flash](qwen-flash-agent-launcher-20261008.md) riproduce un sintomo simile ma usa pesi, cache e geometrie MoE differenti.

### Prova e scelta

Il controllo usa il vero supervisore e il binario `ds4-server-suite` SHA-256 `47638a71d15259f23a6b786504e661bd2d8f99a25967f7f4792e51fbba0e4a46`, una RTX 3090, il profilo salvato e uno stato KV temporaneo separato per avvio. Una richiesta sintetica Responses in streaming con strumenti usa gli stessi byte in tutti i bracci. Il polling di `/v1/manage/status` ogni due secondi è presente nel confronto. Il baseline ha riprodotto incrementi di 64 a circa 30–35 token/s nei primi chunk. Gli artefatti di misura locali sono sotto `performance-results/managed-qwen27-prefill-20261112/`.

Il candidato rimuove l'override soltanto per Qwen27 residente CUDA. Il motore torna a usare il default nativo già documentato e mantiene la scelta dei kernel. Si escludono un nuovo valore hardcoded di 2.048 nel supervisore, 4.096 senza qualifica di memoria e un aumento globale che coinvolga Metal/SSD. MTP viene misurato a parte con lo stesso input perché cambia il percorso di decode e i requisiti del modello. Il controllo indipendente confronta i logits completi del default nativo con il riferimento esplicito a 2.048, mantenendo il criterio bit-exact. Il confronto con 64 è un'ablazione di prestazioni e non sostituisce quel gate.

## Misure HTTP

Una richiesta fredda per braccio, stesso binario e SHA-256 della richiesta `ad0d7d3e8c9f12889f03f28f9b972f42a4420b51510402e7bbbfc95849beac94`. Il server ha contato 4.606 token di input, zero token in cache e 32 di output. Tutti e tre i bracci hanno emesso lo stesso testo greedy. Le misure sono riproduzioni mirate su una singola macchina, senza intervallo di confidenza.

| Configurazione | Prefill | Decode | Primo delta | Risposta completa |
| --- | ---: | ---: | ---: | ---: |
| Gestito con chunk 64 | 32,43 token/s · 142,043 s | 29,13 token/s | 142,096 s | 143,165 s |
| Default nativo 2.048 | 319,21 token/s · 14,429 s | 30,41 token/s | 14,478 s | 15,508 s |
| Default nativo 2.048 con MTP | 295,01 token/s · 15,613 s | 37,31 token/s | 15,671 s | 16,496 s |

- L'API di stato è rimasta attiva in tutti i bracci. Nei 72 campioni del riferimento, il round trip massimo è stato 1,74 ms. Il prefill del candidato esegue due chunk da 2.048 a 778 e 812 token/s. L'allocazione di contesto registrata cresce da 2.999,37 a 3.705,35 MiB a parità di contesto. Il massimo del catalogo CUDA rimane 22.593, esercitato nella prova HTTP.
- Il candidato impiega 7,214 s nei primi 246 token e altri 135,8 ms per salvare il checkpoint. Il riesame sotto distingue il checkpoint disco della richiesta ordinaria dal checkpoint residente dei percorsi agentic. I primi 246 token cadono nel percorso token per token. La coda da 264 righe usa il batch con attenzione legacy, sotto la soglia GEMM di 512 righe.
- MTP con il sidecar locale `mtp-Qwen3.8-27B-Q4_0.gguf` è più veloce in decode in questo campione, con una risposta completa più lenta. Lasciare MTP fuori dal default gestito conserva il profilo target-only e i requisiti di memoria attuali. Questo campione non qualifica il cambio di default su risposte lunghe.
- Con Qwen Image nella suite, il candidato trascorre circa 76 s tra la pubblicazione del modello caricato nel log nativo e l'apertura del servizio. `ds4_bundle.c:ds4_bundle_open` verifica dimensione e SHA-256 di ogni file del bundle anche per la suite manuale. È un costo di avvio separato dal prefill; il dato di 76 s comprende l'intera fase tra le due righe di log.

## Verifiche indipendenti

- `ds4-bench`, ricompilato con `CUDA_ARCH=sm_86`, ha confrontato il default con il riferimento esplicito a 2.048 sullo stesso input, a contesto 22.593. Il comparatore rigoroso ha verificato 248.320 logits FP32 finiti e bit-identici a ciascuna frontiera, 512 e 4.606. I dump dopo 32 passi di decode teacher-forced sono identici byte per byte e contengono tutti i logits finiti. Il primo controllo di copertura del comparatore ha rifiutato le directory contenenti anche i file `.decode.json`. Il risultato rigoroso è stato ottenuto separando i dump, senza modificare soglie o eseguibili.
- `test_server_resident_frontier` sul modello reale ha valutato 180/180 token a freddo senza ripetizioni. Il riuso del prefisso, il ripristino dei checkpoint, il rifiuto di un checkpoint corrotto e il rollback compatto sono risultati bit-exact.
- 32 test di Core/launcher e 19 test mirati di Agent, inclusi avvio e riavvio gestiti con un worker di test che verifica l'assenza dell'override sul profilo CUDA residente. Ruff e `git diff --check` passano. I risultati sono locali e il lavoro resta senza commit.

## Riesame dei 300 e 600 token/s

### Profilo e fonti

Checkout nativo `bb30e2a7f4c45398380aa0a2c396f89ef5d64c5f`, binario CUDA `1b1c6f271d53253f096783a9aa3ee80a319601b5739c22b207651d7b99cf6fcd`, RTX 3090, contesto 22.593, Qwen27 residente, target-only, encoder vision configurato e richiesta solo testuale con strumenti. Il payload freddo è lo stesso del confronto precedente. Un secondo payload cambia solo il testo utente e conserva i 246 token di sistema. Il test usa una directory KV temporanea per processo.

- `ds4_server.c:kv_cache_cold_store_target_request` e `server_session_sync` portano una richiesta Responses ordinaria con tool al confine di sistema prima di salvarlo su disco. Per una richiesta agentic esiste anche `ensure_resident_system_frontier`. Il profilo attuale mostra esplicitamente il salvataggio su disco, non una cattura residente.
- `ds4.c:ds4_session_sync` conserva la valutazione token per token sotto i 512 token totali nei percorsi residenti. Nel suffisso di un prompt lungo usa invece le righe se il blocco ne contiene almeno 128. `ds4_cuda.cu:ds4_gpu_qwen35_full_attention_mrope_rows_tensor` passa al percorso GEMM da 512 righe. La coda da 264 righe ha usato il batch con attenzione legacy.
- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`, `src/transformers/models/qwen3_5/modeling_qwen3_5.py:Qwen3_5GatedDeltaNet.forward`, distingue aggiornamento ricorrente e chunk con stato iniziale e finale. Il gate usa FP32 e l'ordine delle dipendenze deve rimanere quello causale. Questo non implica parità bit-exact fra i percorsi DS4.
- llama.cpp `1a064ab0921238c1daa397d6f4a900ef33884de2`, `src/models/qwen35.cpp`, `src/llama-context.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu` e `tests/test-backend-ops.cpp` separano il grafo GDN/full attention dal microbatch e ammettono MMQ in base a forma, tipo e capability. Il test ggml di una singola operazione non qualifica lo stato completo del checkpoint DS4.
- Strata `6674a0065fb96bacde33e3eb10f91a1df86f95f2`, `src/prefill/prefill.cpp`, `src/prefill/gemm.cu`, `src/prefill/moe_mmq.cu` e `tests/cuda/prefill_mmq_kquant_test.cpp` confermano che cambiare forma e occupazione cambia il percorso veloce. I test MoE/Q4_K non stabiliscono equivalenza per il 27B denso al confine di cache. Il [ledger Qwen](qwen36-performance-ledger.md) documenta il drift al cambio di partizione e i gate storici, compreso il rifiuto di 4.096 come default.

### Misure dirette e profilo

| Stato della cache | Token nuovi | Prefill | Decode | Primo delta | Latenza |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fredda, confine su disco | 4.606 | 311,24 token/s · 14,799 s | 28,27 token/s | 14,851 s | 15,956 s |
| Prefisso di 246 token riusato | 4.360 | 601,16 token/s · 7,253 s | 28,75 token/s | 7,376 s | 8,464 s |

I 246 token freddi hanno preso 7,353 s e il salvataggio su disco 123,8 ms. I due batch da 2.048 hanno prodotto circa 767–791 token/s. Il run diagnostico con sincronizzazioni per layer non registra un batch per il prefisso da 246. Nella coda da 264 attribuisce 1.672 dei 2.189 ms profilati alla full attention. È una misura di fase con sincronizzazioni aggiuntive e resta fuori dal confronto prestazionale HTTP.

La decisione iniziale conserva il prefill qualificato per ciascun backend e il criterio numerico del prefisso. Prima di proporre una soglia diversa per la cache, il controllo diagnostico usa l'opzione nativa `--kv-cache-min-tokens 512` per mostrare il costo della rinuncia al checkpoint corto sia a freddo sia sulla richiesta seguente. Questo controllo non cambia i default del prodotto.

### Confronto con Core e controllo sul checkpoint

| Avvio | Cache | Token riusati | Prefill token/s | Decode token/s | Primo delta | Latenza |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Core, default CUDA | Fredda | 0 | 308,50 | 29,37 | 14,980 s | 16,046 s |
| Core, default CUDA | Seconda richiesta | 246 | 603,10 | 28,38 | 7,352 s | 8,456 s |
| Nativo, minimo cache 512 | Fredda | 0 | 515,53 | 26,95 | 8,995 s | 10,147 s |
| Nativo, minimo cache 512 | Seconda richiesta | 0 | 500,01 | 28,34 | 9,261 s | 10,366 s |

Binario e richieste coincidono fra il controllo nativo, Core e il test con minimo 512. Il risultato greedy di 32 token coincide in tutti i bracci. È una riproduzione direzionale, un processo per braccio, senza intervallo di confidenza. Core e il controllo diretto mostrano lo stesso ordine di grandezza sia a freddo sia con il prefisso riusato. Il minimo cache a 512 riduce il primo TTFT, ma peggiora quello della seconda richiesta con sistema invariato da 7,376 a 9,261 s. È scartato come default. Abbassare il cutoff delle 512 righe o ripartire diversamente i batch richiederebbe un nuovo gate numerico sullo stato ricorrente e sui logits. La coincidenza di una risposta greedy non basta.

### Verifica Metal e protezione dei launcher

Per Flash Next Q2 SSD e Qwen27 SSD, in quest'ordine, il confronto statico con `config/flash-metal-streamed.env.example`, `config/qwen27-metal-streamed.env.example` e i ledger M5 Pro conferma 64 righe e avvio SSD cold. Per Qwen27 residente Metal, il launcher manuale e quello gestito usano 64 righe. Il test di regressione vincola i quattro profili gestiti al prefill effettivo e alla geometria usata dallo stimatore. Il controllo su WSL non è una nuova misura di prestazioni Apple Silicon.

Gate indipendenti rieseguiti su CUDA. Default nativo e `--prefill-chunk 2048` coincidono bit per bit sui 248.320 logits alle frontiere 512 e 4.606, e sui dump dopo 32 passi teacher-forced. Il test di ripristino ha confermato 180/180 token valutati a freddo, senza replay, e parità bit-exact per ripristino e rollback. Passano 39 test Core/launcher, quattro controlli dei testi e due test Playwright del ciclo di avvio. I valori misurati e gli artefatti del gate sono sotto `performance-results/managed-qwen27-prefill-20261112/`.
