# Piano di implementazione: suite multimodale nativa DS4

Data: 20 ottobre 2026. Stato: `IN_PROGRESS`. T00, T01, T05 e T06 sono `DONE` nei rispettivi gate locali; T02 è implementato e verificato su CUDA single GPU, ma resta aperto per la qualifica degli altri backend. T03/T04 sono `IN_PROGRESS`, con accounting e lifecycle Qwen27 CUDA verificati localmente; T07/T08 sono `DONE` nella qualifica locale CPU/CUDA descritta nel [report T07–T08](../performance/native-suite-t07-t08-20261020/README.md). T09/T10 sono `DONE` nei gate di componente CUDA sm_86 del [report T09–T10](../performance/native-suite-t09-t10-20261021/README.md). T11 è implementato e `IN_PROGRESS` per i gate image; T12 è `DONE` nei gate testuali CUDA sm_86 del [report T11–T12](../performance/native-suite-t11-t12-20261021/README.md); T13/T14 sono `DONE` nei gate locali del [report T13–T14](../performance/native-suite-t13-t14-20261021/README.md); T15/T16 sono `DONE` nei gate C/CUDA locali del [report T15–T16](../performance/native-suite-t15-t16-20261022/README.md), con rettifica esplicita del riferimento di accumulo T00. T17/T18 sono `DONE` nei gate di servizio/client CUDA locali del [report T17–T18](../performance/native-suite-t17-t18-20261022/README.md); T19–T22 sono `IN_PROGRESS`, con streaming bounded e strumenti di qualifica/consegna nel [report T19–T22](../performance/native-suite-t19-t22-20261022/README.md). Evidenze e limiti nei report [T00–T02](../performance/native-suite-t00-20261020/README.md), [T03–T04](../performance/native-suite-t03-t04-20261020/README.md) e [T05–T06](../performance/native-suite-t05-t06-20261020/README.md). T05 qualifica il core generico e l'adapter Qwen27 CUDA; T06 qualifica inventario e caricamento dei bundle BF16, senza promuovere profili quantizzati.

La [analisi architetturale](../architecture/native-multimodal-suite-analysis-20261020.md) contiene requisiti, audit del fork, fonti esterne e motivazioni. Leggerla prima dei task, insieme a `AGENT.md`, `PROJECT_INDEX.md` e al [contratto prestazionale](../performance/agent_performance_contract.md).

## 1. Risultato atteso e invarianti vincolanti

Una sola istanza DS4 deve servire LLM, Qwen-Image-2.1-Turbo ed EmbeddingGemma 2, decidendo dinamicamente quali componenti tenere in GPU, spostare o ricaricare, senza perdita di conversazione.

- `I01` Inferenza di produzione nativa in DS4; Python, Transformers, Diffusers, llama.cpp e sd.cpp soltanto per conversione, confronto e sviluppo. Nessun server esterno usato per soddisfare una richiesta.
- `I02` Prima scelta: esecuzione senza eviction quando il budget lo permette. Sono vietati checkpoint e scaricamenti incondizionati.
- `I03` History applicativa conservata. Sospendere significa salvare lo stato necessario e rilasciare allocazioni, non cancellare token o messaggi.
- `I04` Pesi, stato delle sessioni, scratch, cache derivate e temporanei hanno proprietari e durate separati. Nessun cleanup globale durante lo scarico di un componente.
- `I05` Il successo finale arriva dopo il ripristino delle sessioni richieste; valgono anche errori, cancellazione, timeout e client disconnesso.
- `I06` Le scelte di residenza sono generiche. Le forme, i kernel e i payload specifici sono responsabilità dell'adapter del modello.
- `I07` Embedding completi per testo, immagini, video, audio e input interleaved. Nessuna release dichiarata completa con solo testo o con caption/ASR al posto degli encoder.
- `I08` Nessuna riduzione silenziosa di contesto, risoluzione, durata, dimensione vettore o precisione. I profili quantizzati sono artefatti espliciti e qualificati.
- `I09` Semantica in C e API pubbliche ristrette; nessuna nuova architettura C++. Eventuali idiomi CUDA già presenti restano confinati al backend.
- `I10` Nessuna regressione dei percorsi esistenti; `NOT_VERIFIED` non equivale a `PASS`. Backend nuovi e compatibilità legacy hanno matrici distinte.
- `I11` Il lock d'istanza rimane. Co-residenza non implica compute concorrente: primo rilascio con inferenza GPU serializzata e copie asincrone controllate.
- `I12` ExMekh resta unico frontend agentico. DS4 non esegue un tool solo perché riconosce del testo generato: routing e invocazione devono essere espliciti.

## 2. Regole per completare un task

Ogni task parte `PLANNED`, passa a `IN_PROGRESS` e diventa `DONE` solo con i propri gate superati e artefatti verificabili. `BLOCKED`, `FAIL`, `NOT_VERIFIED`, `REJECT` e `NEED_MORE_DATA` vanno conservati separatamente; un'ottimizzazione scartata può chiudere un esperimento, non certificare una funzionalità mancante.

Per ogni task registrare: revisione sorgenti, diff attribuibile, modelli/hash, configurazione, comandi, fixture, output numerici completi pertinenti, memoria e verdetto. Non cambiare golden o tolleranze dopo un fallimento senza nuova diagnosi e nuova baseline dichiarata.

I nomi dei nuovi file sotto indicati sono proposte di responsabilità, non obblighi a creare un file per ogni concetto. Preferire moduli piccoli senza duplicare le primitive già presenti. Qualunque nuova responsabilità deve essere registrata in `PROJECT_INDEX.md`.

## 3. Milestone, ordine e dipendenze

| Milestone | Task | Risultato verificabile |
| --- | --- | --- |
| M0 — Specifiche e riferimenti | T00–T01 | Contratti, artefatti, oracle, soglie e baseline congelati |
| M1 — Runtime e memoria | T02–T05 | Due modelli distinguibili, risorse proprietarie, sospensione e ripresa corrette |
| M2 — Caricamento e primitive | T06–T07 | Formati nativi e kernel comuni verificati su shape reali |
| M3 — Qwen Image nativo | T08–T11 | Generazione/editing e RGBA con schedule Turbo verificata |
| M4 — Embedding multimodali | T12–T16 | Tutte le modalità e combinazioni nello spazio comune |
| M5 — Servizio e client | T17–T18 | API, media e integrazione client usabili senza gestione manuale della memoria |
| M6 — Scheduling efficiente | T19–T20 | Prefetch/streaming misurati e profili numerici qualificati o scartati esplicitamente |
| M7 — Accettazione e consegna | T21–T22 | Stress, recupero, regressioni, packaging e documentazione |

Ordine critico: `T00 → T01 → T02 → T03 → T04 → T05`. Dopo T06/T07 i percorsi image e Gemma sono sviluppabili indipendentemente, ma condividono lo stesso registro di risorse. Non eseguire processi modello voluminosi in parallelo sulla macchina locale. Questa indipendenza descrive le dipendenze tecniche, non autorizza attività o processi concorrenti.

## 4. T00 — Congelare fonti, baseline e criteri numerici

**Dipendenze:** nessuna. **Priorità:** bloccante per qualunque porting. **Stato:** `DONE`.

- [x] `T00.1` Fotografare worktree, modifiche preesistenti, binari e modelli attuali. Non ripulire o riscrivere lavoro dell'utente. HEAD osservato durante la pianificazione: `5e2e99cf3249a5a28efed038bfd4fb2fbda6ce21`.
- [x] `T00.2` Fissare SHA di Transformers, Diffusers, llama.cpp/ggml, Unsloth, sd.cpp, ComfyUI e SGLang per i file effettivamente usati; registrare anche versione vLLM-Omni consultata. Il lock storico della knowledge base non basta per i nuovi modelli.
- [x] `T00.3` Creare manifest con revisione e SHA-256 dei componenti ufficiali Qwen Turbo e Gemma 2, config, tokenizer, processor e profili GGUF candidati. Inventariare ogni tipo tensoriale e dimensione reale, senza scaricare tutto inutilmente in RAM.
- [x] `T00.4` Registrare hardware, sm_86, toolkit/driver, VRAM realmente disponibile, limite WSL, memoria host/pinned disponibile e banda SSD/H2D/D2H sotto carico.
- [x] `T00.5` Congelare baseline Qwen27 testo/vision con MTP acceso/spento e altri LLM CUDA disponibili. Conservare logit dump e stato di sessione alle frontiere canoniche.
- [x] `T00.6` Definire corpus di sviluppo e corpus held-out distinti; congelare input media, rumore latente iniziale, seed, recipe, ranking attesi e condizioni numeriche per ogni componente.
- [x] `T00.7` Definire tolleranze per operazione/dtype e per forward completo, misurando la variabilità dell'oracle. Stabilire i gate percettivi e retrieval prima di valutare quantizzazione o tiling.
- [x] `T00.8` Aprire registro di provenienza/licenze per ogni file portato. Distinguere codice dei kernel, orchestrazione Studio e licenza dei pesi; nessuna importazione automatica dal codice AGPL osservato in Unsloth Studio.

**Gate di completamento:** manifest riproducibili con hash effettivi, oracle eseguibili, budget hardware documentato e soglie concrete già valorizzate. Tutti i riferimenti mobili usati per il porting sono fissati. Un campo critico `TBD` impedisce `DONE`.

**Artefatti:** source lock dedicato, model manifest, experiment baseline, corpus manifest, tolerance manifest e registro provenienza.

## 5. T01 — Congelare i contratti pubblici e interni

**Dipendenze:** T00. **Priorità:** bloccante. **Stato:** `DONE`.

- [x] `T01.1` Definire catalogo configurato, identificativi stabili di modello/profilo, capability, backend e componenti opzionali. Non cambiare il modello predefinito dei launcher esistenti.
- [x] `T01.2` Definire API C minima per descrivere fasi, dipendenze, gruppi di pesi, risorse, picchi e operazioni di prepare/execute/release; niente dettagli tensoriali nel server.
- [x] `T01.3` Definire adapter lifecycle con estimate, load/release, quiesce, checkpoint/restore e validazione; distinguere non supportato, non residente e non pronto.
- [x] `T01.4` Definire `/v1/images/generations`, `/v1/images/edits`, `/v1/embeddings`, media upload/read e discovery, preservando chat/Responses/Anthropic esistenti.
- [x] `T01.5` Specificare l'estensione multimodale embeddings: batch di elementi, contenuti ordinati, riferimenti media, task, dimensioni, normalizzazione e segmentazione esplicita. Congelare una versione del contratto.
- [x] `T01.6` Specificare errori di ammissione, modalità non supportata, inferenza, cancellazione e ripristino. Definire request ID/idempotenza e quando una richiesta è conclusa.
- [x] `T01.7` Stabilire la semantica del video con/senza traccia audio e degli input oltre contesto. Un vettore per elemento composto; nessuna media di segmenti implicita.

**Gate di completamento:** esempi JSON e chiamate client coprono tutte le operazioni; distinguono standard OpenAI da estensione DS4; ogni campo ha semantica, default e validazione. Lo schema rappresenta text-only, media-only e interleaving senza placeholder artificiali lato utente.

**Artefatti:** specifica API, capability matrix e contratti C proposti. Nessun endpoint viene pubblicizzato come già implementato.

## 6. T02 — Separare contesti modello e lifecycle del runtime

**Dipendenze:** T00, T01. **Priorità:** bloccante. **Stato:** `IN_PROGRESS` — implementazione CUDA e gate locali PASS; Metal/ROCm/TP/distribuito `NOT_VERIFIED`.

- [x] `T02.1` Inventariare `g_ds4_shape`, mapping/cache globali CUDA, quality flags, thread pool, stream, handle cuBLAS e cleanup. Assegnare ciascuno a runtime condiviso, modello o operazione.
- [x] `T02.2` Introdurre descrittori immutabili per engine/componente e attivazione del contesto esecutivo sotto lock. Migrare accessi numerici alle shape; eventuali ponti legacy devono essere confinati e verificabili.
- [x] `T02.3` Separare apertura/chiusura del runtime GPU da caricamento/scaricamento dei modelli. `ds4_gpu_cleanup` e rilascio del lock d'istanza avvengono soltanto quando termina il proprietario runtime.
- [x] `T02.4` Rendere mapping, range e cache derivate identificabili per artefatto/componente/formato; invalidare solo il proprietario interessato. Gestire indirizzi GPU che cambiano dopo reload.
- [x] `T02.5` Separare pool di scratch condivisibili da memoria persistente. Impedire alias fra sessioni di modelli differenti e riuso prematuro di buffer.
- [ ] `T02.6` Adeguare i binding backend senza introdurre nuove semantiche nei percorsi Metal/ROCm/TP/SSD esistenti. Il compute concorrente resta fuori dal primo rilascio.

Il bridge legacy consente un solo GGUF LLM per runtime; i componenti nativi usano handle e tensori indipendenti. La verifica locale alterna il LLM con due componenti fixture, prova forward/snapshot/quality e rilascio selettivo. Avvio senza LLM e rebind di engine nello stesso runtime sono implementati; eviction dei soli pesi con engine/sessioni vivi e transazione automatica restano T04/T05. [Inventario e confini](../architecture/native-suite-runtime-lifecycle.md).

**Gate di completamento:** aprire due descrittori, alternarne l'esecuzione seriale e scaricare uno non invalida l'altro. Shape, vocab, cache e quality mode corretti dopo ogni attivazione; baseline dei LLM invariata. Nessun uso globale non classificato resta sul percorso di alternanza.

**Aree:** `ds4.c`, `ds4.h`, `ds4_gpu.h`, runtime CUDA e corrispondenti confini degli altri backend.

## 7. T03 — Registro risorse e accounting su GPU, RAM e SSD

**Dipendenze:** T02. **Priorità:** bloccante. **Stato:** `IN_PROGRESS` — API C e accounting CUDA implementati; inventario automatico RAM legacy e qualifica degli altri backend aperti.

- [x] `T03.1` Registrare byte richiesti/allocati/vivi, proprietario, collocazione, formato, refcount, ultimo evento GPU e possibilità di rilascio per ogni risorsa.
- [ ] `T03.2` Separare storage mmap, RAM residente, RAM anonima e pinned. Rappresentare una copia autorevole senza contare due volte storage condiviso e view.
- [x] `T03.3` Aggiungere prenotazioni per working set, copie in volo, doppio staging e conversioni. Una copia asincrona consuma il budget prima del suo completamento.
- [x] `T03.4` Esporre stime e contatori delle sessioni: capacità allocata, payload salvabile, scratch e cache MTP/ricorrenti. Non usare soltanto il numero di token.
- [x] `T03.5` Definire budget host/device dinamici, margine, limiti pinned e spazio checkpoint; riconciliare contabilità posseduta e memoria globale senza sommarle impropriamente.
- [x] `T03.6` Implementare rilascio selettivo e trimming delle cache ricostruibili tramite il proprietario, senza reset globale del backend.

Registro in `ds4_resources.c/.h`, binding CUDA in `ds4_cuda_accounting.cuh` e query Qwen27 in `ds4_lifecycle.inc`. Le view trattengono lo storage CUDA reale; prenotazioni e cancellazione con eventi hanno test concorrenti host. Storage mmap e pinned sono tracciati, RAM anonima/residente legacy resta esplicitamente ignota (`host_known=false`): T03.2 non è chiuso. La capacità virtuale managed ha un ledger separato dalla VRAM fisica. Driver/cuBLAS/MMQ restano nel campione globale, non vengono inventati come zero. [Contratti, test e limiti](../performance/native-suite-t03-t04-20261020/README.md).

**Gate di completamento:** allocazioni e rilasci tracciati si bilanciano per proprietario; le view non raddoppiano i byte. Reservazioni impediscono oversubscription nei test concorrenti host; cancellare un prefetch restituisce tutti i budget. Le stime espongono chiaramente componenti ignote invece di assumere zero.

**Artefatti:** API di accounting, dump compatto e test di lifetime/alias/overflow dimensionale.

## 8. T04 — Generalizzare sospensione, checkpoint e rilascio pesi

**Dipendenze:** T02, T03. **Priorità:** bloccante. **Stato:** `IN_PROGRESS` — adapter Qwen27 CUDA e gate locali implementati; altre famiglie, MTP embedded e backend restano esclusi dalle nuove API.

- [x] `T04.1` Adattare il meccanismo Qwen27 vision a operazioni lifecycle comuni, preservando identità sessione, callback e modalità MTP.
- [x] `T04.2` Separare rilascio scratch, parcheggio dello stato e rilascio pesi. Supportare anche pesi scaricati con stato sessione mantenuto in GPU, quando richiesto dal piano.
- [x] `T04.3` Inventariare gli adapter delle famiglie LLM correnti, inclusi Qwen27, Flash Next/Mini, DeepSeek e GLM: stato completo, risorse condivise e vincoli TP/SSD. Dichiarare la matrice effettivamente qualificata.
- [ ] `T04.4` Conservare payload KV, stato ricorrente/conv, MTP/DSpark pertinenti, logits/hidden pendenti, identità immagini e mRoPE. Integrare anche RNG/sampler e metadata request/context posseduti dal server.
- [x] `T04.5` Validare fingerprint del modello, ABI, checksum, dimensioni e configurazione prima di mutare una sessione. Mantenere la compatibilità delle cache esistenti oppure introdurre una versione esplicita con rifiuto chiaro.
- [x] `T04.6` Completare il checkpoint prima di distruggere la copia autorevole; conservare il file finché il restore non è validato. Gestire checkpoint multipli e restore parziale senza perdere quelli ancora necessari.
- [ ] `T04.7` Invalidare o ricatturare grafi che contengono indirizzi GPU dei pesi ricaricati. Testare target-only, MTP esterno/embedded e stati multimodali pertinenti.

Le API `trim`, `park/resume`, `weights_release/load` e checkpoint espliciti sono separate. L'envelope lifecycle v1 conserva il fingerprint di directory GGUF/identità dei file aperti, configurazione e checksum; il payload legacy non cambia. Il restore valida un grafo candidato prima del commit, mantiene l'oggetto sessione e conserva i checkpoint dopo gli errori. I gate Qwen27 target-only e MTP esterno/multimodale verificano continuazione bit-exact, più sessioni, corruzione, OOM e budget SSD insufficiente. T04.4 resta aperto per DSpark/altre famiglie e integrazione server transazionale; T04.7 per il MTP embedded e le altre piattaforme. [Inventario ed evidenze](../performance/native-suite-t03-t04-20261020/README.md).

**Gate di completamento:** per ogni famiglia qualificata, continuazione deterministica bit-exact dopo parcheggio sessioni, reload pesi e combinazione delle due operazioni, sul medesimo percorso numerico. Corruzione o spazio SSD insufficiente non distruggono lo stato vivo. Checkpoint conservati dopo restore fallito; nessun callback punta a oggetti liberati.

**Aree:** payload e sessioni in `ds4.c`, wrapper del server, test di checkpoint già presenti.

## 9. T05 — Ammissione, pianificazione e transazione comune

**Dipendenze:** T03, T04. **Priorità:** bloccante. **Stato:** `DONE` per il core comune e l'adapter Qwen27 CUDA single GPU; le qualifiche residue T03/T04 restano aperte.

- [x] `T05.1` Costruire il piano dell'intera operazione, incluse visite ripetute ai componenti e ritorno al LLM, prima di eseguire eviction.
- [x] `T05.2` Implementare scelta senza eviction, cache ricostruibili, sospensione selettiva sessioni, rilascio selettivo pesi e piani streamed. Nessun nome di modello nei criteri generici.
- [x] `T05.3` Confrontare piani ammissibili usando byte, costi di checkpoint/restore e trasferimento; considerare pesi-only con stato residente. Applicare isteresi per evitare oscillazioni.
- [x] `T05.4` Eseguire validate/reserve/quiesce/checkpoint/execute/restore/commit con journal dell'operazione e cleanup idempotente. Il commit del risultato finale è successivo al restore.
- [x] `T05.5` Implementare cancellazione cooperativa, drain degli eventi e gestione della disconnessione senza abbandonare sessioni parcheggiate.
- [x] `T05.6` Rivalutare un'allocazione fallita solo a frontiere che consentono rollback esatto. Non ripetere automaticamente richieste con stato parzialmente avanzato.
- [x] `T05.7` Esporre stato di recupero, conservazione checkpoint e indisponibilità selettiva quando il restore è impedito; distinguere errori device non recuperabili.

**Gate di completamento:** matrice di budget simulati seleziona tutti i piani attesi. Nel caso capiente, zero checkpoint/eviction. Ogni fault iniettato prima/durante/dopo compute termina in stato pronto oppure recuperabile dichiarato, mai in falso successo. Nessuna modifica implicita dei parametri della richiesta.

**Artefatti:** planner C, executor transazionale e test con adapter fittizi più almeno un LLM reale.

**Verifica T05:** `ds4_executor.c/.h`, transazioni runtime e adapter pubblico
`ds4_engine_reclaim`. Matrice host di budget e fault, cancellazione, recupero e
device perso durante compute/restore; Qwen27 CUDA con zero eviction, sessione
parcheggiata e soli pesi scaricati, continuazione bit-exact in tutti e tre i casi.
Il disconnect è un segnale al callback di cancellazione; il collegamento al
protocollo HTTP resta T17. Dettagli nel [report T05–T06](../performance/native-suite-t05-t06-20261020/README.md).

## 10. T06 — Manifest, conversione e loader dei nuovi componenti

**Dipendenze:** T00, T01, T02. **Priorità:** alta. **Stato:** `DONE` per inventario, conversione e loader dei bundle BF16; nessuna qualifica numerica dei nuovi modelli implicita.

- [x] `T06.1` Definire bundle che collega encoder, vision/audio, DiT, VAE, tokenizer, processor e scheduler alle revisioni esatte; rifiutare combinazioni incompatibili.
- [x] `T06.2` Scegliere il formato nativo per ogni componente. Preferire loader GGUF esistente e conversione offline deterministica; valutare safetensors mmap se evita perdita o complessità, senza obbligare un unico contenitore a tutti.
- [x] `T06.3` Implementare mapping dei tensor, shape, stride, dtype, endian/allineamento, scale norm e parametri per layer. Inventario esaustivo: tensor sconosciuti non possono essere ignorati senza una regola motivata.
- [x] `T06.4` Per i GGUF Unsloth leggere il tipo effettivo per tensore e preservare il layout misto. Non assumere che Q4_K_M, Q8_0, INT8 ConvRot, AWQ e BnB condividano packing o kernel.
- [x] `T06.5` Rendere scaricabile ogni componente/gruppo mantenendo mmap e metadata host validi. Evitare espansione completa dei pesi quantizzati durante l'apertura.
- [x] `T06.6` Estendere downloader e discovery con target espliciti, checksum e resume. Scaricare un bundle image/embedding non cambia il modello conversazionale selezionato.

**Gate di completamento:** dump di tutti i tensor e metadata coincide col manifest; conversione ripetuta produce gli stessi byte dove previsto. Errori di componente/shape/precisione falliscono prima dell'inferenza. I picchi di caricamento rispettano i budget e non nascondono una seconda copia integrale.

**Aree:** `download_model.sh`, `gguf-tools/`, loader/catalogo e nuovi manifest.

**Verifica T06:** 1.285 tensori Qwen Image e 1.376 Gemma, inclusi 480 scalari,
confrontati dal loader C con il manifest compilato e l'inventario T00. Hash di
ogni artefatto verificato, apertura mmap bounded, conversione BF16/FP32 ripetuta
identica e GGUF misti verificati su fixture dei tipi supportati. Il profilo
Unsloth Q4_K_M è fissato come candidato `NOT_VERIFIED`: pesi assenti e qualifica
numerica riservata a T20. Binding delle architetture e inferenza restano T07–T16.
Evidenze e comandi nel [report T05–T06](../performance/native-suite-t05-t06-20261020/README.md).

## 11. T07 — Primitive numeriche e kernel CUDA comuni

**Dipendenze:** T00, T02, T06. **Priorità:** alta. **Stato:** `DONE`.

- [x] `T07.1` Estrarre le shape reali dei due modelli e mappare ogni operazione a primitive DS4 già disponibili, cuBLAS o kernel mancante. Separare workload encoder/DiT dal decode GEMV dei LLM.
- [x] `T07.2` Fornire GEMM BF16/FP32 con accumulazioni dichiarate, dequantizzazione bounded e dispatch per shape. TF32/fast math hanno gate separati e non possono cambiare silenziosamente il profilo numerico.
- [x] `T07.3` Implementare/testare RMSNorm a scala diretta e zero-centered separatamente, norm senza scala, GELU tanh/esatta, SwiGLU, residui/gating e RoPE con le rispettive convenzioni.
- [x] `T07.4` Implementare attention tiled/segmentata con softmax online, GQA/MQA, padding e scale esplicite: causale Qwen3-VL, block-causal DiT, bidirezionale locale/full Gemma, forme audio/vision richieste.
- [x] `T07.5` Coprire head dimension 128/256/512 e confini di finestra, batch e lunghezze non multiple del tile; evitare la matrice N×N nel percorso di produzione.
- [x] `T07.6` Fornire convoluzioni e operatori audio/VAE mancanti solo dopo inventario, con reference piccoli e oracle indipendenti. Valutare prima librerie native già presenti e costo di nuove dipendenze C.
- [x] `T07.7` Per ciascun nuovo kernel registrare fonte Unsloth/llama/altro, differenze di dtype/layout/hardware, ipotesi, output completo e benchmark reale. Nessuna trasposizione meccanica di Triton o C++ nel core.

**Gate di completamento:** tutte le primitive necessarie hanno reference indipendente, zero NaN/Inf e soglie T00 rispettate su shape reali e casi limite. Gemma non attraversa fallback FP16. Build sm_86 e reference CPU verdi; nessuna regressione sulle primitive LLM condivise. Le performance vengono misurate dopo la correttezza.

Qualifica: reference CPU e CUDA RTX 3090 sm_86, profilo numerico esplicito;
head geometry reali Qwen/Gemma e confini 257/513 verificati. Librerie suite
separate dal link legacy; misure instabili non promuovono speedup.
Evidenze, dipendenze native e limiti nel [report T07–T08](../performance/native-suite-t07-t08-20261020/README.md).

**Aree:** `ds4_gpu.h`, backend CUDA e moduli numerici specifici; nessun dettaglio kernel nel protocollo HTTP.

## 12. T08 — Encoder Qwen3-VL per Qwen Image

**Dipendenze:** T06, T07. **Priorità:** alta. **Stato:** `DONE`.

- [x] `T08.1` Implementare tokenizer e template esatti del bundle image, padding e token di sistema esclusi dall'output; conservare prompt originali e parametri effettivi.
- [x] `T08.2` Portare il backbone Qwen3-VL previsto dal checkpoint, senza riusare per equivalenza il Qwen27 conversazionale. Includere norm, attention, mRoPE e hidden state richiesti.
- [x] `T08.3` Implementare vision/deepstack, preprocessing e ordine delle reference. Per RGBA verificare compositing sul bianco solo nel ramo vision.
- [x] `T08.4` Esportare gli hidden state pre-normalizzazione finale come nell'oracle image; test esplicito che distingua output pre/post norm.
- [x] `T08.5` Descrivere lifetime di layer/gruppi, output conditioning e cache temporanee. Rilasciare un gruppo soltanto dopo il suo ultimo uso effettivo.

**Gate di completamento:** token, posizioni, maschere, shape e hidden state coincidono con le fixture T00 entro le tolleranze fissate; testo, una reference, multi-reference e RGBA verificati. Il riordino delle reference produce il conditioning corrispondente. Nessuna generazione autoregressiva superflua dell'encoder.

Qualifica: richieste seriali native, otto casi completi e padding a sinistra;
conditioning e confini confrontati con oracle senza allargare T00.
Batching non qualificato, DiT/VAE e HTTP restano nei task successivi.
Evidenze nel [report T07–T08](../performance/native-suite-t07-t08-20261020/README.md).

**Artefatti:** adapter encoder e dump intermedi, inclusa fixture sensibile alla norm finale.

## 13. T09 — DiT single-stream, scheduler Turbo e prefix cache

**Dipendenze:** T06, T07. **Priorità:** alta. **Stato:** `DONE` (componente residente, CUDA sm_86).

- [x] `T09.1` Implementare proiezioni input/testo, timestep embedding, modulazione, 32 blocchi e output usando la configurazione fissata.
- [x] `T09.2` Implementare metadata dei token, RoPE e maschera block-causal senza confondere la versione precedente dual-stream. Validare con maschere piccole materializzate soltanto nei test.
- [x] `T09.3` Implementare primo passo completo e passi target-only con prefix cache per layer; cache proprietaria del solo prefisso e invalidazione per conditioning/profilo.
- [x] `T09.4` Implementare scheduler Euler e otto passi ufficiali, zero terminale, trasformazioni di sigma/time shifting e CFG 1; nessuna recipe base ereditata implicitamente.
- [x] `T09.5` Accettare rumore iniziale esplicito per i confronti e definire RNG nativo stabile per la produzione. Tenere lo stato RNG della generazione distinto da quello del LLM sospeso.
- [x] `T09.6` Eseguire con DiT residente; esporre gruppi e dipendenze per T19. Stimare separatamente prefix cache, attivazioni e workspace del primo passo.

**Gate di completamento:** output di ogni blocco e latenti di ogni passo entro soglie T00; esattamente otto aggiornamenti del latente nella recipe predefinita. Cached/uncached verificati con medesimo input, senza crescita di memoria fra i passi né view che trattengono il forward completo. Peak attention non quadratico per storage di score.

**Artefatti:** grafo DiT, scheduler, dump step-by-step e stime memory-backed.

Qualifica locale: cinque casi, 288 confini oracle per caso bit-exact, otto aggiornamenti e memoria stabile fra passi. Forward full/cached verificati sullo stesso latente; la deriva accumulata della traiettoria interamente uncached reversed resta diagnostica non qualificata, con soglie T00 invariate. Fonti, misure, guasti e limiti nel [report T09–T10](../performance/native-suite-t09-t10-20261021/README.md).

## 14. T10 — VAE Qwen 2.1, immagini reference e RGBA

**Dipendenze:** T06, T07. **Priorità:** alta. **Stato:** `DONE` (immagine singola, CUDA sm_86).

- [x] `T10.1` Portare encode/decode del VAE corretto, normalizzazione dei 64 canali latenti, scaling e convenzioni spaziali/temporali necessarie al caso immagine.
- [x] `T10.2` Preservare alpha nei pixel e nelle reference; controllare clamp, conversione colore, range e orientamento. Rifiutare VAE precedenti incompatibili.
- [x] `T10.3` Implementare un percorso completo di riferimento su risoluzioni gestibili e decode tiled con overlap/blending dichiarati; scegliere tiling tramite il piano di memoria.
- [x] `T10.4` Gestire l'encoding delle reference prima del DiT e il successivo decode, con caricamento/rilascio ripetuto dello stesso componente quando opportuno.
- [x] `T10.5` Integrare PNG e output supportati; perdita di alpha solo su formato che la richiede esplicitamente. Il salvataggio non deve trattenere la VRAM del VAE.

**Gate di completamento:** latenti encode e pixel decode confrontati con oracle, casi RGBA/trasparenti e dimensioni non quadrate. Tiled vs completo supera gate percettivi e controlli sui bordi fissati in T00; nessuna dichiarazione bit-exact non dimostrata. Consumo a tile bounded e rilascio completo dopo il risultato host.

**Artefatti:** adapter VAE, codec output e fixture di bordi/alpha.

Qualifica locale: RGB/RGBA/trasparente 64×96, RGBA 256×768, encode normalizzato, pixel e tiled/full entro T00; PNG e cancellazione prima del risultato verificati dopo release/reload dei pesi. Posterior grezza e feature interne restano diagnostiche. La composizione della pipeline è T11. Evidenze nel [report T09–T10](../performance/native-suite-t09-t10-20261021/README.md).

## 15. T11 — Pipeline image nativa funzionale

**Dipendenze:** T08, T09, T10. **Priorità:** alta. **Stato:** `IN_PROGRESS` — API C e piano completo implementati; gate di qualità separati dalla parità numerica. Il checkpoint oracle fallisce il layout di una fixture T00 con entrambi i seed, come DS4; review umana completa non verificata.

- [x] `T11.1` Comporre text-to-image ed editing con una/più reference, preserving recipe, seed, ordine, size e conditioning.
- [x] `T11.2` Esporre un piano di fasi capace di stimare l'intera richiesta prima del lavoro GPU e di riusare output host/cache valide.
- [x] `T11.3` Validare input, risoluzioni/allineamenti, numero reference e combinazioni supportate. Parametri SDK non implementati, incluse eventuali mask, restituiscono errore esplicito.
- [x] `T11.4` Eseguire inizialmente con fasi seriali e piani di component residency; integrare l'executor comune senza prefetch speculativo.
- [x] `T11.5` Produrre media e metadati riproducibili, dump intermedi opzionali e contatori per caricamento/encoder/DiT/VAE.

**Gate di completamento:** corpus image di T00 passa generazione, editing, multi-reference, testo nell'immagine e trasparenza. Nessun worker esterno o runtime Python attivo nel percorso DS4. Memoria e latenza totale misurate sulla 3090; risoluzioni non qualificate non vengono promesse.

API opache `ds4_qwen_image_pipeline`, resize reference Lanczos dichiarato,
quattro piani residency/cache e pubblicazione dopo il restore. Qualifica
numerica `PASS` sui 56 casi (48 T00 più otto di sviluppo): 448 latenti entro
soglia, di cui 432 bit-exact, e tutti gli output RGBA entro T00. La copertura
per fasi conserva 17 esecuzioni complete con reference e verifica il decoder
finale su 39 DiT invariati; non è un singolo run dell'ultimo binario.
Mediana calda 256×256 con una reference: 12,901 s, cinque campioni, CV 2,41%.
DiT residente e cache GPU quando ammessa; fallback host senza perdita BF16.
Layout del checkpoint e review umana conservano stati distinti nel
[report T11–T12](../performance/native-suite-t11-t12-20261021/README.md).

**Artefatti:** entry point C image, smoke test CLI/engine e report di parità funzionale.

## 16. T12 — Backbone testuale EmbeddingGemma 2 e pooling

**Dipendenze:** T06, T07. **Priorità:** alta. **Stato:** `DONE` — qualifica testuale CUDA sm_86: hidden T00 bit-exact, vettori su 54 casi e tutte le dimensioni entro T00, batch/task/ranking verificati.

- [x] `T12.1` Portare tokenizer, embedding scaling, PLE da proiezione, norm, GELU, residui e proiezione finale, leggendo configurazioni per layer.
- [x] `T12.2` Implementare attenzione bidirezionale locale/full, head dimension e KV head variabili, RoPE e scala 1.0 secondo oracle. Testare inclusività dei bordi della finestra.
- [x] `T12.3` Implementare maschere di padding e pooling esatti del checkpoint, normalizzazione L2 e output 768/512/256/128 con rinormalizzazione dopo il troncamento.
- [x] `T12.4` Mappare task query/document/code/similarity ai prefissi corretti senza duplicazione e senza alterare media o titolo già formattato.
- [x] `T12.5` Supportare batch e input di lunghezza diversa senza attenzione fra elementi; niente KV persistente fra richieste embedding.

**Gate di completamento:** vettori e hidden state su fixture testo/codice/multilingue entro soglie T00; norma finale conforme, dimensioni esatte e ranking verificato. Un elemento produce lo stesso risultato da solo e nel batch entro tolleranza. Milestone interna: non autorizza a dichiarare il supporto multimodale completo.

`ds4_embedding_gemma2` e tokenizer/config compilati offline con hash del bundle.
Ragged batch isolati, nessun padding nelle chiavi o KV persistente. La qualifica
conserva ogni layer/proiezione: tutti i confini dei 48 testi T00 sono bit-exact.
La riduzione RMS riproduce l'ordine eager anche sotto 16 righe; ranking
dev/held-out identico a 128/256/512/768 senza modificare le soglie. Batch,
prefissi task e pooling con padding verificati. Lo stress aggiuntivo da
517 token è ora bit-exact; quello da 1028 mantiene hidden grezzi non bit-exact
ma vettori entro T00, come diagnostica esplicita. Evidenze e limiti nel
[report T11–T12](../performance/native-suite-t11-t12-20261021/README.md).

**Artefatti:** adapter Gemma text, pooling e fixture masking/task/dimensioni.

## 17. T13 — Media e preprocessing deterministico

**Dipendenze:** T00, T01, T06. **Priorità:** alta. **Stato:** `DONE` (qualifica locale Ubuntu/RTX 3090; profilo e limiti nel report).

- [x] `T13.1` Definire oggetto media interno e contenuti ordinati, separando byte originali, decodificati e tensor preprocessati. Integrare il decoder immagini esistente dove compatibile.
- [x] `T13.2` Implementare resize/crop/normalizzazione e patch position per Gemma, con EXIF, alpha e aspect ratio conformi all'oracle.
- [x] `T13.3` Integrare decode video e campionamento frame/timestamp attraverso una libreria nativa con API C, valutando libav o alternativa prima di introdurre dipendenze. Nessun processo di inferenza esterno.
- [x] `T13.4` Integrare audio decode, canali, resampling e feature extractor esatti; testare impulsi, toni, silenzio e sample rate diversi.
- [x] `T13.5` Stimare token e working set prima di ammettere il backbone; limite alle dimensioni decodificate e alle code di frame/audio per evitare espansioni host illimitate.
- [x] `T13.6` Rendere espliciti segmentazione, traccia audio del video e budget visivo. Nessun resize/campionamento/troncamento aggiuntivo invisibile rispetto alla recipe.

**Gate di completamento:** pixel, patch, feature audio, tempi e token count coincidono con i processor fissati. Durate variabili e video a frame rate variabile testati. La pipeline consuma memoria host bounded e rilascia i buffer; errori decode non toccano le sessioni LLM.

**Artefatti:** modulo media/preprocessing, decisione dipendenze e fixture codec/preprocess.

## 18. T14 — Encoder vision e video EmbeddingGemma 2

**Dipendenze:** T12, T13. **Priorità:** alta. **Stato:** `DONE` (qualifica locale Ubuntu/RTX 3090; profilo e limiti nel report).

- [x] `T14.1` Portare vision encoder, posizioni patch, riduzione e proiezioni verso lo spazio del backbone; rispettare padding e soft token validi.
- [x] `T14.2` Implementare immagini singole/multiple e video come frame raggruppati per elemento, con conteggi e ordine originali.
- [x] `T14.3` Inserire i token visivi nelle posizioni previste; validare numero placeholder vs feature prima del forward testuale.
- [x] `T14.4` Esporre fasi encoder/backbone e cache preprocessate con chiavi per revisione, media, resize e precisione. Caricare soltanto il ramo vision quando serve.

**Gate di completamento:** feature intermedie, vettori image-only/video-only e retrieval testo↔immagine/testo↔video rispettano T00. Cambiare un frame o l'immagine modifica il vettore secondo fixture; impossibile accettare media e ignorarli. Batch con numero di frame diverso mantiene isolamento.

**Artefatti:** adapter vision/video e report oracle per ciascuna modalità.

**Evidenze T13–T14:** [report, comandi e limiti](../performance/native-suite-t13-t14-20261021/README.md), [gate completi](../performance/native-suite-t13-t14-20261021/verification.json), [regressioni e memoria](../performance/native-suite-t13-t14-20261021/acceptance.json). Decode/preprocess C con libav/soxr, cache bounded e vision/video BF16 nativi; nessun endpoint nuovo o encoder audio dichiarato. T15/T16 sono completati nella qualifica locale successiva; T17/T18 sono qualificati nel report successivo; T19–T22 restano aperti.

## 19. T15 — Encoder audio EmbeddingGemma 2

**Dipendenze:** T12, T13. **Priorità:** alta. **Stato:** `DONE` (gate locali C/CUDA sm_86).

- [x] `T15.1` Inventariare e portare gli operatori della rete audio del checkpoint, inclusi subsampling, convoluzioni, attenzione e proiezione richiesti dalla configurazione fissata.
- [x] `T15.2` Riprodurre maschere dei frame e selezione dei soft token validi, escludendo padding e rispettando le convenzioni di posizione.
- [x] `T15.3` Implementare audio-only, più clip e testo+audio; descrivere caricamento selettivo della torre audio e durata delle feature.
- [x] `T15.4` Collegare video+audio esplicito senza confondere durata della traccia e ordine dei frame. Nessuna trascrizione usata al posto dell'encoder audio.

**Gate di completamento:** feature/audio hidden e vettori finali entro soglie T00; retrieval testo↔audio e casi silenzio/non-speech/lingue diverse. Lunghezze variabili e padding non contaminano gli altri elementi del batch. Zero NaN/Inf anche sui limiti di durata supportati.

**Artefatti:** adapter audio e fixture intermedie/end-to-end.

## 20. T16 — Embedding multimodali composti e piano selettivo

**Dipendenze:** T12, T14, T15. **Priorità:** bloccante per M4. **Stato:** `DONE` (gate locali C/CUDA sm_86).

- [x] `T16.1` Comporre text/image/video/audio interleaved in una sola sequenza con posizioni, separatori e maschere conformi al modello.
- [x] `T16.2` Produrre un vettore per elemento composto e batch eterogenei. Pooling unico secondo il checkpoint, non media arbitraria di vettori già normalizzati.
- [x] `T16.3` Creare piani che caricano solo gli encoder usati; consentire rilascio degli encoder dopo la produzione delle feature prima del backbone quando il budget lo richiede.
- [x] `T16.4` Gestire contesto comune, errori oltre limite e segmentazione esplicita con output/timestamp corrispondenti. Non dichiarare il massimo posizionale come limite operativo.
- [x] `T16.5` Rendere revisioni, task, dimensioni, profilo e preprocess parte dell'identità degli embedding/cache; evitare mescolamento di spazi differenti negli indici del client.

**Gate di completamento:** tutte le quattro modalità e almeno una combinazione di tutte passano oracle e retrieval. Batch di 1/2/3 elementi, input non multipli dei tile, media ripetuti e ordine diverso verificati. Trace dimostra che gli encoder inutilizzati non vengono caricati e che il vettore deriva realmente da tutti i contenuti richiesti.

**Artefatti:** entry point C embeddings completo e matrice funzionale a quattro modalità.

**Evidenze T15–T16:** [report, precisione del riferimento e limiti](../performance/native-suite-t15-t16-20261022/README.md), [287 casi e gate numerici/retrieval](../performance/native-suite-t15-t16-20261022/verification.json), [regressioni e memoria](../performance/native-suite-t15-t16-20261022/acceptance.json). Audio C/CUDA, composizione tipizzata v2 e piani selettivi verificati; 1450 frontiere audio con errore numerico zero. Il riferimento storico T00 r2 con riduzioni BF16 e i relativi FAIL restano conservati; la qualifica usa un nuovo riferimento separato con accumulo FP32 e soglie T00 invariate. Nessun nuovo endpoint HTTP.

## 21. T17 — Registro modelli, HTTP e gestione dei risultati

**Dipendenze:** T01, T05, T11, T16. **Priorità:** alta. **Stato:** `DONE` nei gate locali di servizio CUDA single GPU; gate qualità T11 invariati.

- [x] `T17.1` Estendere il server a un registro di modelli configurati e routing per operazione/capability, preservando gli alias LLM e i comportamenti di thinking esistenti.
- [x] `T17.2` Collegare Images generation/edit e Embeddings al medesimo executor di risorse; richieste accodate non modificano il contesto attivo durante un'altra esecuzione.
- [x] `T17.3` Implementare upload/riferimenti media e risposta immagini/embedding; dimensionare limiti e lifetime per evitare di trattenere output GPU o interi video in memoria.
- [x] `T17.4` Implementare errori strutturati, cancellazione, timeout, request ID e idempotenza. Un retry di trasporto non deve duplicare una generazione già committata.
- [x] `T17.5` Separare progressi da risultato finale. Inviare il successo soltanto dopo il restore; un media calcolato con restore fallito non diventa un falso successo.
- [x] `T17.6` Rendere osservabili modello/profilo eseguito, piano scelto, cache hit, byte spostati e tempi; discovery distingue configurato/residente senza imporre gestione manuale al client.

**Gate di completamento:** veri client OpenAI funzionano per i contratti standard supportati; client DS4 multimodale passa le quattro modalità. Richieste invalide non mutano sessioni. Chat/Responses/Anthropic e SSE esistenti conservano i propri contratti. Nessun parametro supportato solo nominalmente e poi ignorato.

**Aree:** `ds4_server.c`, risorse testuali/configurazioni del dominio e suite HTTP.

**Evidenze T17:** [report e limiti](../performance/native-suite-t17-t18-20261022/README.md),
[hash e verifiche](../performance/native-suite-t17-t18-20261022/verification.json),
[guida client](../architecture/native-suite-client.md). Sei scenari SDK live:
discovery, quattro modalità, token/PAD/base64, retry concorrenti, editing e
cancellazione per disconnessione. Successo dopo restore e recovery con handle
conservati; fault/restore e deadline verificati nei gate host, senza simulare
perdita hardware HTTP. PNG e riferimenti con geometria nativa ammessa; nessuna
promozione dei gate qualità T11 o dei backend non qualificati.

## 22. T18 — Client semplice e integrazione ExMekh-lib

**Dipendenze:** T17. **Priorità:** alta. **Stato:** `DONE` nei gate locali client/ExMekh e frontend.

- [x] `T18.1` Fornire esempi con un solo client/base URL e chiamate images/embeddings per nome modello. Nessun comando load/unload/checkpoint richiesto all'utente.
- [x] `T18.2` Fornire helper opzionale per embeddings multimodali che gestisce file/media, ordine e serializzazione; mantenere gli embedding testo usabili con SDK OpenAI ordinario.
- [x] `T18.3` Integrare le capability nel client tipizzato ExMekh esistente senza creare un secondo frontend o modificare direttamente wheel compilate. Se serve una modifica alla libreria sorgente esterna, registrarla come dipendenza esplicita con contratto e test. Nel task "Funzionalità extra" prensente file "exmekh_lib/todo.md" del client, sono scritte alcune specifice me la modifca sul client stesso.
- [x] `T18.4` Collegare tool esplicitamente registrati di generazione/editing/embedding alle API; risultato strutturato con media ID, recipe, seed e dimensioni.
- [x] `T18.5` Mostrare i media con il percorso UI già previsto e mantenere la UI responsiva. L'analisi dei pixel da parte del LLM rimane una nuova invocazione esplicita del suo encoder.
- [x] `T18.6` Verificare che client e frontend non debbano sapere se il gestore ha tenuto la KV, parcheggiato la sessione o scaricato i pesi.

**Gate di completamento:** scenario conversazionale reale: LLM → tool image oppure ricerca web con ranking embedding interno → risultato → continuazione dal contesto precedente. Nessun full-prefill quando un checkpoint completo valido era disponibile; nessuna gestione risorse lato client. Le estensioni restano dichiarate e versionate.

**Aree:** esempi client e integrazioni ExMekh già possedute dal progetto; rispettare anche `agent/AGENTS.md` quando applicabile.

**Evidenze T18:** [report](../performance/native-suite-t17-t18-20261022/README.md) e
[patch sorgente ExMekh](../performance/native-suite-t17-t18-20261022/exmekh-lib.patch).
Client/LocalProvider reali, tool registrato e continuazione con codice ricordato
e token cached; anteprima reale True con scala 0.5. Modelli Literal, nessuna
scelta per modello unico, width/height interi. Libreria: 3622 test offline;
frontend/browser: 96 PASS e 1 live preesistente skipped; history/restore: 112 PASS.
Wheel ricostruita dai sorgenti, lock aggiornato e dipendenza esterna registrata.

L'[audit E2E di «Funzionalità extra»](../performance/native-suite-image-e2e-20261022/README.md)
verifica il browser con LLM e generazione GPU reali e corregge la perdita degli
allegati negli snapshot DeepAgent e lo schema image-only. Quattro prove live
passate; miniature, dialog, reload e secondo turno con due anteprime reali.
Libreria offline 3624 PASS e regressioni frontend 162 PASS. Limiti dell'attesa
tool e replay falliti conservati nel follow-up; gate qualità T11 invariati.

Il [riesame clean code successivo](../performance/native-suite-clean-review-20261022/README.md)
uniforma ownership HTTP e serializzazione e, dopo il chiarimento dell'utente,
rimuove la capability embedding autonoma: gli embedding nativi sono interni
alla ricerca web e restano disponibili nell'API tipizzata.

## 23. T19 — Prefetch, streaming per gruppi e mantenimento adattivo

**Perimetro concordato il 22 ottobre:** completare la consegna funzionale e
lasciare aperti gli studi di ottimizzazione. T19.5/T19.8, T20.1–T20.5 e T21.6
restano aperti; nessun profilo alternativo viene promosso senza i propri gate.

**Dipendenze:** T05, T07, T11, T16, T17. **Priorità:** alta dopo la baseline seriale. **Stato:** `IN_PROGRESS`.

- [x] `T19.1` Implementare coda disco→host bounded, riuso mmap/page cache e staging pinned limitato; niente pinning di tutti i modelli come default.
- [x] `T19.2` Implementare copie host→GPU con stream dedicato, eventi di disponibilità/ultimo uso e buffer doppi conteggiati. Il compute aspetta solo il gruppo necessario.
- [x] `T19.3` Scegliere dinamicamente DiT interamente residente, insieme residente più gruppi streamed o streaming completo. Non fissare due blocchi per ogni GPU e workload.
- [x] `T19.4` Applicare lo stesso contratto ai layer encoder e ai componenti Gemma; ogni adapter dichiara gruppi e visite, senza policy Qwen nel gestore.
- [ ] `T19.5` Sperimentare wavefront Qwen3-VL→DiT e VAE→LLM solo con riserva per la fase corrente e il restore. Rilasciare gruppi encoder consumati soltanto dopo aver verificato tutti gli usi, inclusi deepstack/output intermedi.
- [x] `T19.6` Integrare prefetch CPU anticipato su eventi strutturati di tool registrati o hint espliciti; annullarlo se la call non è completata/validata. Non eseguire né caricare risorse arbitrarie da testo parziale.
- [x] `T19.7` Conservare modelli/componenti caldi se compatibili con il ritorno al LLM; applicare costi/isteresi e misure per decidere cosa espellere alla richiesta successiva.
- [ ] `T19.8` Misurare seriale vs overlap, resident vs streaming e prima vs successive richieste. Confrontare policy leading/strided soltanto se la trace mostra un vantaggio plausibile.

**Gate di completamento:** resident e streamed rispettano gli stessi gate numerici; zero use-after-free/eventi stale e nessuna crescita di memoria. Caso capiente: zero eviction, zero ritrasferimenti DiT fra passi. Caso streamed: byte realmente trasferiti coerenti col piano. Overlap promosso solo se migliora la latenza totale oltre il rumore e non peggiora il LLM in corso oltre il gate prestazionale.

**Artefatti:** scheduler di trasferimento, trace temporali e confronto riproducibile fuori profiler.

## 24. T20 — Quantizzazione e ottimizzazioni numeriche misurate

**Dipendenze:** T00, T07, T11, T16; T19 per confronti su streaming. **Priorità:** dopo la correttezza nativa. **Stato:** `IN_PROGRESS`.

- [ ] `T20.1` Valutare encoder Qwen3-VL BF16, Q8/INT8 realmente disponibili e layout misti. Distinguere INT8 storage con compute BF16 da W8A8 Tensor Core; nessun guadagno assunto.
- [ ] `T20.2` Valutare DiT BF16 e profili GGUF Unsloth sulla schedule Turbo, mantenendo encoder/VAE fissi nel confronto. Separare errore di kernel su pesi quantizzati da qualità rispetto a BF16.
- [ ] `T20.3` Confrontare sensibilità per tensore con la proposta ultimi 2–4 layer BF16; scegliere solo su dati held-out, includendo tipografia, editing e alpha.
- [ ] `T20.4` Valutare profili quantizzati Gemma solo con compute BF16/FP32 e gate retrieval per ogni modalità/dimensione; niente fallback FP16 nascosto.
- [ ] `T20.5` Ottimizzare i soli hotspot provati: fusioni norm/RoPE/gating, proiezioni GEMM, attenzione e VAE/audio. CUDA Graph soltanto per shape e indirizzi stabili, con invalidazione al reload.
- [x] `T20.6` Dichiarare fuori dal percorso iniziale step skipping, cache approssimate DiT, KV LLM lossy e dipendenze da istruzioni Hopper/Blackwell. Eventuali ricerche future sono task separati.

**Gate di completamento:** ogni candidato ha `KEEP`, `REJECT` o `NEED_MORE_DATA` con fonti e misure. Un profilo viene offerto al client solo con tutti i gate numerici/qualità/performance superati. Non è obbligatorio promuovere ogni ottimizzazione; è obbligatorio consegnare un percorso nativo corretto, la scelta dei profili supportati e i risultati della valutazione richiesta.

**Artefatti:** ablation per componente, profili pubblici qualificati e registro dei candidati scartati.

## 25. T21 — Qualificazione integrata, guasti e regressioni

**Dipendenze:** T17, T18, T19; tutti i profili promossi da T20. **Priorità:** bloccante per la consegna. **Stato:** `IN_PROGRESS`.

- [ ] `T21.1` Eseguire la matrice funzionale/numerica/memoria dei capitoli successivi, inclusi restore e ripresa agentica su LLM locali qualificati.
- [x] `T21.2` Iniettare fallimenti in checkpoint, allocazione, caricamento, decode media, encoder, ogni frontiera DiT/VAE, output e restore. Verificare cancellazione e disconnessione in ciascuna fase.
- [x] `T21.3` Eseguire almeno 50 alternanze miste dopo warm-up: LLM → image → LLM → embedding di tutte le modalità → LLM, incluse richieste ripetute e nuove.
- [x] `T21.4` Verificare contatori delle allocazioni e risorse per proprietario, file temporanei, handle e cache bounded. Correlare memoria globale con quella posseduta senza attribuire rumore esterno a un leak.
- [x] `T21.5` Eseguire suite esistenti pertinenti CPU/CUDA, server, checkpoint, MTP e vision; verificare build/regressioni degli altri backend disponibili. TP/distribuito secondo le regole del repository, senza avviare prove non autorizzate.
- [ ] `T21.6` Congelare benchmark finali con cinque campioni o più dopo warm-up, cold start separato e misure fuori profiler; registrare i risultati instabili senza promuoverli.

**Gate di completamento:** tutti i casi obbligatori sulla piattaforma di rilascio sono `PASS`; nessun `FAIL` o `NOT_VERIFIED` nascosto nel riepilogo. In ogni guasto recuperabile la sessione continua correttamente; nei guasti non recuperabili l'indisponibilità è esplicita e il checkpoint non viene eliminato prematuramente. Regressioni legacy oltre soglia bloccano il rilascio.

**Artefatti:** report JSON/Markdown, trace, manifest binari/modelli e matrice di qualificazione per backend/famiglia.

**Esito locale del 22 ottobre:** 50 alternanze, equivalenza di residenza,
64 iniezioni alle frontiere native e regressioni CPU/CUDA Qwen27/MTP/vision
PASS. T21.1 resta aperto per la qualità image T11. T21.2 comprende inoltre
17 distacchi socket reali: conditioning, estremi DiT/latenti e tutte le
frontiere VAE nominate, con LLM cached, nessun output parziale e PNG finale
identico. Ogni layer/passo DiT ha iniezione nativa; non si dichiara un
distacco socket distinto per ciascuna ripetizione dello stesso layer/passo.
T21.6 è rinviato su richiesta dell'utente. [Dettagli e limiti](../performance/native-suite-t19-t22-20261022/README.md).

## 26. T22 — Packaging, documentazione e consegna

**Dipendenze:** T21. **Priorità:** bloccante. **Stato:** `IN_PROGRESS`.

- [x] `T22.1` Documentare modelli/profili supportati, download, API standard/estese, limiti e metriche sulla macchina reale. Non presentare stime di memoria come misure.
- [x] `T22.2` Documentare dipendenze native per media, build sm_86, assenza di Python runtime e troubleshooting del recupero sessioni.
- [x] `T22.3` Aggiornare `PROJECT_INDEX.md`, wiki generata e note di provenienza/licenze; preservare frontend unico e default dei launcher correnti.
- [x] `T22.4` Fornire ricette riproducibili per generazione/editing/RGBA, quattro modalità embedding, scenario senza eviction e scenario con checkpoint+reload.
- [x] `T22.5` Eseguire build pulita e comandi di accettazione dal pacchetto finale. Elencare separatamente estensioni e backend non qualificati.

**Gate di completamento:** un utilizzatore esegue gli scenari previsti con un solo server e un solo client configurato, senza controllare la memoria manualmente. Ogni funzionalità dichiarata è collegata a una prova riproducibile; nessun worker esterno necessario. Documentazione e indici passano i controlli.

**Consegna locale:** pacchetto nativo di sviluppo, build da sorgenti isolati,
sette scenari SDK e tool ExMekh con anteprima reale PASS, contesto 2048 invariato;
zero file temporanei alla chiusura. T22 resta `IN_PROGRESS` per la distribuzione
completa finché i gate di rilascio T11/T21 non sono chiusi. Il packager rifiuta
correttamente la promozione a release con gate mancanti.

## 27. Matrice obbligatoria di memoria e ripristino

La matrice usa budget controllati e allocazioni possedute; si aggiungono prove reali sulla memoria libera della 3090. Un budget simulato non certifica una scheda fisica di capacità diversa.

| ID | Preparazione | Risultato richiesto |
| --- | --- | --- |
| R0 | LLM, sessioni e nuova operazione entrano insieme | Nessun checkpoint/eviction; output corretto; componenti caldi riutilizzabili |
| R1 | Serve soltanto rilasciare cache ricostruibili | Sessioni e pesi invariati; bytes liberati effettivamente misurati |
| R2 | Serve parcheggiare una/più sessioni | Payload completi; grafi rilasciati; pesi mantenuti; ripresa identica |
| R3 | Conviene scaricare pesi mantenendo stato GPU | Stato e identità validi; pesi reload; grafi/puntatori corretti |
| R4 | Occorre scaricare pesi e sessioni | Checkpoint, rilascio, inferenza e ripristino completo prima del successo |
| R5 | Entra un componente per volta | Visite encoder/VAE/DiT corrette, picco rispettato e output host bounded |
| R6 | DiT entra intero soltanto dopo l'encoder | Un caricamento DiT, nessun traffico pesi fra gli otto passi |
| R7 | DiT non entra intero | Insieme residente/gruppi streamed corretti; staging incluso nel budget |
| R8 | Editing con prefix cache/reference grandi | Stima del primo passo e cache corretta; nessuna matrice score densa incontrollata |
| R9 | RAM/pinned insufficienti, SSD disponibile | mmap/staging bounded o errore prima della mutazione; niente OOM host nascosto |
| R10 | SSD insufficiente o checkpoint corrotto | Stato vivo conservato oppure recupero esplicito; nessuna cancellazione della history |
| R11 | Pressione GPU esterna impedisce restore | Checkpoint conservato, sessione non dichiarata pronta e nessun falso successo |
| R12 | Richiesta annullata in qualunque fase | Copie/compute drenati alla frontiera valida e risorse restituite |
| R13 | Alternanze ripetute e più sessioni vive | Stato di ciascun proprietario preservato, contatori bounded e nessuna contaminazione |

## 28. Gate numerici, di qualità e prestazionali

### 28.1 Gate numerici

- **N0 — Identità degli input:** hash artefatti/media, token, posizioni, preprocessing, maschere e schedule coerenti con il manifest. Un mismatch qui blocca il confronto degli output.
- **N1 — Primitive:** output completi, finite check e tolleranze per dtype/operazione fissate in T00. Non sostituire max error e relative error con il solo cosine medio.
- **N2 — Modelli:** hidden state e latenti alle frontiere, pooling e vettori; confronto con oracle ufficiale e seconda implementazione quando disponibile. Stesso rumore iniziale per la diffusione.
- **N3 — Ripresa:** modifiche di sola residenza conservano payload autorevoli, logits e continuazione deterministica bit-exact nello stesso percorso numerico, inclusi MTP/stato ricorrente. Verificare RNG e metadata oltre al solo primo token.
- **N4 — Ottimizzazioni:** cache prefix, tiling, fusioni e quantizzazione hanno confronti attribuibili. Le trasformazioni con roundoff differente usano soglie congelate e verifica end-to-end, non una promessa automatica di bit-exact.

Le soglie full-vocabulary dei LLM esistenti restano valide e non vengono allargate da questo piano. Il manifest T00 deve valorizzare anche le soglie specifiche dei nuovi modelli prima che T07–T20 possano essere chiusi.

### 28.2 Corpus di accettazione proposto

Il corpus minimo viene congelato in T00 prima delle ottimizzazioni; può essere ampliato con un nuovo manifest, mantenendo i casi precedenti.

| Area | Casi minimi e verifiche |
| --- | --- |
| Immagini | 24 recipe held-out × 2 rumori iniziali: generazione, tipografia IT/EN, layout, editing singolo/multiplo e alpha; almeno due aspect ratio e una risoluzione superiore al caso base quando ammessa |
| Testo/codice | 48 elementi fra italiano, inglese, multilingue e codice; task query/document/similarity e batch variabili |
| Immagini embedding | 16 elementi con modifiche di contenuto e padding; retrieval incrociato |
| Video embedding | 8 clip con durata/frame rate variabili e ordine dei frame controllato |
| Audio embedding | 16 clip con parlato, suoni non verbali, silenzio e sample rate differenti |
| Input composti | 16 elementi interleaved, inclusa combinazione delle quattro modalità |
| Retrieval | Almeno 32 query per collegamento dichiarato testo↔immagine, testo↔video e testo↔audio, con metriche e limite di regressione fissati in T00 |
| Lifetime | Almeno 50 alternanze miste più matrice R0–R13 e fault injection |

Metriche image: errore sui latenti e output intermedi, LPIPS/SSIM o metriche equivalenti congelate, OCR/layout per prompt appropriati, preservation delle reference e verifica alpha/bordi. Metriche embedding: errore vettoriale, norma, cosine e ranking Recall@k/nDCG su dati fissi. Un'immagine plausibile o un vettore non nullo non bastano.

### 28.3 Gate prestazionali e memoria

- Misurare validazione/preprocess, ammissione, checkpoint, rilascio, H2D/D2H, encoder, primo passo DiT, altri passi, VAE, pooling, restore e risposta finale. La metrica utente include tutto il passaggio LLM → operazione → LLM pronto.
- Distinguere cold start, warm start e cache hit. Confrontare profili con gli stessi input, parametri e stato termico; usare almeno cinque ripetizioni dopo warm-up per promuovere una candidate.
- Conservare campioni, mediana, p10/p90, deviazione e CV. CV > 5% richiede `NEED_MORE_DATA`; non proclamare guadagni comparabili al rumore.
- Le regressioni LLM oltre 5%, o oltre una soglia più stretta già imposta dal relativo harness, bloccano la promozione. Una micro-ottimizzazione non compensa un peggioramento della latenza totale o del restore.
- Il picco deve rientrare nel budget prenotato; contatori per proprietario bounded dopo warm-up e ritorno ai valori attesi dopo rilascio. RSS o memoria globale da soli non certificano un leak.
- Non fissare ora un obiettivo in secondi o una promessa di funzionamento su 4/6/8 GB: manca una baseline DS4 dei nuovi modelli. T00 congela i target misurabili sulla 3090 e T21 ne riporta l'esito.

## 29. Definizione di completamento globale

Il lavoro di implementazione è completo quando T00–T19, T21 e T22 sono `DONE`, T20 ha una valutazione esplicita dei candidati e tutti i profili effettivamente pubblicati hanno superato i rispettivi gate.

Devono essere dimostrati: generazione/editing/RGBA nativi; embedding nativi delle quattro modalità e composti; client semplice; percorso senza eviction; sospensione e scaricamento selettivi; ripristino prima del successo; gestione dei guasti; prestazioni e memoria misurate; conservazione delle funzionalità LLM qualificate.

La matrice dei backend e delle famiglie deve distinguere il supporto architetturale dalla qualificazione sperimentale. Funzioni richieste sul target di rilascio non possono essere omesse con `NOT_VERIFIED`; hardware aggiuntivo non disponibile resta esplicitamente fuori dalla dichiarazione di qualificazione, senza attribuirgli risultati della 3090.

Aggiornamento di implementazione: T00/T01 e i gate locali T05/T06 completati; runtime T02 e adapter T03/T04 verificati su RTX 3090 single GPU, con le qualifiche residue esplicite nei report. T05 espone planner/executor C e adapter lifecycle; T06 espone bundle pinning, conversione offline e loader mmap nativo. Il wrapper vision legacy conserva il proprio percorso; il routing HTTP è ora nel target esplicito ds4-server-suite (T17). Oracle Qwen/Gemma restano strumenti di sviluppo. T07/T08 aggiungono primitive C/CUDA comuni e conditioning Qwen3-VL dedicato; T09/T10 aggiungono DiT Turbo residente con prefix cache e VAE RGB/RGBA completo/tiled, qualificati su sm_86 in API C seriali. T11/T12 implementano la pipeline image seriale con ammissione completa, cache host/device e backbone Gemma testuale, con gate e limiti nel report T11–T12. Gli endpoint Images/Embeddings sono abilitati nel target esplicito T17; T13/T14 aggiungono media/preprocess nativi e vision/video Gemma, con gate e limiti nel report T13–T14. T15/T16 aggiungono encoder audio e composizione completa con piani selettivi, qualificati su C/CUDA sm_86; T17/T18 aggiungono servizio HTTP e integrazione ExMekh con modelli Literal, width/height interi, tool registrati e anteprime persistenti. T19 aggiunge streaming bounded e mantenimento adattivo; T21/T22 aggiungono stress, recupero e pacchetto nativo locale. Restano aperti gli studi di ottimizzazione, la qualifica estesa e i gate qualità T11, come documentato nel report T19–T22. Evidenze nei report T07–T08, T09–T10, T11–T12 e T13–T14.
