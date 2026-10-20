# Suite multimodale nativa DS4: analisi e decisioni

Data: 20 ottobre 2026. Stato: proposta architetturale; nessuna funzionalità implementata con questo documento.

L'obiettivo è eseguire Qwen-Image-2.1-Turbo ed EmbeddingGemma 2 dentro DS4, condividendo GPU, RAM e checkpoint con i modelli linguistici esistenti. La [roadmap operativa](../roadmaps/native-multimodal-suite-implementation-20261020.md) traduce questa analisi in task, dipendenze e condizioni di completamento.

## 1. Requisiti acquisiti e confini

Le istruzioni della conversazione corrente definiscono il perimetro del lavoro.

1. Inferenza nativa nello stesso processo DS4: niente worker Python, Diffusers, llama-server o stable-diffusion.cpp come motori di produzione separati. Questi progetti sono fonti, strumenti di conversione o oracle di sviluppo.
2. Gestione dinamica indipendente da famiglia, operazione e componente. Aggiungere una famiglia richiede il relativo adapter numerico; il gestore della memoria non deve conoscere Qwen o Gemma.
3. Se il budget basta, nessun checkpoint o scaricamento preventivo. Più modelli possono restare residenti.
4. Se serve memoria, liberare risorse ricostruibili, sospendere sessioni e, quando necessario, scaricare pesi o loro parti. La history applicativa non viene cancellata.
5. Ripristinare le risorse necessarie alla ripresa prima di restituire il successo, come nell'operazione vision attuale di Qwen27. Un ripristino fallito è un errore distinto da un'inferenza fallita.
6. Embedding subito per testo, immagini, video, audio e combinazioni; le milestone intermedie testuali non costituiscono il completamento del prodotto.
7. Client semplice, riuso delle API OpenAI dove il contratto lo consente e un'estensione DS4 esplicita per embedding multimodali.
8. Primo target da qualificare: RTX 3090 sm_86 su Linux/WSL. Conservare i percorsi esistenti Metal, CUDA, ROCm, SSD e distribuiti; non dichiarare verificati backend non provati.
9. Restano validi i confini di `AGENT.md`: semantica e scheduling in C, nessuna nuova dipendenza architetturale C++, CPU come riferimento/debug, ExMekh Agent unico frontend agentico.

La suite resta un motore di architetture esplicitamente supportate: non diventa un esecutore automatico di qualunque file GGUF. La decisione dell'utente estende le capacità di DS4 senza eliminare validazione di metadata, shape, dtype e contratti numerici.

## 2. Cosa recuperare dalla chat condivisa

La [chat condivisa](https://chatgpt.com/share/6ad7418a-69cc-83ed-9480-a77ceaafafe8) è stata letta nel browser. È materiale progettuale, non un benchmark locale né una specifica normativa del modello.

| Spunto | Decisione per questo fork |
| --- | --- |
| Separare encoder, DiT e VAE | Adottare: ogni componente ha pesi, output, scratch e durata propri |
| Mantenere il DiT residente durante gli otto passi quando entra | Adottare: evita di ritrasferire la stessa rete a ogni passo |
| Streaming per gruppi con prefetch | Adottare come piano di memoria quando la residenza completa non conviene o non entra |
| Prefetch disco → RAM distinto da RAM → GPU | Adottare con limiti host/device separati e priorità al lavoro in corso |
| Caricamento progressivo encoder → DiT e VAE → LLM | Sperimentare dopo la pipeline seriale corretta, con eventi e prenotazioni di memoria |
| KV LLM sempre residente | Sostituire con decisione dinamica: conservarla quando possibile, checkpoint completo altrimenti |
| Scaricare sempre il LLM prima di image | Escludere: il percorso senza eviction è un requisito esplicito |
| Worker Python come prima integrazione | Escludere dal prodotto: l'utente richiede tutto nativo in DS4 |
| Porting preliminare Qwen27 CUDA | Già presente in questo fork; mantenere e verificare, non reimplementare |
| Contesto iniziale limitato a 8K–16K | Non introdurre come scorciatoia: conservare le capacità attualmente qualificate |
| Stime di 4/6/8 GB e tempi su 4090 | Ipotesi/fonti altrui, non promesse sulla 3090 o su Turbo |
| Ultimi layer encoder in precisione maggiore | Ipotesi di quantizzazione da confrontare con sensibilità per tensore, non regola iniziale |

Ogni passo di denoising attraversa l'intero DiT. Otto passi non significano otto porzioni indipendenti dei pesi. Il risparmio di memoria viene dal lifetime dei componenti e dallo streaming, non dal dividere i pesi per il numero di passi.

## 3. Stato reale di DS4 e ostacoli prioritari

L'audit in lettura ha usato il checkout con HEAD `5e2e99cf3249a5a28efed038bfd4fb2fbda6ce21`, contenente anche modifiche locali preesistenti. HEAD da solo non identifica l'intero contenuto del worktree: la baseline eseguibile futura deve includere hash dei sorgenti e dei binari effettivi.

### 3.1 Base già disponibile

Il [report Qwen27 vision](qwen27-vision-checkpoint-20261001.md), `ds4.h` e le funzioni `engine_vision_suspend_sessions`, `engine_vision_resume_sessions`, `ds4_engine_vision_encode_image` in `ds4.c` documentano e implementano:

- checkpoint completo delle sessioni vive, inclusi stato ricorrente, MTP e informazioni multimodali;
- distruzione dei grafi per liberare davvero la VRAM;
- esecuzione dell'encoder, conservazione degli embedding host e rilascio dei pesi vision;
- ricreazione dei grafi e ripristino prima del ritorno dall'encoder;
- conservazione del checkpoint quando un ripristino non riesce.

È il punto di partenza per il contratto transazionale. Non è ancora uno scheduler generale né un'API per scaricare i pesi del LLM mantenendone il contesto.

### 3.2 Stato globale da separare

`g_ds4_shape` alimenta macro come `DS4_MODEL_FAMILY`, `DS4_N_LAYER`, `DS4_N_EMBD`. Il backend CUDA conserva mapping e cache globali, fra cui `g_model_host_base`, `g_model_device_base`, cache derivate e stream. `ds4_engine_close` chiama `ds4_gpu_cleanup`, chiude thread e rilascia il lock d'istanza.

Di conseguenza, aprire un secondo engine e chiudere il primo non è un protocollo sicuro di alternanza. Si rischiano shape errate, invalidazione di cache, puntatori stale e cleanup delle risorse di un altro modello. Il catalogo multi-modello deve poggiare su ownership esplicita e attivazione controllata del contesto numerico.

È accettabile migrare gradualmente alcune funzioni legacy sotto un contesto attivo protetto dal lock GPU. Non è accettabile chiamare quel ponte da thread asincroni che possono osservare il contesto di un altro modello. I descrittori e le allocazioni residenti devono comunque restare distinti per proprietario.

### 3.3 Il consumo KV non coincide con la chat condivisa

`qwen_graph_alloc` usa `DS4_QWEN_ALLOC` con `sizeof(float)` anche per key/value cache. Nel percorso osservato sono allocazioni FP32 dimensionate sulla capacità del contesto, non soltanto sui token vivi. Sono presenti anche scratch, stato Gated DeltaNet e risorse MTP.

La stima BF16 della chat condivisa non può quindi diventare il budget del fork. Il pianificatore deve interrogare byte allocati, byte vivi serializzabili e byte effettivamente liberabili. Convertire la KV a BF16 sarebbe un esperimento numerico separato, non una conseguenza automatica di questo progetto.

L'[audit memoria del 13 ottobre](inference-memory-audit-20261013.md) distingue già risorse intenzionalmente conservate da ritenzioni evitabili; va mantenuta questa distinzione nel nuovo gestore.

## 4. Qwen-Image-2.1-Turbo: semantica da preservare

Il checkpoint ufficiale comprende encoder Qwen3-VL, DiT single-stream block-causal e VAE dedicato. Il [model index](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/model_index.json) specifica la pipeline e gli otto sigma; la [configurazione DiT](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/transformer/config.json) dichiara 32 blocchi, 32 teste da 128, hidden attention 4096, 64 canali, patch size 1 e causal conditioning.

La [configurazione dell'encoder](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/text_encoder/config.json) richiede un percorso Qwen3-VL dedicato, compresi deepstack vision e mRoPE. Il 27B conversazionale e il suo mmproj non sono sostituti numericamente equivalenti dell'encoder image.

### 4.1 Conditioning e scheduler

La [pipeline Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py) costituisce un oracle semantico: template e padding, esclusione del prefisso di sistema, ordine delle reference e hidden state prima della RMSNorm finale devono coincidere. Per RGBA, la copia destinata al vision encoder viene composta sul bianco; il VAE mantiene i quattro canali. Queste differenze vanno isolate in fixture intermedie.

Il profilo iniziale usa checkpoint Turbo, CFG 1 e schedule ufficiale. Il terminale zero, il time shifting dipendente dalla risoluzione, le convenzioni sui timestep e l'integrazione Euler vanno verificati contro la configurazione scheduler fissata. Copiare una lista di sigma senza verificare la trasformazione applicata dal backend può produrre un sampler differente.

Per confrontare motori diversi serve poter fornire il medesimo rumore latente iniziale: lo stesso intero seed non garantisce la stessa sequenza RNG fra librerie. Le prove del sampler nativo e quelle di equivalenza numerica devono essere separate.

### 4.2 Attention e prefix cache

Nel [transformer Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py) testo e blocchi immagine hanno una maschera block-causal. Il primo passo produce il prefix K/V; i successivi possono elaborare solo i token target. Il conditioning usa modulazione indipendente dal timestep target: senza questa proprietà la cache non sarebbe riutilizzabile.

La cache del prefisso appartiene alla singola generazione, al layer e al conditioning esatto; non è la KV conversazionale. Deve possedere solo i byte del prefisso, senza tenere vive le allocazioni complete del primo forward attraverso view.

Una attention tiled con softmax online deve evitare la matrice densa N×N. Sono necessari casi separati per costruzione del prefisso e passi successivi, padding, immagini multiple e confini non allineati ai tile. La cache è un'ottimizzazione esatta a livello matematico, ma cambiando shape GEMM può cambiare il roundoff: non promettere bit-exact fra cached e uncached senza una prova.

### 4.3 Memoria e ordine delle fasi

Il piano base è preprocessing → encoder/encoding reference → DiT ripetuto → VAE decode → ripristino. Le dipendenze reali possono richiedere il VAE anche prima del DiT per l'editing. Lo scheduler deve rappresentare queste visite multiple, anziché assumere una lista lineare valida solo per text-to-image.

Preferire il DiT completamente residente durante gli otto passi quando il budget totale lo permette. Se non entra, mantenere un insieme residente e trasferire i gruppi restanti; il prefetch deve includere nel budget sia il gruppo attivo sia quello in arrivo e l'eventuale dequantizzazione.

Il [VAE ufficiale](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/vae/config.json) usa quattro canali immagine e 64 latenti con normalizzazione per canale. Il [codice VAE](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py) va usato per verificare convoluzioni, bordi e tiling. Il vecchio VAE Qwen Image/Wan non è intercambiabile. Tiling richiede una prova di qualità: overlap e blending non garantiscono automaticamente l'identità del decode completo.

## 5. EmbeddingGemma 2: quattro modalità nello stesso spazio

La [model card Google](https://huggingface.co/google/embeddinggemma-2) descrive backbone testuale ed encoder vision/audio caricabili selettivamente, spazio comune a 768 dimensioni, contesto operativo 8192 e troncamenti supportati 512/256/128. Le dimensioni e i limiti dell'API vanno derivati dal contratto del checkpoint, non soltanto dal massimo posizionale del file di configurazione.

Il [modello Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py) è un encoder bidirezionale senza KV persistente fra chiamate, con finestre locali simmetriche e layer full-attention. Variano head dimension, gruppi KV e RoPE per tipo di layer; la scala attention osservata è 1.0. Riutilizzare il decode causale LLM sarebbe semanticamente sbagliato.

Il [grafo llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma-embedding2.cpp) è una seconda implementazione utile per PLE da proiezione, norm, residui e proiezione finale. La [conversione Gemma](https://github.com/ggml-org/llama.cpp/blob/master/conversion/gemma.py) evidenzia anche la diversa convenzione della finestra bidirezionale. Il supporto del grafo non prova da solo il supporto HTTP multimodale; la segnalazione [#30082](https://github.com/ggml-org/llama.cpp/issues/30082) motiva un test contro input media ignorati, senza usarla come prova definitiva dello stato di ogni release.

### 5.1 Pooling, precisione e task

I [moduli del checkpoint](https://huggingface.co/google/embeddinggemma-2/raw/main/modules.json) includono pooling e normalizzazione. Definire esattamente quali posizioni partecipano al pooling, gestendo padding, token speciali e media. Dopo il troncamento Matryoshka va applicata nuovamente la normalizzazione L2. Query e documenti devono condividere modello, revisione e dimensione.

La documentazione del modello richiede BF16 o FP32 ed esclude FP16 per il range delle attivazioni. Anche un percorso quantizzato deve conservare conversioni e calcolo compatibili; una dequantizzazione generica verso FP16 non è un fallback valido.

I prefissi per ricerca, documenti, codice e similarità si applicano al testo. L'API deve offrire un parametro task semplice, registrare la trasformazione e impedire applicazioni duplicate; media e istruzioni non devono essere convertiti silenziosamente in caption o trascrizioni.

### 5.2 Immagini, video e audio

L'immagine richiede preprocessing, posizioni patch, encoder, riduzione/proiezione e inserimento dei soft token nel backbone. Il video riusa il vision encoder su frame campionati, con ordine, conteggi e metadati temporali secondo il processor. Un embedding video non è per definizione la media di embedding di immagini indipendenti.

L'audio richiede decode/resampling, feature extractor e rete audio nativa del checkpoint. Va verificata ogni convenzione di finestre, padding, feature, maschere e durata. Per i video con traccia sonora la politica deve essere esplicita: il video visivo e l'eventuale audio sono componenti distinti dell'input, come previsto dall'oracle scelto.

Ogni elemento della richiesta produce un vettore, anche se combina più modalità. Input oltre contesto vengono rifiutati oppure suddivisi solo su richiesta esplicita, con segmenti e timestamp restituiti. Nessun troncamento nascosto per far entrare il lavoro in memoria.

## 6. Cosa trasferire dai motori esaminati

Le fonti seguenti sono state aperte e confrontate con il codice locale. I riferimenti `main`/`master` sono mobili: costituiscono ricerca preliminare datata, non un lock riproducibile. Il task T00 deve fissare commit e hash degli artefatti prima di qualsiasi porting. Il lock storico della knowledge base non include automaticamente il nuovo supporto dei modelli.

| Fonte primaria e codice | Tecnica trasferibile | Vincolo di trasferimento |
| --- | --- | --- |
| [Unsloth RMSNorm](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rms_layernorm.py) e [GEGLU](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/geglu.py) | Riduzioni FP32, fusione elementwise e trattamento degli stride | Triton non diventa una dipendenza runtime; norm Gemma con `weight+1` non equivale alla norm EmbeddingGemma 2 con scala diretta; GELU esatta non equivale alla variante tanh |
| [Unsloth diffusion](https://github.com/unslothai/unsloth/blob/main/studio/backend/core/inference/diffusion.py) e [memory planner](https://github.com/unslothai/unsloth/blob/main/studio/backend/core/inference/diffusion_memory.py) | Separazione companion, attivazioni, component/group offload | Sono orchestrazione Torch/Diffusers, non kernel nativi DS4; lo header Studio osservato è AGPL-3.0-only, quindi nessuna importazione automatica |
| [GGUF Turbo Unsloth](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF?hardware=quadro-rtx-6000) | Layout misto per tensore e calibrazione sulla schedule Turbo | Il file GGUF è il denoiser, non tutta la pipeline; quantizzazione da qualificare separatamente; numeri pubblicati non sono misure DS4 |
| [llama.cpp MMQ](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu) | Dispatch per shape, quantizzazione attivazioni, budget shared memory | Formati Q8_1/GGUF, compute dtype e padding devono coincidere; nessuna equivalenza automatica con INT8/BnB/AWQ |
| [llama.cpp tiled attention](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-tile.cuh) e [dispatch](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-tile.cu) | Softmax online, tile e head size anche 128/256/512 | Maschere, dtype e accumulazioni vanno adattati; il supporto della shape non certifica il range BF16 di Gemma |
| [stable-diffusion.cpp Qwen 2.1](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md) | Componenti, prefix cache e recipe Turbo | Verificare il codice della revisione fissata; non confondere la recipe base con gli otto passi Turbo |
| [stable-diffusion.cpp Qwen precedente](https://github.com/leejet/stable-diffusion.cpp/blob/master/src/model/diffusion/qwen_image.hpp) | Struttura di runner, binding GGUF e attenzione | Il file esaminato espone anche architettura dual-stream precedente: non usarlo come oracle del single-stream 2.1 |
| [Diffusers group offloading](https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/group_offloading.py) | Gruppi, prefetch e lifetime cross-stream | In C servono eventi e ownership espliciti equivalenti; non imitare soltanto l'ordine delle copie |
| [SGLang layerwise offload](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py) | Insieme residente, layer streamed, eventi, budget host/mapping | Strided/leading vanno misurati sulla singola 3090; risultati multi-GPU non si trasferiscono automaticamente |
| [vLLM-Omni layerwise backend](https://docs.vllm.ai/projects/vllm-omni/en/v0.30.0/api/vllm_omni/diffusion/offloader/layerwise_backend/) | Buffer e metadati dei gruppi, separazione stream di copia | Riferimento architetturale, non dipendenza di produzione |
| [ComfyUI Qwen 2.1](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/qwen_image21/model.py) e [encoder](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/qwen_image21.py) | Secondo confronto per binding, conditioning e forward | Repository e configurazioni evolvono; non usare implicitamente i nodi della precedente versione Qwen Image |

La strategia kernel è riusare prima le primitive DS4 verificate, poi cuBLAS per GEMM BF16/FP32, quindi introdurre kernel custom soltanto dove profilo e correttezza ne dimostrano il bisogno. Encoder e DiT lavorano su molti token: le ottimizzazioni GEMV del decode LLM non sono automaticamente le più utili.

Sulla 3090 non basare il percorso su FP8 Tensor Core, NVFP4, TMA o WGMMA. Formati FP8 possono essere storage con conversione, ma il costo di conversione e il workspace fanno parte della misura. Le schede locali [RTX 3090](../hardware/rtx-3090.md), [mappa problema → fonte](../research/guides/problem-to-source.md) e [checklist di trasferimento](../research/guides/rtx3090-transfer-checklist.md) restano il gate prima di proporre un kernel.

## 7. Gestore comune di memoria ed esecuzione

### 7.1 Oggetti minimi

Servono descrittori per modello, componente, gruppo di pesi, sessione, risorsa e operazione. Ogni risorsa espone proprietario, byte, collocazione, formato, ricostruibilità, costo di trasferimento, ultimo uso GPU e durata prevista. Un piano di esecuzione dichiara dipendenze e working set delle fasi.

Lo stato GPU del modello, i pesi immutabili e il contesto della conversazione hanno ownership separate. Per i pesi immutabili conservati in mmap non serve copiare GPU → RAM prima dello scarico: esiste già una copia autorevole. Stati mutabili richiedono invece checkpoint completo.

Non costruire un framework generale di calcolo: una piccola API C per descrivere fasi e risorse è sufficiente. Non inserire nomi di famiglie nei criteri di eviction.

### 7.2 Ammissione e scelta del piano

La condizione di ammissione di ciascuna fase è:

`residenti non rilasciabili + working set della fase + copie in volo + conversioni + riserva di ripristino pertinente + margine <= budget effettivo`.

Evitare doppi conteggi fra memoria libera globale e allocazioni possedute. Misurare anche RAM anonima, mapping residenti, pinned e spazio SSD necessario ai checkpoint. Una mappa virtuale di 30 GB non equivale a 30 GB di RAM residente. La macchina documentata ha 64 GB fisici e WSL con un limite storico di 48 GB: verificarli di nuovo in T00 senza presumere 96/128 GB.

Il gestore considera prima il piano senza eviction; poi cache ricostruibili e piani progressivamente più invasivi. Fra piani ammissibili valuta latenza totale e costo di ripresa. Deve poter mantenere lo stato LLM e scaricarne soltanto i pesi, quando l'adapter lo consente e il costo stimato è migliore del parcheggio della sessione. La preferenza iniziale per sospendere le sessioni prima dei pesi non deve diventare un vincolo cieco.

Evitare oscillazioni continue fra piani con isteresi e costi misurati. Un'allocazione fallita prima della mutazione può far rivalutare un piano; non ripetere ciecamente un kernel fallito o rilanciare una generazione con RNG avanzato. Per risoluzioni eccedenti la capacità, ridurre la qualità richiede una scelta esplicita del client.

### 7.3 Transazione, cancellazione e recupero

Sequenza concettuale: validate → reserve → quiesce → checkpoint se necessario → execute → release/retain → restore → commit result. Il risultato finale rimane privato fino al completamento del ripristino; eventuali eventi di progresso non sono una conferma di successo.

Sincronizzare i kernel alle frontiere valide, completare copie in volo prima di riusare buffer e conservare checkpoint fino al restore validato. Il rollback ripristina anche configurazione attiva, callback, sampler, RNG e metadati server pertinenti. La cancellazione è cooperativa alle frontiere del lavoro, non una pretesa di interrompere arbitrariamente un kernel CUDA.

Se il restore fallisce per pressione esterna sulla GPU, mantenere checkpoint e identificativi di sessione, esporre lo stato di recupero e impedire richieste su sessioni non pronte. Un errore di device non recuperabile non va mascherato da retry riuscito.

### 7.4 Prefetch e riuso

Il prefetch CPU può precedere l'esecuzione solo per modelli già configurati e con budget host. Un indizio di tool call incompleta autorizza al massimo preparazione cancellabile: non esecuzione né acquisizione di input da percorsi generati dal modello.

Il prefetch GPU deve prenotare memoria e rispettare eventi di ultimo uso. Il caricamento anticipato DiT durante l'encoder e LLM durante il VAE è un'ottimizzazione successiva; va mantenuto soltanto se migliora il tempo complessivo senza peggiorare la fase in corso. Gli stessi otto passi riusano il DiT; i pesi streamed si ritrasferiscono a ogni visita.

Per `W_s` byte streamed e banda utile `B`, il solo volume DiT è circa `8 * W_s`, oltre caricamento iniziale e altri componenti. La stima trasferimento è `8 * W_s / B`, non una latenza totale; la sovrapposizione reale va misurata. Tenere l'intero DiT residente può eliminare molto più traffico di una micro-ottimizzazione del kernel.

## 8. Contratto client e integrazione agentica

Un solo server e una sola configurazione client, con selezione `model`. Conservare Chat Completions/Responses, aggiungere Images generation/edit e Embeddings. La [Images API](https://developers.openai.com/api/reference/python/resources/images/methods/generate) e la [Embeddings API](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create) forniscono contratti riutilizzabili; non implicano identiche capacità per ogni modello.

Per gli embedding multimodali progettare un'estensione DS4 versionata del corpo `/v1/embeddings`, con risposta compatibile quando restituisce vettori. Il contratto OpenAI standard non include input media; un piccolo helper opzionale deve nascondere upload e serializzazione senza aggiungere un secondo motore. Il client testuale OpenAI ordinario continua a funzionare.

Preferire riferimenti a media caricati e riutilizzabili per file grandi, più input inline per casi piccoli. Conservare ordine dei contenuti, batch, task, dimensioni e opzioni di segmentazione. Il dettaglio JSON è da congelare in T01; gli esempi della roadmap sono specifiche proposte, non endpoint già disponibili.

Il server valida le capability: una maschera inpainting o un parametro non implementato non può essere accettato e ignorato. `/v1/models` deve distinguere modelli configurati da effettivamente residenti, senza richiedere al client il controllo della residenza.

ExMekh mantiene il controllo del ciclo tool. Restituire media ID, immagine, recipe e parametri riproducibili; mostrare l'immagine non significa inserirne automaticamente i pixel nel contesto del LLM. L'analisi visiva del risultato è una successiva richiesta esplicita. Non eseguire strumenti automaticamente perché il testo generato assomiglia a una tool call.

## 9. Decisioni numeriche e sperimentali

1. Baseline Qwen Image in BF16 e fixture di componenti FP32 quando praticabili; profili GGUF/INT8 separati dopo la correttezza.
2. EmbeddingGemma 2 BF16/FP32: niente fallback FP16. Verificare le norm effettive prima di riusare kernel di altre famiglie Gemma.
3. Nessuna cache che salta approssimativamente blocchi/step, nessuna compressione lossy della KV LLM nel percorso iniziale. Prefix cache e trasferimenti lossless restano nel perimetro.
4. Tiling VAE e quantizzazione hanno gate percettivi/task-specifici distinti dall'equivalenza di una primitiva.
5. Modifica di sola residenza: continuazione LLM deterministica e stato ripristinato bit-exact sul medesimo percorso numerico. Cross-engine e kernel diversi: tolleranze fissate prima del confronto, dump completi e verifica task-specifica.
6. Nessuna promozione su sola somiglianza visiva, cosine medio o microbenchmark. Il [contratto prestazionale](../performance/agent_performance_contract.md) richiede correctness, benchmark completo, warm-up e almeno cinque campioni per la decisione finale.

## 10. Rischi, evidenze mancanti e scelta dell'ordine

| Rischio concreto | Risposta progettuale | Evidenza necessaria |
| --- | --- | --- |
| Stato globale cambiato durante alternanza | Contesti proprietari e attivazione serializzata | Alternanze ripetute fra famiglie con logits originali |
| Restore impossibile dopo successo compute | Piano include il ritorno, recupero e checkpoint conservato | Fault injection su memoria/SSD e richieste successive |
| GPU libera ma RAM WSL esaurita | Budget congiunto, mmap e pinned bounded | RSS, RAM anonima, fault e I/O su cold/warm run |
| Attention O(N²) a 2K immagini | Tiled/segmentata con maschera corretta | Peak e output su prefissi/reference crescenti |
| Gemma risponde solo al testo ignorando media | Test di sensibilità e retrieval incrociato | Cambiare un solo media cambia il vettore nel verso atteso |
| Quantizzato piccolo ma espanso interamente | Accounting storage/compute distinto | Trace delle allocazioni e dei byte dequantizzati |
| Prefetch rallenta LLM o VAE | Priorità e budget, confronto con seriale | Timeline e latenza completa fuori profiler |
| Recipe base scambiata per Turbo | Manifest del profilo e sigma fissati | Dump timestep/latenti degli otto step |
| Riutilizzo improprio di codice esterno | Registro provenienza/licenze per file | Review prima di copiare o portare codice |

Il percorso critico parte dai confini runtime e dal gestore di risorse, poi sviluppa i percorsi numerici separati e li integra. La prima validazione funzionale usa esecuzione seriale; l'overlap si aggiunge su una baseline già corretta. Il prodotto è completo solo con tutte e quattro le modalità embedding, generazione/editing immagini e ritorno affidabile ai LLM supportati: la [roadmap](../roadmaps/native-multimodal-suite-implementation-20261020.md) definisce i gate senza attribuire PASS a lavoro non eseguito.
