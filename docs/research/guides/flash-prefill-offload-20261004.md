# Flash Next: ricerca sul prefill con esperti in offload

Ricerca del 4 ottobre 2026, riferita a DS4 `e8fb251`, RTX 3090 e GGUF Q2
locale. Le proposte riguardano scheduling, riuso dei pesi e gestione delle
copie. Non sono ottimizzazioni già qualificate: questa ricerca non modifica
runtime, modello o launcher. Contesto Flash 128K e MTP restano requisiti.

## Esito dell’esperimento successivo

La [qualifica delle letture anticipate](../../performance/qwen38-flash-prefill-async-20261004.md)
implementa e mantiene la coda bounded sul percorso CUDA Flash. I chunk
più grandi sono stati scartati per drift dei logit. Le altre proposte
rimangono ipotesi da verificare; la ricerca qui sotto descrive la baseline.

## Diagnosi locale e priorità

Il [report della latenza](../../performance/qwen38-flash-prefill-latency-20261004.md)
misura 184,984 → 126,103 s di prefill HTTP dopo la rimozione del tetto
RAM automatico di 16 GiB. È un campione cold per variante, con 4395 token
input e 32 output. Il profiling della baseline attribuisce il 97,4% del
tempo degli stadi al MoE; non è una nuova profilazione della candidate.

Con il nuovo budget RAM, i contatori dell'intera richiesta riportano ancora
circa 230 GiB H2D, 32,7 GiB da SSD, 19,3 s nelle letture host e 7,0 s nella
selezione delle vittime. **Includono il decode**: non sono misure del solo
prefill. I tempi interni possono sovrapporsi e non vanno sommati fra loro.
Il link GPU Gen4 x2 misura circa 2,74 GiB/s H2D con buffer pinned.

La priorità è trasferire meno byte per prompt e nascondere le attese
SSD/host. Una pipeline perfetta non elimina il costo dei byte obbligatori
su PCIe. Verificare la causa del link x2 rimane utile; ridurre il traffico
serve anche su macchine con collegamento x16.

## Funzionalità già presenti e attese residue

| Punto del codice | Comportamento attuale | Opportunità residua |
| --- | --- | --- |
| `ds4.c:qwen4_graph_moe` | Router originale, liste per esperto e riduzione finale per slot | Conservare questo contratto cambiando lo scheduling |
| `ds4_qwen4_cuda.cuh:qwen4_moe_mm` | Routing riusato mid/down; gruppi fino a 64 esperti, due buffer, stream copy/compute ed eventi | La deduplicazione vale nel chunk corrente |
| `ds4.c:qwen4_graph_forward_tokens` e ciclo prefill | Ogni chunk attraversa tutti i layer prima del successivo | Servire più token per caricamento dei pesi |
| `expert_ram_source` | Lettura SSD sincrona sul thread che prepara i kernel | Coda bounded di letture anticipate |
| `expert_demote` | D2H se manca la copia RAM, poi sincronizzazione | Evitare/differire copie di pesi immutabili quando conviene |
| `expert_transfer` | Fallback senza slot RAM con drain e buffer transitorio | Sorgenti host con lifetime protetto da eventi DMA |
| `expert_gpu_victim` / `expert_ram_victim` | Heap ricostruito al cambio di epoch/capacità | Indici per capacità con identiche regole di scelta |

Riferimenti alla revisione auditata: `ds4.c:59769,59944,60473`,
`ds4_qwen4_cuda.cuh:763,789,1091,1113,1134,2908,2963` e
`ds4_cuda.cu:2284`. Cercare i simboli quando cambiano i numeri di riga.
Raggruppare nuovamente i token o aggiungere semplicemente uno stream
ripeterebbe funzionalità già implementate.

## Letteratura pertinente

- [DuoServe-MoE, v2, aprile 2026](https://arxiv.org/html/2509.07379v2):
  distingue la maggiore densità degli esperti nel prefill dalla sparsità
  del decode. Raggruppa i token per esperto e usa una pipeline CUDA
  comunicazione/calcolo. DS4 possiede già parte dello schema; è utile
  soprattutto una politica diversa fra le due fasi.
- [MoE-Infinity, v3, marzo 2025](https://arxiv.org/html/2401.14361v3):
  tracing delle attivazioni, cache e prefetch. Aiuta a valutare il riuso
  su richieste reali, senza garantire un vantaggio nel prefill cold Flash.
  Il codice corrente del repository è successivo al paper.
- [FineMoE, v2, ottobre 2025](https://arxiv.org/abs/2502.05370v2):
  cache/prefetch guidati da routing e prompt. Il repository EuroSys26
  è dichiarato demo, usa Transformers e un altro modello. La predizione
  aggiunge complessità e non è la prima modifica da integrare.
- [FlashMoE, gennaio 2026](https://arxiv.org/html/2601.17063v1):
  carica gli esperti richiesti una volta per batch nel prefill. La politica
  ML di espulsione riguarda soprattutto il decode. L'attention esatta
  dichiarata non prova logit bit-exact DS4: l'accumulo indicizzato può
  avere un ordine diverso.
- [FlexGen, v2, giugno 2023](https://arxiv.org/abs/2303.06865v2):
  considera GPU, RAM e disco insieme. Il suo obiettivo è throughput batch,
  non TTFT dell'agente. Si trasferisce il metodo di analisi degli accessi;
  compressione dei pesi/KV rimane fuori dal perimetro.

I risultati numerici pubblicati non diventano obiettivi TPS locali:
modelli, precisioni, memoria e workload differiscono.

## Repository e sorgenti ispezionati

Sono stati letti sorgenti a commit fissati oltre ai README. Il
[manifest](flash-prefill-offload-20261004.sources.json) conserva SHA completi,
URL, licenze dichiarate e hash SHA-256 dei 20 file acquisiti. I sorgenti
sono artefatti ignorati in `misc/prefill-research-20261004/`;
nessun codice esterno viene importato.

| Repository | Commit abbreviato | File di ingresso | Spunto trasferibile |
| --- | --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) | `6f32ec070f23` | `src/prefill/prefill.cpp`, benchmark prefill del 28 settembre | Chunk adattivi e scratch proporzionato alla richiesta |
| [Flash su una 3090](https://github.com/DominikBucko/qwen38-flash-next-3090) | `10b24c282f59` | `docs/how-it-works.md`, `compressed_tensors_moe/stream_v2.py` | Letture NVMe fuori dal thread dei kernel, eventi DMA |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | `09fd6da3f54a` | `core/aio/archer_aio_thread*.cpp`, `core/prefetch/task_scheduler.cpp` | Worker persistenti e priorità della domanda reale |
| [Anemll Flash llama.cpp](https://github.com/Anemll/anemll-flash-llama.cpp) | `eafc89e7035a` | `src/llama-context.cpp` | Lettura anticipata del prossimo banco e microbatch separati |
| [FineMoE EuroSys26](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26) | `80717e96e6f5` | `finemoe/runtime/expert_cache.py`, `model_offload.py` | Ranking/prefetch; non il runtime completo |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | `a5d7ad90479c` | `kt-kernel/python/experts.py`, `fp8_layerwise_transport.cpp` | Contratto di chunking/trasporto; formati diversi |

Strata prende parte della cache esperti come scratch e sceglie chunk in
base alla memoria. Il report distingue scheduling e cambiamenti di
arrotondamento; segnala differenze anche cambiando chunk. Il confronto
dello stato GDN non sostituisce i logit completi.
Fonte: [benchmark e gate Strata](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-28-prefill-speed/README.md).

La [documentazione KTransformers](https://ktransformers.net/en/docs/optimization-techniques/layerwise-prefill)
descrive ogni chunk che percorre i layer, con pipeline conversione/copia/
repack. Chunk maggiori riducono i passaggi sui pesi ma aumentano la VRAM
temporanea. Non descrive l'intero prompt per layer senza chunk. Formati,
Marlin e piattaforma Gen5 differiscono da DS4 Q2/sm_86; non copiare le soglie.

DominikBucko usa un worker per il tail NVMe e attende la fine DMA prima di
sovrascrivere il bounce buffer. Prepara copie durante il calcolo degli
esperti residenti. Il checkpoint è W4A16/FP8PLE, il profilo singola GPU
usa KV INT8 e riporta divergenze rispetto a BF16. Si trasferisce lo
scheduling, non KV o GEMM CPU/GPU.
Fonte: [architettura e numerica](https://github.com/DominikBucko/qwen38-flash-next-3090/blob/10b24c282f59348037096a2259d3ba0948f1b202/docs/how-it-works.md).

MoE-Infinity espone worker persistenti e una coda con priorità; la domanda
reale rimuove prefetch duplicati/obsoleti. In DS4 partire da esperti già
scelti dal router. Un pool non rende sicura la cache condivisa esistente.
Fonte: [scheduler](https://github.com/EfficientMoE/MoE-Infinity/blob/09fd6da3f54a114feeb3b67248a68ce759ca7b88/core/prefetch/task_scheduler.cpp).

Anemll pianifica gli esperti selezionati e prepara il banco successivo mentre
calcola quello corrente; separa microbatch GEMM e batch di prefill.
Il codice conferma `std::async`: riprendere il principio con C e un pool
bounded, senza introdurre C++. Gli offset GGUF evitano un nuovo sidecar.
Conservare la riduzione DS4 per slot.
Fonte: [planner ed executor](https://github.com/Anemll/anemll-flash-llama.cpp/blob/eafc89e7035af1f0d7dad500a100a38591ff2477/src/llama-context.cpp).

## Interventi proposti per DS4

La priorità d'impatto non coincide con l'ordine di implementazione: prima
rimuovere attese con cambiamenti piccoli, poi provare il riuso più ampio.
Ogni voce resta un'ipotesi da misurare sul modello locale.

| Priorità d'impatto | Intervento | Ipotesi falsificabile |
| --- | --- | --- |
| Alta | Più token per caricamento degli esperti | Calano H2D byte/token e TTFT cold senza drift |
| Alta | Pipeline SSD→RAM→GPU con worker persistenti | Calano gap GPU e attese host senza più byte letti |
| Media | Politica prefill distinta dal decode | Calano D2H e attese senza peggiorare il decode successivo |
| Media | Indici delle vittime per capacità | Cala il tempo host con identiche vittime e traffico |
| Successiva | Prefetch del prossimo layer | Nasconde miss utili senza saturare PCIe/SSD |

### Pipeline di lettura esatta

Provare una coda di miss già noti dal router, con 2/4/8 worker diagnostici.
Il runtime possiede la cache; i worker riempiono slot riservati e pubblicano
il completamento senza scegliere vittime o mutare mappe. Riutilizzare uno
slot host dopo la fine H2D e uno GPU dopo il consumo del kernel.
Completamento I/O e fine DMA sono condizioni diverse.

`cuda_model_stage_read_from` può chiudere O_DIRECT e cambiare `direct_fd`
nel fallback: non chiamarlo da più worker con lo stesso stato mutabile.
Servono ownership o descriptor separati, fallback coordinato, letture
parziali/errori, drain su cancellazione e teardown.

Accodare lavoro GPU indipendente prima di attendere SSD. Oggi
`qwen4_graph_moe` prepara il routed mid prima dello shared mid; letture
anticipate e calcolo shared possono sovrapporsi preservando dipendenze e
riduzione finale. Verificare numericamente il nuovo scheduling.

### Demozioni distinte per fase

I pesi sono immutabili: la demozione conserva una copia utile, non un
aggiornamento necessario alla correttezza. Valutare D2H+attesa contro
lettura SSD futura e riuso; separare staging prefill e cache utile al decode.
Non disabilitare tutte le demozioni: si potrebbero aumentare le letture del
turno successivo. La precedente prova che rendeva espellibili copie RAM
degli esperti GPU è stata REJECT: serve una nuova ipotesi distinta.

### Riuso su una finestra maggiore

Verificare chunk 256/512/1024/2048 con RAM attuale, ctx128K e stessi prompt.
Il precedente chunk1024 non dava un vantaggio stabile: non promuoverlo
alla cieca. Un cambio può modificare dispatch GEMM, GDN e arrotondamenti
anche lasciando i pesi intatti. Gate sui logit prima di benchmark lunghi.

Se chunk maggiori introducono drift, esplorare una finestra di riuso più
ampia mantenendo le suddivisioni numeriche originali: per layer, eseguire
attention/GDN nelle porzioni originali, conservare input MoE, unire il piano
delle letture e servire le porzioni con gli stessi kernel. È una proposta
DS4 non qualificata dai repository. Richiede buffer e verifiche di causalità,
carry GDN, convolution state, PLE, MTP e steering. Fermare la finestra alle
frontiere checkpoint complete di tutti i layer.

Per contenere lo scratch, valutare mid→down dello stesso gruppo prima del
successivo, mantenendo risultati per `(token, slot)` e riduzione originale.
Non usare somme atomiche per esperto. Il guadagno sui trasferimenti non
è implicito: contatori e benchmark decidono.

### Tuning successivo

Ottimizzare gli heap se la selezione rimane costosa, conservando pinning,
ordinamento e tie-break: confrontare le vittime. Anticipare tutti i 512
esperti spreca banda sui blocchi piccoli; misurare densità e utilità.
Un predittore può anticipare letture, mai sostituire il router originale.

Il [resoconto primario TokenQL](https://github.com/eiomra/tokenql/blob/main/ARTICLE.md)
riporta predizione accurata ma pochi miss utili, coalescing più lento e
16 worker peggiori di code piccole. È evidenza dell'autore su altro hardware,
non un risultato riprodotto o una soglia DS4.

## Soluzioni escluse o a priorità bassa

Escludere pruning/deferral, router modificati, KV INT8, nuove quantizzazioni
di pesi/attivazioni e attention approssimata. Il
[checkpoint albucino](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE/blob/main/README.md)
e il GGUF Q2 locale hanno formati/precisioni diversi. MMQ/Marlin/fused
non sono sostituzioni numericamente trasparenti. Attention, PLE e GDN
non dominavano il prompt profilato. Non usare AMX, DDIO o numeri RTX 5090
come capacità già verificate dell'i9/3090 locale.

## Gate economici prima dei benchmark lunghi

1. Congelare build, modello, budget, cache e link PCIe; distinguere contesto
   allocato da token nuovi elaborati.
2. Testare lifetime, RAM=0/poca RAM, errori I/O, cancellazione e teardown.
3. Primitive e full-vocabulary/teacher-forced su prefissi brevi, chunk e
   frontiere non allineate. Sole copie/scheduling: obiettivo bit-exact.
4. Suite direction cold e continuazione/resume, due campioni per caso:
   scarta direzioni sbagliate, non promuove KEEP.
5. Solo dopo: warm-up e almeno cinque misure senza profiler; TTFT, prefill,
   decode, H2D/D2H/SSD per fase, hit/miss, wait, VRAM/RAM, mediana/p90/CV.
6. Checkpoint sistema/skill/turno, MTP verify/rollback e Qwen27: logit,
   token e assenza di replay. Percorsi non testati restano NOT_VERIFIED.

La matrice finale distingue prompt breve cold, prefisso caldo, resume e
input da 8K/32K/128K token. Non eseguirla tutta per scegliere una direzione.
Non cambiare payload checkpoint, contesto o MTP per migliorare i numeri.
Più utilizzo GPU senza ridurre la latenza della richiesta non basta.
