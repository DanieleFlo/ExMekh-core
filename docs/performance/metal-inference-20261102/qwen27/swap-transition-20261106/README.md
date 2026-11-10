# Qwen27: swap nella transizione verso Image

Baseline `60e9400`, Apple M5 Pro 24 GiB, 6 novembre 2026. Riserva host
`DS4_METAL_HOST_RESERVE_BYTES=0`, cache automatica, contesto 8192, prefill64,
target-only, stessa recipe Image256/otto passi e stessi cinque input Gemma
dello stress T08. Nessun cambiamento a pesi, precisione, stato o criteri.

Aggiornamento9novembre: stress Qwen27 PASS50cicli+warmup nello stesso
processo, crescita swap0 e chiusura regolare. Le prove lunghe salvano ogni
ciclo validato e riprendono con `qualify_metal_server.py --resume`;
arresto/ripresa reali verificati. Pin e risultati in `resume-verification.json`.
Il [lavoro residuo T08](../../../metal-apple-silicon-20261031/t08-20261106/remaining.md)
conserva gli altri gate aperti. Le misure del9novembre erano senza commit;
l’integrazione del10novembre è descritta nel riesame.

## Riproduzione e profilo prima del codice

La prova T08 originaria si ferma nel warmup con crescita swap globale
1809254973B. La nuova riproduzione usa il medesimo primo ciclo tramite
`qualify_suite_release.py --bounded --cycles 1`: è diagnostica, non uno
stress50cicli. Si ferma nuovamente con `SKIP_RESOURCE_LIMIT`,47.697s,
crescita swap1571228222B, server exit0, nessun cleanup forzato.
I raw sono in `build/metal-t08-review-20261106/swap-investigation/baseline/`.

Il profilo separato inizia dopo il checkpoint della risposta. Il worker
è in `ds4_qwen_image_encode` e `ds4_weight_group_load`, soprattutto nella
copia mmap→Metal di `DS4MetalNumeric.copyTo`. La cache SSD del LLM viene
rilasciata nella successiva chiusura del server; non si attribuisce il
picco al kernel DiT. Una seconda prova diagnostica registra il piano prima
dell'esecuzione tramite le API pubbliche `ds4_qwen_image_job_plan` e
`ds4_resources_dump`, senza cambiare il planner o il suo risultato.

## Fonti rilette per questa discrepanza

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py`: cache conv e
  ricorrenza GDN, percorsi chunk/recurrent e frontiere FP32. L'allocatore
  non autorizza a cambiare o ricostruire diversamente la sequenza numerica.
- Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
  `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`:
  offload text encoder→transformer→VAE e dtype espliciti. Il lifecycle è
  trasferibile dopo salvataggio verificato, non il modello di RAM CUDA.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/arena_budget.h`, `window.cpp`, `stream.cpp`, `stream.h`,
  `metal/tests/arena_budget_test.cpp`: quota anonima distinta da pagine
  mmap, floor file cache, costo dell'intero processo, fill incrementale,
  `F_NOCACHE` per il trasporto SSD. Le riserve7GB/15%/40%, mlock e layout
  MoE non vengono importati. Non copre la suite Image/Gemma di DS4.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/llama-mmap.cpp`, `ggml/src/ggml-metal/ggml-metal.cpp`: mmap shared,
  advisory distinti dal possesso dei buffer e sincronizzazione prima del
  rilascio. Il backend non certifica il nostro picco anonimo né il piano
  dei tre modelli; nessun criterio numerico viene importato.
- XNU `f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`,
  `osfmk/vm/vm_pageout.c:vps_calculate_filecache_min` e selezione delle
  code: floor dinamico e preferenza per anonimi quando i file sono sotto
  il floor o la coda inactive_external è troppo piccola. `bsd/vm/vm_unix.c`
  e `osfmk/kern/host.c`: contatori correnti e cache delle statistiche Mach.
  I campioni globali restano misure; non si scrive alcun sysctl.
- [Apple, setPurgeableState](https://developer.apple.com/documentation/metal/mtlresource/setpurgeablestate(_:)):
  `Empty` è appropriato solo per contenuti definitivamente scartabili,
  dopo i consumer. Il rilascio esplicito delle regioni SSD esiste già;
  non spiega da solo l'ammissione che mantiene i pesi vivi.

Ricerca locale riletta: Q-P23/Q-P25 per restore atomico e memoria restituita,
I-P04/S5 per il floor XNU, I-P05 per prefetch16MiB; `ds4_lifecycle.inc`,
`ds4_resources.c`, `ds4_executor.c`, `ds4_qwen_image_pipeline.c`,
`ds4_bundle.c`, `ds4_weight_cache.c`, wrapper Metal legacy e Numeric.
I-P04 protegge già il sampler Numeric; il legacy SSD usa ancora floor0.
Le vecchie prove con riserva2GiB non provano la sicurezza del profilo zero.

Prima di scegliere l'intervento si confrontano: costo delle pagine file
dell'upload, esattezza del credito di reclaim, disponibilità del legacy
SSD e piano/vittime effettivamente selezionati. Non si introduce un nuovo
margine fisso né si alza la soglia swap256MiB. Gli hash delle fonti e delle
prove vengono conservati nel manifest della campagna.

## Intervento scelto prima dell'implementazione

Il secondo replay conferma `variant=10,victims=0`: picco device2122870752B,
host1904672B contro capacità host2128330752B. Restano soltanto3555328B
dopo le due domande. Il piano mantiene quindi l'intera cache Qwen e conta
i buffer di destinazione, ma non il working set delle pagine mmap lette
dall'upload dell'encoder. Il profilo mostra proprio quella copia mentre
inizia lo swap. Il floor nel sampler Numeric non prenota i nuovi file
che la copia deve rendere attivi durante l'esecuzione.

Ipotesi da verificare: rendere esplicito il costo transiente della sorgente
in memoria unificata fa scegliere un piano che rilascia i pesi ricostruibili
quando le due copie fisiche non stanno insieme. Il backend Numeric espone
il costo attraverso la propria API; il planner Image lo aggiunge al picco
host per componente, usando il gruppo massimo realmente caricabile.
Gli upload sono sincroni e sequenziali: non si sommano tutti i pesi del
modello. CPU/CUDA conservano i rispettivi costi esistenti. Non è una nuova
allocazione host né un margine fisso: è la domanda del trasporto mmap→Metal.

Alternative: alzare la soglia swap o reinserire2GiB nasconderebbe il costo;
forzare sempre l'espulsione impedirebbe combinazioni che entrano davvero;
estendere il floor al legacy SSD resta una verifica distinta, non una
spiegazione automatica del piano Numeric già floor-aware. Cambiare il
trasporto in pread/F_NOCACHE richiede un audit e una misura propri.
Prima test del costo/backend e piano con input isolati, poi replay breve,
gate indipendenti e50cicli sul prodotto, se i guard consentono la prova.

## A — costo dell'upload isolato

I test host del costo pubblico e dell'ammissione passano. La prima fixture
di reclaim era incompleta (mancavano quiesce/restore obbligatori): corretta
la fixture, senza allentare il validatore. Il replay reale con il solo costo
aggiunto resta `SKIP_RESOURCE_LIMIT`,45.8s e crescita swap1122752267B.
Non è una correzione qualificata; nessuno speedup viene dedotto dal delta
fra due campioni globali. Il piano del candidato viene registrato a parte.

## B — nuovo confronto del sampler legacy, prima del codice

Riletti nuovamente Strata `arena_budget.h:footprint_cap/arena_budget` e
`stream.cpp:make_arena` al pin sopra, XNU `vps_calculate_filecache_min` e
la scelta della coda anonima, wrapper `ds4_gpu_metal_memory_query` e
`DS4MetalNumeric.refreshAvailability`. Il minimo protetto riguarda la
competizione fra anonimi e file, anche quando gli anonimi vengono riempiti
da `pread/F_NOCACHE`; non è limitato alle copie da mmap. Strata distingue
infatti l'arena anonima dal pack residente file-backed. Le sue quote fisse
e mlock restano incompatibili con questa richiesta.

Il sampler legacy SSD può continuare a crescere fino a14.3GiB usando
floor0; il planner Numeric riceve poi una lettura con floor escluso. Il
secondo non può restituire retroattivamente la RAM già impegnata dal primo.
Scelta B: usare la stessa disponibilità sopra il floor per i buffer
anonimi dello streaming. Il percorso residente mmap mantiene la propria
contabilità file-backed. La funzione del sampler viene nominata per il
tipo di storage, non per il metodo con cui i pesi sono stati letti.
Nessuna percentuale o riserva aggiuntiva: si usa ancora tutto il residuo
ammissibile con margine0. Verificare B insieme ad A, poi ablation di A per
non conservare costi inutili se B è sufficiente; nuovi gate completi prima
di promuovere il prodotto.

## Esito B e scelta C prima del codice

A+B resta `SKIP_RESOURCE_LIMIT`,46.6s, crescita swap1416951235B. Nel
diagnostico A il piano si limita a ridurre la cache Image: device1583344032B,
host520001120B, vittime0. La copia continua a caricare successivamente molti
gruppi da mmap; riservare un gruppo non limita le pagine file che restano
attive dopo le copie. I due interventi non hanno quindi chiuso il problema.

Nuovo confronto C: riletti `Strata stream.cpp:open/read/load_dense` allo
stesso pin (F_NOCACHE, pread bounded nella destinazione), wrapper ggml
`llama-mmap.cpp` per advisory contro letture autorevoli e ordine della
sincronizzazione; DS4 `ds4_file_reader.h`, reader SSD Qwen, bridge Numeric
e `ds4_weight_group_load`. Transformers/Diffusers mantengono esattamente
gli stessi tensori: cambiano soltanto il trasporto dei byte e il costo di
memoria. [Apple fcntl(2)](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
documenta F_NOCACHE; non è un'opzione di approssimazione numerica.

Scelta: il bundle conserva identità, selezione, descriptor e publication;
il backend Metal legge gli intervalli verificati direttamente nello storage
shared con pread/F_NOCACHE e chunk bounded, senza faultare un secondo
working set mmap. Hook opzionale fuori dal layout v1 del backend bundle;
CPU/CUDA mantengono il percorso mmap/staging esistente. UNSUPPORTED è
ammesso solo prima di scrivere; letture fallite non pubblicano residenza.
EOF, errori e fence continuano a richiedere cleanup/retry espliciti.

Il costo A viene ritirato: con trasporto diretto non esiste quella copia
mmap da prenotare. B resta candidato separato fino alla verifica. Non si
usa madvise come garanzia di RAM restituita, non si sommano riserve
arbitrarie e non si allarga il monitor. Prima byte/range/errori isolati,
poi gli stessi gate completi e la campagna50cicli.

## Esito C e nuova indagine D sulle pagine inactive

C+B supera byte BF16/FP32, offset17, coda oltre8MiB, EOF, failure/fallback
e12test bundle. Il profilo reale conferma `readTo/pread`, quindi il percorso
diretto è effettivamente usato. Tuttavia il ciclo si ferma ancora48.7s con
crescita swap3445683651B. Il trasporto da solo non spiega la capacità
fisica sovrastimata; C e B vengono ritirati dal prodotto e conservati come
patch/binario diagnostici. Non si conserva una nuova API senza beneficio
dimostrato per questa correzione.

Nuovo audit D prima del codice: XNU allo stesso pin, `vm_pageout.c` calcola
`inactive_external_count = vm_page_inactive_count - vm_page_anonymous_count`
prima di scegliere file o anonimi. `host.c` esporta inactive totali e
internal_page_count (anonimi attivi+inattivi); il contatore esatto della
sola coda inactive anonima non è esposto da questi sysctl pubblici.
Il sampler DS4/Strata limita inactive con il totale file-backed, ma ciò
non impedisce di contare pagine anonime quando i file sono invece attivi:
`min(inactive,file_total)` non è la coda dei file inactive.

Scelta D: limite inferiore verificabile per i file inactive =
`max(inactive_total - anonymous_total, 0)`, poi il limite file/floor già
esistente. Si usa il massimo fra anonimi Mach e contatore corrente per non
ridurre la deduzione con uno snapshot cached; free/speculative restano
correnti. Può escludere anche anonimi attivi, ma non vende anonimi inactive
come RAM liberabile senza compressione. Non è un margine fisso, non cambia
il cap richiesto, e la cache può crescere quando memoria reale viene
restituita. Le assunzioni FP32/cache di Transformers e dtype/offload
Diffusers non cambiano. Strata resta confronto per il costo anonimo, non
un oracle della coda XNU; ggml lascia tali pagine al kernel. Si verifica D
da solo, mantenendo il trasporto e l'ABI originali, prima di qualsiasi bundle.

## Esito D e prova isolata E prima del codice

D da solo si ferma48.0s, crescita swap1333065155B. La cache passa da7.35
a14.3GiB durante il forward; il nuovo limite alle pagine inactive non chiude
il picco e non viene integrato in questa correzione.

Nuovo isolato: dieci allocazioni da256–265MiB, scrittura identica, rilascio
tramite API Numeric pubbliche. Con sola scrittura CPU la memoria ritorna
al footprint iniziale~8MB. Dopo un consumer Metal completato, la medesima
liberazione porta `currentAllocatedSize` a720896B ma lascia il footprint
a323–333MB, contro7.8MB iniziali. Il processo non possiede più il buffer
secondo il registry, ma il driver/comando completato trattiene le pagine.
Raw e probe sono in `release-before/` e `release-gpu-before/`; entrambi
PASS con crescita swap0, senza modello e cap3GiB.

Riaudit E: wrapper Strata/ggml di allocazione, sincronizzazione e free ai
pin sopra; Q-P25 `qwen4_stream_discard` e i suoi gate locali, che distinguono
regioni ancora riutilizzabili da storage definitivamente eliminato;
contratto Numeric di completamento sincrono e `destroy_storage` dopo fence.
Apple documenta `MTLPurgeableStateEmpty` per dati definitivamente scartabili.
La differenza è osservata nel lifecycle, non in un kernel o nel formato.

Scelta prima del codice: nel solo proprietario Numeric Metal, marcare
Empty il buffer che sta per essere distrutto dopo il completamento dei
consumer. Non svuotare buffer vivi o comuni, non anticipare i fence e non
attribuire credito a RAM che il driver trattiene. A–D sono ritirati;
verificare E da solo sullo stesso isolato e sul ciclo reale originale.

## Esito E e indagine F sulla retention dei comandi

E non riduce il footprint dell'isolato GPU e il ciclo si ferma45.7s,
crescita swap1561591808B. Non viene conservato. Il segnale è dunque la
retention successiva al consumer, non il solo stato purgeable del buffer.

Riaudit F prima del codice: Strata `metal/src/mtl_rt.mm:Context::run`
usa `commandBufferWithUnretainedReferences`, anche nel percorso sincrono
commit/wait. Il percorso differito conserva invece esplicitamente i
proprietari/graveyard fino al fence. ggml `ggml-metal.cpp` distingue
operazioni async e synchronize prima di free; tale scheduler non è il
contratto sincrono Numeric e non viene copiato. Il fetch aggiuntivo
`ggml-metal-context.m` al pin non è disponibile, nessun contenuto dedotto.
Il runtime legacy DS4 espone già lo stesso tipo di comando come diagnostica,
ma non viene modificato.

[Apple retainedReferences](https://developer.apple.com/documentation/metal/mtlcommandbuffer/retainedreferences)
richiede che l'applicazione mantenga i riferimenti fino al completamento.
Numeric possiede già tutti i root in `_buffers`, i dummy e le pipeline nel
contesto, e i temporanei MPS restano vivi fino a `completeCommand`, che
attende sempre. Slices/viste non trasferiscono proprietà; nessuna
distruzione è consentita prima del fence del registry. Non si cambia
alcun dtype, kernel, ordine o confine BF16 delle implementazioni HF.

Scelta F: una factory privata del solo owner Numeric per comandi senza
retention aggiuntiva, usata sia dal dispatch dei kernel sia da MPS. I
proprietari esistenti continuano a coprire l'intero intervallo GPU. Prima
confronto dell'isolato identico, poi regressioni e gate completi; se il
footprint non cambia anche questo candidato non spiega la discrepanza.

## Esito F e costo fisico completo G prima del codice

F non cambia il footprint residuo immediato: ritirato. Una pausa soltanto
diagnostica di100ms nell'isolato lo riporta da323322648B a54755928B. La
restituzione fisica è differita rispetto a `free/currentAllocatedSize`,
anche dopo il consumer completato; non viene inserita alcuna sleep nel
prodotto e non si assume che il ritardo abbia un limite universale100ms.

Riaudit G: riletti i percorsi alloc/free e commit/wait Strata, il registry
DS4 che non accredita direttamente global_free al rilascio, le copie
Numeric e il planner. Il costo A contava solo il gruppo sorgente mmap;
il refill può sovrapporre anche lo storage del predecessore ancora in
ritiro. Queste sono due domande fisiche distinte, assenti dal limite del
solo cache resident. ggml/Strata non certificano la restituzione immediata
del nostro storage Metal. Semantica/dtype HF e i fence rimangono invariati.

Scelta G: il backend espone il workspace fisico del trasporto dei pesi,
parametrizzato dal gruppo massimo: una sorgente mmap più un predecessore
in ritiro per Metal, zero extra per i backend con i propri costi esistenti.
Il planner Image prenota tale domanda host insieme a cache/attivazioni,
per componente e soltanto se mancano pesi. Non alloca padding, non impone
un cap cache globale e non forza sempre l'espulsione: decide il planner
con costi espressi in byte del modello. A–F restano ablation scartate.
Questa stima resta da validare sul picco reale e sullo stress prolungato;
un PASS dell'isolato non certifica da solo il50cicli.

## G rifiutato per ammissione; H sul budget condiviso

G si ferma con `NOMEM` alla prima richiesta Image,41.7s, crescita swap0.
Il piano prende il massimo per location e poi considera il restore dei
pesi Qwen: aggiungere il grande workspace in HOST lo somma al massimo
DEVICE del restore, benché le due fasi non siano simultanee. È un rifiuto
di ammissione, non la qualifica della pipeline.

Riletti prima di H `phase_peak`, `ds4_execution_plan_build` e la riserva
transazionale, il budget unificato, i costi `linear_workspace` Metal e i
contratti upstream già fissati. H esprime il workspace nel budget del
dispositivo shared che lo deve sostenere, come le altre domande Metal:
si prenota il massimo fra fase Image (cache+trasporto+workspace) e restore,
senza sommare domande di fasi disgiunte. È un bound del working set, non
una rivendicazione di storage Metal allocato per le pagine sorgente.
CPU/CUDA conservano zero extra e la propria contabilità separata.
La nuova prova diagnostica distingue piano, rilascio, riserva ed esecuzione.

## H: piano corretto, campione stale dopo il rilascio

Il diagnostico H ammette `variant=1,victims=2` (pesi Qwen), picco device
15479677952B e host1904672B. Prima del rilascio device allocato16315572872B,
dopo1257494784B; tuttavia `global_free` resta2745090048B device e1712390144B
host in entrambi i dump. La successiva riserva fallisce con NOMEM.
Il ciclo40.9s ha crescita swap0 e non esegue l'encoder. È una discrepanza
diversa dal picco originario: la capacità misurata non viene aggiornata
dopo il reclaim del backend legacy.

Riaudit I prima del codice: `ds4_resources.c:release` vieta giustamente
di accreditare global_free dai byte logici; `DS4MetalNumeric.releaseAddress`
rilegge già la disponibilità, `ds4_gpu_legacy_model_release` e
`lifecycle_suspend` invece no. Strata `window.cpp:avail_now/recheck` e
`stream.cpp:make_arena` al pin fissato rivalutano prima del riempimento;
ggml sincronizza prima del rilascio ma non certifica il nostro sampler.
I contratti HF di stato e rounding restano immutati.

Scelta I: i proprietari Metal rinfrescano il registry dopo aver eliminato
pesi/sessioni, mantenendo intatti limiti, margini e lease. Nessun credito
aritmetico inventato nel registry. Una lettura fallita conserva lo stato
di rilascio effettivo e invalida la capacità tramite il reconcile esistente,
così l'ammissione fallisce e il recovery mantiene gli owner corretti.
Il percorso CUDA non cambia. Verificare H+I sugli stessi input e sul prodotto.

## H+I e scelta J del piano a costo minimo

Il ciclo H+I torna a SKIP_RESOURCE_LIMIT45.8s/+1415640515Bswap. La disponibilità
globale varia e può nuovamente ammettere la variante minima senza espulsioni;
il costo aggiuntivo non risolve in modo robusto questa priorità. H viene
ritirato, mentre I corregge il rifiuto stale realmente isolato e resta.

Riaudit J prima del codice: `phase_peak` include già byte caricati, costo
dei miss DiT ripetuti e costo di restore delle vittime. Il planner però
preferisce incondizionatamente qualsiasi variante senza espulsioni anche
quando il suo costo è119GB contro44GB di quella che rilascia pesi Qwen
ricostruibili. Strata `stream.cpp:make_arena` e `window.cpp` gestisce la
quota del working set attivo; Diffusers ha offload per fase. Non si
importano layout, quote o criteri numerici. ggml async/sync non decide la
priorità di questa suite. I contratti di stato/checkpoint DS4 e HF
rimangono gli stessi e vengono verificati dopo la transizione.

Scelta J: in memoria unificata il planner confronta i costi completi già
calcolati, compreso il reclaim, e conserva l'isteresi. Un piano con
espulsioni deve quindi guadagnare il proprio costo, non viene sempre
forzato. Il registry espone la proprietà unified tramite API pubblica;
i backend con RAM host/device separate mantengono la precedenza storica
senza espulsioni. Non c'è un flag utente o una seconda semantica del modello.
Il trasporto, i workspace e i kernel tornano originali: A–H sono ablation,
la correzione da verificare è soltanto scheduling J più refresh I.

## J+I: rilascio fisico differito; K prima del codice

J+I seleziona il piano con reclaim ma si ferma NOMEM43.8s, swap0. Il
diagnostico separato conferma che Metal allocated scende a1259552768B;
la capacità host effettiva cresce invece da7852654592B a12631965696B
dopo100ms e17430495232B dopo200ms, fino a17.7GB. La contabilità logica
non è più stale: il kernel restituisce fisicamente le pagine in differita.
Dopo la sola attesa diagnostica e il refresh, la riserva riesce.

Riaudit K: Strata `window.cpp:avail_now` e fill/recheck dell'arena,
ggml synchronize/free, XNU e i nuovi campioni, `release` del registry,
callback dell'executor e refresh Numeric. Il completamento GPU non
certifica il ritorno immediato della RAM. Non si accredita la capacità
prevista, non si ripete una fase numerica e non si cambia la semantica HF.

Scelta K: solo dopo vittime effettivamente rilasciate e NOMEM della riserva,
l'executor usa un callback del proprietario per rinfrescare capacità e
riprova l'ammissione per al massimo1s, controllando cancellazione a ogni
passaggio. Il poll1ms non è una pausa incondizionata e non alloca RAM;
appena entra la domanda misurata, si procede. UNSUPPORTED non attende,
gli errori di telemetria vanno al recovery normale. Callback opzionale:
CPU/CUDA e client senza refresh conservano il comportamento precedente.
Il bound1s limita il costo di un reclaim che non restituisce capacità;
non è un'ipotesi che ogni driver completi entro quel tempo. Nuove prove
sul prodotto senza il secondo diagnostico artificiale, poi stress50.

## Bundle finale candidato: C + I + J + K

Il diagnostico con attesa dimostra che la riserva è possibile dopo il
reclaim, ma l'upload mmap della variante residente raggiunge nuovamente
il limite swap durante Image (61.8s totali). È una nuova fase rispetto al
tentativo che teneva la cache Qwen: adesso il picco viene dalla copia del
componente residente, non dalla mancata espulsione del LLM.

Riletti prima del bundle Strata `stream.cpp:open/read/load_dense`, wrapper
e free/commit in `mtl_rt.mm`, ggml mmap/Metal ai pin fissati e tutti i test
del trasporto C. L'upload mmap di un componente residente porta in memoria
anche le sue sorgenti oltre alle destinazioni; il solo gruppo massimo
non vincola il working set file accumulato dal preload. F_NOCACHE/pread
diretto evita quella seconda copia fisica, senza cambiare un byte numerico.
Il precedente C falliva perché manteneva ancora il LLM: non ne provava
l'inutilità in una transizione che assegna la RAM al modello attivo.

Scelta: C (trasporto bounded, stesso test byte/range/EOF/fallback già
passato), I (refresh dopo rilascio), J (costo completo su RAM unificata),
K (ammissione rinnovata solo se la restituzione fisica è ancora in corso).
Il sampler e tutti i workspace/kernel restano originali, nessuna riserva
fissa e nessuna soglia modificata. CPU/CUDA conservano trasporto e
precedenza storici; il nuovo refresh UNSUPPORTED non introduce attese.
Prima un ciclo sul prodotto finale, quindi gate numerici indipendenti,
regressioni e50cicli senza profiler; niente promozioni dai diagnostici.


## C+I+J+K e scelta L: ammissione dopo la cache opzionale

Il bundle si ferma ancora SKIP_RESOURCE_LIMIT47.6s, crescita1800740536B:
il trasporto diretto è effettivamente usato (sample `readTo/pread`) ma il
preload dell'encoder viene dimensionato con la capacità prevista prima del
rilascio fisico. Il floor XNU dei file cambia quando i pesi GPU non sono più
wired. Il picco di restore include inoltre la precedente cache LLM opzionale.
J/K non risolvono questo errore di fase e vengono ritirati dal prodotto.

Riaudit L prima del codice: Transformers ca8c871ede62b521a75a2ddfc5e1968ac9c8e097
`src/transformers/models/qwen3_5/modeling_qwen3_5.py` separa lo stato ricorrente/KV
dai pesi immutabili. Diffusers aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e
`src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py:183` usa sequenza
text_encoder→transformer→vae. Strata46c152a1310f77cd2f0ce2c7cd213962df14fe83
`metal/src/engine/window.cpp` (avail_now/recheck), `stream.cpp` (make_arena)
assegna RAM al lavoro attivo; le sue quote fisse, C++ e layout non si importano.
llama.cpp dde879bc4f954b809892fd0d660515f42536394a
`ggml/src/ggml-metal/ggml-metal.cpp` sincronizza prima del free: il fence non
certifica il ritorno fisico delle pagine. XNU f6217f891ac0bb64f3d375211650a4c1ff8ca1ea
`osfmk/vm/vm_pageout.c` ricalcola filecache_min con lo stato wired/anonymous.
Hash e copie dei percorsi restano in sources.json. Ricontrollati localmente
`ds4_qwen4_metal_stream.inc:ds4_gpu_qwen4_stream_resize`,
`ds4.c:qwen_metal_stream_rebalance`, `ds4_lifecycle.inc` e Q-P23/Q-P25.
Resize0 conserva reader/layout e può ricrescere prima del forward; lo stato
sessione non dipende dagli indirizzi dei pesi cached.

Scelta L: API engine stretta per svuotare solo la cache SSD ricostruibile,
prima di creare il piano del lavoro nativo. Il backend attende il calo misurato
del footprint (massimo1s, nessuna pausa se non possiede storage), poi riconcilia
la capacità reale. Il timeout non accredita RAM prevista. Il planner Image
rimane quello originale e legge la capacità dopo il cambio di fase. Non si
espellono modello, checkpoint o KV; la cache torna dinamica al prossimo forward.
CPU/CUDA/residente restituiscono UNSUPPORTED prima di modificare alcunché.
C e I restano candidati nel bundle, J/K rimossi per limitare la responsabilità.


## L e nuova discrepanza M: floor differito e pagine speculative

L rilascia effettivamente14GiB di cache: footprint16.56→2.06GB in~430ms,
ma il ciclo si ferma SKIP_RESOURCE_LIMIT46.8s. Durante il preload encoder
il floor rimane128831pagine anche dopo il rilascio, poi sale458930solo
quando riparte vm_pageout_scan. Non è una cache di sysctl: il sorgente
calcola il minimo soltanto dentro quel loop. Attendere il solo free non
rende corrente il floor. Inoltre pageable_external include le speculative,
mentre il sampler le somma al free e le lascia nel bound delle inactive:
con statistiche Mach cached può accreditarle due volte.

Riaudit M prima del codice: stessi pin HF/Diffusers (nessuna operazione
numerica cambia), Strata arena_budget/window/stream e ggml mmap/Metal.
Le riserve statiche di Strata non si importano. Nuovi sorgenti XNU
f6217f891ac0bb64f3d375211650a4c1ff8ca1ea: `osfmk/vm/vm_page.h:1419`
definisce AVAILABLE_NON_COMPRESSED_MEMORY; `osfmk/vm/vm_resident.c:166–184`
include speculative nei pageable; `osfmk/vm/vm_pageout.c:2211–2242,3247–3250`
calcola il floor solo durante la scansione con10/27su macOS.
`bsd/kern/kern_descrip.c:3588–3620` conferma flag F_NOCACHE condiviso dal dup;
`bsd/vfs/vfs_cluster.c:4582–4650,4765,5183` conferma che NOCACHE disabilita
già readahead e scarta i nuovi UPL. Aggiungere F_RDAHEAD non spiegherebbe
questa discrepanza. Riletti I-P04/S5, Q-P23 e policy/test locali.

Scelta M: Numeric conserva il maggiore tra floor misurato e floor che XNU
ricalcolerebbe sui contatori pageable/free correnti (10/27, senza contare
speculative due volte; throttled escluso). Il calcolo non spende un floor
stale ridotto da un precedente modello wired. La disponibilità rimane free
+speculative+inactive file oltre il floor, ma esclude speculative dal bound
file per non duplicarle. Il legacy mantiene floor0; la deduplicazione delle
speculative corregge lo stesso conteggio anche lì. Nessun valore sysctl è
modificato, nessuna riserva fissa o soglia nuova. Validare aritmetica, quindi
ciclo reale prima di altri gate. C/I/L restano candidati, J/K ritirati.


## M, isolamento del trasporto e scelta N

M non chiude ancora il ciclo: SKIP44.4s; il diagnostico separato ammette
variante1/vittime0/picco14607261696B contro host15056060416B. Il rilascio
Qwen funziona, ma il preload produce contemporaneamente pagine file e
anonime. Il trace separato conferma13892143104B di trasporto C con flag
F_GETFL40000 (F_NOCACHE): non è fallback mmap o flag perso nel dup.
Non c'è evidenza per attribuire il costo a un altro processo.

L'isolato1GiB mostra lettura allineata F_NOCACHE senza crescita dei file;
lettura non allineata (+4097file/+32destinazione) cresce invece di57345pagine
file/speculative (~896MiB), oltre alla destinazione anonima. Il secondo
passaggio già cached non è un controllo freddo e resta separato nei raw.
I file sono `load-*.log`, sorgenti/binary diagnostici conservati negli ignored.

Riaudit N prima del codice: XNU f6217f891ac0bb64f3d375211650a4c1ff8ca1ea
`bsd/vfs/vfs_cluster.c:cluster_read_direct` (offset_in_file/iovbase,
misaligned e fallback cluster_read_copy) tratta l'intero vettore non
allineato tramite UBC anche con NOCACHE. Non basta aver scelto pread.
Strata46c152a `metal/src/engine/stream.cpp` e `mtl_rt.mm` separano arena
allineata, trasporto e GPU; GGUF/layer/routed non sono i range BF16 DS4B.
llama.cpp dde879bc `src/llama-mmap.cpp` e wrapper Metal conservano invece
pagine mmap e fence, senza promettere I/O diretto dei nostri range.
Transformers ca8c871 e Diffusers aea7b70 nei percorsi fissati restano la
semantica invariata: si copiano gli stessi byte prima di renderli residenti.
Riletti reader locale, bridge Numeric, gruppo pubblicazione/identity e I-P05.

Scelta N: buffer host allineato8MiB riutilizzabile posseduto dal backend
Numeric, registrato come costo host prima di pianificare. Ogni range viene
letto da un offset pagina allineato e copiato nel relativo offset Metal;
solo la pagina finale EOF può richiedere I/O parziale. Non si legge oltre
il file, non si pubblica un gruppo parziale, il descriptor resta bundle-owned.
CPU/CUDA conservano il fallback esistente. I test byte/range/EOF vengono
ripetuti prima del ciclo completo; non si modifica alcuna soglia o kernel.


## Primo ciclo completo del candidato N

PASS160.186s controller,146.349s ciclo: Image57.450s (load2.016/2.172/0.125s,
encoder1.771s,DiT38.385s,VAE10.299s), embedding58.127s e ritorni Qwen corretti.
Crescita swap globale0, pressione massima2, minimo host2573680640B;
riserva0, soglia256MiB/2campioni immutata. Il residuo host è misurato,
non una riserva imposta. Nessun cleanup forzato, server exit0.
Profilo diagnostico conservato; non è lo stress50 né uno speedup appaiato.
Fonte/binary specifici nel report raw `aligned-transport-candidate/server/report.json`.
I precedenti candidati falliti rimangono ablation, non risultati promossi.
Ora gate indipendenti: Flash/Qwen32+8 includono trim API e ricrescita,
confronto di tutti i logits e payload con il controllo congelato; Image512,
Gemma e regressioni, poi50alternanze senza profiler.


## Confini del candidato da qualificare

- Bundle possiede descriptor, hash, range e pubblicazione: il trasporto opzionale
  non espone indirizzi backend e non cambia il layout v1 del weight_backend.
- Numeric possiede storage, contatori e capability; UNSUPPORTED prima di toccare
  byte conserva il trasporto CPU/CUDA precedente. Metal possiede staging8MiB
  allineato, la sua registrazione HOST e il rilascio. Il costo è presente prima
  del piano, non viene inventata una nuova riserva del sistema operativo.
- Engine possiede cache e sessioni: la suite usa soltanto trim_cache/reclaim.
  Il backend attende al massimo1s su footprint effettivo, senza credito previsto;
  il successivo forward riammette la cache dinamica rispettando il ceiling.
- Registry/executor e kernel sono tornati ai contratti precedenti: J/K sono
  ablation ritirate. Restano il refresh dei proprietari e la policy dei contatori
  realmente misurati. Nessun cambiamento a pesi, dtype, maschere o riduzioni.

Il primo ciclo precede soltanto l'inizializzazione del contatore peak della
staging a8MiB; i gate e lo stress usano il binario ricostruito finale. Non si
attribuiscono prestazioni al contatore né ai soli confronti statici.


## Gate sul candidato congelato prima dello stress

Fondazioni Metal resident/SSD, Qwen27/Flash lifecycle, features, risorse,
executor, bundle, factory, planner cache e HTTP PASS. Python86test in14.944s;
Numeric CPU48test in5.078s (14skip per backend/geometrie), bundle12PASS.
ASan/UBSan della policy e del lifecycle Numeric PASS; compilazione CPU isolata,
Ruff e diff-check PASS. La sola prova del file upload Metal passa in0.422s
con BF16/FP32, offset17, più di8MiB, coda e EOF.

Flash8.1s/Qwen27 8.2s:10/10artefatti ciascuno esatti sul controllo precedente,
compresi tutti i logits e lo stato. Ora il probe chiama trim_cache prima del
forward e verifica frontier/ricrescita oltre al park/release/restore esistente.
Image512213.5s:11/11artefatti nativi esatti, nuovo confronto PASS sul medesimo
oracle indipendente CPUencoder/MPSDiT/MPSVAE. Gemma12.3s:491/491file float
identici al controllo;480/480frontiere esatte anche sull'oracle indipendente,
vettori128 max_abs5.96e-8,RMSE6.47e-9,cosine_min~1.0.

Nessun tempo sopra viene qualificato come speedup appaiato. Gli shader sono
invariati. `frozen-candidate.json` conserva hash di23sorgenti e6binari;
il server normale di fase è stato ricostruito dopo i diagnostici. Lo stress
Qwen50+warmup è avviato senza profiler con gli stessi input,contesto8192,
prefill64,riserva0,soglia swap256MiB/2campioni e timeout dichiarato4×.


Il confronto Image riusa il medesimo SHA256 fixture512 del controllo, non
noise T00 mancanti rigenerati. Un errore iniziale del solo harness contava
9file `.f32` contro10attesi; incluso il `.state` i10artefatti Flash sono
esatti. Nessuna differenza numerica o esecuzione modello fallita in quel caso.
I log conservano il primo errore e la correzione. Le due asserzioni sin/cos
Metal già preesistenti restano debito separato: gli shader non sono cambiati.


## Prima prova lunga fermata: cache inattiva, scelta O prima del codice

21cicli misurati +warmup completati, tutti gli output e ritorni corretti,
poi stop ordinato anticipato2906.5s: server exit0, nessun cleanup forzato,
crescita swap0. Il report controller FAIL documenta lo stop manuale, non
un PASS50 né un nuovo superamento del guard. Dati/patch/hash precedenti
rimangono in `stress-qwen27*` e `frozen-candidate.json`.

Il controllo finale invariato richiede lo stesso owner nativo dopo il
warmup: l'analisi anticipata trova due stati. Variante Image4:device0,
HOST41838725B,roots10. Varianti1/3:device675480832B,
HOST41838837B,roots14. La differenza coincide con pesi VAE inattivi più
112B dei loro buffer, non con workspace o prenotazioni rimaste vive.
I piani adattivi cambiano legittimamente con la RAM ma la suite cede solo
la cache LLM, conservando quella dell'adapter non attivo quando il piano
non ha bisogno di espellerla. Questa asimmetria contraddice lo stato idle
richiesto dallo stress e lascia cache non necessaria al lavoro attivo.

Riaudit O: Transformers ca8c871ede62b521a75a2ddfc5e1968ac9c8e097
`models/qwen3_5/modeling_qwen3_5.py`: stato conv/KV distinto dai pesi.
Diffusers aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e
`pipelines/qwenimage21/pipeline_qwenimage21.py:183,856` dichiara
text_encoder→transformer→vae e maybe_free_model_hooks dopo il risultato.
Strata46c152a `metal/src/engine/window.cpp:avail_now/make_arena` e
`stream.cpp` assegnano il working set al consumer attivo; non certificano
la cache dell'API DS4 né richiedono di chiudere il tokenizer. ggml dde879bc
`ggml/src/ggml-metal/ggml-metal.cpp:free/synchronize` separa rilascio
pesi e fine del runtime. I pin completi/hash sono quelli in sources.json,
i percorsi locali e le righe sono stati riletti per questa discrepanza.
Locale: I-P04/I-P05, `suite_victims/suite_drop_image`, pipeline/component
release_weights e controllo `snapshots[4:]` del qualificatore.

Scelta O: nel medesimo handoff Metal SSD già supportato da trim_cache,
cedere anche i pesi dell'adapter inattivo prima di creare il piano attivo.
Image espone release_weights nel proprio owner, che rifiuta un job aperto,
rilascia encoder/DiT/VAE e conserva tutti i metadata/tokenizer per riuso.
La suite usa questa API e i metodi Gemma esistenti; CPU/CUDA e adapter
senza handoff mantengono i loro cache-victims opzionali. Il criterio idle
esatto, guard swap e criteri numerici restano invariati. Gate indipendenti,
prova di ricaricamento, poi un nuovo stress50 dall'inizio, separato dal21.


## Ripresa del9novembre e richiesta di checkpoint dei test

Il candidato O passa i gate rimasti in esecuzione: Flash8.2s/Qwen8.3s,
20artefatti esatti; due Image512 in415.7s,11artefatti esatti sul controllo
e altri11identici dopo release/reload, Busy con job aperto/idempotenza e
residency0verificati. Gemma12.3s,491file esatti. Confronto Image indipendente
ripetuto PASS. Riletti anche Transformers `models/qwen4_exp/modeling_qwen4_exp.py`
al pin ca8c871 e Diffusers transformer/VAE QwenImage21 al pin aea7b70:
stato conv/ricorrente, dtype, maschere e cache temporali rimangono dei job;
il rilascio riguarda solo pesi immutabili. Fonti e hash aggiunti al manifest.

L'utente chiede esplicitamente ripresa dei test lunghi dopo interruzione.
Prima del codice del qualificatore: i cicli di `qualify_suite_release.py`
iniziano conversazioni distinte; la continuazione cached viene verificata
all'interno dello stesso ciclo. JSONL viene flushato ma non ricaricato,
non fsyncato e non protegge compatibilità/artefatti. Il controller rifiuta
una directory esistente e conserva i campioni VM in memory.jsonl.

Scelta: journal durevole per ogni ciclo completato, identità di binario,
modelli/configurazione/policy/harness/fixture, lock contro avvii duplicati,
scritture atomiche e verifica degli artefatti prima della ripresa. Ripetere
solo il ciclo incompleto, con warmup del nuovo processo. Un errore numerico
o uno stato idle diverso rimangono fallimenti, non checkpoint validi.
Distinguere50cicli funzionali cumulati da50cicli continuativi nello stesso
processo: solo questi ultimi chiudono il gate lifetime continuo. I dataset
precedenti al protocollo e i21del vecchio binario non sono importabili.
Nessun kernel, stato di inferenza, soglia o corpus viene modificato.


## Nuovo limite del9novembre: ammissione della cache Qwen, candidato P

La prima prova reale del checkpoint (`resume-proof-server`) fallisce prima
che Image sia invocata: primo prefill318token20.799s, decode0token, richiesta
22.390s. La cache automatica cresce6.04→7.06→10.15→12.33GiB; segue un rifiuto
per221759488B, capacità host/device0 per pressione4. Swap globale+789976186B,
server exit0, nessun cleanup forzato. Zero cicli convalidati, conservati come
tali. Non è la precedente discrepanza dell'upload encoder. I94test Python
del checkpoint/controller passano, ma non sostituiscono la prova reale.

Riaudit P prima del codice, pin completi e hash invariati in `sources.json`:
Transformers ca8c871ede62b521a75a2ddfc5e1968ac9c8e097,
`src/transformers/models/qwen3_5/modeling_qwen3_5.py:574,641` distingue
conv/recurrent state dai pesi immutabili; nessuna ricostruzione dello stato
è ammissibile per recuperare RAM. Strata46c152a1310f77cd2f0ce2c7cd213962df14fe83,
`metal/src/engine/arena_budget.h:footprint_cap/arena_budget/arena_grow`,
`window.cpp:1262–1290` e `stream.cpp`: l'arena anonima non è memoria file
residente; il floor file entra nel limite del footprint e il fill verifica
la disponibilità corrente. Quote7GB/15%/40%, mlock e layout MoE non si
trasferiscono al27B denso né al profilo con riserva0. llama.cpp
dde879bc4f954b809892fd0d660515f42536394a,
`src/llama-mmap.cpp`, `ggml/src/ggml-metal/ggml-metal.cpp:29–126` e
`ggml-metal-device.cpp`: wrapping shared e buffer posseduti hanno lifetime
separati, il working set Metal non certifica che XNU espella i file prima
di comprimere/swapparne gli anonimi. Non importa un criterio numerico.

XNU f6217f891ac0bb64f3d375211650a4c1ff8ca1ea,
`osfmk/vm/vm_pageout.c:2211–2242,2621–2624,3247–3250`,
`vm_page.h:AVAILABLE_NON_COMPRESSED_MEMORY` e `vm_resident.c` confermano:
il floor protegge i file anche durante fill da pread, è ricalcolato solo
nel pageout e include le speculative nei pageable. Ricerca locale riletta:
Q-P23/Q-P25, I-P04/S5, candidati B/M/N e `qwen_metal_stream_query_budget`,
`qwen4_stream_alloc`, sampler/test. B non correggeva il vecchio problema
Image, ma non qualificava questo nuovo caso di crescita Qwen prima di Image.
La nuova provenienza legge l'intero GGUF per SHA256 prima del lancio: può
rendere visibile il problema attraverso la file cache, ma non è ancora una
causa misurata con ablation. Nessun hash o controllo viene rimosso.

Differenza osservata: Numeric rispetta il floor, il sampler legacy usato
per la crescita SSD conta ancora file protetti come capacità per buffer
anonimi. Scelta P: campione sopra il floor anche nel legacy quando è in
modalità SSD; il percorso residente mmap conserva il campione precedente.
Riusa la policy M corrente, non aggiunge una riserva o un limite percentuale
alla RAM fisica. Il helper è nominato per storage posseduto, perché non è
specifico di mmap. Cache ancora dinamica, riserva0, guard immutati. Prima
aritmetica isolata, lifecycle e replay reale con gli stessi input; quindi
gate indipendenti e prova d'interruzione/ripresa. Se il replay fallisce,
registrare il nuovo risultato e riauditare, senza promuovere P.


## Uso della ripresa

Per i test lunghi gestiti da `tools/qualify_metal_server.py`, aggiungere
`--resume` alle opzioni del controller (prima del separatore `--`), mantenendo
le medesime opzioni e le due directory `--output`. Il controller inoltra la
ripresa al client `tools/qualify_suite_release.py`. Non cancellare le directory.
Il journal `cycles.jsonl` è l'autorità sul progresso: ogni ciclo PASS e i suoi
artefatti sono sincronizzati su disco prima di iniziare il successivo.
`summary.json` espone `completed_cycles`, `required_cycles`, `segments` e
`longest_continuous_cycles`. I tentativi server successivi sono conservati in
`attempts/0001`, `attempts/0002`, ecc.; PNG e vettori completati sono verificati
e riusati. Un ciclo interrotto prima del PASS viene rieseguito. Un nuovo
processo esegue il proprio warmup, escluso dai cicli misurati.

La ripresa rifiuta binari, modelli, fixture, corpus, criteri o harness diversi,
artefatti completati alterati e avvii duplicati. Se il ciclo ha fallito un gate,
il marker permanente `gate-failure.json` ne impedisce una promozione attraverso
il retry. Una coda JSONL interrotta viene archiviata e soltanto l'ultima riga
incompleta è scartata. Anche l'avvio interrotto prima del primo manifest client
è recuperabile, purché non vi siano cicli non versionati. Non vengono importati
i21cicli del binario precedente al handoff O.

Cinquanta cicli divisi fra processi possono chiudere il controllo funzionale
cumulativo (`PASS_SEGMENTED`), conservando le verifiche numeriche e idle,
ma lasciano `continuous_lifetime=NOT_VERIFIED`. Cinquanta cicli nello stesso
processo sono ancora necessari per promuovere il lifetime continuo T21/T08.
Il controller non si collega a un server estraneo rimasto sulla porta: un
processo ancora attivo o un lock posseduto viene segnalato prima del carico.


## Esito della prova reale di ripresa, prima della qualifica lunga

`resume-proof-v2-server` si interrompe intenzionalmente via SIGTERM dopo il
primo PASS durevole: controller NOT_VERIFIED177.470s, client INTERRUPTED1/3,
server exit0, nessun cleanup forzato. `--resume` conserva il primo tentativo,
valida il manifest e completa soltanto i cicli1e2: PASS309.631s,3/3cumulativi,
segmenti1+2. Il PNG0conserva SHA256 e mtime nanosecondi, non è rigenerato.
Tutti e3i cicli restituiscono owner500device0/HOST41838725B/roots10 e nessuna
prenotazione viva. Pressione max1; delta swap globale62915B nel primo tentativo,
398459B nel secondo, sotto il guard congelato256MiB/2campioni: non si dichiara
swap zero. Questi sono3cicli bounded, non un PASS50 continuativo.

Durate misurate: cicli153.638/153.732/135.772s, Image61.170/61.764/61.821s,
embedding56.120/56.117/56.029s. Campioni insufficienti per una promozione
prestazionale; condizioni VM diverse dal replay fallito, nessuno speedup
appaiato né attribuzione quantitativa del delta al solo candidato P.

Riesame della provenienza prima del run lungo: oltre alle opzioni esplicite,
il runtime eredita gli override nativi DS4/Metal e i percorsi delle librerie.
La ripresa deve rifiutare anche il cambiamento di tali valori. Registrarli
nell'identità attraverso SHA256 dei valori evita di inserire eventuali segreti
nei report. I soli percorsi di controllo generati per tentativo sono esclusi;
la modalità phase-server rimane esplicita. Non cambia alcun ambiente passato
al runtime, operazione, dispatch o criterio: è una verifica della provenienza.
Il meccanismo di journal/ripresa provato sul server resta identico; aggiungere
una prova isolata di rifiuto dell'override cambiato prima del run50.


Controlli sulla versione finale:95test Python in15.767s,1skip previsto;
make all, policy/risorse/executor/bundle/Numeric/planner Image/HTTP e lifecycle
Metal di entrambe le famiglie PASS. Il primo avvio dei test bundle usava il
Python di sistema privo di jsonschema: log conservato, ripetizione PASS con
il virtualenv del progetto. Analogamente il solo test del socket localhost
richiedeva i permessi fuori sandbox; ripetuto PASS con i permessi necessari.
Non erano errori del prodotto e nessun controllo è stato disabilitato.
`resume-verification.json` distingue gate, prova di ripresa e stress ancora
in corso; conserva pin/hash e i limiti della copertura. Lo stress50 usa il
candidato P con gli stessi corpus, contesto8192, prefill64 e soglie congelate.


## Chiusura dello stress Qwen27:50cicli continuativi

`stress-qwen27-final/report.json` PASS6956.335s,50cicli misurati più1warmup,
un solo processo/server_instance, client/server exit0, nessun cleanup forzato.
Contesto8192,prefill64,noMTP,Image256otto passi,cinque modalità Gemma con
128/256/512/768dimensioni, cache automatica e riserva0. Corpus, soglie
numeriche, guard pressione4 e swap256MiB/2campioni invariati.6847campioni VM:
pressione max2, crescita swap0, minimo host1625800704B. Non vengono sommati
i21del vecchio binario o i3della prova bounded di ripresa.

Tutti i51record hanno output corretti, continuation cached dopo Image e
Gemma, owner500device0/HOST41838725B/roots10 e prenotazioni vuote. Verificati
nuovamente dopo la chiusura: catena del journal, hash di tutti i PNG/vettori,
ordine0..50, criterio idle esatto e conteggio50continuativo. I30sorgenti e
5binari del manifest sono rimasti identici durante la prova. Lo stress usa
solo Imagevariant4: non certifica da solo ogni variante adattiva; le prove
indipendenti512e di release/reload rimangono evidenze distinte.

Latenze dei50cicli misurati: ciclo medio135.648s,mediana135.524s,
min134.337/max139.084s,CV0.682%; Image61.142s,CV1.285%; embedding57.251s,
CV1.503%. Nel profilo interno Image le medie dei quattro slot sono
0/4.224/43.792/10.462s (reference non usate/prompt encoder/DiT/VAE).
`phases.json` separa153prefill (media2.052s,primo freddo19.486s) e153chunk
decode (media3.644s). I TPS di quei chunk riguardano tool call vincolate,
non sono un benchmark di generazione libera né uno speedup appaiato.

Il gate locale di stress Qwen27 è chiuso. T08 completo resta IN_PROGRESS:
F08, asset congelati T00 mancanti, due assert sin/cos preesistenti e copertura
di altri hardware/backend sono elencati nel documento `remaining.md` T08.
Nessun commit. La verifica breve Flash del percorso condiviso è separata.


## Regressione finale condivisa Flash

Un ciclo bounded Flash→Image→Gemma→Flash passa in168.030s, controller
239.011s incluso fingerprint completo del GGUF e avvio;
Image65.633s,embedding54.670s. Continuation cached363/404token, codice
conservato, owner500device0/HOST41838725B/roots10, prenotazioni vuote.
185campioni: pressione max1,swap growth0,
minimo host7892369408B,server exit0,nessun cleanup forzato.
È una regressione del codice condiviso, non una nuova qualifica Flash50
né una chiusura di F08. Tutte le istanze della campagna sono terminate.
Il manifest finale conserva il confronto di30sorgenti/5binari e il dettaglio
dei test. Nessuna modifica a shader, precisione, soglie o corpus; nessun commit.


## Riesame e integrazione10novembre

I commit `e46c4ad`, `6b29099` e `6721055` separano trasporto/lifetime,
handoff Metal e qualificatori. Il [riesame](../../../../architecture/metal-swap-resume-clean-code-review-20261109.md)
e `review-verification-20261110.json` riportano105test Python PASS(1skip),
nuovi gate esatti Flash/Qwen e Image512/reload indipendente, prova Metal
reale di riuso dopo closeBUSY, fondazioni e sanitizzatori PASS.
Gemma e il nuovo HTTP completo richiedono più RAM disponibile; CUDA resta
PENDING su richiesta dell'utente. I due assert sin/cos e il NOMEM della
fixture con riserva2GiB sono espliciti. Lo stress50storico non viene
riattribuito al binario del riesame e T08 completo rimane IN_PROGRESS.
