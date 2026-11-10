# I-P04 — Budget continuo della cache Image

## S1 — Profilo della memoria e fonti prima del codice

Le richieste Image a cap7.5GiB mantengono almeno5.26–6.15GiB host
nel primo screeningI-P02, con riserva configurata2GiB. L'attuale pianificatore
sceglie solo frazioni1/2,1/4,... dei pesi per la cache; il cap DiT7.115GB
resta identico anche se l'admission consentirebbe un intervallo intermedio.
Il profilo DiT I-P01 contiene429stack in copia e216in proiezioni, con codice
DiT invariato in I-P02. Aumentare la quota utile può ridurre ricaricamenti,
ma il guadagno viene misurato separatamente dal kernel e dalle copieI-P03.

Fonti rilette il4novembre prima del codice:

- Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
  `models/transformers/transformer_qwenimage21.py:forward`,
  `pipelines/qwenimage21/pipeline_qwenimage21.py:__call__`,
  `models/autoencoders/autoencoder_kl_qwenimage21.py:decode`: stessi pesi
  rivisitati, fasi e picchi distinti, nessuna variazione numerica necessaria.
- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
  `models/qwen3_vl/modeling_qwen3_vl.py`: encoder precede DiT, nessun
  cambiamento semantico per la scelta di quali pesi restano residenti.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/arena_budget.h`, `stream.cpp`, `window.cpp`,
  `mtl_rt.mm:require_fit` e test dense/parity: capacità disponibile meno costi
  futuri e riserva, campione al momento dell'uso. Le quote7GB/15% e il
  footprint cap del M5Max non sono il nostro contratto2GiB. La cache MoE
  non implementa il planner encoder/DiT/VAE né i suoi restore.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/llama-mmap.cpp:impl`, `ggml/src/ggml-metal/ggml-metal-device.cpp`
  (working set/capability), `ggml-metal-ops.cpp` e `test-backend-ops.cpp`:
  mapping lazy e limiti hardware non giustificano un cap percentuale fisso
  né un cambiamento di dtype o arrotondamento.
- Locale: `ds4_resources.c:available_after,can_reserve` aggrega la RAM
  unificata, rispetta limiti espliciti/margini/lease; `ds4_executor.c:plan_build`
  calcola picchi e restore e preferisce evitare espulsioni; Qwen27Q-P23
  campiona memoria attuale e riserva2GiB. `ds4_weight_cache.c:limit` conserva
  una slot grande per progredire e arrotonda implicitamente a gruppi interi.

Scelta: mantenere la selezione iniziale dell'executor e ampliare soltanto
la variante streamed ammessa, usando ricerca binaria in byte per componente.
Ogni candidato passa dallo stesso planner pubblico con picchi/restore/margini;
non sono ammesse nuove vittime per ottenere più cache. Aggiornare memoria e
costo della variante prima di creare l'esecuzione, poi descrittori immutabili.
Un hook Numeric opzionale aggiorna le misure senza riscrivere cap espliciti.
Solo Metal lo espone: CPU/CUDA conservano la selezione discreta attuale.
La rivalutazione avviene a ogni nuovo job, non durante un kernel o una fase.
Non promette una quantità costante di RAM disponibile né uso di byte inferiori
alla dimensione di un gruppo. Sotto pressione continua a valere l'admission;
nessun minimo o riserva viene abbassato.

Verifica di provenienza: il vecchio manifesto29ottobre riportava un hash
arena_budget diverso. Prima dell'implementazione è stata acquisita la URL raw
al commit46c152a e confrontata byte per byte con la copia locale: coincidono,
SHA25652df7df71bd5e6623fe86c3830b432a39ce89c6c681ffce567b60b19f8973fca.
Il nuovo audit usa questi byte verificati; il vecchio manifesto resta storico.

## S2 — Pianificatore isolato

`make test-qwen-image-cache-budget` PASS: tre cache per componente con
scratch diverso nelle quattro fasi. Il limite in byte raggiunge la capacità
ammessa dal budget fisico unificato o dal cap device; un byte aggiuntivo
richiederebbe una nuova espulsione o violerebbe il limite. La risorsa vittima
rimane registrata/live e non viene rilasciata durante il planning. Le altre
varianti rimangono invariate; l'admission impossibile conserva il rifiuto.
CPU espone UNSUPPORTED al refresh e conserva la policy precedente.
Numeric CPU46 PASS (13skip) e lifetime PASS dopo l'estensione del callback.

Riletti anche `metal/tests/arena_budget_test.cpp` e
`window.cpp:make_arena`: Strata conserva l'arena pianificata quando la memoria
cresce e la riduce quando cala prima del fill. Il nostro piano viene creato
per ogni job e ricampiona prima dell'admission; se la memoria nel frattempo
cala, le varianti più piccole e i rifiuti restano disponibili. Il planner
conserva tutti i picchi di checkpoint/restore. L'espansione precede anche
la creazione dell'esecuzione di fallback che rilascia cache preesistenti.
Nessuna modifica ai descrittori dopo la creazione dell'esecuzione.

## Piano di conferma fissato prima delle coppie

I-P03 è ritirato. Baseline I-P02 (Conv tile, cache discreta, copia seriale)
contro I-P02+I-P04 con copia seriale. Cap device comune10GiB, riserva host2GiB,
stesse fixture e semantica. Una coppia di warmup0 e cinque coppie alternate1..5,
nessun profiler. Il cap è un ceiling comune; la capacità globale resta campionata
ed è l'admission a decidere se ciascun caso è ammesso. Nessun guard viene ridotto
in caso di rifiuto. Registrare byte e gruppi, tutti gli artefatti, fasi, processo,
memoria e crescita swap. I casi automatici restano separati dal confronto a cap
comune, senza attribuire la loro variazione di RAM al solo kernel.

Screening seriale: cap7.5GiB84.639s, encoder13.118/DiT60.997/VAE10.523s.
Automatico79.499s, encoder16.303/DiT52.680/VAE10.516s;96caricamenti DiT e
41.711GB logici contro159/69.192GB della cache discreta. Margine minimo4.225GiB,
swap invariato. Un precedente automatico con I-P03 aveva9.25MiB di crescita swap
pur mantenendo4.16GiB disponibili: conservato come esplorazione, escluso dalla
qualifica senza swap. Nessuna modifica di policy per attribuire quel singolo
contatore globale al modello; ripetizione seriale controllata senza crescita.

## S3 — Gate del wrapper finale e variabilità I/O

Dopo il ripristino del pool autorelease, Numeric e Gemma sono ripetuti sul
binario finale:480/480frontiere e vettori PASS; Numeric46 conserva soltanto
le due asserzioni sin/cos già riprodotte sul controllo.

Il primo RGB finale passa ma misura122.189s contro93.931s del precedente
RGB esplorativo; encode2.005/2.009, encoder19.864/15.266,
DiT89.849/66.147 e VAE10.468/10.509s. Stessi contatori, memoria del piano,
caricamenti e tutti gli output. Entrambi sono profili funzionali, esclusi

dalle coppie a cap comune. Il VAE conserva il guadagno col pool corretto;
la differenza si trova nelle fasi con trasporto pesi.

Nuovo riesame per questa differenza prima di altre modifiche: Diffusers al
pin sopra conserva blocchi/prefixKV e fasi, Transformers conserva vision e
text separati; Strata `stream.cpp:load_dense` e `arena_budget.h` distinguono
bytes tenuti e capacità del file cache; ggml `llama-mmap.cpp` usa advice lazy,
senza garantire che i byte mappati siano fisicamente residenti fra processi.
Locale: precedenti profili I-P01, tentativoI-P03 e contatoriH2D descrivono byte
logici, non letture SSD né latenza dei fault. Il nuovo campione contiene323
stack memmove e304nelle proiezioni, include attesaGPU e ha finestra diversa.
La riserva minima è5.166GB decimali contro6.215GB nel precedente RGB.
Nessuna evidenza di una nuova operazione numerica o di un nuovo dispatch
encoder/DiT; l'origine di ogni stallo I/O non è attribuita. Nessun kernel,
cache limit o criterio numerico viene modificato per spiegare il confronto
non appaiato. La conferma resta affidata alle cinque coppie senza profiler.

Il warmup baseline al cap10GiB viene rifiutato dal preflight del probe
(exit2,1.029s, nessuna inferenza): il cap esplicito eccede la capacità iniziale
che il probe ottiene dopo la riserva. Il monitor osserva12.206GB decimali
host prima dell'apertura, parte dei quali è riservata al sistema. Riletto
`tests/probe_qwen_image_pipeline.c` al controllo del cap, `suite_numeric_create`
e `ds4_resources_available`: il rifiuto precede la scelta della variante,
non è una regressione numerica del planner. Il raw `cap10-baseline-0` è conservato.
Prima delle nuove coppie il cap comune viene fissato a8.5GiB, per entrambi
i binari. La riserva2GiB, i criteri, il modello e la recipe restano invariati;
nuovo warmup0 e cinque coppie1..5 con etichetta `cap85`. Nessun campione
rifiutato entra nella distribuzione prestazionale e nessun guard è ridotto.

## S5 — Nuova differenza: file cache e swap nell'automatico finale

Dopo le cinque coppie accettate senza crescita swap, l'automatico finale
viene interrotto dal monitor a27.847s: swap+367.914MB, pressione2, disponibilità
riportata minima4.341GB, senza cleanup forzato. È `SKIP_RESOURCE_LIMIT`, non
qualifica automatica. Il precedente automatico seriale senza pool corretto
resta una prova storica e non sostituisce questo gate finale.

Nuovo audit prima della correzione: XNU
`f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`,
`osfmk/vm/vm_pageout.c:vps_calculate_filecache_min` e selezione delle pagine
nel pageout. Quando pageable_external_count scende sotto vm_page_filecache_min,
il kernel privilegia gli anonimi e riattiva pagine file. Il semplice
free+min(inactive,file) può quindi contare pagine che il kernel protegge
prima della compressione/swap. Il Mac espone210202pagine da16384byte di
minimo al campione, variabile nel tempo. `bsd/vm/vm_unix.c` conferma i
contatori UINT correnti; il minimo è letto in modo compatibile4/8byte.
Strata46c152a `arena_budget.h/window.cpp` e relativi test riconoscono il
minimo file cache ma usano7GB/15% e regole del M5Max, incompatibili col nostro
margine2GiB. ggml dde879bc `llama-mmap.cpp` lascia il file cache al kernel;
non certifica il nostro totale reclaimable. Diffusers/Transformers ai pin
precedenti non cambiano alcuna operazione o frontiera BF16. Locale: le prove
Q-P23 spiegavano i contatori Mach cached, non il file-cache floor di questo
nuovo caso; campionamento e report globali restano dati, non risultati numerici.

Scelta prima del codice: nel solo owner Numeric Metal, i cui pesi sono
copiati da mmap, contare free+min(inactive,max(file−filecache_min,0)). Il
minimo protetto non diventa una nuova riserva arbitraria: la riserva host resta
2GiB e viene speso tutto l'eccesso effettivamente disponibile. Il campionamento
legacy dei LLM con reader SSD esplicito resta invariato. Nessun sysctl viene
modificato, né si allargano soglie o si disabilita il monitor. Confrontare
aritmetica isolata e nuovi gate automatici prima di qualificare questa correzione.

Il probe aveva inoltre trasformato la disponibilità iniziale in un cap
permanente (`baseline+available`), impedendo al successivo refresh di usare
RAM restituita al sistema. Dopo il nuovo audit, il percorso Metal del solo
probe usa il limite hardware come ceiling automatico; il cap esplicito resta
un massimo sotto quel ceiling e l'admission usa sempre capacità globale
campionata, floor e riserva. CPU/CUDA del probe conservano il comportamento
precedente. Questa correzione diagnostica precede le nuove prove e non
trasforma i vecchi rifiuti in campioni validi. Nuovi binari e raw separati.

## S6 — Controllo finale del bundle dopo il floor

Le misure9.471% di S4 restano associate al campionamento precedente e non
sono riattribuite automaticamente al binario con il floor. La nuova conferma
confronta il bundle finale (Conv2D+cache continua+floor) con un controllo che
usa lo stesso sampler, probe, backend e pool, ma Conv scalare e cache discreta.
Il controllo disabilita soltanto le due pipeline opzionali nella copia privata
del backend e compila il pipelineC al HEAD4fef55f. Nessun flag nel prodotto,
nessuna differenza di input, precisione o criteri. Cap comune massimo8.5GiB;
l'admission può scegliere una quota minore quando la capacità sicura lo richiede.
Una coppia warmup0 e cinque coppie alternate1..5, senza profiler. Il guadagno
misurato riguarda il bundle; le ablation I-P02/I-P04 precedenti restano distinte.
La richiesta automatica viene ripetuta tre volte con swap growth esattamente0.

Le prime tre richieste automatiche col floor passano109.342–117.455s,
senza crescita swap e con tutti i bit identici. Il campione di RAM differisce
dalle vecchie prove: i tempi non sono un confronto di speedup contro79.499s.
Le quote crescono quando la capacità sicura aumenta; una riserva apparente
di file cache protetto non viene spesa in buffer anonimi. Il floor viene
letto ogni volta, non fissato al campione210202. Test policy host PASS,
compatibilità del sampler legacy a floor0 verificata dalla stessa aritmetica.
Si ripetono Numeric/Gemma e RGB/RGBA prima delle cinque coppie del bundle finale.

## Esito finale — Bundle qualificato, senza commit

Warmup0, poi cinque coppie alternate1..5 con campionamento floor-aware e
ceiling8.5GiB identici. Tutti110artefatti sono bit-exact contro la baseline
nativa originale; gate indipendenti PASS, riserva minima monitorata4.398GiB
e nessuna crescita swap. Non è una qualifica universale di tutte le risoluzioni.

| Fase | Mediana controllo s | Mediana finale s | Delta appaiato geometrico, IC95 |
| --- | ---: | ---: | --- |
| Encoder |14.956|14.942|−0.106% [−3.307,+3.276] |
| DiT |87.154|90.120|+4.835% [+1.715,+8.050] |
| VAE |53.960|10.406|−80.698% [−80.733,−80.655] |
| Operazione |156.165|115.846|−25.214% [−26.848,−23.665] |
| Processo |169.374|128.624|−23.424% [−24.729,−22.115] |

CV operazione1.792/2.820%. Bootstrap appaiato20000resample, seed20261104,
come nel piano. Dati completi in `final-bundle-performance.json`.

Riesame specifico del delta DiT prima della decisione: HF/Diffusers ai pin
sopra mantengono i medesimi blocchi e dtype; Strata arena_shrink confronta la
capacità al momento dell'uso, ggml mmap non rende costante la residenza delle
pagine. Il controllo mantiene7.115GB di limite DiT dal piano discreto iniziale;
il finale ricampiona prima dell'admission e sceglie4.747–6.213GB nel carico
osservato. I caricamenti salgono159→173–194 e i byte logici totali
83.620→89.727–98.888GB. Questa è una differenza osservata di budget/lavoro,
non una nuova differenza aritmetica. Non si forza una quota superiore al
campione corrente per recuperare throughput. I kernel DiT sono identici;
il guadagno KEEP è del bundle completo. Il9.471% della vecchia ablation
cache non viene attribuito alla release dopo la correzione del sampler.

Il finale supera tre automatici testo e i due editing RGB/RGBA,57artefatti
esatti nel complesso, oltre ai110delle coppie. Gemma480/480 e vettori128PASS.
CPU46PASS (13skip), Metal46con il solo metodo sin/cos preesistente (due
asserzioni,3skip), riprodotto nel controllo. Policy memoria, planner cache,
executor/restore, risorse, lifetime, bundle e fondazioni Image PASS.

`final-gates.json` e `final-regressions.json` conservano gli esiti finali.
Le sorgenti e i due binari sono congelati nel manifest dei grezzi
`build/metal-perf-image-20261104/cache/floor/final-freeze.json` e nel riepilogo
artefatti. Le prove precedenti, i rifiuti e lo swap esplorativo restano separati.
CPU/CUDA mantengono il planning discreto; il probe cambia ceiling solo su
Apple. Il sampler SSD legacy resta a floor0 e la sua aritmetica è verificata;
nessun nuovo benchmark LLM/CUDA/distribuito è dichiarato in questa campagna.

## Riesame finale dei confini di responsabilità

Il totale utile della cache viene letto da `memory.bytes[DEVICE]` del gruppo,
anziché da `load_cost`: il costruttore assegna a entrambi gli stessi `weights[i]`.
La modifica elimina la dipendenza dall'unità del costo, senza cambiare i byte
ammessi. Nessun kernel, dispatch o scheduling viene modificato in questo
riesame equivalente. Build prodotto, planner, policy, lifetime, executor e
risorse PASS. Un nuovo gate indipendente automatico sul binario riesaminato
passa115.457872s,11artefatti bit-exact, minimo host6468829184B e swap growth0.
È una conferma numerica/resource, non un campione appaiato aggiuntivo.

`reviewed-gate.json` registra il gate; `final-artifacts.json` separa sorgenti
e binari congelati delle cinque coppie da quelli dopo il riesame. Le misure
del bundle restano attribuite a `probe-floor` e `probe-floor-baseline`, senza
riassegnarle implicitamente al nuovo hash del prodotto.

## Profilo successivo sul prodotto riesaminato

Un'ulteriore richiesta automatica viene campionata nel DiT per8s, ogni10ms,
32s dopo l'avvio. Dei666campioni del thread principale,427sono nella copia
`ds4_weight_group_load`→`nm_copy`→`memmove` e217nelle proiezioni BF16,
inclusa l'attesa GPU. Le copie da mapping possono comprendere stalli da
page fault: questi conteggi non misurano banda SSD né utilizzo hardware GPU.
Confermano il trasporto come area residua da investigare, senza introdurre
un altro cambiamento di dispatch in questo ciclo.

La richiesta profilata passa il gate indipendente in119.381015s,
con11artefatti nativi identici, minimo host5754929152B e crescita swap0.
È esclusa dalle cinque coppie senza profiler. `final-profile.json` conserva
conteggi, hash del campione, gate e binario; i controlli prestazionali restano
quelli congelati sopra.
