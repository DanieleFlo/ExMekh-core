# Qwen Flash: esperti ibridi, PCIe e KV paginato

Backend Flash CUDA su una GPU. Il launcher Flash standard abilita i percorsi
qualificati; le chiamate dirette conservano i default riportati sotto. GGUF, precisione delle attivazioni, PLE BF16,
selezioni QSA, ordine della riduzione MoE e formato logico dei checkpoint
restano quelli del percorso esistente.

## Controlli

Richiedono `DS4_CUDA_QWEN4_TIERED_CACHE=1`; il modello deve essere Flash.
Il motore comunica questa capability prima di registrare il model map.
Variabili ereditate non attivano questi allocatori sul modello denso 27B.

| Variabile | Default | Significato |
| --- | --- | --- |
| `DS4_CUDA_QWEN4_HYBRID_MOE` | 0 | Scheduler di esperti completi CPU/GPU |
| `DS4_CUDA_QWEN4_DECODE_GRAPHS` | 0 | Due grafi per layer per decode e verifica MTP da 1..3 righe; richiede anche la capability CUDA singola GPU dei grafi DS4 |
| `DS4_CUDA_QWEN4_GRAPH_DEFAULT_STREAM` | 1 con grafi | Replay sulla coda ordinaria del forward; `0` conserva il precedente stream separato per gli A/B. La cattura resta su stream separato |
| `DS4_CUDA_QWEN4_DECODE_RANKS` | 1 con MoE ibrido | Indici persistenti delle vittime RAM/VRAM per forward da 1..8 righe; `0` conserva gli heap ricostruiti per layer. Stessa frequenza, età, preferenza RAM e tie-break per indice |
| `DS4_CUDA_QWEN4_PCIE_FRAC` | auto | Quota dei miss assegnata a staging GPU, `0..1` oppure `auto` |
| `DS4_CUDA_QWEN4_CPU_WORKERS` | profilo/4 | Override `1..8`, limitato alle CPU visibili |
| `DS4_CUDA_QWEN4_HYBRID_PREFILL` | 0 | Abilita l'override manuale anche per batch di 9..8192 righe; richiede qualifica separata del prefill |
| `DS4_CUDA_QWEN4_TUNING_PROFILE` | assente | Preferenze distinte decode/MTP e prefill |
| `DS4_CUDA_QWEN4_KV_STREAMING` | 0 | KV storico FP16 in RAM, working set esplicito in VRAM |
| `DS4_CUDA_QWEN4_KV_RESIDENT` | 32768 | Celle per layer, multiplo di quattro, almeno `sel_stride` |
| `DS4_CUDA_QWEN4_KV_DECODE_BATCH` | 0 | Per 1..8 righe: dopo lo spill unisce append K/V e indici QSA in una pubblicazione; prima dello spill registra l'evento dell'append e pubblica il mirror RAM dopo aver avviato l'attenzione GPU. Checkpoint, rollback e release drenano le copie pendenti |
| `DS4_CUDA_QWEN4_QSA_REGISTER_KEYS` | 0 | Selettore radix esatto con chiavi nei registri per 1..8 righe e 8192..33792 blocchi; ordine degli indici originale e fallback fuori intervallo |
| `DS4_CUDA_QWEN4_EXPERT_PROFILE` | assente | File di conteggi appresi associato al modello |
| `DS4_CUDA_QWEN4_PROFILE_PRELOAD` | 1 | Precarica esperti completi dal ranking del profilo valido |
| `DS4_CUDA_QWEN4_HOST_PRELOAD` | 1 | Precarica il complemento CPU se entra nel budget RAM |
| `DS4_CUDA_QWEN4_CACHE_STATS` | 0 | Contatori di tier, memoria e attese |
| `DS4_CUDA_QWEN4_EXPERT_POLICY` | recent | `recent`, `lru` oppure `stable`: triplette complete con quote per layer e senza eviction GPU |
| `DS4_CUDA_QWEN4_PLE_OVERLAP` | 0; 1 con stable | CUDA: legge le PLE dopo aver avviato il primo layer, prima del loro consumo |
| `DS4_CUDA_QWEN4_PLE_PREFETCH` | 0; 1 con stable | Legge le righe BF16 del chunk successivo con pool persistente durante il chunk corrente |
| `DS4_CUDA_QWEN4_KV_PREFILL_STAGE` | 0; 1 con stable | Workspace condiviso di un layer KV per batch da 32 righe, fino a 262144 celle; fallback selezionato se non disponibile |
| `DS4_CUDA_QWEN4_KV_PREFETCH` | 0; 1 con stable | Copia del prefisso del layer successivo su stream separato dopo il consumo del workspace corrente |
| `DS4_CUDA_QWEN4_KV_PREFETCH_BUFFERS` | 1 | `1..2` workspace identici: uno come Strata, secondo buffer opzionale per confronto |
| `DS4_CUDA_QWEN4_STAGING_WORKERS` | 0; 2 con stable | `0..4` worker persistenti per le copie CPU RAM→pinned del prefill |

`auto` senza un profilo compatibile conserva lo scheduler CUDA precedente.
Un profilo può disabilitare separatamente una delle due famiglie. Contesto,
chunk, fingerprint del modello, GPU, driver e numero di CPU partecipano alla
chiave di calibrazione, insieme ai budget espliciti di cache, policy di
eviction, stager CPU, sovrapposizione PLE, grafi di decode e configurazione KV.
La chiave dipende anche dalla versione dello scheduler: 35 per chunk oltre
256, versioni precedenti compatibili per chunk più piccoli. Il launcher
standard usa il percorso GPU qualificato e non carica profili CPU automatici.

Esempio riproducibile per il solo decode/MTP:

```sh
DS4_CUDA_QWEN4_TIERED_CACHE=1 \
DS4_CUDA_QWEN4_HYBRID_MOE=1 \
DS4_CUDA_QWEN4_PCIE_FRAC=1 \
DS4_CUDA_QWEN4_DECODE_GRAPHS=1 \
DS4_CUDA_QWEN4_CPU_WORKERS=4 \
./ds4 --cuda -m gguf/Qwen3.8-Flash-Next-Q2.gguf --ctx 2048 -p 'Scrivi una funzione C.'
```

La calibrazione va generata con il binario qualificato corrente, lo stesso
contesto, chunk e budget. I vecchi file `tuning-v12.txt` appartengono a una
versione precedente e vengono rifiutati dalla nuova chiave dello scheduler.
Per utilizzare un nuovo file di calibrazione:

```sh
DS4_CUDA_QWEN4_TIERED_CACHE=1 \
DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=20 \
DS4_CUDA_QWEN4_HYBRID_MOE=1 \
DS4_CUDA_QWEN4_PCIE_FRAC=auto \
DS4_CUDA_QWEN4_TUNING_PROFILE=/percorso/del/profilo-qualificato.txt \
./ds4 --cuda -m gguf/Qwen3.8-Flash-Next-Q2.gguf --ctx 2048 --prefill-chunk 256 -p 'Scrivi una funzione C.'
```

Per sperimentare la capacità KV, aggiungere `DS4_CUDA_QWEN4_KV_STREAMING=1`
e, facoltativamente, `DS4_CUDA_QWEN4_KV_RESIDENT=32768`. Questa modifica
invalida la calibrazione precedente: usare un override manuale qualificato
oppure ricalibrare con lo stesso contesto e budget. Il report prestazionale
distingue le configurazioni che migliorano velocità e quelle ancora
sperimentali; nessun launcher viene modificato automaticamente.

## Piano di placement e lifetime

La policy `stable` riprende le quote per layer di Strata. Seleziona tutti e
tre i pesi di ogni esperto dal primo chunk del router (o da un profilo
compatibile) entro il budget effettivo, con quote ponderate dai byte del
layer. Durante il prefill i residenti non vengono sostituiti dai miss; i buffer di staging
eseguono il complemento. La RAM ordinaria può così conservare gli esperti
assenti dalla GPU senza demozioni ricorrenti. Il piano è fisico e appartiene
all'engine; i checkpoint conservano il formato logico precedente. Un cambio
di dominio può favorire la policy recente: le modalità restano confrontabili
con lo stesso budget.

La quota stabile considera anche il limite fisico dell'allocatore CUDA, i
pesi densi non ancora caricati e lo scratch lazy. Al termine del prefill,
dopo tutti i consumatori, si prepara il ranking recente per il decode:
gli hit desiderati vengono protetti globalmente prima di promuovere i miss.
Questo lavoro è incluso nel tempo di prefill/TTFT. Decode e MTP fino a otto
righe mantengono la policy recente; il ranking fisso riguarda il prefill.

Il router produce gli ID originali. Per ogni forward si classifica ogni
tripletto gate/up/down una sola volta. Un hit GPU richiede tutte e tre le
proiezioni; un hit parziale viene trattato come miss completo. Le proiezioni
già residenti restano protette durante il consumo. Gli esperti più riusati
tra le righe del batch hanno precedenza nel tier PCIe.
Le liste dei contributi di ogni esperto sono costruite con una scansione
lineare degli ID originali; non si riscansiona il batch per ogni esperto.

Si avvia la fase GPU residente, si caricano i miss CPU da RAM/SSD bounded,
si inviano i gruppi a un pool persistente, quindi si avvia la fase PCIe GPU
mentre i worker calcolano. Ogni contributo CPU viene scritto nello slot
originale `(token, expert)` dallo scatter di un output CPU compatto; si
trasferiscono solo i contributi CPU. Lo scatter e la riduzione finale usano lo stesso
ordine del percorso CUDA. Gli esperti condivisi restano sulla GPU.

La CPU legge IQ2_XXS e Q2_K originali con attivazioni float, inclusi i blocchi
Q2_K parzialmente utilizzati dalla proiezione down. Fino a quattro righe che
condividono un esperto riusano la dequantizzazione, conservando l'accumulazione
float di ogni riga. Il GGUF locale contiene un layer MTP Q4_K/MXFP4: quello
mantiene il percorso CUDA. Batch oltre 8192 righe usano il prefill precedente.
La v23 qualificata era ancora limitata a 256 righe; l'estensione a 8192
è nel sorgente successivo e deve completare i propri gate e benchmark.
I buffer FP32 CPU sono allocati solo per quote che prevedono lavoro CPU;
con quota PCIe 100% non sottraggono VRAM alla cache degli esperti.
L'accumulo CPU riproduce le 32 lane CUDA: quattro prodotti consecutivi per
lane, stride 128, FMA esplicite e riduzione nello stesso ordine. Il percorso
AVX2 raggruppa quattro righe senza cambiarne l'aritmetica. Gate e up sono
calcolati dalla CPU in FP32; un piccolo kernel GPU applica la stessa SiLU
originale, poi la CPU calcola down in FP32. Non si quantizzano le attivazioni.
Questa frontiera evita le differenze dell'esponenziale e del reciproco CPU
che facevano fallire i controlli stretti di stato. La v21 supera primitive,
checkpoint/replay, rewind e MTP con quote CPU. Il tuner richiede comunque
un report superato con identità corrispondente al binario utilizzato.

Pool, scratch, routing, output e staging sono persistenti. Un task porta
esplicitamente l'indice della propria partizione: l'identità del worker serve
solo a scegliere lo scratch. Tutti i task vengono drenati anche su errore;
reset e teardown attendono consumatori e copie. La cache RAM protegge i range
CPU durante il forward; la cache VRAM protegge i range GPU nello stesso epoch.
Il readback delle attivazioni parte soltanto se il placement contiene
contributi CPU, su stream separato da quello dei residenti già avviati.
Una selezione tutta-hit GPU non scarica attivazioni anche con quota CPU.

I miss PCIe vengono trasferiti su uno stream di copia dedicato, in gruppi
di 16 esperti per decode/MTP e 64 per prefill. Due buffer VRAM alternati e
gli eventi di copia/consumo consentono di iniziare il calcolo di un gruppo
mentre il successivo viene preparato. Il prefill conserva il pool SSD con
letture anticipate bounded per triplette eterogenee. Il pool pinned aggrega
più proiezioni prima di riutilizzare un buffer: non attende un evento per
ogni piccola proiezione. La configurazione PCIe 100% evita il readback delle
attivazioni destinato alla CPU.
Anche il prefill ordinario usato come fallback raggruppa le copie da RAM
ordinaria: mantenere il vecchio scheduler non deve reintrodurre un'attesa
per ogni piccola proiezione. La fine del gruppo pubblica l'evento che
protegge il riutilizzo dello staging pinned.
Le demozioni D2H si accodano negli stessi gruppi. I descrittori di destinazione
RAM sono bounded e persistenti; prima di pubblicare una sorgente RAM ancora
in preparazione si attende l'evento e si completa la copia host. La fine
del gruppo e il reset drenano tutte le destinazioni prima del riutilizzo
o della liberazione delle arene.

Le classifiche di eviction sono persistenti e separate per dimensione della
proiezione. Le chiavi sono copiate al principio dell'epoch e non cambiano
durante l'uso dell'heap; una nuova ammissione rende la voce protetta e viene
scartata dagli altri heap quando raggiunge la testa. Il tie-break resta
identico alla scansione lineare, verificata da un controllo opzionale nei test.
Questo evita di ricostruire decine di migliaia di slot a ogni alternanza
gate/up/down del piano unico.
Anche le classifiche RAM sono separate per dimensione e usano chiavi
immutabili. Danno precedenza all'eviction dei pesi con una copia GPU;
le demozioni vengono riclassificate prima della scelta. Il controllo
lineare opzionale verifica sia RAM sia VRAM sotto pressione di memoria.

Lo staging usa il numero di righe dell'operazione corrente, indipendente
dalle liste del router GEMM nativo. Dopo un prefill grande, il decode ibrido
non ricostruisce quelle liste: riutilizzarne `route_T` inviava erroneamente
le piccole copie a worker CPU. Il percorso fino a otto righe ora conserva
le copie inline; un test prepara prima liste da 37 righe e verifica questo
passaggio per tutte le quote CPU/GPU, senza cambiare le tolleranze numeriche.
`DS4_BENCH_TIER_DECODE=1` registra separatamente traffico, job dello staging,
attese e tempo di scelta delle vittime nella sola fase decode.

Gli esperti usano RAM ordinaria. Il pool DMA comune a esperti e KV contiene
due buffer pinned da 64 MiB. Un evento protegge ogni riutilizzo dopo H2D/D2H.
Non si registra una seconda copia integrale del modello. La RAM disponibile
limita l'arena e sottrae il backing KV; con budget zero si usa staging SSD/GPU.

## KV per sessione

Per il prefill con policy stabile, il workspace identico al layout originale
consente di elaborare il batch completo. Il prefetch prepara lo storico del
successivo layer di attenzione mentre GPU e MoE completano il corrente;
solo le nuove righe del chunk vengono aggiunte dopo il relativo
append. Eventi separati proteggono fine copia e fine consumo. Restore,
rollback e rilascio invalidano i prefissi preparati e drenano le copie prima
di modificare o liberare il backing. Tutti i workspace sono sottratti al
budget disponibile per gli esperti.

Il prefetch PLE mantiene il calcolo originale degli hash, compreso EOS e gli
snapshot MTP. Il risultato anticipato è utilizzato solo se modello, token,
posizione e storia n-gram coincidono. Errori di lettura non sono ignorati;
cancellazione, reset e teardown drenano i worker prima del rilascio.

Il backing conserva K/V FP16 logici in RAM. Pagine da quattro celle vengono
deduplicate dopo la selezione QSA; una mappa logica individua gli slot VRAM.
Si proteggono tutti gli hit del gruppo prima di scegliere le vittime. I miss
sono trasferiti con DMA esplicito, raggruppando slot fisici contigui. I kernel
originali risolvono gli indici attraverso la tabella pagine; selezione,
causalità, split e riduzione rimangono quelli originali.
Prima del primo spill, quando lo storico entra tutto nel budget residente,
gli slot hanno indici fisici identici agli indici logici. Append copia i
byte FP16 originali da device a device e mantiene il mirror RAM; attention
usa i kernel residenti senza readback delle selezioni né H2D del working set.
Al primo spill si pubblica la mappa anche se tutte le selezioni risultano hit.
Una scrittura di checkpoint abbandona l'identità fisica e ricostruisce le
pagine dai dati logici; rollback conserva causalità in entrambe le modalità.

Lo streaming riguarda i layer target. Il KV del predittore NextN resta
residente: il percorso MTP esistente inizia dalla frontiera del prompt senza
prefill del proprio storico, quindi non soddisfa l'invariante di append
contiguo del backing paginato. Questa scelta preserva il comportamento MTP
attuale; il suo KV continua a essere incluso nel budget VRAM e nei checkpoint.

Se l'unione richiesta da un batch supera il budget, si dividono le query in
gruppi bounded. Una singola query che supera la capacità fallisce con una
diagnosi, senza eliminare selezioni. Indexer e block keys restano in VRAM.
Le pagine usate sono protette fino alla fine del consumo sullo stream CUDA.

Append trasferisce i nuovi byte FP16 target in RAM. Rewind/rollback invalidano
pagine e byte target oltre la frontiera ripristinata; lo stato NextN segue le
copie e i checkpoint residenti esistenti. Save/restore leggono e scrivono
gli stessi span logici dei checkpoint precedenti; le mappe fisiche vengono
ricostruite dopo restore. I backing appartengono alle sessioni, le cache pesi
all'engine. Sessioni diverse non condividono pagine KV.

Le API di lettura/scrittura e copia sincrona supportano gli span logici.
Views, accesso a un puntatore device contiguo, copie async/xdev e conversioni
fisiche incompatibili vengono rifiutati. Il batching nativo intersessione
con tabelle di puntatori contigui usa il fallback ordinato per sessione quando
il KV streaming è attivo; non è stato portato al nuovo layout.

## Profili e misure

Si conta ogni scelta una volta per layer/riga, al gate, e si salva un payload
di soli conteggi con fingerprint e checksum. File incompatibili o corrotti
sono ignorati. La sostituzione del file avviene con rename del temporaneo.
Il ranking iniziale considera frequenza e byte del tripletto; il profilo
iniziale cede gradualmente alla finestra recente di 128 righe. La policy
recente e l'override LRU esistenti restano disponibili per i confronti.

Il ripristino conserva il payload logico validato della cache esperti fino
al forward successivo. Una riserializzazione immediata resta byte-identica,
anche se l'engine possiede altri slot fisici acquisiti dopo il checkpoint.
Il formato non cambia e la residenza fisica continua a rispettare i budget
correnti.

`tools/perf_qwen_flash_strata.py` congela identità, misura forward completi,
calibra quote `0,.25,.5,.75,1` e worker `1,2,4,8`, confronta logits completi
e produce JSON/CSV. `paired` alterna A/B e B/A per cinque campioni, con warm-up
in ogni processo. Il filtro numerico del tuner è diagnostico: non sostituisce
primitive, teacher-forced loss, fixture di qualità e regressioni MTP/vision.
Un risultato inconcludente disabilita la preferenza; una regressione congiunta
oltre il 5% disabilita le famiglie aggiornate. Nessun dato esterno di TPS vale
come risultato.

Il formato `DS4QWEN4TUNE2` distingue decode ordinario (T=1), verifica MTP
(T=2..8) e prefill (T>8), con pool persistenti separati. La griglia MTP usa
le impostazioni iniziali del modello, verifica l'esecuzione multi-riga e
misura il tempo totale di decode. Conserva le scelte decode/prefill del
profilo compatibile. I file V1 continuano a fornire decode/prefill e usano
il percorso GPU precedente per MTP, finché questa famiglia non viene
calibrata. Il nuovo test host v53 verifica loader, fingerprint, dati troncati,
range, valori non finiti e override senza richiedere il modello o CUDA.

La v55 estende lo scope MTP anche alle proposte di una sola riga e all'intero
ciclo speculativo: T=1 dentro MTP deve usare la scelta MTP, mentre T>8 resta
prefill. Lo scope precedente viene ripristinato anche sui ritorni di errore.
La v57 prepara ciascuna famiglia al primo uso, mantiene pool e metadata,
riduce lo staging dei prefill sotto 1024 righe a gruppi di 16 esperti e
restituisce il solo staging GPU alla fine del prefill. Evita di sottrarre
VRAM alla cache per famiglie mai utilizzate o workspace della fase conclusa.
Non alloca o libera staging ordinario durante i token di decode.

Il confronto alternato v57, cinque coppie fuori profiler, qualifica la
configurazione automatica sul workload breve: prefill 504,07→587,23 (+16,50%),
decode totale 33,85→36,64 (+8,24%). Con MTP iniziale invariato: prefill
491,36→581,16 (+18,28%), decode totale 35,27→36,18 (+2,58%). Il guadagno
MTP di decode è inferiore alla variabilità; il guadagno qualificato è il
prefill. Tutti i confronti del vocabolario completo sono bit-identici.
La policy stabile separata peggiora invece il prefill breve del 14%:
resta sperimentale, senza sostituire la policy recente automaticamente.

Nel server i contatori della richiesta sommano le differenze di ciascun
tratto di inferenza sotto mutex, escludendo il lavoro intercalato di altre
richieste. Un batch decode con più membri può condividere upload e hit:
questi costi compaiono separatamente come `shared decode batches`, senza
dividerli arbitrariamente o sommarli tra richieste. La memoria è la
fotografia dell'engine al termine dell'ultimo tratto. Un reset dei contatori
marca la misura incompleta. Acceptance MTP continua a essere riportata dai
contatori di generazione già presenti. La nuova attribuzione successiva alla
v23 supera il test di due richieste intercalate, batch condiviso e reset dei
contatori nella v24. Il log include letture SSD per proiezione oltre ai byte.
La memoria del KV paginato distingue celle VRAM e workspace di append/mappa;
lo scheduler riporta separatamente i propri workspace VRAM. I buffer generali
del grafo e il KV NextN residente continuano a figurare nel budget del grafo.

## Provenienza e 4 BIS

Riferimento architetturale: Strata commit
`1735d6471df29b42c26170efaac1f1446a58640f`, licenza
[MIT](https://github.com/Niko1221/Strata/blob/1735d6471df29b42c26170efaac1f1446a58640f/LICENSE).
Sono stati analizzati `session.cpp`, `verify.cpp`, `generate.cpp`,
`expert_cache.cpp`, `kv_stream.cu` e il pool CPU. Non sono state copiate
implementazioni Strata. Il percorso KV locale usa copie esplicite per WSL.

I grafi registrano mixer/GDN o proiezioni dell'attenzione e il mixer FFN fino
al router. Posizioni, selezioni QSA, copie esplicite KV e placement dinamico
restano eager. Snapshot e buffer residui fanno parte della chiave; replay e
fallback conservano gli scambi host e la numerica dei kernel. I consumatori
sono attesi prima della distruzione della sessione e dell'invalidazione.

Il gate vision/thinking v53 ha trovato un errore di lifetime nei grafi:
il rollback MTP scambia i buffer ricorrenti live/snapshot, mentre una chiave
che distingue solo residui e flag può riprodurre un kernel sul vecchio banco.
La v54 include nella chiave stato/history live, snapshot e snapshot2,
oltre al residuo alternativo. Entrambi gli assetti vengono conservati e
riutilizzati; otto slot per layer/isola coprono le forme 1/2/3 righe con i
due banchi. Il formato dei checkpoint resta logico e invariato.
Il test con contatori GPU alternati verifica replay reali e assenza di
scritture nel banco inattivo. La fixture API invariata passa sia con grafi
da soli sia con tutti i flag/CPU 50%; il vecchio binario passa usando copie
di restore, confermando la causa. Le misure MTP v53 con grafi non promuovono
prestazioni o qualità, anche se riferimento e candidato coincidevano.

La v26 raggruppa le due pubblicazioni K/V e le due letture ID/conteggi QSA
con un evento per coppia. Il prefill misto CPU usa lo stesso ordine warp-FP32
anche per gli esperti GPU, evitando che gli arrotondamenti dipendano dalla
residenza fisica durante restore o tra sessioni. Il prefill GPU al 100%
mantiene i GEMM originali. Questa variante CPU richiede calibrazione distinta;
un costo maggiore conserva il percorso GPU precedente. I gate stretti di
checkpoint, restore, snapshot e rewind del prefill misto v26 sono passati.
La qualità completa v26 registra però differenze di distribuzione e un
argmax tool diverso. Il tuner richiede anche il report teacher-forced
compatibile con il binario e lo screen KL/argmax su tutte le righe: questa
variante viene esclusa dalle preferenze automatiche, anche se passa i gate
di stato. Il prefill GPU qualificato resta il percorso scelto.

La [scheda 4 BIS archiviata](../experiments/qwen27-4bis-20261008/documentation/qwen27-kv-capacity-4bis-20261006.md) descrive il percorso
27B implementato su successiva richiesta dell'utente: modalità esplicita per
maggiore capacità, scelta del compromesso dai benchmark, precisione e intero
contesto preservati. La v27 ha completato 65536 token effettivi più otto righe
teacher-forced: 414,10 TPS di prefill e 0,69 TPS di decode. Il confronto del
modello intero con i kernel originali su un layer contiguo copiato dalla RAM
produce logits bit-identici e KL zero. È un riferimento aritmetico, distinto
da una misura con tutti i layer KV residenti in VRAM. Le nove primitive con
oracle CUDA residente a 64K e i gate MTP reali passano anche nella build v30.
La modalità offre maggiore capacità con un costo di circa 1,45 s/token sul
link locale. L'8 ottobre l'utente ha ritirato l'esperimento: codice e report
sono nell'[archivio 27B](../experiments/qwen27-4bis-20261008/README.md);
nessun backend di capacità o streaming dei pesi 27B resta nella build.

## Liste compatte del prefill — incremento successivo al ritiro 4 BIS

`DS4_CUDA_QWEN4_COMPACT_LISTS=1` è una variante sperimentale, disattivata
inizialmente. Lo scheduler ibrido riservava `esperti × token × slot` interi
per le liste, benché il router selezioni ogni esperto al massimo una volta
per token. La variante riserva `esperti × token`; non cambia i kernel,
l'ordine delle righe o la riduzione finale. Con 512 esperti, 8192 righe
e dieci slot libera 144 MiB di workspace, disponibili per la cache.

Gli input diretti dell'API possono contenere un esperto ripetuto nello stesso
token. I conteggi originali determinano una crescita sufficiente della lista
prima del consumo, dopo aver drenato i lettori precedenti. Nessun contributo
viene troncato. Decode T=1 e verifica fino a otto righe continuano a usare i
kernel per riga, senza crescite ordinarie durante i token.

La primitiva v64 confronta gate/up/SwiGLU, down con padding e shared expert
bit per bit contro il layout precedente, anche con duplicati che eccedono
la capacità iniziale. La chiave del tuner passa alla versione 29 e include
il flag; le vecchie preferenze incompatibili conservano il percorso precedente.
Il guadagno end-to-end richiede misure nuove e non deriva dalla sola memoria
risparmiata.

## Copie anticipate bounded nel prefill

`DS4_CUDA_QWEN4_PREFILL_AHEAD=1`, con policy `stable` e ring attivo,
anticipa le prime due porzioni del ring mentre l'attenzione già accodata
viene calcolata, prima del readback del router. Il budget del ring non
cambia: nessuna seconda allocazione VRAM. Gli eventi di consumo precedenti
ordinano il riuso; la cache non cambia epoch prima del termine dei lettori.

Gli hint provengono dall'ultimo chunk completo dello stesso layer e modello.
Il primo chunk freddo conserva la selezione ordinaria; un miss della
predizione conserva il caricamento successivo al router. Non si modifica
alcuna scelta, lista o riduzione. Gli hint sono dell'engine, non dello stato
logico di sessione. Cleanup e prefill-finish drenano eventuali upload
abbandonati prima di liberare lo scratch. Il tuner versione 31 include
tutti i controlli dell'anticipazione e delle liste compatte. Quando le
nuove modalità sono disattivate conserva la chiave v28 già qualificata;
le preferenze automatiche esistenti non vengono invalidate senza motivo.

Il [report dell'8 ottobre](../performance/qwen-flash-prefill-strata-20261008.md)
distingue la variante indiscriminata v65 dagli hint v66 e dai risultati
delle cinque coppie per le liste compatte.

## NextN e selezione per fase

La v68 ha provato NextN embedded di default per Flash CUDA. La build corrente
mantiene l'attivazione automatica sperimentale con `DS4_QWEN4_MTP_AUTO=1`,
perché i benchmark mostrano una regressione ripetibile sulla prosa;
`--mtp` resta disponibile e `--no-mtp` permette il confronto con il decode ordinario. Si attiva dopo
la validazione del target: non riguarda 27B, CPU, ispezione o modelli senza
head. Le opzioni library azzerate, bench ed eval restano ordinarie. Il
percorso automatico preserva il sampling tramite la verifica esatta; la profondità
adattiva greedy corrente propone uno/due token. Non è DSpark v2.

Il prefill mantiene un proprio scheduler. Il profilo `DS4QWEN4TUNE2` separa
token singolo, verifica MTP e prefill: una quota PCIe condivisa esplicita può
essere utile per confronti ma non rappresenta i migliori parametri di tutte
le fasi. Il profilo deve corrispondere alla chiave di modello, contesto,
chunk e workspace. NextN non promuove gli altri flag sperimentali.

La fase viene impostata esplicitamente dalle entrate di prefill, NextN e
verifica e ripristinata al ritorno, anche in errore. Il prefill di 1–8 righe
usa così le proprie preferenze e prepara la cache decode al termine; non
eredita la quota MTP dal solo numero di righe. Le primitive dirette senza
scope conservano il classificatore precedente. Il MTP nativo Flash e NextN
indicano la stessa testa: la profondità adattiva esistente usa l'acceptance,
senza una decisione CUDA automatica di attivazione/disattivazione basata
sui tempi del decode.

I gate v68 verificano il default senza attivazioni embedded/esatte esplicite,
il rewind da resample, il `--no-mtp` del server e il normale 27B. Il report
dell'8 ottobre distingue i test del percorso GPU dai confronti ibridi:
gli stessi output greedy non garantiscono un guadagno su ogni workload.

`DS4_CUDA_QWEN4_MTP_RESIDENT=1` riserva gli esperti completi del NextN nel
budget pesi esistente, prima degli esperti target. La policy stabile divide
il budget residuo tra i layer target; le altre policy mantengono i propri
criteri di ammissione. Gli slot del drafter sono protetti in tutte le
selezioni di vittime, comprese quelle lineari e gli indici persistenti.
Il costo iniziale e la cache target più piccola richiedono un confronto
end-to-end; la riserva non è attiva inizialmente. Se non entra interamente
nella quota, resta il percorso bounded. La capability deriva dal modello
Flash e dall'effettiva presenza di NextN nella sessione, senza inferire che
l'ultimo layer di un modello qualsiasi sia un drafter.

I checkpoint continuano a memorizzare dati logici. I suggerimenti di
residenza GPU/RAM dei pesi precaricati vengono esportati solo quando esiste
la corrispondente cronologia reale del router. Nessuna scelta viene inventata
per riempire quella cronologia, né si serializzano pin permanenti o puntatori.
La residenza fisica dipende dall'engine e dal budget corrente al restore.

Le proiezioni dense Flash CUDA usano al massimo 256 colonne token per chiamata
cuBLAS anche quando il prefill del modello usa 2048/4096 righe. Questo conserva
la riduzione numerica originale; il batch MoE resta grande per riusare pesi e
trasferimenti. La coda mantiene il dispatcher precedente, senza nuova
quantizzazione delle attivazioni. Il [gate del launcher](../performance/qwen-flash-agent-launcher-20261008.md)
isola il drift del batch cuBLAS dalla policy di cache e conserva i confronti
full-vocabulary prima/dopo la correzione. Le preferenze tuner per chunk grandi
usano versione 35; quelle per chunk fino a 256 restano compatibili.

Il launcher Flash usa ora il profilo standard qualificato: prefill 2048,
ring bounded, KV storico FP16 in RAM con 32768 celle residenti, policy
stabile e ranking/grafi CUDA. Prefill e MTP usano GPU (quota PCIe 1) e MTP
nativo conserva la propria profondità adattiva. Non servono flag sperimentali;
gli override espliciti restano controlli di configurazione e il profilo
legacy ricostruisce il precedente avvio. La promozione riguarda il launcher
locale verificato, non cambia automaticamente i percorsi CPU/Metal/27B.
Il contesto predefinito attuale è 65536; un contesto diverso è richiesto tramite
`DS4_CTX`. I benchmark storici a capacità 131072 conservano il loro valore esplicito.

Il prefill KV ausiliario del drafter Strata (E-9 `Prefill::draft_kv`, con
fallback `MtpDrafter::prefill`) non è ancora portato. DS4 mantiene il suo
inizio NextN alla frontiera del prompt. La residenza completa dei suoi
esperti non basta a qualificare l'intera pipeline: preparare lo storico
KV richiede verifiche proprie su numerica, mRoPE, checkpoint e acceptance.
