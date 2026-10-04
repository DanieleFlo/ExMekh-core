# Flash Next Q2: residency CUDA sulla RTX 3090 — 3 ottobre 2026

## Stato

**Esperimenti diagnostici, non qualificati per il percorso predefinito.**
I quattro flag nuovi sono disattivati per default. Il GGUF, i suoi tensori,
la precisione KV, le formule e le soglie dei test restano invariati.
Nessun risultato Strata viene considerato una validazione di DS4.

## Diagnosi e provenienza

Macchina locale: RTX 3090 sm_86, i9-14900K, 64 GB fisici, WSL con 48 GB e
16 CPU logiche. Modello: `gguf/Qwen3.8-Flash-Next-Q2.gguf`, 147207127040 byte,
41,73 GiB principali/MTP e 95,37 GiB di n-gram BF16 letti dal disco.

Il loader registra la mappa host. In `cuda_model_range_ptr`, una mappa
registrata viene risolta direttamente prima del caricamento nella cache.
`DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=16` limita una cache, ma non trasferisce
automaticamente 16 GiB di pesi in VRAM. Il percorso originale misurato
raggiunge circa 0,40 token/s anche con contesti di 16 e 32 token.

Fonti consultate prima degli esperimenti:

- [Post RTX 3090/128 GB](https://www.reddit.com/r/LocalLLaMA/comments/1wwamjw/strata_rtx_3090_128_ram_qwen_38_flash_next/): risultati riferiti a Strata, UD-Q3_K_XL, cache esperti e MTP; hardware/RAM e quantizzazione differenti. Fonte di ipotesi, non baseline DS4.
- [Strata, dettagli](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md): GPU per pesi densi/esperti frequenti, CPU per miss, copie/overlap e speculazione. Sono trasferibili le idee di residency; i formati, il packer e i kernel aritmetici non sono intercambiabili automaticamente con questo Q2.
- [club3090 MoE cache](../research/forks/club3090-llama-moe-cache.md): la precedente decisione di non portarlo si riferiva al 27B denso, interamente residente. Flash Next MoE mostra il problema di accessi host per cui il fork è pertinente.
- [llama.cpp](../research/platforms/llama-cpp.md): separazione GEMV/GEMM e scheduling degli esperti; nessuna requantizzazione delle attivazioni è introdotta.
- Primitive riusate da DS4: arena pesi bounded, range lookup, scratch per tier, buffer prefill dello streaming e kernel Qwen4 esistenti.

## Variabili isolate

1. `DS4_CUDA_QWEN4_DENSE_CACHE=1`: copia byte per byte i tensori densi attraverso l'arena esistente. Gli esperti mantengono il loro placement.
2. `DS4_CUDA_QWEN4_STAGE_EXPERTS=1`: per T <= 8, copia soltanto gli esperti selezionati e fornisce puntatori per slot agli stessi dot product. Router, ordine degli slot, accumulazione e shared expert sono invariati.
3. `DS4_CUDA_QWEN4_EXPERT_CACHE=1`: conserva i range selezionati nell'arena bounded; i miss continuano a usare le copie transitorie. Riserva 4 GiB del limite per i pesi densi. È una cache statica diagnostica, senza eviction adattiva.
4. `DS4_CUDA_QWEN4_PREFILL_STAGE=1`: per T >= 64, trasferisce i tensori completi nel buffer streaming già gestito dal backend. Il layout e i kernel prefill restano identici. Lo scratch dei tile è distinto dal buffer dei pesi.

I flag rispettano 0/false/off/no. Gli esperimenti usano esplicitamente
`DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=16`; non attivare la cache diagnostica
senza il budget finito: il codice lascia disabilitate le nuove cache se il limite è illimitato. Metal, ROCm e i launcher non sono modificati.

## Prime misure direction

Contesto allocato 64, frontiere 16/32, teacher-forced 4 token per frontiera,
due ripetizioni, medesimo testo `speed-bench/promessi_sposi.txt`. Queste misure
non promuovono una candidate e non certificano conversazioni lunghe.

| Variante | Decode osservato |
| --- | ---: |
| Originale | 0,40 token/s |
| Solo pesi densi in VRAM | 2,74–2,81 token/s |
| Densi + copie degli esperti selezionati | 3,36–3,62 token/s |
| Con cache degli esperti, primo passaggio | 5,54–7,44 token/s |
| Con cache degli esperti, ripetizione calda | 32,14–32,97 token/s |

I quattro dump per ogni candidate coincidono su tutti i 248320 logit,
compresi quelli dopo decode; massimo errore assoluto 0 e argmax identici.
Le suite CUDA Flash Next contro gli oracle double passano con staging,
cache e prefill staging attivi. Sono stati conservati binari e output.

Il primo run slow della baseline è INVALID: l'ultimo campione è stato
contaminato da una suite CUDA concorrente. Il run pulito è stato completato
in sequenza, senza altri carichi GPU. La prima registrazione hardware è
avvenuta dopo il decode: non misura il picco VRAM né il link sotto carico.
Non se ne traggono conclusioni sul PCIe.

## Gate di promozione

Richiesti: warm-up e almeno cinque ripetizioni con CV <= 5%, logit del
medesimo GGUF, continuazioni nuove, frontiere più grandi, suite kernel,
MTP, checkpoint e regressioni pertinenti. Una failure o un gate non
verificato impedisce la promozione. Le tolleranze e i golden non cambiano.
I profiler Nsight non risultano disponibili nel PATH locale.

Gli artefatti completi sono in
`performance-results/qwenflash-3090-20261003/` (ignorati da Git), incluso
`experiment.json`. Il branch è `codex/qwen-flash-3090`.

## Verifiche successive

Baseline pulita a frontiere 16/32, 4 token teacher-forced, un primo sweep
scartato e cinque sweep misurati: mediana 0,40 token/s a entrambe,
CV 1,77%. Candidate: mediana 35,50 a 16 e 33,59 a 32; CV rispettivamente
7,27% e 3,59%. Il punto a 16 è **NEED_MORE_DATA**, non un risultato
prestazionale qualificato. Tutti e quattro i dump rimangono bit-exact.

La registrazione durante il run candidate conserva 88 campioni: picco
VRAM complessiva 10288 MiB (desktop incluso), RSS del processo 44286300 KiB,
VmSwap del processo sempre 0. Due campioni della baseline al 100% di utilizzo
GPU e tutti i campioni candidate mostrano PCIe Gen4 x2; il dispositivo dichiara
Gen4 x16 massimo. La causa fisica dello slot/link non è stata verificata.
Il primo monitor idle resta separato e non viene usato come picco.

Gate esistenti, soglie invarianti:

- Suite CUDA `tests/test_qwen4_cuda`: PASS, con tutti i flag e limite 16 GiB.
- `tests/test_qwen4_mtp_limits.py`: PASS per one-row, two-rows/depth-three e unfused;
  il testo greedy è identico con e senza MTP.
- `tests/test_qwen4_logit_dump.py`: PASS a 6/132/136 righe; massimo scarto fra
  all-row e teacher-forced 0 / 0,000561714 / 0,000305176, limite originale 0,03.
- `tests/test_qwen4_checkpoint_replay.py`: inizialmente FAIL, tools-omit turno 2
  con cached_tokens=0. Non è stato qualificato quel risultato. Una regressione
  statica ha confermato due assertion sul guard famiglia preesistente del server:
  includeva Qwen27 ed escludeva Qwen4 nonostante il renderer visibile Qwen4 già
  presente. Il guard ora include entrambe, mantenendo esclusioni Responses/
  Anthropic, reasoning aperto, error/length e MTP/KV campionati. La regressione
  statica e il test originale completo di nove turni, riavvii e disk eviction
  passano. Il caso prima fallito riusa 3348 token e passa da 56,76 a 9,53 secondi.

Il controllo full-vocabulary contro la CLI ricompilata dal CUDA originale
congelato è PASS su 200 posizioni. Anche i quattro dump delle frontiere
prestazionali 64/128 e dopo 16 token coincidono byte per byte. Le continuazioni B-tree e TCP provengono dal corpus Flash esistente;
sono input canonici, non oracle numerici del servizio hosted.

## Nota sulla scheda madre

L'inventario Windows identifica MSI MAG Z790 TOMAHAWK MAX WIFI (MS-7E25),
RTX 3090 sul bus `00000000:01:00.0`, driver 610.62. Le
[specifiche MSI](https://mx.msi.com/Motherboard/MAG-Z790-TOMAHAWK-MAX-WIFI/Specification)
dichiarano PCI_E1 CPU x16 (x8 se M2_1 è occupato), PCI_E3 chipset x4,
PCI_E2 x1. La condivisione documentata con M2_1 non spiega da sola il x2
misurato. Slot effettivo, riser, contatti e negoziazione non sono stati
verificati; nessuna impostazione BIOS/driver o configurazione hardware è
stata cambiata durante questi test.


## Gerarchia adattiva e checkpoint: estensione richiesta dall’utente

La cache statica precedente è superata. Il prototipo attuale implementa
`DS4_CUDA_QWEN4_TIERED_CACHE=1` esclusivamente per Flash CUDA su una GPU,
con budget pesi finito. Il loader lascia la mappa principale **non pinnata**;
non vengono preparati preventivamente interi tensori di esperti. I pesi densi
vanno in VRAM, mentre tutti i percorsi routed, compresi i batch oltre 8 righe,
forniscono ai kernel pesi effettivamente trasferiti in VRAM. Nessun puntatore
host non registrato può essere usato come fallback per un peso denso.

Il primo test aveva individuato proprio un fallback non valido per T > 8:
è stato corretto prima dei test modello. I fallimenti intermedi sono conservati;
non sono stati cambiati oracoli, golden o tolleranze.

### Policy e trasferimenti

- La graduatoria nasce vuota e viene appresa dalle scelte effettive del router,
  per tensor/layer. Tre istogrammi esatti conservano gli ultimi 32, 64 e 128 token.
  La policy scelta usa frequenza sugli ultimi 128 token, normalizzata sul numero
  di righe effettivamente osservate all’avvio; la graduatoria cambia a ogni route.
  La storia più vecchia di 128 token non contribuisce mai alla frequenza.
  Finestre 32/64 e una regola Pearson che rileva cambiamenti sono confronti
  diagnostici: sotto pressione, la finestra 128 è risultata più veloce.
- La VRAM protegge **tutti** gli esperti della richiesta prima di scegliere una
  vittima, compresi i hit successivi e le selezioni duplicate. La vittima è
  l’esperto con frequenza recente minima, con LRU a parità di frequenza.
  Una richiesta più fredda della vittima usa VRAM transitoria e non distrugge
  il working set caldo. Gli slot mantengono a budget la capacità realmente allocata.
- La RAM è una cache separata di byte quantizzati originali, con allocazioni
  pinned e limite predefinito di 16 GiB (`DS4_CUDA_QWEN4_RAM_CACHE_MB`). Le copie
  già presenti in VRAM hanno priorità di eviction dalla RAM, per dare spazio
  al livello intermedio. Se un esperto esce dalla VRAM e la sua copia RAM non
  esiste più, viene demotato quando il budget lo consente.
- I miss RAM leggono il GGUF tramite le primitive allineate di DS4 e O_DIRECT,
  quando disponibile. Gli esperti sono immutabili: il livello SSD è il GGUF
  originale e non richiede scritture di eviction. Sotto pressione si usa uno
  staging transitorio sincronizzato, senza rubare buffer ancora in uso.
- Il prefill conserva layout, selettori e kernel originali, ma trasferisce
  soltanto gli esperti effettivamente utilizzati. Gli esperti non scelti non
  vengono letti per completare artificialmente il buffer della GPU.

`DS4_CUDA_QWEN4_EXPERT_CACHE_MB` restringe il budget VRAM per i test;
`DS4_CUDA_QWEN4_EXPERT_POLICY=lru` e
`DS4_CUDA_QWEN4_EXPERT_WINDOW=32|64|128|adaptive` sono confronti diagnostici.
`DS4_CUDA_QWEN4_CACHE_STATS=1` stampa i totali senza logging di ogni eviction;
`DS4_CUDA_QWEN4_EXPERT_TRACE` conserva le selezioni originali del router.
La RAM non è esclusiva: una copia può esistere in entrambi i livelli, ma le
copie GPU sono vittime preferenziali della RAM. Budget RAM, buffer temporanei
e page cache del sistema operativo sono misure distinte.

### Persistenza della graduatoria

Il payload Flash può aggiungere una sezione versionata contenente storia
recente, frequenza ricostruibile, recency e indicazioni dei livelli. Non salva
puntatori, indirizzi di allocazione o pesi duplicati. Il reader accetta anche
checkpoint Flash precedenti senza la sezione. Il formato della sezione viene
validato interamente prima di cambiare la policy; quella del figlio non deve
restare nel genitore ripristinato.

Il restore ricarica subito la graduatoria. La residenza fisica dei pesi viene
ricostruita **su richiesta**, rispettando i budget correnti; non copia
automaticamente parecchi GiB durante il caricamento del KV. Le copie
immutabili ancora utili possono essere conservate, ma frequenza e recency
vengono riportate allo stato salvato.

### Correttezza e integrazione verificate

- Policy C: pinning, vittime, capacità miste, startup, cambio di distribuzione,
  conservazione dei conteggi e sparizione completa della storia oltre 128 token.
- Suite CUDA completa con VRAM/RAM da 1 MiB: sostituzioni, demotion, miss da file,
  oracoli numerici originali, round trip metadata e rifiuto di lettura troncata.
  Il test sintetico usa pread buffered perché modifica i pesi durante la prova;
  i test GGUF reali registrano O_DIRECT attivo.
- **200 posizioni × 248320 logit bit-exact con la CLI originale congelata**:
  due continuazioni tecniche da 32 posizioni e 136 righe che attraversano il
  chunk. Cache GPU/RAM limitate a 512 MiB, 56,3 GB utili di letture esperti,
  eviction realmente esercitate. Errore massimo 0; nessun valore non finito.
- Checkpoint del server: testo e immagini, decode normale e MTP, skill e subskill
  annidate a tre livelli, ripresa del subagente in una sessione nuova,
  token/logit identici, continuazione identica, file corrotto respinto senza
  mutare lo stato. Il test aggiunto confronta anche la storia degli esperti
  del genitore prima e dopo restore.
- Vision: proiettore Flash Q8_0 di ggml-org, 616703104 byte, SHA256
  `b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad`.
  Non viene usato il proiettore 27B. Le prove di checkpoint con immagini non
  qualificano da sole la qualità dell’encoder quantizzato rispetto al checkpoint HF.
- MTP one/two rows e unfused: PASS. Replay HTTP di nove turni con thinking,
  tool e omissione/echo reasoning, riavvii e budget KV: PASS; il restore conserva
  più di 3000 token di prefisso e la graduatoria.

### Selezione dell’algoritmo

Con cache abbastanza grandi da contenere il working set breve, LRU e finestre
recenti hanno lo stesso numero di miss: quel carico non può scegliere una policy.
Il replay di 9792 richieste reali con 1 GiB GPU e 2 GiB RAM distingue invece le
policy: adaptive riduce i byte H2D di circa 7,8% rispetto a LRU e le letture SSD
stimate di circa 26,7%. La finestra fissa 128 riduce ulteriormente i byte SSD,
ma aumenta quelli H2D. Non c’è ancora prova di un vincitore universale.

Fonti primarie: [Caffeine W-TinyLFU](https://github.com/ben-manes/caffeine/wiki/Efficiency),
[ARC](https://www.usenix.org/conference/fast-03/presentation/arc-self-tuning-low-overhead-replacement-cache),
[MoE-Infinity](https://arxiv.org/abs/2401.14361). Motivano recency/frequenza,
adattamento e tracing; il prototipo non modifica il router né importa i loro
kernel senza gli oracoli DS4. Il replay è una stima dei byte utili e non misura
TPS, allineamento I/O o overlap; il confronto end-to-end usa un secondo prompt
tecnico, mantenendo identiche selezioni e quantizzazione.

La prima direction del percorso a tre livelli misura 6,57/8,79 token/s al primo
passaggio e 29,33/29,55 nella ripetizione calda, a frontiere 64/128 con 16 token
teacher-forced. Questi numeri non certificano 20 TPS su qualunque contesto.
Le prestazioni finali devono rispettare warm-up, cinque campioni e CV ≤ 5%.


### Scelta misurata e varianti scartate

Cinque sweep misurati dopo il warm-up, cache 1 GiB VRAM / 2 GiB RAM, prompt
tecnico distinto, 16 token teacher-forced a frontiere 64/128:

| Policy | 64 token, mediana / CV | 128 token, mediana / CV |
| --- | ---: | ---: |
| Finestra recente 128 | 3,96 TPS / 4,19% | 3,46 TPS / 3,57% |
| Finestra variabile Pearson | 3,13 TPS / 1,90% | 3,27 TPS / 3,16% |

La finestra 128 è quindi il default della cache diagnostica. Il ranking resta
adattivo e locale: non è LFU sull’intera conversazione. Non viene dichiarato
un optimum per ogni possibile contesto/hardware. Il replay di soli byte non
ha prevalso sulle misure end-to-end.

È stato provato anche il popolamento VRAM dal buffer prefill tramite D2D.
Suite CUDA, 200 posizioni full-vocabulary e checkpoint passavano, ma la
mediana calda scendeva da 30,07/29,77 a 19,30/18,01 TPS (frontiere 64/128).
Quella variante è **REJECTED** e rimossa. Il prefill apprende la graduatoria;
la residenza viene stabilita su richiesta, senza perturbare un working set caldo.

### Parità dei checkpoint con Qwen 27B

Due guard preesistenti escludevano Flash dalla cattura del prefisso di sistema
e del punto prima del reasoning. Il test nel client ExMekh superava il suo
limite invariato di 600 secondi; i log mostrano un rebuild da zero di 4611 token
per canonicalizzare una coda di 67 token. Quel run è FAIL, non validato.
La regressione statica è prima fallita e poi passata con l’inclusione di Flash.
Flash usa il payload completo (con vision e graduatoria), mentre Qwen 27B può
usare il frontier compatto residente su GPU. Non si estende impropriamente
l’API compatta qwen35 a qwen4exp.

Il gate del payload ora esercita anche il frontier pre-thinking: cattura,
continuazione, rollback e confronto bit-exact di token/logit/storia esperti.
I test reali del client restano necessari per qualificare i passaggi HTTP.
Metal, ROCm e inferenza distribuita non sono disponibili su questa macchina:
**NOT_VERIFIED**, nessuna qualificazione di quegli hardware è dedotta dai test CUDA.


### Misure finali della policy selezionata

Medesimo Q2, testo letterario, ctx allocato 256, frontiere 64/128, 16 token
teacher-forced, primo sweep escluso e cinque sweep misurati:

| Configurazione | 64 token (mediana / CV) | 128 token (mediana / CV) |
| --- | ---: | ---: |
| Originale | 0,41 TPS / 2,20% | 0,41 TPS / 0,00% (CSV arrotondato) |
| Cache selezionata, limite 16 GiB | 29,19 TPS / 6,57% — NEED_MORE_DATA | 28,49 TPS / 4,67% |
| Cache selezionata, limite 20 GiB | 30,27 TPS / 4,68% | 29,50 TPS / 4,05% |

Il primo passaggio a 20 GiB misura 6,82/9,01 TPS: non equivale al caso caldo.
I quattro dump full-vocabulary delle frontiere a 16 e 20 GiB sono bit-exact
con l’originale, compresi argmax e token teacher-forced. Il contesto breve
occupa solo il proprio working set: picco GPU complessivo 13266 MiB, RSS
14633600 KiB, swap 0. Il budget 20 GiB non forza copie inutili. Il maggiore
budget viene esercitato anche con contesto 16384, vision, MTP e client reale;
i picchi di quel run sono registrati separatamente.

Per riprodurre il percorso diagnostico: `DS4_CUDA_QWEN4_TIERED_CACHE=1`,
`DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=20`, RAM cache predefinita 16 GiB. Facoltativo:
`DS4_CUDA_QWEN4_CACHE_STATS=1`. Rimane disabilitato nei launcher predefiniti:
una misura breve PASS non certifica conversazioni arbitrarie o altri backend.


Il replay reale successivo conferma la correzione dei frontier: prefisso
4515 token catturato in 185 ms, richiesta 4545 in 160 ms; il checkpoint dopo
la risposta valuta soltanto 62 token (2,56 s). Al riavvio del client il sistema
viene ripristinato in 117 ms e il prefill valuta solo 30 token (1,57 s).
Entrambi i run del test ExMekh restano tuttavia **FAIL**: nel primo il server
respinge argomenti generati per `mock-skill`, il cui schema richiede un oggetto
vuoto senza proprietà aggiuntive; nel secondo respinge `mock-tool`, non ammesso
nello scope corrente della skill. Nessuno schema viene rilassato né alcun argomento
scartato per trasformare la failure in PASS. La causa completa di generazione/
protocollo richiede un trace dedicato; non viene attribuita ai checkpoint
senza tale evidenza. Il percorso sperimentale non viene promosso nei launcher.


Le sei prove vision/memoria nel client ExMekh installato sono **PASS**:
colori rosso/blu e memoria di due immagini con tool reali, per Chat Completions
e Responses. Segreti e ricevute casuali vengono recuperati dopo l’eliminazione
della nota dal disco, senza reintroduzione nel prompt utente; output strutturato
e eventi SSE verificati. Anche MTP finale one-row, two-rows/depth-three e
unfused è PASS, con continuazione greedy identica al percorso senza MTP.

La prima prova HTTP SSD ha usato una directory errata nel **runner**: il test
cerca `agentic.*/*.dsk` sotto il root del cache, ma era stato passato il percorso
interno `/tmp/ds4-agentic.*`. Le tre assertion sui file sono quindi **INVALID
setup**, non evidenza di failure del checkpoint; i log mostrano i `.dsk`
creati. Il quarto caso aveva invece raggiunto il limite di risposta 64 token
(`incomplete`) dopo un restore effettivo del parent. I quattro test vengono
ripetuti invariati con `--kv-disk-dir` e il root corrispondente, con trace.

Il run completo client/vision/HTTP, limite pesi 20 GiB, registra 814 campioni: picco GPU 16014 MiB, RSS 21313948 KiB, swap 0 KiB; PCIe Gen4 x2 durante il carico.


### Esito della verifica richiesta

Con il root SSD corretto, i test HTTP invariati sono:

- `test_allowed_skill_checkpoint_and_return`: **PASS**.
- `test_nested_skill_files_are_consumed_in_return_order`: **PASS**.
- `test_z_corrupt_checkpoint_is_rejected_without_consuming_it`: **PASS**,
  senza skip e senza consumare il file corrotto.
- `test_opaque_agent_context_switch_restores_parent_with_distinct_registry`:
  **FAIL** sulla risposta finale `incomplete` contro `completed`, limite 64 token.
  Il trace conferma restore di 382 token, registry del parent ripristinato,
  comprensione corretta del risultato del worker e risposta finale troncata.
  Il ripristino non viene confuso con il PASS dell’intero test.

Il test del client completo resta **FAIL** per argomenti/scope generati.
Il gate globale di promozione è quindi **NOT QUALIFIED**: cache disabilitata nei
launcher, stesso modello Q2 e nessuna riduzione dell’intelligenza ottenuta tramite
omissione di esperti, modifica del router, compressione KV o cambio di quantizzazione.
I successi numerici e dei checkpoint restano registrati separatamente dai fallimenti
end-to-end. Artifact `final-qualification.json` raccoglie stato, hash e misure;
log e trace delle prove fallite sono conservati. Tutti i processi del modello avviati
per queste verifiche sono stati arrestati.


## Audit upstream e ripresa del 4 ottobre 2026

Il fetch di `antirez/ds4/main` restituisce ancora
`0aaea5a238fb41a35106a551e73c8409dfb751ac`, lo stesso commit fissato nel merge.
Il confronto distingue `upstream/main` → `HEAD` dalle modifiche sperimentali
non ancora committate. Non viene eseguito un nuovo merge.

- Le 135 funzioni `qwen4_*`/`ds4_qwen4_*` del core hanno corpo identico.
- Il file CUDA Flash importato differisce soltanto per il guard sm_86 del kernel
  attention raggruppato, già documentato nel report di merge dopo un fault
  riproducibile. Nessuna funzione upstream è stata eliminata. Il guard mantiene
  i kernel scalar/split esistenti; non viene rimosso per aumentare la velocità.
- Tile FP16 per IQ2/Q2/Q4/MXFP4, doppia componente TF32, scelta GEMV/GEMM,
  tile schedule degli esperti, GDN, fusioni decode e flush anticipato della
  command queue sono già upstream e conservati.
- Nel CUDA condiviso il dispatch Q2_K/IQ2_XXS/MXFP4 MMQ resta upstream. Q4_K usa
  il dispatch locale che supera gli oracle sm_86; quello generico era stato
  respinto dal gate numerico. Il loader selettivo no-copy e la risoluzione dei
  pesi host sono conservati. La copia predefinita aggiunta per Qwen27 denso
  riguarda `qwen35`, non rende residenti i 41,73 GiB Flash.
- Upstream non contiene la gerarchia adattiva Flash VRAM/RAM/SSD richiesta qui.
  La registrazione dei pesi host non è una cache VRAM; la VRAM poco utilizzata
  del comando originale è coerente con quel percorso.
- Anche `s->prefill_cap = ctx_size` nella sessione Flash era già upstream.
  Riportava un chunk diverso da quello del grafo; ora si espone `cap_tokens`.
  Il sync Flash usa già la capacità del grafo. Il reader dei payload Flash non
  vincola il campo storico del chunk; viene aggiunto un replay del vecchio header.

Fonti: [kernel Flash del commit importato](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4_qwen4_cuda.cuh),
[core del commit importato](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4.c),
[CUDA condiviso](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4_cuda.cu).
Il confronto per funzione e i diff sono salvati in `prefill-vram/upstream-*`.

Le modifiche ancora sotto gate fanno usare al prefill puntatori diretti agli
slot adattivi, evitando di ricopiare i hit in un intero layer transitorio.
Ogni miss non ammesso viene comunque eseguito in VRAM transitoria. La tabella
puntatori vive fuori dallo scratch del tile schedule. Il budget pesi automatico
viene fissato alla prima allocazione dei pesi, dopo il contesto, con una riserva
workspace di 2 GiB; una riserva densa prima conteggiata due volte è corretta.
I flag restano diagnostici e disattivati per default durante la qualifica.


### Budget degli esperti rispetto al contesto configurato

Il GGUF locale dichiara `qwen4exp.context_length=262144`. Il budget automatico
usa `cudaMemGetInfo` dopo aver allocato il KV, lo stato ricorrente, i buffer del
prefill e il modello ausiliario. Il limite dei pesi è la VRAM libera meno 2 GiB
per workspace, più i byte già residenti della cache. Il budget degli esperti è
quel limite meno i pesi fissi effettivi del GGUF. Un `--ctx 100000` lascia così
più spazio agli esperti di `--ctx 262144`, senza cambiare il KV o il router.

La riserva fissa iniziale di 4 GiB era insufficiente: il primo prefill reale
4699 token con MTP/vision terminava HTTP 500 al layer 40 perché gli esperti
occupavano lo spazio dei pesi densi successivi. Questo run è **FAIL**. La
riserva ora viene calcolata dalla directory dei tensori: 5,14 GiB per il Q2
locale, comprese le proiezioni MTP. Le righe embedding raccolte dalla CPU e la
tabella PLE su disco sono escluse; gate/up/down instradati sono contabilizzati
nella cache degli esperti. Anche ogni nuova arena deve lasciare fisicamente
lo spazio dei pesi fissi non ancora caricati, oltre al workspace.

Il log espone contesto, chunk, pesi fissi e budget degli esperti. La capacità
di esperti completi è un intervallo perché gate/up/down dei diversi layer,
MTP incluso, hanno dimensioni e formati quantizzati diversi. Gli slot delle
statistiche runtime sono singole proiezioni, non altrettanti esperti completi.

La capacità dipende dal contesto **allocato**, non dal numero di token già
usati: avviare il server al massimo e usare solo 100k token non restituisce
automaticamente il KV riservato alla cache. La prova comparativa usa contesti
allocati 16384/100000/262144 e frontiere identiche 256/512, con warm-up e cinque
campioni. Non equivale a una prova di qualità su un prompt lungo 100k token.
La correzione e la cache restano sotto gate fino ai risultati numerici e live.

Nella ripresa, la suite locale/mock degli agents con lo scheduler del progetto
supera **715 test**, senza skip. Il run seriale precedente ha quattro failure
di avvio del provider nella steering recovery, mentre gli stessi quattro casi
isolati passano; si conservano entrambe le evidenze senza attribuire questa
interferenza ai pesi del modello. I test reali sono riportati separatamente.


Il preload dei pesi fissi è necessario anche dopo il conteggio esatto: la
variante densa lazy fallisce a 100k presso i layer 46/47. Ridurre la granularità
dell’ultima arena da solo non basta; con cache a 16,39/17,17 GiB rimangono 1,90
GiB liberi e gli ultimi pesi non sono ancora residenti. Entrambi i tentativi
sono **FAIL**. Il percorso aggiornato prepara invece tutti i pesi fissi dopo
il contesto e prima di ammettere gli esperti. Un test CUDA azzera il backing
file dopo il preload e verifica che le proiezioni continuino a usare i valori
originali, anche ricreando la configurazione del contesto.

Misure del percorso aggiornato, RTX 3090 sm_86, stesso Flash Next Q2, chunk 256,
MTP e vision disabilitati nel benchmark, frontiere reali 256/512 token. Un
warm-up e cinque campioni per frontiera; tutti i CV sono inferiori al 2%:

| Contesto allocato | Quota esperti VRAM | Byte esperti effettivamente riservati | Prefill caldo 256 / 512 token |
| --- | ---: | ---: | ---: |
| 16384 | 14,75 GiB | 14,02 GiB | 91,96 / 121,28 TPS |
| 100000 | 12,03 GiB | 11,32 GiB | 73,92 / 95,38 TPS |
| 262144 | 6,85 GiB | 6,14 GiB | 56,22 / 64,71 TPS |

100k guadagna il 31,48%/47,40% rispetto alla capacità massima sullo stesso
prefill breve. I sei dump da 248.320 logits sono bit-exact rispetto al percorso
host-mapped originale, con argmax identico, cosine 1 e overlap top-20 1. Il
primo runner confrontava anche il campo `ctx` del dump e si fermava sul
metadata diverso: **INVALID runner assertion**, logits già identici; la
capacità attesa è ora verificata separatamente e gli altri metadata restano
vincolati. Non si cambiano soglie numeriche né golden.

I picchi GPU complessivi sono 22978/23006/23044 MiB; swap 0. Le quote dichiarate
sono limiti, non una richiesta di allocare esperti inutili: i byte effettivi
risentono delle arene, del workspace live e del working set. Il vantaggio non
è un benchmark su un prompt pieno da 100k né una misura decode a quel contesto.
Artifact: `prefill-vram/context-budget-prepared-results.json`, CSV sincronizzati,
monitor hardware a 1 Hz e dump delle frontiere. Il comando completo MTP/vision
e i gate live sono verificati separatamente prima di qualsiasi promozione.


Il gate di checkpoint del percorso con budget dinamico passa con MTP, vision,
contesto 16384 e prefisso testuale popolato da 512 token (611 con l’immagine).
Skill, subagent, sessione ricreata, continuazione e tre livelli annidati
ripristinano 248.320 logits bit-exact; anche la cronologia recente degli esperti
viene ripristinata. Corruzione rifiutata senza mutazione. Il replay legacy del
chunk passa: il primo fixture modificava erroneamente il wrapper VMCP della
vision invece dell’header interno Flash, ed è conservato come **INVALID
fixture**. Le soglie del gate restano identiche.
Artifact: `prefill-vram/context-budget-live/checkpoint-context-budget-mtp-vision.log`.

Due prompt reali di 4699 token, senza prefix hit, con MTP e vision e il comando
utente (ctx 16384, chunk 256), terminano rispettivamente in 273,38 e 230,30
secondi, includendo 8 token di output. Il prefill mostra prevalentemente chunk
15–24 TPS dopo il primo caricamento. Non è una misura di decode a 20 TPS.
La prima suite Agent reale termina **10 PASS, 2 FAIL, 4 NOT_VERIFIED**: passano
health, UI, chat, reasoning, chiamata del tool, persistenza e tutte le quattro
combinazioni immagini/PDF × Chat/Responses. Falliscono cronologia tra turni
(il padre genera `exit_agent` del figlio) e argomento `items` di tipo errato.
Le quattro prove di misuse scadute non qualificano il recupero.
Artifact: `prefill-vram/context-budget-live/agent-live.xml` e trace del server.

Il filtro durante il campionamento escludeva esplicitamente `QWEN4`, quindi
il parser finale rifiutava scope/argomenti solo dopo averli generati. La nuova
grammatica riconosce `<function=...>`/`<parameter=...>`, applica nomi e schema
registrati, mantiene i dati stringa con gli stessi delimitatori del parser
nativo e permette blocchi consecutivi. Il replay MTP usa gli stessi confini di
ragionamento della generazione ordinaria. La suite server e i gate focalizzati
ASan/UBSan/LSan passano; i gate HTTP con oracle e Agent reale sono **PENDING**.
Il difetto è nell’integrazione locale dei vincoli, non nei kernel importati.
Non si promuove la cache sulla base di questa sola correzione statica.
