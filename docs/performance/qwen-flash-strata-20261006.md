# Qwen Flash — pipeline ispirata a Strata, RTX 3090 / WSL

Misure del 6–7 ottobre 2026. Le nuove modalità restano disattivate per default.
I risultati si riferiscono al GGUF locale, non ai TPS pubblicati da Strata.

Il decode a 64K è stato corretto nella v32: l'indice persistente delle vittime
VRAM/RAM conserva la policy esatta, ma evita di ricostruire heap globali per
token. I 14 gate selezionati passano; un oracle CPU indipendente esercita 4096
epoche e il controllo CUDA confronta le vittime con la scansione originale.

| v32, contesto effettivamente usato | Baseline prefill TPS | v32 prefill TPS | Baseline decode TPS | v32 decode TPS |
| --- | ---: | ---: | ---: | ---: |
| 65536 + 64 token | 157,91 | 226,08 | 15,10 | 16,64 |
| 131072 + 8 token | 155,74 | 199,71 | 9,88 | 14,85 |

Logits completi identici alla frontiera e dopo il decode, KL zero. A 64K il
tempo CPU delle vittime passa da 2722,619 ms nella v31 a 172,462 ms nella v32.
Il campione 64K include una sonda diagnostica degli API CUDA; il 128K è fuori
profiler. Sono singoli campioni freddi con baseline compatibile riutilizzata,
non la promozione a cinque A/B. Artefatti:
`kv-decode-v32-64-diagnostic/report.json`, `kv-decode-v32-128/report.json`.

La build precedente v30 migliora il decode ordinario del 13,54% rispetto alla
baseline iniziale nei cinque A/B alternati. Frontiera effettiva 256, chunk
256, 64 righe teacher-forced, due warm-up per processo, policy recente e
quota PCIe 1: mediana 34,49 → 39,16 TPS, CV 2,03%/1,27%; prefill caldo
542,77 → 534,39 TPS (−1,54%). Dieci confronti dell'intero vocabolario sono
bit-identici. Dati: `decode-v30-initial-corrected/final-paired.json`.
Questo guadagno non va sommato ai risultati delle versioni precedenti.

Il confronto che cambia soltanto l'abilitazione dei grafi sulla stessa v30,
con identici scheduler, KV e QSA, misura 38,52 → 39,97 TPS (+3,76%), CV
2,86%/1,17%, prefill −0,96%, logits bit-identici. Dati:
`decode-islands-v30-corrected/graphs-paired.json`. La v29 isolata era neutra:
spostare il replay sulla coda ordinaria del forward risolve un costo reale
dell'integrazione. Il selettore QSA conserva le chiavi nei registri e il
mirror KV differisce la pubblicazione del decode senza bloccare l'attenzione.

I 12 gate v30 selezionati e il test server passano, inclusi 68 confronti
teacher-forced bit-identici, primitive QSA/KV, stato, MTP e protezione 27B.
La prova Flash v30 a 65536 token effettivi e 64 righe decode è bit-identica
alla baseline: prefill 157,91 → 233,75 TPS (+48,03%), decode 15,10 →
10,81 TPS (−28,41%). La baseline diagnostica compatibile è riutilizzata;
questo campione non promuove le prestazioni. Il KV è a 32768 celle residenti:
la regressione decode è risolta nella v32 sopra. Dati storici:
`kv-decode-v30-64-sustained/report.json`.
La prova v30 a 131072 token effettivi più otto righe decode è completata:
prefill 155,74 → 204,60 TPS (+31,37%), decode 9,88 → 10,53 TPS (+6,58%),
due confronti dei logits completi bit-identici. Tempo totale 848,9 → 645,6 s.
È un singolo A/B a cache fredda, distinto dalla promozione a cinque campioni.
Dati: `kv-decode-v30-128/report.json`.
La matrice PCIe corrente, il profilo e HTTP/vision finali sono in corso.
La policy `stable` v29 peggiorava il
prefill breve caldo del 13,19%; per quel workload resta sperimentale. La
policy recente sopra conserva il prefill entro −1,54%. Le misure storiche
seguenti documentano gli incrementi e i fallimenti, senza sostituire i gate
del binario corrente.

La prova 27B v27 completa 64K effettivi più otto righe teacher-forced:
414,10 TPS di prefill e 0,69 di decode. Il confronto numerico del modello
intero con attention originale su un layer contiguo è bit-identico; questo
riferimento viene copiato dalla RAM e non misura tutti i KV residenti in
VRAM. Il risultato è maggiore capacità con un costo di decode importante.

La nuova ispezione del decode Strata ha individuato l'assenza dei grafi CUDA
sul percorso Flash. La v25 aggiunge due isole per layer. A 256 token
effettivi, 64 righe di decode, chunk 256 e due warm-up per processo, cinque
A/B alternati rispetto alla v24 ordinaria danno:

| Metrica | v24 ordinaria, mediana TPS | v25 grafi + scheduler GPU, mediana TPS | Differenza |
| --- | ---: | ---: | ---: |
| Prefill a cache calda | 545,86 | 545,37 | −0,09% |
| Decode steady | 35,09 | 37,83 | +7,81% |

CV decode 2,52% e 1,20%; ogni campione candidato supera tutti i campioni
baseline. Dieci confronti dei logits completi sono bit-identici.
I dati sono in `decode-graphs-v25-paired/graphs-paired.json`. È un confronto
dell'incremento grafi/scheduler con la v24 ordinaria, distinto dalla baseline
iniziale e dalla prova a 64K. Non dimostra ancora il risultato a contesto lungo.

Nella v26 passano le primitive KV e ibride, il prefill misto CPU con i gate
stretti e stato/checkpoint/MTP con grafi. Le 68 righe teacher-forced con
grafi e quota PCIe 0,5 sono tutte bit-identiche alla baseline iniziale
(KL e delta loss zero). Il gate con grafi e KV passa con un budget
residente valido: 1024 è stato rifiutato dal controllo `sel_stride >= 2052`,
prima dell'inferenza; la ripetizione usa 4096. Non sono state cambiate
tolleranze o golden. Contesto lungo e isolamento del solo effetto dei grafi
restano da completare.

| 65536 token effettivi, chunk 1024, otto righe decode | Prefill TPS | Decode steady TPS | Logits completi |
| --- | ---: | ---: | --- |
| Baseline iniziale compatibile, riutilizzata | 201,78 | 12,62 | riferimento |
| v26, grafi e batch separati K/V e QSA | 235,76 | 12,11 | bit-identici |
| v28, pubblicazione K/V e QSA unificata | 236,01 | 12,27 | bit-identici |

Dati: `kv-decode-v26-64/report.json`, `kv-decode-v28-64/report.json`.
Misure diagnostiche singole, non una qualifica di velocità a cinque A/B.
La v29 aggiunge il selettore QSA con chiavi nei registri, estratto come idea
dal kernel Strata: 72 confronti passano anche l'ordine degli indici, i pareggi
a zero e i limiti del dispatch. Il probe della primitiva dimezza il costo
a 64K e lo riduce di circa due terzi a 128K; il guadagno del token intero
va misurato separatamente.

## Identità e condizioni

- Checkout iniziale `1239542282ebcc724a94f754c181f530113ea993`, branch
  `codex/qwen-flash-strata-20261006`; la modifica preesistente a
  `PROJECT_INDEX.md` non appartiene a questo lavoro.
- Strata `1735d6471df29b42c26170efaac1f1446a58640f`, MIT. Riferimento
  architetturale, nessun codice copiato; URL dei file nel manifest baseline.
- RTX 3090 24 GiB, CUDA 12.4, driver 610.88, i9-14900K con 16 CPU visibili,
  Ubuntu 24.04 WSL; circa 44 GiB disponibili prima dell'inferenza.
- PCIe Gen4 x2 confermato sotto carico. Probe pinned 64 MiB, due warm-up e
  cinque campioni: H2D mediana 3,025 GiB/s, D2H 3,067 GiB/s. Il report storico
  da 2,74 GiB/s usa un buffer diverso: non dimostra un cambiamento hardware.
- `Qwen3.8-Flash-Next-Q2.gguf`, 147207127040 byte, SHA256
  `b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`.
  PLE BF16, IQ2_XXS/Q2_K target, Q4_K/MXFP4 NextN invariati.
- Baseline congelata in `performance-results/flash-strata-20261006/baseline`.
  Manifest, diff, hash dei sorgenti e binari sono nella stessa directory di
  risultati; gli incrementi misurati hanno copie immutabili `candidate-vN`.

## Correzioni della pipeline

Lo scheduler esegue triplette complete: prima gli hit GPU, poi CPU e gruppi
PCIe sovrapposti, infine la riduzione negli slot originali. Il pool CPU riusa
la dequantizzazione fino a quattro righe senza quantizzare le attivazioni.

La prima variante prefill perdeva circa il 40%. La diagnosi ha individuato
parti mancanti: letture SSD anticipate per triplette eterogenee, staging in
due gruppi, aggregazione delle copie pinned, dimensionamento dei kernel sugli
esperti realmente attivi e ricostruzione ripetuta della classifica delle
vittime a ogni alternanza gate/up/down. La sola selezione delle vittime
occupava 14,5 s su un benchmark da 42,7 s; con heap per dimensione scende a
circa 0,316 s. Sono stati eliminati anche i cicli ripetuti sul routing.
Gli heap RAM applicano la stessa correzione sotto pressione.
La verifica successiva ha trovato un altro percorso incompleto: con l'arena
RAM ordinaria, il prefill di fallback riutilizzava lo staging ogni due piccole
proiezioni. La build v10 raggruppa anche quelle copie prima di pubblicare
l'evento di fine consumo. I contributi CPU vengono trasferiti in forma
compatta, senza includere gli slot GPU.
Il test HTTP lungo v10 ha evidenziato circa 359000 demozioni: la pipeline
attendeva ancora ciascuna copia D2H prima di proseguire. La build v12 accoda
anche le demozioni e pubblica i byte RAM al confine del gruppo o quando una
sorgente viene richiesta prima. Un oracle di pattern indipendenti verifica
180 MiB di demozioni intercalate a upload, lettura in flight e riuso oltre
entrambi i buffer. Il reset drena le copie prima di liberare la RAM.

Il KV usa copie esplicite, pagine da quattro celle e il working set QSA
originale. Prima dello spill mantiene slot identici agli indici logici;
dopo lo spill risolve i blocchi selezionati e divide i batch bounded. Il
predittore NextN resta residente perché il percorso attuale non prefilla
il proprio storico. Il batching nativo intersessione usa il fallback ordinato.

## A/B alternati: incremento v5

Cinque coppie A/B e B/A, processi freschi, due warm-up per processo, un
campione misurato, fuori profiler. Prompt `promessi_sposi.txt`, contesto
allocato 2048, chunk 256, otto token teacher-forced, budget pesi 20 GiB,
stessi pesi e impostazioni. Variante: scheduler abilitato, quota PCIe 1,
prefill abilitato; KV streaming e profilo persistente disabilitati.

| Frontiera | Metrica | Baseline TPS | v5 TPS | Delta |
| ---: | --- | ---: | ---: | ---: |
| 256 | Decode ordinario | 32,03 | 35,45 | +10,68% |
| 512 | Decode ordinario | 26,84 | 28,98 | +7,97% |
| 256 | Prefill caldo | 178,53 | 164,36 | −7,94% |
| 512 | Prefill caldo del suffisso di 256 token | 232,16 | 234,19 | +0,87% |

CV decode: baseline 0,52% / 3,22%, variante 2,07% / 1,61%. Il guadagno
decode supera la variabilità. Il prefill a 256 token è variabile e ancora
peggiora; non costituisce una promozione generale del nuovo prefill.
La mediana del primo prefill a cache esperti fredda passa da 34,00 a 50,38
TPS (+48,18%); la variabilità della baseline fredda è maggiore. La mediana
del benchmark completo passa da 21,90 a 17,60 s (−19,65%). Non è TTFT HTTP.
I venti confronti dei logits completi alle frontiere e dopo decode sono
bit-identici. Dati: `pipeline-v5-ab-paired.json`.

## A/B finale v10, scelta automatica

Stesse condizioni e cinque coppie alternatamente A/B e B/A. La scelta
automatica abilitata riguarda soltanto decode; prefill conserva lo scheduler
ordinario con il raggruppamento corretto delle copie da RAM.

| Frontiera | Metrica | Baseline TPS | v10 TPS | Delta | CV baseline / v10 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Decode ordinario | 30,59 | 34,73 | +13,53% | 3,84% / 1,38% |
| 512 | Decode ordinario | 25,27 | 29,09 | +15,12% | 3,78% / 3,85% |
| 256 | Prefill caldo | 171,00 | 168,64 | −1,38% | 0,39% / 3,32% |
| 512 | Prefill caldo del suffisso | 222,64 | 222,98 | +0,15% | 0,71% / 0,38% |

Il guadagno decode supera la variabilità; nessuna delle altre metriche
mostra un rallentamento oltre il 5%. I venti confronti full-vocab prima e
dopo gli otto token teacher-forced sono bit-identici. Queste misure
qualificano il guadagno nel decode ordinario, non un aumento universale
di TTFT, prefill, MTP o contesto lungo. Dati: `auto-v10-ab-paired.json`.

L'A/B successivo sulla build v12 conferma il guadagno del percorso automatico:
decode 30,97 → 34,46 TPS a 256 (+11,27%; CV 1,65% / 0,92%) e 25,47 → 28,47
a 512 (+11,78%; CV 3,54% / 3,13%). Prefill caldo 172,21 → 169,49 (−1,58%)
e 224,81 → 221,63 (−1,41%). Tutti i venti confronti sono bit-identici.
La ricalibrazione completa sceglie ancora decode GPU, prefill ordinario;
chiave `ed4833ccf469e4fc`. Dati: `auto-v12-ab-paired.json`,
`calibration-v12/calibration.json`, `tuning-v12.txt`.

## KV e profilo

Primo attraversamento reale: 16384 token, 12288 celle residenti per layer,
chunk 256, otto token teacher-forced. Full logits prima e dopo decode
bit-identici. Un campione: prefill 145,87 → 158,09 TPS, decode 25,38 → 22,82
TPS, totale 118,6 → 107,9 s. Qualifica capacità e numerica; non promuove
prestazioni. Dati: `kv-spill-16k/report.json`.

La build finale supera anche 32768 token effettivi con 32768 celle residenti,
compreso lo spill nel decode degli otto token successivi. Logits completi
bit-identici prima e dopo decode. Un campione: prefill 145,06 → 151,00 TPS,
decode 20,53 → 22,28 TPS; totale 232,4 → 221,7 s. È una qualifica di capacità
e numerica, non una promozione prestazionale. Dati: `kv-long-v12/report.json`.

A 65536 token effettivi, resident 32768, i logits completi sono ancora
bit-identici, ma il campione automatico è più lento: prefill 126,85 → 82,00
TPS, decode 19,39 → 11,62 TPS; totale 523,3 → 804,3 s. L'arena esperti
automatica del candidato è 31,18 GiB e registra 1380207 demozioni, mentre
preserva la riserva per PLE/agenti/checkpoint. Questo profilo di memoria è
corretto numericamente e resta sperimentale per velocità; non viene
promosso dal solo aumento di capacità. I byte KV/esperti sono nei log.

Il profilo iniziale usa codice, prosa e tool; ciascuno dei 48 layer target
conta 1480 scelte (una per router/token/slot), NextN zero nel workload senza
MTP. File valido di 200736 byte, fingerprint/checksum verificati. Reload e
cambio di dominio nelle tre fixture restano bit-identici. File troncato,
fingerprint diverso e checksum errato vengono ignorati, conservando logits
bit-identici; il salvataggio produce un file nuovamente valido. La sorgente
congelata del profilo resta invariata durante i confronti.

Confronto diagnostico a pari budget 20 GiB, due warm-up e cinque campioni:
policy recente/LRU/profilo appreso hanno prefill caldo 175,79/177,01/174,39
TPS a 256 e 227,51/230,15/228,31 a 512. Il primo prefill del profilo raggiunge
56,40 TPS contro 31,03 della policy recente, ma sposta I/O nell'avvio: totale
45,2 contro 36,0 s. Non è una promozione automatica. Dati: `profile-v6/report.json`.

## Correttezza e stato di promozione

Vision finale v12: pass tutti i 16 gate Flash/27B, target/MTP. Quattro
frontiere multimodali confrontano anche gli ID token e l'intero vocabolario
e sono bit-identiche. Skill, restore in sessione ricreata, tre livelli di
checkpoint annidati e continuazione restano bit-identici; payload corrotto
viene rifiutato senza mutare la sessione. Passano tutti i 40 casi API:
colori, immagini multiple, append/cached continuation, thinking, JSON,
tool, SSE e Anthropic. Il 27B eredita i flag Flash senza attivarne i percorsi.
Dati: `vision-v12/report.json`.

Primitive CPU con riferimento FP64 indipendente, padding Q2_K, tutti i
65536 codici FP16 e righe raggruppate: pass, anche ASan/UBSan e build scalare.
Primitive CUDA, slot originali, duplicati, cache parziale, tre gruppi di
staging, RAM zero, file troncati: pass. Il KV supera blocchi remoti, gruppi
oltre il working set, sessioni indipendenti, restore con altro budget,
primo spill con soli hit e rewind dentro una pagina.

Gli incrementi v3/v4/v5 superano i gate ordinari e MTP sul percorso GPU:
snapshot, restore/reuse, checkpoint di prefill, cancellazione e rewind.
La riserializzazione dei metadati esperti conserva il payload logico
validato anche se la cache fisica ha acquisito slot dopo lo snapshot.
KV-only e profilo persistente sono bit-identici sulle 68 posizioni delle
fixture codice/prosa/tool, con vocabolario completo di 248320 logits.

Le quote CPU non sono promosse. In v6, quota 0,5 con KV supera primitive e
conserva tutti gli argmax delle 68 posizioni, ma fallisce i gate stretti di
checkpoint/replay: 18 assert ordinari e 21 MTP. Le tolleranze non sono state
allentate. KL massimo teacher-forced: 4,08e-9 codice, 6,54e-10 prosa,
7,66e-8 tool; delta loss rispettivamente −6,37e-6, +2,92e-6, +1,41e-6.
Questi diagnostici non sostituiscono i gate falliti. Il tuner misura anche
le quote CPU, ma richiede un report di stato superato e lo stesso binario
prima di ammetterle tra le preferenze automatiche.

La build v10 conserva l'accumulo CPU a otto lane: il tentativo di emulare
l'accumulo warp CUDA aumentava il costo e non risolveva i gate. Con quota
0,5 i fallimenti di stato sono 13 ordinari e 17 MTP. Le 68 posizioni
teacher-forced conservano tutti gli argmax; KL massimo codice/prosa/tool
3,33e-9 / 3,08e-10 / 1,45e-8 e delta loss +7,74e-6 / +5,23e-6 / +6,94e-6.
La qualifica v10 passa gli altri 14 gate su 16. Le primitive CPU originali
passano nuovamente native, ASan/UBSan e build scalare. Dati: `v10/qualification.json`.

La calibrazione v10 misura tutte le 20 combinazioni quota/worker. Le quote
CPU già escluse dai gate usano frontiere diagnostiche 16/32 con due warm-up
e cinque campioni; non certificano il prefill richiesto e non sono eleggibili.
Le quattro configurazioni PCIe 100% usano 256/512. Il tuner sceglie decode
GPU quota 1 e due worker, con guadagno minimo del 15,16% nel suo screening;
prefill ordinario. Il controllo congiunto passa il limite del 5% sulle due
metriche. Le preferenze restano sperimentali e legate alla chiave
`9a800df97794d231`; non si applicano ad altri contesti/budget. Dati:
`calibration-v10/calibration.json`, `tuning-v10.txt`.

I gate 27B con variabili Flash ereditate proteggono l'esclusione del nuovo
percorso. Cinque coppie ordinarie precedenti sono bit-identiche, con decode
sostanzialmente invariato (+0,06% / +0,29%). Su richiesta successiva, la
capacità estesa 27B è stata implementata in questa sessione: parametro
`--qwen-kv-capacity-mode off|grow|hybrid-attention`, FP32 originale e copie
esplicite sotto WSL. La [scheda 4 BIS archiviata](../experiments/qwen27-4bis-20261008/documentation/qwen27-kv-capacity-4bis-20261006.md)
descrive memoria, riduzione attention e gate già superati. Il test a 65536
token effettivi v27 passa capacità e logits completi rispetto al riferimento
contiguo per layer; misura circa 1,45 secondi/token. La modalità rimane
esplicita: non è un miglioramento di decode sul percorso residente.
Questa sezione è storica: l'8 ottobre il 4 BIS è stato ritirato e rimosso
dalla build, come descritto nell'[archivio](../experiments/qwen27-4bis-20261008/README.md).

Qualifiche 32K/64K/128K effettive, misure HTTP e gate vision sono registrati
separatamente; la loro esecuzione finale è ancora in corso. Allocare 128K
non costituisce evidenza di aver usato 128K token.

## HTTP agentico a 128K allocati, 4395/4401 token effettivi

Una richiesta per condizione, quindi diagnostico e non promozione A/B.
Stessi pesi, chunk 256, massimo 32 output token, greedy, MTP caricato.
La API Flash con tool usa il fallback target del percorso constrained
preesistente: questi casi non misurano cicli MTP eseguiti.

| Caso | Baseline TTFT s | v10 con KV s | v12 con KV s |
| --- | ---: | ---: | ---: |
| Cache esperti fredda, 4395 token | 96,185 | 110,321 | 100,526 |
| Prompt nuovo, cache esperti calda, 4401 token | 101,899 | 129,210 | 116,031 |
| Prefisso riusato, 4387 token cached | 1,037 | 0,646 | 0,667 |

Il batching D2H recupera parte del rallentamento, ma il prompt nuovo v12
rimane più lento del 13,9% circa. La combinazione con KV streaming non è
promossa per questo workload. Il prefix hit migliora nel singolo campione;
non basta a generalizzare il vantaggio. Dati: `http-v10-baseline/report.json`,
`http-v10-candidate/report.json`, `http-v12-candidate/report.json`.

## HTTP con MTP effettivo, cinque coppie A/B

Fixture senza tool, reasoning `none`, 34/40 token input effettivi, 32 output,
ctx 8192, chunk 256. Baseline/candidato alternati A/B e B/A in processi
freschi. Candidato: quota PCIe 1, scheduler decode, KV streaming. Il caso
ripetuto ha zero cached token: il prompt è inferiore alla soglia 128 della
prefix cache e non viene presentato come prefix hit.

| Caso | TTFT baseline s | TTFT v12 s | Riduzione | Throughput HTTP dopo primo token, baseline / v12 |
| --- | ---: | ---: | ---: | ---: |
| Freddo | 5,553 | 3,405 | 38,69% | 9,373 / 9,583 |
| Prompt nuovo a esperti caldi | 1,387 | 0,901 | 35,02% | 14,535 / 15,461 |
| Prompt ripetuto a esperti caldi | 0,381 | 0,371 | 2,43% | 49,811 / 52,732 |

Il throughput include protocollo/finalizzazione HTTP e usa gli output token
dopo il primo divisi per il tempo HTTP rimanente: non è il decode del kernel.
Sul caso ripetuto il guadagno è +5,86%, con CV 1,05% / 1,04%. I guadagni
TTFT dei primi due casi superano la variabilità, ma non si generalizzano al
prompt agentico lungo. Tutti i 15 confronti di testo e token usage coincidono;
questi controlli integrano i gate full-logit e di stato, senza sostituirli.
Ogni processo registra cicli MTP eseguiti e draft accettati nel proprio log.
In tutti i dieci processi i contatori sono identici: 39 cicli, 35 accepted,
89,7% secondo il contatore MTP esistente.
Dati: `http-mtp-v12-paired-v2/report.json`.
