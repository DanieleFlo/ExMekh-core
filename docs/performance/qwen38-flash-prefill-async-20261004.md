# Flash Next CUDA: letture anticipate degli esperti

Qualifica locale del 4 ottobre 2026, baseline `e8fb251`, RTX 3090,
CUDA 12.4, `sm_86`, i9-14900K e WSL2. Verdetto **KEEP** per il
prefill Flash con cache CUDA VRAM/RAM/SSD. Le
[evidenze JSON](qwen38-flash-prefill-async-20261004.json) conservano campioni,
statistiche, comandi, hash e limiti della qualifica.

## Risultato

| Workload | Baseline | Candidate | Esito |
| --- | ---: | ---: | --- |
| Flash, 256 token nuovi, cache esperti vuota, mediana di 5 processi | 20,48 token/s | 32,16 token/s | +57,0%; latenza circa 12,50 → 7,96 s |
| Flash, richiesta HTTP dell’agente, 4395 token input, un campione | 117,034 s prefill | 109,929 s prefill | −6,1%; evidenza direzionale |
| Stessa richiesta, prefill + 32 token output | 123,630 s | 116,562 s | −5,7%; stesso testo prodotto |
| Flash, cache calda, frontiere 256/512/768/1024 | 128,75 / 158,73 / 135,30 / 135,35 token/s | 128,26 / 158,20 / 136,21 / 135,09 token/s | Differenze inferiori all’1%; nessun guadagno dimostrato |
| Qwen27, frontiere 256/512, prefill | 32,54 / 413,95 token/s | 32,58 / 417,39 token/s | Nessuna regressione osservata |
| Qwen27, decode teacher-forced | 30,79 / 30,69 token/s | 31,10 / 30,62 token/s | Differenze intorno all’1% |

I tempi del primo workload derivano dal CSV arrotondato a due decimali,
`256 / token_s`, ed escludono l’avvio del processo. Le coppie cold riducono
la latenza del 31,7–39,1%. Il coefficiente di variazione è 3,75% per la
baseline e 2,27% per la candidate. Il JSON include minimo, p10, mediana,
media, p90, massimo e deviazione standard.

Il prefill lungo rimane lento: l’intervento nasconde una parte delle letture,
ma non riduce sensibilmente i byte H2D. Il precedente
[report del link PCIe](qwen38-flash-prefill-latency-20261004.md) misura
Gen4 x2 e circa 2,74 GiB/s con buffer pinned. Non è stata misurata una
nuova percentuale di utilizzo SM.

## Modifica mantenuta

La [ricerca precedente](../research/guides/flash-prefill-offload-20261004.md)
individua le letture sincrone sul thread che prepara i kernel come attesa
residua. La coda persistente prende spunto da MoE-Infinity e dal runtime
Flash/3090 esaminato; DS4 anticipa esclusivamente esperti già selezionati
dal router originale, senza predizione. Le revisioni delle fonti sono nel
[manifest](../research/guides/flash-prefill-offload-20261004.sources.json).

- `ds4_read_queue.h` contiene una coda host C con quattro worker di default
  e al massimo sedici richieste vive. Il caller possiede payload e buffer;
  `wait` pubblica il risultato e `stop` drena e unisce i worker.
- `ds4_qwen4_cuda.cuh` riserva slot della cache RAM esistente e li protegge
  con l’epoch corrente. I worker leggono byte quantizzati immutati, usando
  descriptor duplicati e fallback O_DIRECT privato per worker. Non
  accedono a CUDA, mappe della cache, ranking o contatori condivisi.
- Il thread dell’inferenza attende il risultato prima della copia H2D,
  aggiorna i metadati e alimenta altre letture. Il piano segue lo stesso
  ordine di proiezioni e gruppi del percorso precedente. Kernel, gruppi,
  eventi copy/compute e ordine della riduzione restano quelli precedenti.
- Il teardown e il cambio del descriptor del modello drenano la coda
  prima di liberare gli slot o sostituire il file. Gli errori di lettura
  vengono propagati; uno slot incompleto non viene passato ai kernel.
- RAM=0, assenza di un victim sicuro o impossibilità di avviare i worker
  conservano il percorso sincrono. Le richieste usano gli slab esistenti
  e rispettano lo stesso budget RAM; non aggiungono un secondo pool pinned.

L’intervento interessa il prefill `qwen4exp` CUDA con cache tiered. Il
percorso Qwen27 rimane separato. Non cambiano pesi, quantizzazione,
attention, KV, protocollo, payload dei checkpoint, capacità Flash 128K o
impostazioni MTP dei launcher. L’anticipazione può cambiare il momento
dell’ammissione/eviction nella cache RAM, senza cambiare i pesi eseguiti.

La statistica `expert_read_ms` somma adesso anche durate di worker
contemporanei: **non è un tempo wall-clock**. `host_wait_ms` misura le
attese dell’inferenza per la pubblicazione delle letture. Nella richiesta
HTTP candidate i due valori sono circa 24,71 e 1,79 s; comprendono anche
il decode e non vanno sommati ai tempi degli altri stadi.

I diagnostici `DS4_CUDA_QWEN4_SYNC_READS=1` e
`DS4_CUDA_QWEN4_IO_WORKERS=1..8` permettono confronti. Solo quattro worker
sono qualificati su questa macchina; altri numeri non sono nuovi profili
prestazionali promossi.

## Correttezza e checkpoint

| Gate eseguito | Esito |
| --- | --- |
| Build CLI, server, bench, eval e test CUDA `sm_86` | PASS |
| Coda: concorrenza gated, limite, completamento, error payload, riuso, drain, restart | PASS; 10 ripetizioni ASan/UBSan sulla coda |
| Policy esperti e budget memoria Linux | PASS |
| Primitive CUDA Flash con RAM 32 MiB, RAM=0 e letture sincrone forzate | PASS |
| File troncato, decode e nuovo caso prefill | PASS: errore propagato prima del consumo dello slot |
| Flash, 200 posizioni full-vocabulary/teacher-forced, 198656000 byte float32 | Identità byte per byte con la baseline |
| Flash, quattro frontiere prefill della suite slow | Logit float32 identici |
| Qwen27, logit alle frontiere prefill e dopo 16 token teacher-forced | Identici alla baseline |
| Flash e Qwen27: checkpoint sistema/skill, corruzione e rollback compatto | PASS bit-exact |
| Prefill di sistema con frontiera interna al chunk, entrambi i modelli | PASS, nessun replay dei token già elaborati |
| HTTP reale con proiettore vision caricato e flag `--mtp` | Un solo prefill, 4395 input + 32 output, testo identico |
| Suite Python del motore | 199 PASS, 188 sotto-test PASS, 41 SKIP/NOT_VERIFIED |

I test serializzano i processi del modello, rispettando l’instance lock.
Un primo tentativo Python usava l’interprete di sistema e coincideva con
un benchmark: mancavano dipendenze opzionali e il lock rifiutava
l’ispezione CLI. Il rerun seriale con `agent/.venv/bin/python` supera i
test eseguiti. Un argomento errato al wrapper del benchmark Qwen27 è
stato corretto prima dell’avvio del modello; il rerun è incluso nel JSON.

## Misure e limiti

Il modello Flash è il GGUF Q2 locale da 147207127040 byte; Qwen27 è
`Qwen3.8-27B-UD-Q4_K_S.gguf`. I modelli sono gli stessi della baseline;
non sono stati riletti interamente per un nuovo hash. Dimensione, mtime,
hash dei sorgenti e binari congelati sono registrati nel JSON.

Per la suite slow vengono scartati due sweep di warm-up e misurati i
cinque successivi, senza profiler. Ogni frontiera Flash aggiunge davvero
256 token; non si misura un semplice hit della KV. I CV Flash caldi sono
inferiori al 2,1%. Il benchmark Qwen27 include il costo del ripristino della
frontiera, particolarmente visibile a 256 token: non è throughput puro
delle moltiplicazioni.

Le cinque coppie cold partono ciascuna da un processo nuovo dopo le suite
di preriscaldamento. La cache esperti applicativa è vuota; O_DIRECT evita
la page cache quando supportato, senza garantire lo stato freddo del
controller SSD. Ordine A/B alternato. Tutti i dieci run leggono e copiano
13997798400 byte, con 14495514624 byte di slab RAM allocati: il guadagno
non deriva da un budget più grande o da token omessi.

Il confronto HTTP usa il fixture catturato dall’agente, tool e sistema
inclusi, non la sola stringa «ciao». La capacità allocata è 131072 token,
chunk 256, temperature 0 e limite esplicito di 32 output; lo stato
`incomplete` è quindi atteso. È un solo campione per variante. Il budget
RAM automatico è 32,28 contro 32,36 GiB: i contatori non coincidono
perfettamente, perciò questa coppia resta evidenza direzionale.

I contatori HTTP includono prefill, decode e shutdown: circa 230 GiB H2D
per entrambi, 32,7 GiB da SSD e più demotions RAM nella candidate.
Non sono contatori isolati del prefill. La sovrapposizione SSD non risolve
il costo delle copie PCIe né quello delle demotions/ranking. I prossimi
esperimenti devono ridurre o riusare questi trasferimenti mantenendo
chunk e ordine numerico, invece di limitarsi ad aggiungere worker.

Metal, ROCm, distribuito, altre GPU, swap/RSS/VRAM di picco campionati,
occupancy, tutte le azioni tool/vision e sessioni lunghe rimangono
**NOT_VERIFIED**. Allocare 128K non qualifica un prompt pieno da 128K.
La massima lunghezza effettivamente verificata qui è 4395 token input;
non viene rivendicata una nuova qualifica di tutti gli scenari agentici.

## Esperimento scartato: chunk più grandi

A parità di prompt da 1024 token, una prova cold misura 68,69 / 68,43 /
89,06 token/s con chunk 256 / 512 / 1024. Il secondo passaggio su ciascuna
frontiera non esegue nuovo prefill e i suoi TPS sono esclusi.

Chunk 512 e 1024 cambiano tutti i 248320 logit; differenze assolute massime
rispettivamente 0,5843 e 1,1307, pur mantenendo l’argmax. Verdetto
**REJECT** per questo task: rimane chunk 256. Una misura più veloce non
soddisfa il vincolo di preservare la correttezza numerica della baseline.

Prima del commit sono stati ricostruiti i binari di produzione. Le sezioni
CPU `.text`, dati e CUDA `.nv_fatbin` coincidono byte per byte con la
candidate misurata; gli hash completi differiscono dopo la sola correzione
di un commento di ownership e la rigenerazione delle informazioni di debug.
Il JSON conserva entrambe le provenienze.

Gli artefatti locali sono in
`performance-results/flash-prefill-async-20261004/`; non sono nuovi golden.
Il JSON versionato conserva i comandi e il fixture numerico necessario a
ricostruire i confronti.
