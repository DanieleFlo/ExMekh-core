# Piano di azione: Flash Next veloce, Qwen27 protetto

## Perimetro e verdetto — 4 ottobre 2026

L'audit indica che la gerarchia VRAM/RAM/SSD è una base valida, ma il percorso Flash CUDA non sfrutta ancora bene la sovrapposizione fra I/O, trasferimenti e calcolo. Riempire la VRAM non basta: occorre ridurre il tempo in cui la GPU aspetta pesi, routing o checkpoint. Non ci sono elementi per promettere i TPS delle repository esterne sul GGUF e sull'hardware locali.

Analisi statica del checkout `codex/qwen-flash-3090`, HEAD `bd89b13f0c839d3d9e17c7afab1f8a4dcf1b98e6`, incluse le modifiche non committate. Letti codice, diff, istruzioni, corpus locale e artefatti già prodotti. Non sono stati avviati modelli, benchmark, build o suite di inferenza durante questo audit; non sono stati modificati codice eseguibile, configurazioni, pesi o server. È stata letta una fotografia hardware mentre era presente un processo server preesistente.

I PASS riportati sotto appartengono agli esperimenti precedenti, non a nuove esecuzioni. Il manifest `final-qualification.json` contiene hash diversi da quelli attuali di core CUDA Flash e server: non certifica automaticamente l'intero working tree odierno. Il report successivo sul budget del contesto ha una copertura separata e più limitata.

## 1. Evidenze hardware e limiti del confronto

- RTX 3090, 24.576 MiB; i9-14900K; WSL espone 16 CPU logiche e circa 47 GiB utilizzabili. La documentazione recente registra 64 GB fisici e limite WSL di 48 GB. La topologia virtuale non basta per dedurre l'affinità ai P-core/E-core dell'host.
- Fotografia durante l'audit: 23.343 MiB occupati, GPU 39%, PCIe Gen4 x2, massimo dichiarato Gen4 x16. È un singolo campione, non una misura di throughput o una media del workload. Conferma però il link x2 già registrato sotto carico negli esperimenti precedenti.
- Prima priorità hardware: verificare slot, eventuale riser, negoziazione e configurazione motherboard. La causa non è dimostrata. Otto volte meno lane di x16 non significa automaticamente otto volte meno TPS.
- GGUF locale documentato: Flash Next Q2, 147.207.127.040 byte, circa 41,73 GiB di pesi principali/MTP e 95,37 GiB di PLE BF16 su disco. Il nome Q2 non significa che tutti i tensori abbiano lo stesso formato.
- Non aumentare semplicemente la RAM pinned: Windows, WSL, cache del filesystem, PLE, agente e checkpoint condividono un budget finito. Non modificare WSL/BIOS durante la prima fase software.

Riferimento locale: [report residency e qualifiche](../performance/qwen38-flash-3090-residency-20261003.md).

## 2. Cosa è stato fatto bene

1. Separazione `qwen35`/`qwen4exp`; il percorso residente del 27B non è stato sostituito con quello degli esperti Flash.
2. Copie dei byte quantizzati originali e mantenimento dei kernel aritmetici; protezione preventiva degli hit prima di scegliere le vittime; gestione della capacità reale degli slot.
3. Budget VRAM calcolato dopo l'allocazione del contesto e preload dei pesi fissi prima degli esperti. Il conteggio reale dei tensori corregge la vecchia riserva densa insufficiente.
4. Riutilizzo nel prefill dei puntatori agli slot residenti, evitando copie aggiuntive degli hit; fallback esplicito in VRAM per i miss.
5. Stato della policy versionato nei payload; validazione prima di mutare i metadati della cache; ricostruzione della residenza su richiesta. Le copie fisiche dei pesi immutabili non devono essere serializzate nel checkpoint.
6. Estensione dei guard di sistema/pre-thinking a Flash senza chiamare impropriamente l'API compatta del 27B. Separazione fra successo del restore e successo dell'intero test agentico.

Le evidenze storiche comprendono 200 posizioni × 248.320 logit bit-exact, test sotto pressione delle cache, MTP e checkpoint annidati/vision. Sono buone prove mirate; non dimostrano ancora qualità e prestazioni per ogni lunghezza e scenario.

## 3. Colli di bottiglia e rischi visibili nel codice

I numeri di riga si riferiscono al working tree osservato e potranno cambiare.

| Priorità | Evidenza | Conseguenza e azione |
| --- | --- | --- |
| P0 | `ds4_cuda.cu:852–895`: un mapping registrato può essere risolto prima del caricamento nella cache. `DS4_CUDA_QWEN4_TIERED_CACHE` resta opt-in. | Impostare solo il limite pesi non garantisce residency. Registrare all'avvio il percorso realmente attivo e i byte per tier. Qualificare il nuovo percorso prima del default. |
| P0 | `ds4_qwen4_cuda.cuh:1132–1181`: lettura CPU degli ID e sincronizzazione; trasferimenti; seconda sincronizzazione; soltanto dopo parte il kernel. Gate/up e down ripetono il percorso. | Riutilizzare un piano di routing per layer/forward, con staging pinned persistente e dipendenze tramite eventi. Non eliminare le barriere senza sostituire le garanzie di lifetime. |
| P0 | `ds4_qwen4_cuda.cuh:2682–2762`: readback di counts e dell'intera tabella `NE × cap`, ricostruzione CPU degli ID, caricamento dei miss, attesa, GEMM. | Il prefill copia metadati sovradimensionati e aspetta tutti i pesi. Leggere una sola volta `T × NS` ID o una lista compatta dei miss; riusare il risultato per gate/up/down. Con NE=512, NS=10 e cap=T, la sola tabella lista è 51,2 volte gli ID selezionati. |
| P0 | `ds4_qwen4_cuda.cuh:888–933`, `ds4_cuda.cu:2283–2317`: miss con `pread` sincrono; demotion D2H con attesa; copie sullo stream del decode. | Parte dell'I/O può sovrapporsi a lavoro già accodato, ma manca una pipeline dedicata fra gruppi di esperti e GEMM. Usare worker I/O bounded, doppio/triplo buffer e stream di copia separato. |
| P1 | `ds4_qwen4_cuda.cuh:847–885`, `ds4_qwen4_expert_cache.h:74`: RAM 16 GiB predefiniti, allocazione pinned per slot, scansioni lineari per vittima. | Possibile costo CPU crescente sotto pressione. Misurare prima di sostituire la policy; riusare slab per classi di capacità e indici di eviction preservando frequenza, recency e tie-break. Budget RAM derivato dalla memoria effettivamente disponibile. |
| P1 | `ds4_qwen4_cuda.cuh:2723–2726`: rimane l'allocazione di staging dimensionata per interi tensori anche con puntatori agli hit. | Non si ricopia più tutto, ma si riserva ancora spazio per il layer. Passare a gruppi bounded può liberare VRAM per working set o chunk maggiori. |
| P1 | `ds4.c:58537–58667,59894–59936`: letture PLE BF16 per riga; su Linux le richieste piccole sono seriali, le grandi creano/joinano thread; gather prima del forward. | Valutare pool persistente, cache bounded delle righe originali, deduplicazione e prefetch del prossimo chunk noto. Non cambiare BF16 in FP8. Preservare storia PLE e rollback MTP. |
| P1 | `ds4_server.c:15836–15927`: Flash usa il payload completo nelle frontiere residenti dove il 27B può usare uno stato compatto. | Funzionalità disponibile, ma costo proporzionale allo stato salvato. Misurare D2H/H2D, byte e tempo di capture/restore; progettare una frontiera Flash specifica, con fallback completo. |
| P1 | `ds4_cuda.cu:2467–2545,38245`: alcune decisioni dell'allocator condiviso leggono direttamente il flag Flash globale. | Il guard alto livello è utile ma non isola tutto. Derivare una capability effettiva da famiglia/backend/topologia e verificare Qwen27 anche con variabili Flash ereditate dall'ambiente. È un rischio da coprire, non una regressione numerica dimostrata. |

La cache e la sua cronologia sono globali nel backend (`ds4_qwen4_cuda.cuh:724–743`). Prima di promettere più sessioni o cambi di contesto in-process, definire ownership: pesi condivisibili, policy per engine o per sessione, budget di tutte le sessioni, restore senza interferenze. I test single-session non bastano.

## 4. Interpretazione delle misure esistenti

| Esperimento storico | Risultato | Limite |
| --- | --- | --- |
| Decode originale, frontiere molto corte | circa 0,40 TPS | Percorso con pesi host-accessible; non è un roofline GPU. |
| Cache a 20 GiB, frontiere 64/128 | circa 30,27/29,50 TPS caldi; 6,82/9,01 al primo passaggio | Working set breve, teacher-forced; non equivale a una conversazione nuova lunga. |
| Prefill, ctx allocato 16K, chunk 256, prompt 256/512 | 91,96/121,28 TPS caldi | Senza MTP/vision; non misura prompt da 16K. |
| Stesso prompt, ctx allocato 100K e 262K | 73,92/95,38 e 56,22/64,71 TPS | Più KV riservata lascia meno esperti in VRAM; nessuna prova di throughput su 100K token reali. |
| Prompt agentico reale 4.699 token, MTP/vision | 230–273 secondi per richiesta inclusi 8 token di output | Evidenza del problema pratico; non separa da sola I/O, prefill e decode. |

Non usare la ripetizione identica del prompt per rappresentare il caso generale. Distinguere processo appena avviato, pesi già caldi con prompt nuovo, cambio di dominio e prefix-cache hit. Non svuotare globalmente la page cache del sistema per simulare un cold run.

## 5. Fonti esterne: idee trasferibili e incompatibilità

Fonti consultate il 4 ottobre 2026. Le pagine `main` sono mobili: fissare commit e licenze dei singoli file prima di un port. Il codice DS4 è stato ispezionato direttamente; il confronto esterno riguarda model card, documentazione tecnica, configurazioni e albero dei sorgenti, non una qualifica locale di quei motori.

| Fonte primaria | Da studiare | Cosa non trasferire automaticamente |
| --- | --- | --- |
| [Model card albucino](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE/blob/main/README.md) | Provenienza e runtime associati | W4A16 AutoRound e PLE FP8 differiscono dal Q2/PLE BF16 locale. Sostituirli violerebbe il perimetro. Nessuna abliteration. |
| [Runtime singola 3090: architettura](https://github.com/DominikBucko/qwen38-flash-next-3090/blob/main/docs/how-it-works.md) | Copie per gruppi, prefetch I/O, pipeline prefill, graph e buffer persistenti | Calcolo CPU degli esperti e KV INT8 cambiano il percorso numerico. La verifica INT8 dichiarata non comprende ancora likelihood teacher-forced. |
| [Runtime doppia 3090: profilo prefill](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/configs/agent-128k-prefill.env) | Tradeoff chunk 8K/cache esperti e overlap | Migliora prefill sacrificando decode: non esiste una singola configurazione ottima per tutto. TP/EP e budget non sono parametri da copiare su una sola GPU. |
| [Strata: dettagli tecnici](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md) | Chunk adattivi, streaming del layer successivo, pool di lettura, RAM complementare alla VRAM, metriche per richiesta | Formati, quantizzazione KV e calcolo CPU/GPU possono produrre rounding differente. La documentazione segnala output dipendente da placement e drafting. Non usare i suoi TPS come target certificato per DS4. |
| [llama.cpp CUDA](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda) | MMVQ/MMQ, scheduling MoE, attention e GDN; confrontare prima `cuda/mmq/` già presente | Non reimportare kernel già integrati. Cambi di aritmetica richiedono gate separati. |
| [club3090 MoE cache](https://github.com/noonghunna/llama.cpp/tree/stack/club3090-moe-cache) | Budget, scatter e overlap; riesaminare il commit nel corpus locale | La precedente esclusione riguardava il 27B residente; Flash offloaded è un caso diverso. Nessun benchmark del fork è una misura locale. |

La soluzione CPU+GPU per gli esperti è una ricerca successiva, non il primo intervento: richiede un kernel per gli esatti formati GGUF locali, nuovi oracle numerici e una valutazione della banda RAM/CPU realmente esposta da WSL. Il percorso attuale può migliorare molto nello scheduling senza questo cambiamento.

## 6. Sequenza operativa e criteri di uscita

### Fase 0 — Congelare baseline e proteggere Qwen27

- Salvare manifest di commit, diff/untracked rilevanti, hash sorgenti/binari e identità degli artefatti modello già disponibili. Non usare un vecchio PASS per un binario diverso.
- Tenere distinti interventi cache/runtime Flash e interventi schema/protocollo condivisi. Conservare rollback del singolo esperimento senza cancellare modifiche preesistenti.
- Definire capability per modello/backend: Qwen27 residente, Flash tiered su singola CUDA, altri percorsi preesistenti. Nessuna attivazione universale tramite flag ambientale.
- Inventariare allocator, stream, scratch, snapshot e cache condivisi; fallback esplicito sulle configurazioni non qualificate. Il launcher attuale è dedicato al 27B e al suo proiettore: un profilo Flash deve scegliere modello e mmproj corretti.

Uscita: baseline riproducibile e matrice di isolamento. In questa fase basta analisi statica/model-free.

### Fase 1 — Diagnosi corta del percorso critico

- Verificare il link PCIe con telemetria durante il prossimo run breve; approfondire l'anomalia x2 prima di tarare profili hardware. Nessuna promessa di correzione software del link.
- Aggiungere tempi per PLE, routing/readback, scelta vittima, SSD, demotion, H2D, GEMM/GEMV, attention/GDN, sampling, checkpoint. Contare byte utili e byte fisici/allineati, miss unici, picchi RAM/VRAM, page fault/swap e acceptance MTP.
- Contatori per fase/richiesta, non soltanto cumulativi allo shutdown. Usare eventi CUDA per le durate device; le letture CPU del profiler non devono serializzare ogni kernel nelle misure finali.
- Acquisire una timeline corta con gli strumenti disponibili, o strumentazione locale se Nsight non è disponibile. Un processo modello alla volta, nessuna interferenza col server dell'utente.

Uscita: separare limiti di PCIe/SSD, CPU di gestione e GPU; ipotesi falsificabile per il primo intervento. Non eseguire ancora matrici 32K–262K.

### Fase 2 — Pipeline degli esperti, byte e matematica invariati

- Creare una richiesta di routing unica per layer/forward e riutilizzarla nelle tre proiezioni; deduplicare i miss identici, soprattutto MTP e piccoli batch.
- Buffer host pinned persistenti, pool I/O bounded, letture aggregate dove gli offset lo permettono; A/B direct I/O e buffered senza cambiare i dati.
- Caricare gruppi sullo stream di copia mentre la GPU esegue gruppi pronti e shared expert. Il router effettivo decide sempre tutti gli esperti da eseguire; una previsione può soltanto anticipare I/O.
- Segnalare via eventi fine copia/fine consumo; non riutilizzare né espellere uno slot in flight. Copie demotion fuori dal percorso critico quando la dipendenza lo permette.
- Scrivere risultati negli slot originali e conservare ordine di riduzione, top-k, tie-break, logits e stato. Non raggruppare cambi di kernel matematico in questo intervento.

Uscita: correctness breve PASS, riduzione misurabile dei gap e miglioramento end-to-end sul workload nuovo/caldo pertinente. Scartare varianti lente prima della suite lunga.

### Fase 3 — Prefill e memoria per scenario

- Esplorare progressivamente chunk 256/512/1024/2048/4096/8192, fermandosi prima dei budget di memoria o tempo. Il default interno è già 8192; il comando locale misurato usa 256: non serve alzare ciecamente il default.
- Con 512 esperti e top-10, 256 token producono circa 5 assegnazioni per esperto in media sull'intero layer; 8192 ne producono circa 160. Il vantaggio del riuso deve essere verificato sulla distribuzione reale e sul workspace.
- Dimensionare staging per gruppi e non per l'intero layer; scegliere chunk e quota esperti insieme. Profili riproducibili: piccoli suffissi/tool, prefill lungo, sessione agentica con riuso, MTP/vision.
- Budget RAM basato su disponibilità WSL/container e riserva host; distinguere byte unici, duplicati GPU/RAM, pinned, cache filesystem e checkpoint. Valutare demotion rispetto a rilettura: non sempre una copia D2H conviene.
- PLE: pool persistente e prefetch del prossimo chunk noto, conservando BF16 originale. Cambi di contesto non devono invalidare indici o creare prefissi sbagliati.

Uscita: configurazione per capacità hardware e scenario, con fallback. `ctx` allocato e token effettivi sono assi distinti. Un'autoconfigurazione non deve ridurre silenziosamente il contesto richiesto.

### Fase 4 — Parità dell'agente e costo dei checkpoint

- Chiudere i gate HTTP/client rimasti aperti: scope padre/figlio, tipi degli argomenti, structured output e MTP. Sono problemi di integrazione separati dai kernel. Non allentare schema o oracle per farli passare.
- Provare sistema residente e SSD, pre-thinking, tool-result canonicalization, reasoning omesso/eco, HDS, skill sorelle, nesting, subagent, cambio registry, riavvio client/server, eviction, cancellazione e file corrotti.
- Includere immagini/PDF, cronologia multimodale, context boundary, restore in sessione nuova, budget diversi e payload legacy. Verificare KV/ricorrenza/PLE/mRoPE/logits e policy esperti, non soltanto il testo finale.
- Misurare byte e tempo di capture/restore Flash a contesto crescente. Poi introdurre frontiera compatta per rollback locale se le righe KV rimangono valide; checkpoint durevoli e cambio sessione devono conservare tutto lo stato necessario.
- Riutilizzare scenari e contratti del 27B per Flash. I file KV del 27B non possono essere caricati nel modello Flash: fingerprint e famiglia devono continuare a impedirlo.

Uscita: successo numerico e applicativo distinti, nessun full-prefill del parent quando esiste uno snapshot valido. Una risposta troncata non certifica il completamento del test, anche se il restore è corretto.

### Fase 5 — Qualifica progressiva e portabilità

- Prima direzione breve: due workload differenti, due campioni e budget temporale stretto, oltre ai gate numerici pertinenti. Non basta per promuovere.
- Prima dei contesti lunghi stimare costo da TPS osservati sul prompt nuovo e dal numero di righe realmente da prefìllare/ripristinare. Se supera il budget stabilito, interrompere la matrice e tornare alla diagnosi; non attendere ore per un risultato già prevedibile.
- Poi warm-up e almeno cinque misure, CV <= 5%, secondo il contratto della repo. TTFT, prefill, decode p50/p90, memoria, I/O e qualità; benchmark finali fuori dal profiler.
- Matrice progressiva: prompt 128/512/2K/8K/32K/100K/fino alla capacità nativa, separati da ctx allocato; nuovi prompt, cambi di dominio, prefix hit/miss, MTP on/off, testo/vision, tool/JSON e restore. Non lanciare tutto il prodotto cartesiano: scegliere prima i casi che stressano ciascun meccanismo e poi le frontiere.
- Testare cambi di shape/chunk e code 1/2/3/8/9 token; il comportamento MTP e i percorsi GEMV/GEMM non devono dipendere da una lunghezza favorevole.
- Qwen27: regressioni numeriche, performance breve, checkpoint, MTP/vision e API condivise. Metal, ROCm, SSD streaming e multi-GPU conservano il percorso precedente; qualificare sul rispettivo hardware prima di dichiarare supporto. Distribuito secondo l'autorizzazione prevista dalle istruzioni della repo.

Uscita: un candidato viene promosso solo per hardware/scenari verificati. GPU utilization/occupancy sono strumenti diagnostici: un collo di bottiglia di banda può lasciare SM poco occupati senza indicare un errore. L'obiettivo è tempo utile per richiesta, a correttezza invariata.

## 7. Backlog successivo, solo se giustificato dai profili

CUDA Graph per sottografi stabili dopo aver rimosso readback/allocazioni dinamiche dal percorso; ottimizzazioni attention/GDN e MMQ per le shape davvero dominanti; gestione KV paginata o tiered senza nuova quantizzazione, solo se il costo del KV supera i benefici persi nella cache esperti. Conservare il guard sm_86 del kernel attention escluso dopo il fault: non riattivarlo soltanto per aumentare la velocità.

La prima implementazione consigliata comprende Fasi 0–2: osservabilità e pipeline degli esperti, con breve gate numerico e protezione del 27B. Chunking, PLE e snapshot seguono i tempi osservati, non una riscrittura generale del motore.

## 8. Provenienza locale dell'audit

Hash SHA-256 letti durante l'audit:

- `ds4.c`: `aedb37027ffd52d7942e1fd0008a10fc6b688400225d80136fd902ddcba3f545`
- `ds4_cuda.cu`: `7a2292162064917b99c551fbf7452c6154d7ae93f40aa2bb0a1418e5bf8d0fed`
- `ds4_qwen4_cuda.cuh`: `b7794f86fe4264d0082c66b95909399bdee74b349d7268bf9a2c52e61ce7d881`
- `ds4_qwen4_expert_cache.h`: `514010854c82d3391d7b38869e9309c8c44e0f17c6e8483111cee22b64efc16d`
- `ds4_server.c`: `24cba2729397d69aaedaf69ade211e08b968373da96bc50657d00edde0ae46dd`

Artefatti precedenti consultati: `performance-results/qwenflash-3090-20261003/final-qualification.json` e `prefill-vram/context-budget-prepared-results.json`. I risultati locali ignorati da Git restano evidenze di quella macchina, non fixture pubbliche garantite.

## 9. Prima implementazione e qualifica — 4 ottobre 2026

Il [registro di implementazione e qualifica](../performance/qwen38-flash-pipeline-20261004.md) descrive il codice risultante, i comandi riproducibili, i gate e le prestazioni. La baseline è stata congelata dal working tree di questo audit, senza eliminare modifiche preesistenti.

Completati isolamento della capability Flash, routing compatto per coppia mid/down, copia/calcolo a gruppi con due buffer protetti da eventi, allocazione RAM a slab con headroom e ranking delle vittime equivalente alla scansione. Il lavoro su memoria e gestione CPU è stato anticipato dopo le misure brevi; aumentare il chunk a 1024 da solo non ha dato un vantaggio sufficiente. Il launcher aggiunge la scelta esplicita Flash e conserva Qwen27 come default.

PLE, pool I/O dedicato, snapshot Flash compatti e CUDA Graph rimangono interventi condizionati da nuovi profili; il percorso aritmetico e i checkpoint completi sono conservati. Non è avviata la matrice lunga 32K–262K. Qualifica e limiti degli scenari effettivamente provati sono nel registro, separati dalle evidenze storiche.

Gate numerici full-vocabulary, regressione Qwen27, checkpoint Flash compositi, HTTP agentic e client tipizzato superati localmente. La candidate è stabile nelle cinque misure warm; la baseline è instabile, quindi il verdetto performance resta NEED_MORE_DATA. Non è una qualifica completa della matrice di contesti/hardware; campioni e limitazioni sono conservati nel registro e nel relativo JSON.
