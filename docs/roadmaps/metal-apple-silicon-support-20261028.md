# Piano di supporto Metal del fork su Apple Silicon

Data: 28 ottobre 2026; avvio implementazione e revisione delle priorità:
29 ottobre 2026. Stato: `IN_PROGRESS`.
I singoli gate riportano soltanto le prove effettivamente eseguite; il piano
non attesta supporto completo finché la matrice obbligatoria resta aperta.

## 1. Obiettivo e macchina di riferimento

Il piano copre i modelli LLM, image ed embedding e i flussi fondamentali di ExMekh.
Portare su Metal le funzionalità aggiunte dal fork per
**Qwen3.8-27B `qwen35`** e **Qwen3.8 Flash Next `qwen4exp`**, conservando le rispettive
semantiche di inferenza, API, strumenti, output vincolato, checkpoint e modalità
multimodali. **Qwen-Image-2.1-Turbo, EmbeddingGemma 2 e i flussi applicativi
fondamentali di ExMekh fanno parte obbligatoria della verifica**, con pesi
reali e risultati controllati. Il completamento della sola chat non soddisfa
l'obiettivo del fork. In questo piano «Flash» indica Qwen Flash Next; DeepSeek Flash e
i kernel Flash Attention sono concetti distinti.

La macchina rilevata durante la pianificazione è un **MacBook Pro M5 Pro,
Mac17,9, 24 GB**, con macOS 26.6.2. La disponibilità inizialmente indicata era
**17–18 GB generalmente liberi**, ma i tentativi del 27B residente sono stati
rifiutati dal guard. La nuova sequenza non attende che l'utente liberi quella
quantità di RAM: punta allo streaming SSD esplicito entro il budget misurato.
La stessa implementazione deve adattarsi ad altri Mac Apple Silicon tramite
capacità effettive del device e memoria,
senza dipendere dal nome M5 Pro.

**Priorità corrente richiesta dall'utente il 29 ottobre:** avviare prima
**Qwen3.8 Flash Next Q2 in streaming SSD su Metal**, poi provare
**Qwen3.8-27B UD-Q4_K_S in streaming SSD**, con lo stesso contesto massimo
di **8192 token (8k)**. Una sola sessione e un solo modello grande per volta.
Il limite include sistema, strumenti, cronologia, input e output; sostituisce
il precedente tetto locale di 2048 token e la preparazione Quick Chat a 4096.
Il primo profilo è target-only, prefill 64, reasoning disattivato, senza
vision né suite attiva. La prima integrazione frontend resta Quick Chat.

La capacità configurata a 8192 e il funzionamento con un prompt breve sono
gate distinti dall'uso effettivo di un contesto prossimo a 8k. Il supporto
funzionale completo resta l'obiettivo del piano; le prove testuali iniziali
non chiudono MTP, vision o i gate image/embedding.

Il profilo esistente `config/qwen27-metal-quick.env.example` conserva la
precedente preparazione residente 4096/64 e le relative evidenze. I profili
streamed 8192/64 di entrambe le famiglie sono ora implementati e qualificati
localmente (§9); i launcher ordinari mantengono il default 512/64.

## 2. Risultati dell'ispezione del fork

Riferimento sorgenti: `837894c2af5aaf3382dfc0a0750689baee73363e`.
All'inizio della pianificazione il worktree era pulito. Nessun GGUF è stato
trovato nella directory locale `gguf/`; reperimento e verifica dei modelli
andranno affrontati prima dei futuri test reali, senza download in questa fase.

| Area | Situazione osservata | Conseguenza per il piano |
| --- | --- | --- |
| Build Metal | Il [Makefile](../../Makefile) seleziona già Metal su Darwin | Eseguire una baseline di compilazione e link; non assumere che tutte le aggiunte del fork siano già collegate correttamente |
| Qwen27 | `ds4_engine_open()` in [ds4.c](../../ds4.c) rifiuta esplicitamente backend diversi da CUDA | Serve il porting effettivo del percorso `qwen35`, prima di rimuovere il rifiuto |
| Primitive Qwen27 | Le implementazioni `ds4_gpu_qwen35_*` di attention, KV e GDN sono in [ds4_cuda.cu](../../ds4_cuda.cu); non risultano corrispondenti in [ds4_metal.m](../../ds4_metal.m) | Inventariare anche i simboli condivisi necessari alla build; non risolvere con stub che simulano successo |
| Quantizzazione 27B | Il GGUF UD contiene Q3_K e sei formati IQ oltre a F32/Q8_0/Q4_K/Q5_K/Q6_K | Verificare ogni operazione e formato; le routine Metal esistenti non coprono automaticamente questo inventario |
| Vision 27B | Guard CUDA esplicito e rami mRoPE che su Apple restituiscono errore | Portare normalizzazione, posizioni, scrittura KV, catch-up MTP e restore multimodale |
| Flash Next | Grafo Metal dedicato e [kernel Qwen4](../../metal/qwen4.metal) già presenti | Conservare e qualificare il percorso esistente, estendendo le funzioni del fork |
| Flash con poca memoria | Il controllo dell'engine rifiuta `--ssd-streaming` per `qwen4exp`; la cache tiered del fork è esposta sul backend CUDA | Lo streaming degli esperti DeepSeek/GLM non rende già disponibile lo streaming Flash su Metal |
| Launcher | [start-ds4-server.sh](../../start-ds4-server.sh) forza `--cuda`, MTP e default di contesto elevati; il launcher agent Qwen27 attiva anche la suite | Occorrono selezione backend e profili di memoria coerenti in tutti gli entrypoint |
| Lifecycle della suite | [ds4_lifecycle.inc](../../ds4_lifecycle.inc) esclude Apple e limita l'adapter a Qwen27 CUDA | Separare checkpoint agentici esistenti dal nuovo park/release/restore della suite; estendere quest'ultimo ad entrambi i modelli |
| Calcolo suite | [ds4_numeric.h](../../ds4_numeric.h) espone CPU/CUDA; [ds4_server_suite.inc](../../ds4_server_suite.inc) crea il contesto CUDA | Immagini ed embeddings richiedono un backend numerico Metal e il relativo collegamento al servizio |

Le guide [Metal upstream](../upstream/METAL.md) e
[Flash Next](../upstream/QWEN38_FLASH_NEXT.md) sono riferimenti ereditati.
Le qualifiche CUDA dei report del fork non sono qualifiche Metal.

## 3. Budget: working set dello streaming nella memoria disponibile

Le dimensioni seguenti provengono dall'[audit Qwen27](../architecture/qwen38-compatibility.md)
e dalla [guida Flash](../upstream/QWEN38_FLASH_NEXT.md). Descrivono artefatti o
pesi, non il picco di memoria dell'intero processo.

| Componente | Dimensione documentata | Implicazione |
| --- | --- | --- |
| Qwen27 UD-Q4_K_S | 15.358.213.024 byte, circa **14,30 GiB** | Secondo obiettivo locale in streaming dei pesi densi; il caricamento residente richiede un budget distinto |
| Sidecar MTP 27B | 1.369.590.656 byte, circa **1,28 GiB** | Non attivarlo nel primo test; pesi condivisi e selettivi vanno misurati |
| Vision 27B F16 | 927.607.488 byte, circa **0,86 GiB** | Servono anche attivazioni e workspace; prevedere esecuzione per fasi |
| Flash Next Q2 | File circa **137,10 GiB**, pesi residenti documentati **41,73 GiB** | Il percorso residente documentato supera nettamente la memoria locale |
| N-gram Flash | Circa **95,37 GiB**, già letti da disco | Non precaricarli e non confondere la loro gestione con lo streaming degli esperti |

Ridurre il contesto risparmia KV e parte dei temporanei; **non riduce i pesi
né elimina lo stato ricorrente**. Nel layout FP32 corrente del 27B, a 2048
token la sola KV dei 16 layer full-attention vale circa **256 MiB**;
stato GDN e convoluzione dei 48 layer ricorrenti aggiungono circa **151,5 MiB**
indipendenti dal contesto. A 8192 token la stessa KV vale circa **1 GiB**;
lo streaming dei pesi non elimina questo costo. Restano scratch, tokenizer, logits, mapping,
checkpoint, eventuale MTP e allocazioni del backend. Sono stime ricavate da
`DS4_SHAPE_QWEN38_27B` e `qwen_graph_alloc()`, non misure Metal.

Prima di ogni futuro run calcolare il picco incrementale dell'operazione:

`pesi effettivamente residenti + stato sessione + scratch + staging + copie/checkpoint temporanei + overhead misurato`.

- [x] Rappresentare CPU e GPU come consumatori della **stessa RAM fisica** su
  Apple Silicon: una view no-copy non è una seconda copia; un buffer di
  staging realmente separato lo è. Distinguere mapping virtuale e pagine residenti.
- [x] Usare disponibilità host, pressione memoria e capacità Metal insieme.
  Come limite prudente, partire dal minore fra margine host e margine rispetto
  al working set raccomandato, sottraendo una riserva iniziale di **2 GiB**.
  Evitare doppie sottrazioni delle risorse già contabilizzate. La riserva è
  una policy iniziale da calibrare, non una proprietà hardware.
- [x] Non interpretare `recommendedMaxWorkingSetSize` come RAM libera: Apple
  lo descrive come soglia indicativa per mantenere le prestazioni. Registrare
  anche `currentAllocatedSize` e `maxBufferLength` nel report del device.
  Riferimento: [documentazione memoria MTLDevice](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize).
- [x] Se il budget non basta, restituire un errore di ammissione con stima e
  componente limitante. Niente oversubscription intenzionale, bypass dei guard
  o riduzioni silenziose di contesto, precisione, risoluzione o output.

**Decisione aggiornata:** qualificare prima Flash streamed con capacità
8192, poi il 27B streamed alla stessa capacità. Inventariare per ciascuno
pesi non evictable, massima unità di lettura, cache, stato e transitori;
stimare il picco con prefill 64 e conservare la riserva host di 2 GiB.
Il cap iniziale della cache Flash è 2 GiB, non il budget dell'intero processo;
verificarne la sufficienza sul GGUF reale. Cache o staging doppi si sommano
solo quando sono storage distinti, includendo entrambi quando coesistono.
Se il profilo non entra, ridurre il working set tramite scheduling esplicito
o registrare il limite; non diminuire silenziosamente contesto o precisione.

## 4. Milestone e dipendenze

Ogni task parte `PLANNED`. Diventa completato solo con il proprio gate e
artefatti riproducibili. `PASS`, `FAIL`, `NOT_VERIFIED` e
`SKIP_RESOURCE_LIMIT` devono restare distinguibili.

| Task | Dipendenze | Risultato richiesto |
| --- | --- | --- |
| T00 — Inventario e baseline | Nessuna | Inventario delle funzioni fondamentali del fork, matrice di accettazione §10.1, simboli mancanti, stime e riferimenti fissati |
| T01 — Build e avvio portabili | T00 | Binari Metal collegabili e launcher con profilo locale esplicito |
| T02 — Memoria unificata | T00; integrazione con T01 | Accounting e ammissione affidabili, primitive testabili senza GGUF completo |
| T03 — Qwen27 testo | T01, T02; prova locale streamed dopo T06-B | Quantizzazioni, grafo ibrido, prefill e decode Metal corretti |
| T04 — Funzioni agentiche | Gate testo della singola famiglia: T06-A per Flash, T03/T06-B per 27B | API, vincoli, skill, checkpoint e ripresa su entrambi i modelli; Quick Chat Flash per prima |
| T05 — MTP e vision | T04 e gate testo della singola famiglia | Percorsi opzionali corretti, bounded e ammessi secondo il budget |
| T06-A — Flash streamed, priorità 1 | T01, T02, audit del grafo Flash | Avvio reale Metal in streaming a 8192/64, kernel e stato verificati, poi Quick Chat |
| T06-B — Qwen27 streamed, priorità 2 | Gate iniziale T06-A; primitive T03, T01/T02 | Streaming dei pesi densi e prova dei kernel 27B sullo stesso contesto 8192/64 |
| T07 — Suite nativa Metal | T01/T02 per prove standalone; T04/T05 e streaming T06 della singola famiglia per integrazione locale | Inferenza reale image/embedding, lifecycle di entrambi i LLM e flussi ExMekh completi |
| T08 — Qualifica e documentazione | Gate dei task precedenti | Matrice per modello, funzione, hardware e livello di evidenza |

Ordine operativo corrente: **fondamenta T00–T02 già disponibili → T06-A
Flash streamed 8k → T04 Quick Chat Flash → T06-B/T03 Qwen27 streamed 8k
→ T04 Quick Chat 27B → T05 e integrazione T07 → T08**. Il 27B residente
non è più un prerequisito dell'avvio locale. I gate numerici già ottenuti
restano validi nel proprio ambito; la qualifica reale rimane aperta.
Le dipendenze non richiedono processi modello contemporanei o lavoro con subagent.
L'inventario, gli oracle e le prove standalone image/embedding di T07 iniziano
già dopo T01/T02: non devono attendere il completamento dello streaming Flash.

## 5. T00–T02: fondamenta, build e controllo della memoria

- [x] Congelare commit, versione clang/SDK/macOS, capacità device, modelli,
  hash, unità GB/GiB e spazio SSD. Inventariare anche le dipendenze dei launcher
  e i comandi con `--print-command`, senza aprire modelli.
- [x] Costruire la matrice `modello × funzione × backend`: compilato,
  primitiva numerica verificata, inferenza reale verificata, limiti risorse.
  Collegare ogni funzione fondamentale alla matrice §10.1, ai test esistenti
  e ai relativi oracle; registrare esplicitamente eventuali lacune di copertura.
  Confrontare l'upstream fissato dal fork prima di progettare nuovi kernel.
- [x] Eseguire `make -j2` nella futura implementazione; censire errori di
  compilazione/link, riferimenti CUDA non protetti e simboli Qwen27 mancanti.
  Eseguire la build CPU in una directory separata per evitare il riuso di
  oggetti con flag diversi; usarla come controllo, senza inferenza CPU grande.
- [x] Rendere il launcher comune consapevole del backend, con scelta automatica
  Metal su Darwin e override esplicito validato. Applicare i default CUDA solo
  al percorso CUDA; mantenere `--print-command` utilizzabile senza modelli.
- [x] Introdurre un profilo locale di qualifica: una sessione, contesto 512,
  prefill 64, target-only e moduli opzionali espliciti. Esaminare anche i due
  wrapper in `agent/`: non devono riattivare 26k, MTP o suite implicitamente.
  I nomi di eventuali nuove opzioni saranno decisi nell'implementazione.
- [x] Integrare il registro di risorse con le allocazioni Metal, alias, mmap,
  staging e completamenti dei command buffer; vietare il riuso prima della fence.
  Collegare i budget logici host/device a un vincolo fisico condiviso, senza
  introdurre due budget indipendenti che possano spendere due volte la RAM.
- [x] Interrogare le capacità, non il nome del chip. Conservare un percorso
  generale corretto; specializzazioni M3/M5/Metal 4 soltanto quando disponibili.
  Per BF16 preservare storage e arrotondamento anche se serve conversione
  esplicita in FP32, senza sostituirli silenziosamente con FP16.
  Riferimento: [rilevamento delle funzionalità Metal](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions).

**Gate:** build/link Metal, dry-run dei launcher, ammissione/rifiuto sotto budget
sintetico, alias conteggiati una volta, allocazioni parziali ripulite e comandi
GPU completati prima del rilascio. Nessun requisito di GGUF completo.

**Evidenze T00–T02 aggiornate:** [report T00–T03](../performance/metal-apple-silicon-20261029/t00-t03/README.md)
e relativo registro riproducibile. Inventario di 866 tensori e hash completo del
Qwen27 locale verificati; le identità e prove dei modelli suite restano nelle
evidenze precedenti. L'integrità del Flash locale rimane un gate T06-A.
La pressione host è best-effort e viene segnalata come sconosciuta se non
disponibile; le pagine inattive anonime non sono tutte considerate recuperabili.
Budget ordinario, alias, staging, fence e cleanup sono verificati con fixture;
capability Qwen rilevate dal device/pipeline e replay generale senza Metal 4.
I percorsi ottimizzati upstream conservano le proprie policy; altri chip Apple
e regressioni CUDA/ROCm restano non verificati.

## 6. T03: porting del 27B senza alterare il modello

### 6.1 Confronti ripetuti con Strata durante la scrittura dei kernel

**Regola utente aggiornata il primo novembre 2026:** ogni nuova differenza
numerica o regressione di prestazioni riapre il confronto pertinente con
Transformers/Diffusers, Strata, llama.cpp/ggml e le ricerche già presenti nel
repository, inclusi casi analoghi CUDA/Metal. Prima della modifica registrare
revisione/file, discrepanza, assunzioni trasferibili e scelta nel report della
diagnosi. Il riesame precedente non spiega automaticamente una nuova differenza.
La regola permanente è registrata anche in `AGENTS.md` alla radice.

Prima di scrivere o modificare ogni famiglia di kernel, confrontarla con
[Strata al commit 46c152a1310f77cd2f0ce2c7cd213962df14fe83](https://github.com/Niko1221/Strata/tree/46c152a1310f77cd2f0ce2c7cd213962df14fe83/metal),
fissato il 29 ottobre 2026. Le precedenti analisi CUDA del fork restano
riferimenti storici distinti. Consultare i kernel Metal e i rispettivi test,
non dedurre prestazioni Apple dai benchmark NVIDIA.

- [ ] **S1, prima dell'implementazione:** confrontare layout quantizzato,
  shape, precisione, tiling, accessi memoria, riduzioni e occupazione con il
  kernel Strata pertinente. Registrare file/commit, soluzione riusabile,
  incompatibilità e ragione della scelta; evitare di riscrivere una soluzione
  già trasferibile. Verificare licenze e attribuzioni prima di importare codice.
- [ ] **S2, primo kernel corretto:** ripetere il confronto sul codice scritto,
  inclusi wrapper, dimensione dei threadgroup, scratch e sincronizzazione.
  Eseguire l'oracle numerico indipendente prima delle misure di velocità.
- [ ] **S3, tuning:** confrontare nuovamente le scelte di Strata su shape e
  dtype identici, con input congelati e più campioni brevi. Misurare latenza,
  byte temporanei e traffico; includere il costo di eventuali conversioni.
  Se non è possibile eseguire Strata nelle stesse condizioni, dichiarare
  confronto statico e non rivendicare uno speedup misurato contro Strata.
- [ ] **S4, integrazione:** riesaminare fusioni, dispatch, riuso dello scratch
  e lifetime nel forward reale; ripetere numerica e misura dell'intera fase.
  Registrare i quattro controlli per ciascuna famiglia, anche dopo una revisione
  sostanziale. Nessuna ottimizzazione viene mantenuta con drift non spiegato.

I riferimenti iniziali comprendono `metal/kernels/quant.metal`,
`gemv.metal`, `dense_mm.metal`, `gdn.metal`, `gdn_prompt.metal`,
`qsa*.metal`, `moe*.metal`, il budget in `metal/src/engine/arena_budget.h`
e i test `metal/tests/*parity*`. Per un'operazione image/embedding non coperta
da Strata, documentare l'assenza e applicare comunque i quattro riesami alle
primitive condivisibili; l'oracle del modello resta obbligatorio.

Per **Qwen-Image-2.1-Turbo**, su indicazione dell'utente del 29 ottobre,
il riesame comprende anche
[Transformers](https://github.com/huggingface/transformers) e
[llama.cpp](https://github.com/ggml-org/llama.cpp). Prima di costruire o
modificare una famiglia di kernel, consultare Transformers per la semantica
degli encoder compatibili e llama.cpp/ggml per le primitive Metal pertinenti,
verificando la copertura effettiva del modello e delle operazioni. Conservare
[Diffusers](https://github.com/huggingface/diffusers) come riferimento della
pipeline image, di DiT, VAE, scheduler e conditioning. Fissare commit, file,
licenze e attribuzioni; confrontare shape, layout, accumulo FP32, storage e
frontiere BF16, tiling, workspace e sincronizzazione nei quattro riesami
S1–S4. L'assenza di un'implementazione compatibile va registrata; una primitiva
di un altro modello non prova l'equivalenza della pipeline Qwen Image. Gli
oracle indipendenti e i gate numerici/qualità restano obbligatori.

### 6.2 Implementazione e gate del 27B

- [x] Inventariare per ogni tensore tipo, shape e uso: embedding, output,
  matvec di decode, prefill multi-riga e MTP. Coprire Q3_K, IQ2_XS, IQ2_S,
  IQ3_XXS, IQ3_S, IQ4_NL e IQ4_XS; verificare inoltre tutti i tipi già presenti
  nel target e Q4_0 del sidecar. Riutilizzare codice Metal/ggml compatibile
  solo dopo verifica del layout e della licenza, fissandone la revisione.
- [x] Mantenere i pesi quantizzati mmap-backed. Dequantizzazione limitata a
  tile/singole proiezioni, con cap esplicito: nessuna espansione dell'intero
  modello in F16/F32 o conversione lossy per aggirare formati mancanti.
- [x] Implementare i binding `ds4_gpu_qwen35_*`: Q/K norm, RoPE parziale,
  gated full-attention, KV, convoluzione causale e Gated DeltaNet. I kernel
  Flash Next sono una fonte da valutare, non una prova di equivalenza di shape.
- [x] Riutilizzare lo scheduling C `qwen_graph_*`; portare decode e prefill
  causale a chunk, stato persistente, reset e cleanup. Verificare esplicitamente
  gli alias scratch oggi motivati dall'implementazione CUDA.
- [x] Rimuovere i guard CUDA-only soltanto per combinazioni effettivamente
  supportate; conservare il rifiuto precoce per le altre.

**Implementazione T03 e gate sintetico:** [report T00–T03](../performance/metal-apple-silicon-20261029/t00-t03/README.md).
Quantizzazioni, oracle e grafo ibrido ridotto passano; replay dello stesso
decode e alias SwiGLU bit-exact, chunk 1–5 con logits/stato entro le soglie
congelate. La shape ridotta e i pesi Q8_0 sintetici non sono il modello 27B.
La vision Qwen27 Metal è implementata e il suo encoder è qualificato separatamente.
**Gate locale T03/T06-B PASS**: modello reale, replay/prefisso, otto posizioni
con logits completi, Quick Chat e occupazione 2048/4096/8176. Il confronto
Strata S1–S4 resta documentato separatamente, senza dedurlo dai test locali.
Vedi [evidenze T06–T07](../performance/metal-apple-silicon-20261031/t06-t07/README.md).

**Gate numerico:** blocchi quantizzati confrontati con decoder scalare
indipendente; attention e GDN con oracle ridotti FP64/FP32; replay attraverso
confini dei chunk. Congelare tolleranze prima del confronto. Per il modello
reale usare pochi prompt brevi e 8–16 posizioni teacher-forced con logits
completi. Non richiedere bit-exact CUDA/Metal; richiedere invece replay dello
stesso percorso riproducibile e nessuna divergenza numerica inspiegata.

**Gate locale aggiornato:** dopo Flash, caricamento reale 27B target-only
con streaming T06-B e capacità 8192/prefill 64, prompt breve, 8–16 token,
estensione del prefisso e chiusura corretta, senza pressione memoria sostenuta.
Seguono le prove di occupazione effettiva del contesto (§11). Un eventuale
run residente è una qualifica separata, non un prerequisito locale.
Se non ammesso, registrare il limite senza dichiarare completato il supporto reale.

## 7. T04: portare le funzionalità del fork, oltre la chat

**Implementazione aggiornata al 31 ottobre 2026:** T06 e Quick Chat reali
passano su entrambe le famiglie; il replay agentico del GGUF 27B completo
passa con tre livelli, isolamento, cancellazione e sessione portabile su SSD,
con logits bit-exact. Il gate esteso API/frontend resta `IN_PROGRESS` e
distinto dai test del server nativo e del frontend mock.
La fixture ibrida ridotta esegue il grafo di produzione su Metal e verifica
`padre → skill → ritorno`, cancellazione, checkpoint SSD in una nuova sessione,
metadati RNG, KV, stato ricorrente e frontier MTP. Evidenze e riproduzione:
[qualifica T04–T05](../performance/metal-apple-silicon-20261031/README.md).

La [revisione Clean Code](../performance/metal-apple-silicon-20261031/clean-code-review/README.md)
sposta il budget vision nel backend proprietario dei buffer, rende canonico
lo stato MTP inattivo negli export e separa l'adapter del riferimento dai
parser condivisi degli artefatti. La regressione successiva comprende 25
gruppi PASS, 1210 test frontend PASS e i due encoder reali PASS; il gate
integrato sui LLM completi resta aperto.


- [ ] Riutilizzare server e frontend ExMekh: Chat Completions, Responses,
  streaming SSE e percorso Anthropic, dove previsti dal contratto corrente.
  Separare i dialetti Qwen27 e Flash: rendering, tool call e maschere devono
  continuare a seguire la propria famiglia.
- [ ] Verificare tool obbligatorio/opzionale, allowlist `allowed_tools` e
  `allowed_skills`, JSON Schema, output strutturato, richieste invalide e
  protocol-looking text. Usare schemi piccoli nei test locali.
- [ ] Verificare `agent_context_id`, ingresso/ritorno skill, isolamento del
  figlio e ripresa del padre. Salvare KV **e** stato ricorrente, timeline,
  logits, RNG e stato speculativo pertinente; non limitarsi ai token.
- [ ] Coprire riuso KV, checkpoint su SSD, restore in una nuova sessione,
  restart del server, corruzione/troncamento, cancellazione, steering durante
  una richiesta e replay/idempotenza dell'API interattiva.
- [ ] Provare compaction e history su transcript artificialmente piccoli.
  Misurare il prompt completo, inclusi sistema, strumenti e ragionamento:
  8192 è il tetto complessivo, non lo spazio disponibile per il solo utente.
  Contare nuovamente il prompt con il tokenizer e il renderer della famiglia;
  il conteggio Quick Chat 27B non qualifica il dialetto Flash. Allineare
  capacità server e frontend e rifiutare gli input oltre budget senza troncarli.
- [x] Adattare i test esistenti con backend e budget espliciti, senza
  trasformare un percorso host-only o mock in evidenza di inferenza Metal.

**Gate:** sequenza breve `padre → skill → ritorno`, tool e JSON conformi,
continuazione equivalente dopo restore, cancellazione recuperabile e un giro
del client ExMekh reale. Ripetere su ciascun modello eseguibile; per Flash
locale il gate dipende da T06.

## 8. T05: MTP e vision con picchi controllati

**Implementazione aggiornata al 31 ottobre 2026:** MTP Qwen27 usa ora il
contratto di copia Metal anche fuori da un batch esplicito; l'allocazione dei
buffer opzionali è atomica, non azzera il target e condivide lo stimatore con
ammissione e report memoria. La fixture Metal copre accettazione/rifiuto,
rollback RNG, catch-up, limite di contesto e ripristino multimodale. I test
Flash includono ora il predictor del GGUF completo: la verifica target Metal
procede per righe causali e gli otto token speculativi hanno logits identici
al decode ordinario. La regressione finale combina immagine reale e MTP in
64,09 s. Le prove sintetiche restano distinte da questi gate reali.

Vision Qwen27 ammette Metal, preserva il proiettore **5120** e rifiuta quello
Flash **2560** sul target incompatibile. Entrambi riusano il preprocessing
Qwen uint8 bicubic e il pianificatore di dimensioni con arrotondamento Python;
la modalità raw cambia soltanto la normalizzazione. I checkpoint multimodali
Qwen usano VMCP v2: le versioni precedenti vengono rifiutate senza mutare la
sessione. L'encoder verifica il
picco host/device dopo il parcheggio delle sessioni. Una prova di rifiuto
memoria assicura il ripristino di tutte le conversazioni.

Gli encoder reali sono verificati separatamente con una sola immagine e
64–128 token effettivi, confrontando sia patch identiche sia l'intero percorso
immagine con Transformers e gli stessi pesi GGUF. Soglia coseno minima **0,99**,
fissata prima delle prove. I report conservano anche i fallimenti iniziali del
preprocessing. Il gate locale LLM+MTP/vision è ora **PASS** dopo T06: entrambi
i GGUF completi eseguono otto token target/speculativi, replay e restore,
con il sidecar 27B corrispondente e i rispettivi encoder. Flash ha delta logits
massimo zero; Qwen27 4,768e-6 entro le soglie congelate. Nessuna inferenza LLM
su CPU o riduzione implicita di contesto è usata per chiuderlo. Comandi,
correzioni e limiti nel [report T06–T07](../performance/metal-apple-silicon-20261031/t06-t07/README.md).

- [x] Portare MTP 27B con sidecar corrispondente, cache indipendenti,
  cattura/ripristino dello stato ricorrente, verifica e catch-up. Provare
  accettazione e rifiuto dei draft, rewind, contesto quasi pieno e cancellazione.
- [x] Conservare la semantica di sampling dichiarata: l'accettazione greedy
  e il sampling esatto non sono intercambiabili. Riutilizzare i controlli
  esistenti e non promettere speedup senza misure.
- [x] Portare vision 27B riusando l'encoder compatibile e mantenendo il suo
  proiettore 5120; rifiutare il proiettore Flash incompatibile. Implementare
  mRoPE interleaved [11,11,10], posizioni assolute e aggiornamenti KV MTP.
- [x] Preservare fingerprint immagine, span, delta mRoPE e checkpoint completi.
  Provare `testo → immagine → testo → restore → testo` e immagine cambiata con
  medesimo testo. Iniziare con una sola immagine da 64–128 token visivi.
- [x] Dimensionare il picco encoder/LLM e, se necessario, salvare e rilasciare
  lo stato prima dell'encoder, poi ripristinarlo. Non mantenere copie host/GPU
  duplicate della stessa attivazione per comodità.
- [x] Qualificare gli equivalenti MTP/vision Flash sul grafo Metal esistente
  e, dopo T06, sul percorso streamed. MTP e vision devono poter essere
  disabilitati esplicitamente senza compromettere il testo.

**Gate:** test sintetico mRoPE senza modello, verifiche MTP di stato, un breve
caso reale per funzione quando ammesso e nessuna perdita di conversazione dopo
errore o cancellazione. Le combinazioni non eseguite restano `NOT_VERIFIED`.

## 9. T06: streaming SSD Metal, prima Flash e poi Qwen27 a 8k

Lo streaming di questa milestone riguarda **i pesi da SSD**, distinto dallo
streaming SSE dei token HTTP. I sorgenti comprendono una cache sincrona per
layer condivisa dai trasporti Flash e Qwen27 in
[`ds4_qwen4_metal_stream.inc`](../../ds4_qwen4_metal_stream.inc), che legge
pesi densi ed esperti selezionati e sincronizza prima del rilascio. Il 27B
usa lo stesso contratto di storage per i propri pesi densi, conservando il
grafo e i kernel della famiglia. Le qualifiche sui GGUF reali seguono sotto;
i test sintetici e di lifecycle restano evidenze distinte.

### 9.1 T06-A: completare e avviare Flash streamed

- [x] Inventariare byte non evictable, esperti, attention, MTP, cache e
  transitori del Q2 reale. Calcolare il minimo working set; includere
  `maxBufferLength`, paginazione, spazio SSD e costi di apertura/riscaldamento.
  Verificare prima che download e GGUF siano completi: metadata, directory
  tensor, offset/dimensioni entro il file e identità/hash dell'artefatto.
  La sola presenza del file da circa 137 GiB non prova la fine del download.
- [x] Conservare il percorso Metal residente per Mac con memoria sufficiente.
  Sulla macchina locale fermare il tentativo residente prima di allocarlo.
- [x] Implementare un adapter Metal per cache esperti con cap in byte e I/O
  bounded. Riutilizzare la policy C e le code esistenti dove compatibili;
  eventi CUDA, memoria pinned, grafi CUDA e ottimizzazioni PCIe non si
  trasferiscono direttamente alla memoria unificata.
- [x] Leggere gli esperti selezionati dal router in buffer riutilizzabili;
  gestire pinning degli slot in uso, cache miss, eviction e drain. Non usare
  paging/swap incontrollato del GGUF come sostituto di uno streaming esplicito.
- [x] Completare e verificare il caricamento per layer dei pesi densi e
  non-routed insieme agli esperti. Includere embedding, output head e
  l'unione degli esperti selezionati dal chunk di prefill da 64 token:
  può essere maggiore del working set del singolo token. Se un'unità supera
  il cap, suddividere esplicitamente le letture/l'esecuzione conservando
  ordine, routing e numerica; non affidarsi al paging dell'intero GGUF.
- [x] Prima implementare caricamento e sincronizzazione corretti; introdurre
  prefetch/overlap soltanto dopo parity e lifetime verificati. Preservare
  gli n-gram su disco e il loro recupero dopo letture fallite. Il primo avvio
  può usare il percorso sincrono; overlap e riuso dei buffer hanno misure e
  gate propri e non sostituiscono la prova di correttezza.
- [x] Eseguire prove sintetiche con cache volutamente piccola e letture
  fallite; confrontare logits e stato con il percorso residente su fixture
  ridotte. Su Mac capiente aggiungere il confronto col medesimo GGUF reale.
- [x] Preparare un profilo esplicito Flash Metal streamed **8192/64**,
  target-only, reasoning none, vision e suite disattivate; allineare
  `DS4_CTX` e `CONTEXT_LENGTH`, rendere visibili cap cache e opzioni I/O.
  Verificare in dry-run il launcher Flash e poi Responses/Quick Chat sul
  server nativo. Il wrapper agent Qwen27 forza la propria famiglia e non
  deve essere usato per lanciare Flash.

**Gate iniziale T06-A:** primitive, policy, I/O e cap verificati; apertura
Flash reale con capacità **8192** e prefill **64**, prompt breve e 1–4 token
dopo ammissione, poi 8–16 token, riuso del prefisso e cleanup. Registrare
logits e stato su posizioni congelate, non soltanto testo plausibile.
Prima del passaggio al 27B eseguire anche un turno Quick Chat Flash reale;
il riempimento progressivo fino a 8k resta un gate esplicito in §11.

### 9.2 T06-B: aggiungere lo streaming dei pesi densi del 27B

- [x] Dopo il gate iniziale Flash e Quick Chat, inventariare il working set
  minimo 27B a **8192/64**: embedding/output, proiezioni dei 64 layer,
  quantizzazioni UD, KV, GDN/convoluzione e scratch. Separare pesi evictable
  dallo stato persistente, che deve sopravvivere al cambio layer.
- [x] Introdurre letture SSD esplicite per gruppi/layer nei binding Metal
  `qwen35`, con buffer posseduti e cap in byte; riusare il trasporto Flash
  solo dove il contratto coincide. Il 27B è denso: non ha routing esperti
  da simulare e conserva grafo ibrido, formati e kernel numerici propri.
- [x] Coprire tutti gli accessi ai pesi, anche embedding e output head,
  senza riscaldare o rendere residente l'intero mapping. Mantenere il
  percorso residente mmap-backed; nessuna copia integrale o conversione
  lossy del GGUF. Fence prima del riuso, cleanup dopo errore/cancellazione
  e checkpoint completi devono rispettare gli stessi owner del runtime.
- [x] Estendere l'ammissione e la stima del contesto per famiglia e modalità:
  conteggiare il working set realmente pianificato, staging e stato 8k.
  Non sostituire i 14,30 GiB del file col cap Flash finché tutti gli accessi
  27B non sono instradati nello storage bounded. Conservare riserva di 2 GiB
  e guard di capacità; rifiutare le combinazioni ancora non supportate.
- [x] Verificare su fixture ridotte pesi/logits/stato contro il percorso
  residente, confini dei chunk e dei layer, cache piccola, letture
  corte/fallite e ripresa. Applicare §6.1 se si modificano i kernel;
  la correttezza Flash non dimostra quella di attention/GDN/quant UD 27B.
- [x] Preparare il profilo 27B streamed **8192/64**, con le stesse opzioni
  iniziali Flash, quindi smoke reale T03 e Quick Chat. Confrontare i due
  modelli per memoria/I/O/tempi sullo stesso contesto; logits e qualità si
  confrontano con il riferimento della rispettiva famiglia.

**Gate T06-B:** inferenza reale 27B streamed, primitive e logits entro le
tolleranze T03, prefisso e stato corretti, cleanup e un turno Quick Chat
su capacità 8192. Ripetere poi il riempimento effettivo previsto in §11.
Per entrambi i modelli, un rifiuto per risorse o timeout resta
`SKIP_RESOURCE_LIMIT`/`NOT_VERIFIED`; la build, i test sintetici e l'apertura
da soli non certificano kernel o inferenza a 8k funzionanti.

**Esecuzione locale T06 PASS (31 ottobre 2026):** entrambi i GGUF completi,
8192/64 e cache SSD 2 GiB, otto token con logits completi e stato verificato,
Quick Chat reale, occupazione 2048/4096/8176 e alternanza con Gemma reale.
Evidenze, primi tentativi falliti e limiti ancora aperti sono nel
[report T06–T07](../performance/metal-apple-silicon-20261031/t06-t07/README.md).
Il confronto residente dello stesso GGUF su Mac più capiente resta una
qualifica hardware separata; le fixture ridotte residente/SSD passano.

**Primo ciclo prestazionale Flash, 2 novembre:** letture demand degli esperti
parallele in una coda C bounded, senza nuovi buffer di staging o modifiche
ai kernel DS4 ereditati. Sul subset M5 Pro, 8192/64/cache 1 GiB, cinque coppie
alternate misurano prefill +89–98% e decode +16–18% alle frontiere 64/128.
Gate isolati, logits/stato completi, MTP e profilo standard cache 2 GiB
passano. Il [report prestazionale](../performance/metal-inference-20261102/flash/README.md)
conserva audit S1–S4, campioni, limiti e controllo 512 token; nessuna nuova
qualifica di throughput 8k occupati, altri modelli o consumer live estesi.

## 10. T07: lifecycle, embeddings e immagini della suite

Questa milestone verifica funzionalità fondamentali del fork. Una consegna
testuale intermedia è possibile, ma il piano non è completato finché le prove
reali image, embedding e di integrazione obbligatorie non sono superate.
Sviluppare e verificare gli adapter standalone appena disponibili le primitive,
poi collegarli al LLM; non subordinare tutto il lavoro suite alla capienza Flash.

- [x] Estendere runtime e lifecycle a Metal: fence, ownership, checkpoint,
  park, rilascio selettivo dei pesi e ripresa. Aggiungere adapter distinti
  per Qwen27 e Flash; preservare runtime e lock fino all'ultimo proprietario.
  Non scambiare questi metodi per le API di checkpoint agentico già esistenti.
- [x] Verificare separatamente il lifecycle Qwen27 target-only e Flash con
  shape complete e stato sintetico su buffer Metal reali: restore bit-exact
  in nuova sessione, errore di allocazione parziale, checkpoint corrotto,
  troncato o mancante, metadata intatti sui guasti e cleanup del ledger.
  Cap device di 1 GiB, riserva host di 2 GiB, famiglie eseguite in sequenza.
  Questo gate senza GGUF non chiude l'alternanza con inferenza reale.
- [x] Implementare un backend `ds4_numeric` Metal con API C invariata per
  buffer, copie, GEMM, norm, RoPE, attention tiled, reorder, pooling e
  convoluzioni. Conservare accumulo FP32 e punti di arrotondamento BF16.
  Verificare capability e risultati prima di usare percorsi accelerati.
- [x] Per ogni nuova famiglia di kernel Qwen Image o revisione sostanziale,
  registrare il confronto con Transformers, llama.cpp/ggml e Diffusers
  secondo §6.1, indicando codice riusabile, incompatibilità e scelta finale
  prima della scrittura e del tuning.
  Il [riesame del primo novembre](../performance/metal-apple-silicon-20261031/t06-t07/upstream-numerics.md)
  fissa le revisioni HF/ggml e Strata prima dei nuovi probe BF16. Il backend BF16 integrato conserva gli ordini di accumulo/riduzione e
  le funzioni fissate: testo Gemma 81/81 frontiere esatte su tutte le dimensioni,
  media 480/480 e Qwen RGB/RGBA 69/69. Generazione, RGB e RGBA completi
  passano contro encoder/DiT CPU e VAE MPS esplicita. I client rimanenti
  e la qualità estesa restano gate distinti e aperti. La matrice finale del
  riesame distingue S1–S4 per famiglia; la baseline di 14 primitive ha due
  campioni bounded, senza speedup rivendicato contro Strata.
- [x] Collegare factory e build della suite al backend selezionato, eliminando
  dipendenze CUDA/cuDNN dal link Metal. Portare anche FFmpeg/soxr, percorsi
  librerie e packaging macOS; nessun server Python sostitutivo.
- [x] Portare prima EmbeddingGemma testo, poi immagini, audio, video e
  interleaving; successivamente Qwen-Image encoder, DiT e VAE, generazione ed
  editing. Mantenere i gate numerici e di qualità già previsti dalla roadmap
  della suite, compresi quelli image ancora aperti su CUDA.
- [x] Usare il planner per alternare componenti, pesi e stato LLM su SSD
  quando non co-residenti. Limiti di risoluzione/durata e profili numerici
  diversi devono essere richiesti esplicitamente e qualificati.
- [x] Verificare `LLM → embedding/immagine → LLM`, anche dopo cancellazione,
  errore di allocazione o lettura. Pubblicare il risultato solo dopo il restore
  richiesto; memoria insufficiente deve lasciare utilizzabile la conversazione.
  Payload/logits completi esatti su entrambe le famiglie, recovery HTTP e
  cancellazione al primo latente con retry verificati; lo stress esteso resta
  una qualifica distinta.
- [x] Esporre in discovery le capacità effettive. Una funzione implementata
  ma non ammissibile localmente deve riportare il limite; il launcher di base
  deve restare utilizzabile durante il porting della suite.

**Gate richiesto:** inferenza reale con i pesi dei modelli configurati,
generazione ed editing image completi, embedding testuali/multimodali utili e
restore della conversazione. Iniziare sul Mac locale con i profili piccoli
della sezione 11; usare esecuzione standalone e rilascio seriale dei componenti
per contenere il picco. Se una prova non entra, mantenere quel gate aperto e
ripeterlo su Apple Silicon più capiente. Mock, primitive isolate e rifiuti OOM
non sostituiscono il funzionamento dei modelli. Stress e qualità estesa restano
gate ulteriori, distinti dallo smoke funzionale.

### 10.1 Matrice obbligatoria delle funzioni fondamentali

La matrice deve essere mantenuta per backend, modello, profilo e hardware.
Ogni riga richiede output reali verificabili, criteri congelati e un verdetto;
una semplice risposta HTTP 200 non basta. Le righe riferite al LLM si ripetono
su Qwen27 e Flash, con stato separato per ciascuno. Le righe dei modelli suite
si eseguono standalone e, per l'alternanza, con ciascun LLM.

| ID | Funzione fondamentale | Prova reale minima e criteri di accettazione |
| --- | --- | --- |
| F01 | LLM, MTP e vision | Testo e riuso del prefisso; MTP on/off con rollback; immagine seguita da testo e restore. Logits/stato confrontati secondo T03/T05, modalità distinte |
| F02 | Image: generazione | Qwen-Image-2.1-Turbo, prompt e seed fissati, encoder → tutti gli otto passi DiT → VAE → PNG. Dimensioni/recipe corrette, valori intermedi finiti e confronto con oracle; immagine ispezionata per contenuto e artefatti |
| F03 | Image: editing e RGBA | Una reference RGB 256×256, modifica descritta e reference RGBA per il caso alpha; ordine degli input, geometria e canali conservati secondo la recipe. Confrontare conditioning, latenti/RGBA e preservation con riferimenti congelati; restituire il PNG originale tramite Files |
| F04 | Embedding testuali | EmbeddingGemma 2 su stringhe/token, batch ragged e task supportati. Dimensioni 128/256/512/768, float/base64, vettori finiti, norma L2 e ordine corretti; confronto numerico e piccolo ranking query/positivo/negativi rispetto all'oracle |
| F05 | Embedding multimodali | Una piccola immagine, audio breve, video breve e un input interleaved ordinato. Usare gli encoder reali, controllare processor/maschere, vettori e ranking rispetto agli oracle delle modalità; coprire video con/senza traccia audio secondo il contratto |
| F06 | Alternanza e memoria | `LLM → embedding → LLM → image generation/edit → LLM`, con una richiesta per volta. Verificare token/cache/contesto preservati, continuazione rispetto alla baseline senza tool, pubblicazione dopo restore e rilascio di pesi/scratch/staging |
| F07 | Tool, skill e checkpoint | Tool obbligatori/opzionali, JSON vincolato, `padre → skill → ritorno`, isolamento dei contesti e riuso KV. Ripetere dopo restore da SSD/restart e dopo compaction su transcript piccolo |
| F08 | Editing interattivo del fork | Flusso XML v2 con lettura/provide, create, edit multiplo, delete, cancellazione e replay. Il client reale opera solo in un workspace temporaneo di fixture; verificare contenuto finale, call/exchange ID e ripresa LLM |
| F09 | Ricerca/ranking tramite embedding | Eseguire il percorso di ranking ExMekh su corpus locale fissato di testo/immagini, con embedding nativi reali e dati di ricerca riproducibili. Verificare ordine/corrispondenza dei risultati e fallback dichiarato se il servizio non è disponibile |
| F10 | API, media e discovery | Client standard e tipizzato sulla stessa base URL; upload/download/riuso file con checksum, mapping degli errori, recipe/seed/dimensioni e capability effettive. Nessun parametro ignorato o risultato parziale pubblicato |
| F11 | Frontend e persistenza | Dal client/agente reale invocare `generate-image`, salvare PNG originali e mostrare preview; riaprire la chat e ritrovare messaggi/media. Verificare allegati vision espliciti, steering/cancellazione e continuità della conversazione |
| F12 | Guasti e risorse | Cancellazione, disconnessione, timeout, retry/idempotenza, input corrotto, budget insufficiente ed errore di lettura/restore. Nessun doppio job o perdita di contesto; checkpoint recuperabile e contatori bounded dopo cleanup |

- [ ] Riutilizzare `tools/qualify_qwen_image_pipeline.py`, gli oracle encoder/DiT/VAE
  e le fixture di qualità image; conservare la distinzione tra difetto del
  checkpoint, regressione Metal e requisito di qualità ancora aperto su CUDA.
- [ ] Riutilizzare `tools/qualify_embedding_gemma2.py`,
  `tools/qualify_gemma_visual.py`, `tools/qualify_gemma_multimodal.py` e le
  tolleranze fissate, verificandone l'applicabilità a Metal prima dei run.
  Precalcolare gli oracle pesanti altrove; non caricarli insieme ai modelli locali.
- [ ] Collegare i gate di servizio a `tests/test_suite_http_live.py`,
  `tests/test_suite_exmekh_live.py`, `tools/qualify_suite_release.py` e ai test
  interattivi/agentici esistenti. Creare selezioni brevi senza modificare i
  criteri di correttezza per ottenere un PASS.
- [ ] Per la prima verifica di alternanza eseguire un ciclo completo; dopo
  il warm-up ripetere fino a tre cicli brevi per identificare crescita della
  memoria. Lo stress di almeno 50 alternanze previsto dalla suite resta una
  qualifica estesa separata, non certificata dai tre cicli locali.
- [ ] Salvare PNG originali, reference e confronti, vettori/ranking, risposte
  HTTP, traccia del client e ledger di memoria. Nessuna riga può essere chiusa
  con soli log di apertura del modello o con risultati mock.

## 11. Strategia di test contenuta

**Nuova fase richiesta il 2 novembre 2026:** ottimizzare l'inferenza Metal
un modello alla volta, prima Flash Next Q2 streamed e poi Qwen27 streamed.
Il [piano prestazionale e registro](metal-inference-performance-20261102.md)
definisce profilazione, riesame delle fonti per ogni intervento, ottimizzazioni
di kernel e workflow, dispatch distinti per prefill/decode e valutazione
cumulativa anche dei piccoli incrementi. La campagna include misure e gate
numerici indipendenti sul modello completo pertinenti alle modifiche;
i consumer live estesi sotto sospesi restano una qualifica successiva.
La preparazione del piano non costituisce una nuova misura prestazionale.

**Preferenza precedente del 2 novembre 2026, riesame offline:** sospendere le prove live
rimanenti. Continuare il riesame dei bug d'inferenza con controlli statici,
input isolati e fixture sintetiche. Dedicare una sessione successiva ai TPS
prima di riprendere i consumer live. F08 rimane `NOT_VERIFIED`; F10/F11
passano sul subset locale già eseguito su entrambe le famiglie. T07 resta
aperto; nessun nuovo full-model gate viene avviato durante questo riesame.
Esiti e limiti sono nel [report offline del 2 novembre](../performance/metal-apple-silicon-20261031/t06-t07/offline-inference-review-20261102.md).

**Preferenza utente aggiornata, 31 ottobre 2026:** usare la RAM disponibile
lasciando circa 2 GiB per agente e sistema. Nei nuovi run il working set è
pianificato dalla memoria reale; nessun cap fisso del processo a 10 GiB o
della suite a 4 GiB è imposto dal profilo automatico. Restano attivi la
riserva host, lo stop per pressione critica e per crescita sostenuta dello
swap. Un warning di pressione, da solo, non equivale a fallimento. Questo
aggiorna la policy di risorse, senza cambiare recipe, output o soglie numeriche.
I tentativi precedenti mantengono il verdetto e i limiti allora applicati.

Non eseguire automaticamente `make test` o benchmark aggregati senza prima
inventariare modelli richiesti, memoria e timeout dei singoli target.

| Livello | Configurazione proposta | Evidenza ottenibile |
| --- | --- | --- |
| L0 — Statico/build | Nessun modello; `make -j2`, link, dry-run launcher e test host mirati | Portabilità, protocollo, ammissione, strutture dati |
| L1 — Primitive GPU | Blocchi/tensori sintetici e tile; budget iniziale massimo 1 GiB per processo, verificato prima del run | Quant, GDN, attention, mRoPE, lifetime e streaming |
| L2 — Smoke streamed | Prima Flash, poi 27B; capacità 8192, prefill 64; prompt semplice 32–128 token; prima 1–4, poi 8–16 token generati; una sessione | Ammissione e avvio reali, primi logits/decode, prefisso e chiusura; non ancora contesto pieno |
| L3 — Quick Chat e funzioni fork | Capacità 8192/prefill 64 per ciascun modello; prompt completo misurato con il proprio renderer; Quick Chat prima, poi casi tool/JSON/checkpoint | Correttezza funzionale di base, frontend e restore |
| L4 — Occupazione effettiva 8k | Dopo L2/L3, stesso profilo 8192/64: prompt totali circa 2k, 4k e prossimi a 8k, lasciando un budget output esplicito; casi oltre limite rifiutati | Prefill attraverso più chunk, posizioni KV alte, continuazione/restore, logits/stato e picco con contesto realmente utilizzato |
| S1 — Embedding reale | Pesi Gemma, 2–3 testi brevi e batch piccolo; smoke a 128 dimensioni, poi tutte le dimensioni supportate nello stesso processo | F04: vettori, norma, formato e retrieval; una dimensione output minore non riduce i pesi del modello |
| S2 — Embedding multimodale reale | Una immagine piccola; clip audio 1–2 s; video breve con pochi frame secondo il processor; un interleaving. Una richiesta per volta, entro il budget del modello | F05/F09: funzionamento degli encoder e ranking; nessuna trascrizione/caption sostitutiva |
| S3 — Image reale | Una generazione e un editing 256×256, `n=1`, seed fissato, otto step e CFG 1 della recipe; reference editing 256×256 valida. Rilascio del LLM quando necessario | F02/F03: pipeline completa e file verificati; la sola risoluzione ridotta non riduce i pesi |
| S4 — Integrazione del fork | Un ciclo F06 dal client ExMekh, capacità LLM 8192 e transcript breve, output tool compatti, poi guasti/cancel e fino a tre cicli se ammessi | F06–F12: funzioni applicative, ripresa e memoria |
| L5 — Esteso | Mac con memoria sufficiente; stessi artefatti; confronto resident/streamed, corpus e stress completi | Tutti i gate non eseguibili localmente, qualità image/retrieval estesa e prestazioni; contesti oltre 8k richiedono un piano distinto |

Il limite locale **8k (8192 token) riguarda l'intera conversazione LLM**,
output incluso. Nei test vicini al limite riservare prima i token di risposta
e misurare il prompt effettivo; allocare una KV da 8k non prova di averla usata.
Eventuali diagnosi a contesto inferiore sono run separati e non chiudono L4.
Le sequenze degli encoder, i frame e le griglie latenti hanno propri limiti e
stime di memoria: una chat piccola non garantisce che il modello image o
embedding entri. Nessun troncamento implicito dei media per far passare il test.

- [ ] Una sola istanza di modello grande; eseguire i riferimenti pesanti
  altrove o in momenti separati. Nessuna grande inferenza CPU su macOS.
- [ ] Preparare una selezione breve dai test esistenti: `test-qwen4-kernels`,
  `test-qwen4-q2`, n-gram, `test_qwen35_mrope`, session state, risorse/runtime,
  server interattivo, client structured e suite reali image/embedding elencate
  in §10.1. Controllare prima shape, picchi e
  numero di ripetizioni: «senza GGUF» non significa necessariamente leggero.
- [ ] Prevedere **120 secondi iniziali per singolo caso LLM/embedding** e
  **300 secondi per singolo caso image**, con apertura inclusa e tempi delle
  fasi registrati. Sono cap iniziali, non stime delle prestazioni Metal.
  Suddividere la verifica in sessioni di massimo **10 minuti**, mantenendo
  obbligatori tutti i gruppi S1–S4: il limite di sessione non elimina i test.
  Consentire SIGINT/cleanup; se occorre terminazione forzata, non contare il cleanup PASS.
  Un timeout registra `NOT_VERIFIED`; eventuali estensioni sono run separati
  motivati, non retry infiniti. Non abbassare automaticamente il test fallito.
- [ ] Campionare footprint del processo, allocazioni Metal possedute, memoria
  disponibile, pressione, swap iniziale/finale, byte letti e picco. Arrestare
  il test se il margine pianificato è esaurito o pressione/swap crescono in modo
  sostenuto; non pretendere swap assoluto zero su un desktop già avviato.
- [ ] Un warm-up e al massimo una/due ripetizioni misurate dei casi scelti.
  Registrare load time, TTFT e decode per diagnosi, senza soglie di velocità
  inventate o confronto diretto con la RTX 3090.
- [ ] Conservare comando, commit, backend/capability, modello/hash, token
  effettivi, contesto, prefill, opzioni, tempi, picchi e verdetto in un report
  JSON compatto. Distinguere timeout, memoria, artefatto assente ed errore reale.

## 12. T08: criteri di consegna e regressioni

Aggiornamento6novembre: la [campagna T08](../performance/metal-apple-silicon-20261031/t08-20261106/README.md)
usa riserva host0su istruzione dell'utente. Chiusi localmente50cicli Flash,
alpha512numerico con oracle CPUencoder/MPSDiT/MPSVAE e Gemma480/480.
F08resta aperto su entrambe le famiglie; lo stress Qwen si ferma durante
il warmup per crescita swap oltre soglia. Asset/corpus estesi, altri chip
e backend restano nella [lista residua](../performance/metal-apple-silicon-20261031/t08-20261106/remaining.md).
Questi esiti non modificano i criteri di consegna seguenti.

**Prima consegna: Flash Metal streamed a 8k.** Sono richiesti build,
primitive numeriche e I/O bounded, smoke target-only con capacità 8192/64,
Quick Chat reale e prove L4 con contesto effettivamente utilizzato, prefisso,
cancellazione e checkpoint/restore. Misurare RAM, storage Metal e byte SSD;
un errore di ammissione corretto prova il guard, non l'inferenza funzionante.

**Seconda consegna: Qwen27 Metal streamed allo stesso contesto 8k.** Sono
richiesti streaming dei pesi densi T06-B, gate numerici T03 e gli stessi
gate locali L2–L4/Quick Chat della prima consegna. Riportare separatamente
quali kernel e percorsi sono verificati per ogni famiglia.

**Terza consegna: funzioni LLM complete per modello.** Oltre al testo,
servono MTP, vision, ripresa skill, JSON/tool, cancellazione e cache corretti.
Per Flash si qualificano distintamente residente e streamed. Il supporto
locale di Flash richiede almeno un run reale nel budget, non solo la build.

**Quarta consegna: suite Metal completa.** Entrambi i LLM devono alternarsi
correttamente con gli adapter supportati, con numerica, recovery, qualità e
API verificate. I limiti della singola macchina rimangono espliciti.
Tutte le righe F01–F12 devono avere un esito documentato: i gate reali image,
embedding e client sono obbligatori per completare il piano. Un limite locale
sposta la prova su hardware adeguato e lascia aperta la relativa qualifica;
non trasforma una funzione fondamentale in opzionale. Le prime tre consegne
sono risultati intermedi e non costituiscono completamento del fork su Metal.

- [ ] Mantenere separata la matrice M1/M2/M3/M4/M5 secondo le capacità
  effettivamente supportate. Iniziare con M5 Pro locale e almeno un secondo
  Mac di famiglia diversa quando disponibile; non estendere la qualifica
  empirica a hardware mai provato.
- [ ] Proteggere Metal upstream, SSD delle famiglie già supportate, CPU
  reference, CUDA e ROCm da modifiche condivise. Sulla macchina locale usare
  primitive/regressioni leggere; i modelli grandi restano `NOT_VERIFIED`.
- [ ] Per modifiche comuni, prevedere un gate CUDA su macchina dedicata e
  compilazione degli altri backend disponibili. Le regole di `AGENT.md`
  richiedono di chiedere accesso alla macchina CUDA e autorizzazione prima
  dei test distribuiti: questi ultimi non appartengono allo smoke locale.
- [ ] Aggiornare README, launcher, discovery e guide con capacità e limiti
  realmente qualificati, senza modificare retroattivamente i risultati CUDA.
  Conservare ExMekh come unico frontend e aggiornare `PROJECT_INDEX.md`.

### 12.1 Registro di esecuzione

- 29 ottobre 2026: inventario iniziale e `make -j2` eseguiti. La baseline
  fallisce in `ds4_bundle.c` per nomi dei timestamp nascosti dalle feature
  macro POSIX e per `posix_fadvise` assente su macOS. Nessuna inferenza avviata.
- Strata fissato al commit sopra; inventariati kernel e test Metal disponibili.
  I gate S1–S4 e F01–F12 restano aperti fino alle rispettive evidenze.

- 29 ottobre, implementazione: build Metal e suite native riuscite; launcher
  Metal 512/64, budget unificato, Qwen27 UD/GDN/mRoPE e attention collegati.
  `ds4_numeric` Metal conserva BF16 e contabilizza anche l'espansione FP32
  delle matrici correnti; test di ammissione confrontato col picco reale.
- Flash: primo streaming Metal sincrono per layer, cap predefinito 2 GiB,
  esperti selezionati compattati senza nuovi kernel numerici e n-gram su disco.
  Fixture con oracolo scalare, 512 esperti, cap e lettura fallita passano.
  Il percorso residente conserva la propria arena; lo streaming usa scratch
  privato per park/release/restore della suite. Test con storage Metal reale
  e shape Flash completa: payload bit-exact, checkpoint corrotto e restore
  senza RAM lasciano intatto lo stato vivo; sessione vuota e ripresa verificati.
  Prefetch/overlap e parity del modello reale restano aperti.
- Gemma reale: testo, immagini, audio, video e interleaved; dimensioni
  128/256/512/768, piccolo ranking, ripetizioni, cancellazione/riprova e budget
  insufficiente. Output finali entro soglie congelate; frontiere BF16 interne
  non tutte bit-exact contro l'oracolo CPU: gate stretto ancora FAIL.
- Qwen Image BF16: generazione e editing RGB 256×256 completati, otto passi,
  output RGBA e latenti finiti, picchi posseduti 3,43/3,70 GiB entro piano e
  zero storage numerico vivo a fine caso. PNG ispezionati; confronto numerico
  con oracle image, reference RGBA e gate alpha/qualità ancora aperti.
  Il profilo DiT Q4 resta non eseguibile dagli adapter BF16 già nella baseline.
- Qwen27 reale rifiutato prima del caricamento: 14,76 GiB necessari contro
  6,57 GiB incrementali disponibili dopo margine di 2 GiB. Flash reale ancora
  in download in un terminale dell'utente. CUDA non disponibile, confermato
  dall'utente; nessuna qualifica CUDA/ROCm o distribuita rivendicata.
- [Report e hash delle evidenze](../performance/metal-apple-silicon-20261029/README.md),
  incluso riesame S1–S4 statico con Strata e limiti del confronto. Le checkbox
  di implementazione non chiudono i gate reali F01–F12, che restano distinti.
- 29 ottobre, continuazione: riserva host di 2 GiB applicata anche ai probe
  suite Metal. Prima divergenza Gemma isolata mediante replay PLE di due testi
  e testimoni FP64; trace del proprio backend riprodotte esattamente, gate
  BF16 inter-backend ancora FAIL. Gemma reale, ripetizione e recovery passano.
- Qwen Image: editing con reference RGBA 256×256 e 16 livelli alpha, otto
  passi, seed fissato; PNG e latenti finiti, picco numerico 3,70 GiB entro piano,
  storage finale nullo. Completamento in 243,77 s con apertura inclusa, senza
  pressione sostenuta rilevata. Preprocessing verificato con Pillow 12.3 fissato;
  oracle image indipendente e qualità/alpha estesi restano aperti. Riferimenti
  Transformers/llama.cpp/Diffusers fissati e confronto S1 registrato prima di
  nuovi kernel; non è un riesame S2–S4 o un benchmark comparativo.
- Nuovo tentativo Qwen27: guard di ammissione attivo, 14,76 GiB necessari
  contro 9,42 GiB disponibili dopo riserva. Inferenza LLM ancora non verificata;
  Flash principale ancora assente. [Registro della continuazione](../performance/metal-apple-silicon-20261029/continuation/README.md).
- 29 ottobre, task senza Flash: build Metal/suite e build CPU completa in
  directory separata PASS; dieci test launcher e quattro gruppi host PASS.
  Lifecycle Qwen27 target-only e Flash verificati con shape complete e buffer
  Metal reali, senza GGUF/forward: restore in nuova sessione, metadata,
  fallimento di allocazioni parziali, checkpoint corrotto/troncato/mancante,
  park/release/resume e ledger ripulito. Cap device di 1 GiB, riserva host di
  2 GiB. Registrati disponibilità host, working set raccomandato, allocazioni
  Metal e max buffer come misure distinte. Nessun gate di inferenza LLM o
  alternanza reale chiuso. [Evidenze riproducibili](../performance/metal-apple-silicon-20261029/offline-foundations/README.md).
- 29 ottobre, preparazione Quick Chat richiesta dall'utente: dipendenze
  frontend bloccate installate su macOS, startup HTTP 200 e rendering UI
  verificati. Quarantasette test frontend mirati, undici test launcher e
  regressioni host del protocollo interattivo PASS. Il contatore duplicava
  un'istruzione assente nel server: risorsa allineata e confronto regressivo
  fra macro C e risorse del frontend. Sui quattro prompt iniziali congelati
  il rendering frontend/C è byte-exact; Quick Chat usa 2497 token senza
  reasoning e 2495 con reasoning, prima dell'output. Il profilo generico usa
  rispettivamente 4460/4458 token e non è il profilo iniziale scelto.
- Lo smoke target-only 4k è rifiutato prima dei pesi: 15,22 GiB necessari,
  5,65 GiB utilizzabili dopo riserva di 2 GiB. Inferenza Quick Chat, logits,
  tool reali e alternanza suite restano `NOT_VERIFIED`; nessun guard aggirato.
  L'utente libererà la memoria in seguito: questa fase termina con il profilo
  e i comandi pronti. [Registro Quick Chat](../performance/metal-apple-silicon-20261029/quick-chat/README.md).
- 29 ottobre, primo avvio dopo la preparazione richiesto dall'utente:
  smoke 27B Metal 4096/64, target-only, quattro token massimi. Il guard
  rifiuta prima dei pesi: 15,22 GiB necessari contro 8,78 GiB utilizzabili,
  con riserva host di 2 GiB e swap a zero. Nessun forward eseguito.
  Audit del loader: nessun prefetch CPU sul mapping Metal e controllo prima
  del riscaldamento/binding dei pesi; nessun bypass applicato. Lo screenshot
  dell'utente riporta 14,04 GB utilizzati e 10,26 GB di file nella cache.
  [Tentativo conservato separatamente](../performance/metal-apple-silicon-20261029/quick-chat/live-attempt-01.json).
- 29 ottobre, revisione richiesta dall'utente dopo i rifiuti del 27B
  residente: priorità a Flash Next Q2 in streaming SSD Metal, poi 27B
  UD-Q4_K_S in streaming dei pesi densi, entrambi con contesto massimo
  8192/prefill 64. Il precedente tetto 2k e il profilo preparatorio 4k non
  definiscono più l'obiettivo locale. Introdotti T06-A/T06-B, Quick Chat
  Flash prima del 27B, gate distinti per capacità allocata e occupazione
  effettiva fino a 8k. Non si attende la liberazione dei 17–18 GB per il
  percorso residente; riserva e guard restano attivi. Questa revisione
  aggiorna il piano, senza nuove prove di inferenza o modifiche ai launcher.
