# Ottimizzazione dell'inferenza Metal, un modello alla volta

Data: 2 novembre 2026. Stato: `IN_PROGRESS`; primi cicli Flash e Qwen27 `LOCAL_PASS`.
Modello attivo: **nessuna istanza; stress Qwen27, ripresa e regressione Flash chiusi il9novembre**.
Aggiornamento6novembre: l'utente elimina la riserva host fissa2GiB per la
ripresa. Il nuovo profilo usa `DS4_METAL_HOST_RESERVE_BYTES=0` in engine,
adapter e monitor. Costi reali di grafo/restore, floor XNU e pressione
critica restano distinti dal margine eliminato. Le evidenze precedenti
conservano il loro profilo originario; [campagna T08](../performance/metal-apple-silicon-20261031/t08-20261106/README.md).
Aggiornamento4novembre: continuazione senza commit; Flash e Qwen27 hanno
chiuso i rispettivi cicli locali. I-P02/I-P04 qualificano Conv VAE e RAM
dinamica Image, con file cache protetto e margine2GiB.
[Esito del ciclo](../performance/metal-inference-20261102/qwen-image/continuation-20261104.md).
Aggiornamento5novembre: I-P05 anticipa fino a16MiB del prossimo gruppo DiT.
Cinque coppie al ceiling2GiB misurano richiesta−8.838%, con CV8–9%.
I gate finali RGB/RGBA/Gemma passano e il confronto nativo512è bit-exact.
Il [report I-P05](../performance/metal-inference-20261102/qwen-image/group-prefetch-20261105/README.md)
conserva la variabilità, il piccolo delta di swap globale all'avvio di un
campione e la qualifica indipendente512ancora aperta. Nessun commit.
Riesame successivo del5novembre: l'utente autorizza l'integrazione con commit
distinti dopo i test. Convoluzioni, capacità/planner e prefetch sono separati.
Il [riesame finale](../architecture/metal-image-prefetch-clean-code-review-20261105.md)
registra i nuovi hash,35artefatti Image esatti e i limiti conservati.
Il 2 novembre l’utente autorizza il passaggio a Qwen27. Il primo ciclo è
concluso nel subset locale: Q-P03 KEEP, letture dense grandi parallele,
baseline e audit propri nel
[registro Qwen27](../performance/metal-inference-20261102/qwen27/README.md).
Q-P02 (tile UD generalizzato) è ritirato; kernel e criteri numerici conservati.
Il backlog residuo resta esplicito; Gemma conserva la campagna QUEUED.
Nel primo ciclo i kernel ereditati erano invariati. La continuazione del
3 novembre riscrive le proiezioni UD e qualifica MTP4; il costo I/O resta
rilevante anche dopo il miglioramento del bundle.

**Riesame successivo del 3 novembre:** l’utente richiede ora refactoring,
verifiche e commit distinti per scopo. Il divieto di commit della campagna
qui sotto resta cronologia della raccolta prestazionale; la nuova richiesta
autorizza l’integrazione dopo il riesame. Il report è nel
[riesame kernel/streaming](../architecture/metal-qwen27-kernel-stream-clean-code-review-20261103.md).

**Mandato aggiornato del 3 novembre:** continuare Qwen27 cercando almeno
**1 token/s**, senza cambiare pesi/semantica o allargare i gate numerici e
**senza commit**. Il lavoro comprende esplicitamente la **riscrittura dei
kernel Metal**, dei wrapper e del dispatch quando giustificata dalle misure;
non si limita a cache e I/O. Riscrivere un kernel è un'implementazione da
confrontare e qualificare, non un risultato prestazionale già acquisito.

Sequenza attiva della continuazione:

1. Concludere il confronto della residenza bounded a budget dichiarati e
   profilare di nuovo prefill/decode, conservando il controllo a2GiB.
2. Riscrivere e misurare almeno una proiezione UD rilevante: decode e
   prefill hanno geometrie diverse. Confrontare GEMV/GEMM, decode cooperativo
   dei blocchi, riuso dei pesi tra righe e costi di preparazione. Ripartire
   dai risultati Q-P02, che respingevano solo il tile generalizzato allora
   provato e non ogni possibile riscrittura o dispatch per forma.
3. Valutare esplicitamente la tecnica CUDA **Q8_1+R8** su Metal: quantizzazione
   delle attivazioni, doppio residuo/accumulatore e costo completo del kernel.
   Prima verificare numerica sugli input isolati; i gate CUDA storici e DP4A
   non certificano né equivalenza né vantaggio sul Mac. Se l'approssimazione
   non soddisfa i criteri Metal, provare una riscrittura che conservi FP32.
4. Valutare submit anticipato/overlap e il bundle contro B0 e Bi, con gate
   completo indipendente e warmup+5 coppie. MTP esatto è una modalità separata,
   da dichiarare insieme ad acceptance e token effettivamente emessi.

Ogni punto ripete audit upstream/locali prima del codice, isolati e modello
completo dopo; conservare esperimenti negativi e limiti, anche se1TPS resta
irraggiungibile nel budget/hardware disponibile. Il target1TPS non autorizza
a sostituire la misura reale con un microbenchmark o a ridurre la correttezza.

**Aggiornamento operativo del 2 novembre:** l'utente richiede una campagna
rapida per la sola parte Flash Metal già ereditata da DS4: pochi tentativi
mirati, inizialmente al massimo due direzioni se motivate dai profili.
Lo streaming e le aggiunte del fork conservano il ciclo completo. Il
[registro Flash](../performance/metal-inference-20261102/flash/README.md)
documenta baseline/profilo e confronto delle fonti per ciascun intervento.

Questo piano applica la richiesta utente del 2 novembre e affianca la
[roadmap di supporto Metal](metal-apple-silicon-support-20261028.md).
I gate locali T03/T05/T06 e i subset numerici T07 sono già documentati nel
[report T06–T07](../performance/metal-apple-silicon-20261031/t06-t07/README.md).
La correttezza di quei subset è il punto di partenza; F08, stress, qualità
estesa e hardware non provato conservano i rispettivi esiti aperti.

## 1. Regole permanenti

- Ottimizzare il tempo della richiesta: TTFT, prefill token/s, decode token/s
  e ms/token; per embedding latenza/throughput e per image tempi delle fasi e
  della pipeline. Registrare anche memoria, I/O e recupero dopo errori.
- Studiare ogni volta il collo di bottiglia attuale: kernel, dispatch, CPU,
  sincronizzazioni, letture, allocazioni, cache, restore e workflow possono
  avere costi diversi. Dopo un miglioramento, profilare di nuovo.
- **Ogni intervento**, anche di scheduling o workflow, richiede un confronto
  pertinente e registrato con fonti e ricerca locale prima di implementare.
  L'audit iniziale non vale come autorizzazione tecnica per i kernel seguenti.
  Ogni nuova discrepanza numerica o regressione riapre l'indagine specifica.
- Conservare pesi, quantizzazioni, precisione, semantica del sampling, maschere,
  stato, recipe e soglie numeriche congelate. Verificare ordine delle operazioni
  e frontiere BF16/FP32 anche per fusioni e cambi di chunk.
- Valutare anche incrementi piccoli e combinazioni coerenti. Nessun requisito
  del +1–2% per ogni singolo kernel. Un incremento deve essere misurabile;
  una differenza indistinguibile dal rumore resta da confermare.
- Un modello attivo alla volta, una sola grande istanza. Le regressioni di
  codice condiviso sugli altri modelli si eseguono in sequenza e non aprono
  una seconda campagna di tuning.

## 2. Ordine e avanzamento

Si mantiene la priorità Flash → Qwen27 della roadmap. Seguono i modelli suite,
ciascuno con metriche proprie; non attribuire TPS LLM alla generazione image.

| Ordine | Modello/modalità iniziale | Stato prestazionale | Condizione per passare oltre |
| --- | --- | --- | --- |
| 1 | Flash Next Q2, Metal streamed, target-only | `LOCAL_PASS`, F-P01 KEEP nel subset misurato | Ciclo rapido concluso; residui documentati, prove estese restano distinte |
| 2 | Qwen3.8-27B UD-Q4_K_S, Metal streamed | `LOCAL_PASS`, Q-P25 e riesame M-P48 integrati | Misure Q-P25con margine2GiB: tre richieste auto64+16 target1.49–1.61TPS e MTP4 4.09–4.31TPS. Il nuovo profilo T08usa margine0e mantiene separate misure e limite swap dello stress. |
| 3 | EmbeddingGemma 2, testo poi media | `QUEUED` | Gate BF16 e ranking conservati, guadagni misurati per modalità |
| 4 | Qwen-Image-2.1-Turbo, encoder/DiT/VAE | `LOCAL_PERF_PASS`, I-P01–I-P05;512NUMERIC_PASS | 256×256, otto passi. I-P05 su5coppie al ceiling2GiB: DiT−11.601%, richiesta−8.838%[−16.093,−1.314], CV8–13%. T08aggiunge alpha512indipendente CPUencoder/MPSDiT/MPSVAE entro le soglie, margine0e crescita swap0. Corpus esteso distinto. |

La chiusura di una campagna non richiede inseguire ogni microsecondo:
documentare hotspot residui, esperimenti scartati e il motivo per cui la
prossima direzione non è conveniente o richiede altro hardware. Il semplice
successo di un microbenchmark non chiude il modello.

## 3. Ciclo da ripetere per ciascun intervento

1. **Congelare il caso.** ID univoco, commit e diff, hash di sorgenti/binario,
   modello e artefatti, prompt/token effettivi, seed, modalità, shape, dtype,
   contesto allocato e occupato, chunk, cache e budget. Conservare B0, baseline
   iniziale della campagna, e Bi, ultima versione accettata.
2. **Misurare e attribuire.** Baseline senza profiler; poi trace separata.
   Ordinare gli hotspot per quota del tempo reale e beneficio massimo plausibile.
   Distinguere attesa host, lavoro GPU, I/O, submit e allocazioni. Tempi di
   worker concorrenti e intervalli sovrapposti non si sommano al wall-clock.
3. **Riesaminare le fonti per quel caso.** Compilare la scheda §4 con i file
   realmente letti. Cercare anche implementazioni nuove su internet, risalendo
   a codice, test, issue/PR e documentazione primari.
4. **Scegliere un'ipotesi falsificabile.** Descrivere causa, intervento minimo,
   metrica attesa, costi aggiuntivi e condizioni che smentirebbero l'ipotesi.
   Confrontare almeno le alternative pertinenti: kernel, scheduling o riuso.
5. **Implementare un cambiamento attribuibile.** Separare le componenti quando
   possibile. Se sono inseparabili, dichiarare il pacchetto e le interazioni;
   prevedere ablation utili senza imporre una soglia a ogni componente.
6. **Verificare correttezza.** Stessi input isolati contro oracle indipendente,
   code/shape e confini rilevanti; poi gate indipendente sul modello completo.
   Ripetere replay, logits completi/teacher-forced, stato e percorsi toccati
   secondo i criteri congelati. Un mismatch riporta al punto 3 prima di correggere.
7. **Misurare fuori dal profiler.** Microbenchmark comprensivo di conversioni,
   preparazione e sincronizzazioni; poi prefill/decode e richiesta completa.
   Confrontare con Bi e B0, riportando miglioramento locale e cumulativo.
8. **Decidere e aggiornare.** `KEEP`, `REJECT`, `NEED_MORE_DATA` o
   `PENDING_BUNDLE`; salvare anche i tentativi negativi. Aggiornare registro,
   backlog e prossimo hotspot prima di iniziare il kernel successivo.

Per le famiglie di kernel mantenere i quattro riesami S1–S4 della roadmap:
prima del codice, dopo la prima parity, durante il tuning e all'integrazione.
Un riferimento già acquisito può essere riletto al medesimo commit; va
comunque verificata e descritta la sua pertinenza al nuovo intervento.

## 4. Scheda delle fonti obbligatoria, per esperimento

La scheda precede il codice nel report dell'esperimento e contiene:

- Operazione/fase/shape/formato, differenza osservata e prova che la riproduce.
- Per ciascuna fonte: URL, repository, SHA completo, path e simboli letti;
  hash dei file acquisiti quando disponibili, data di consultazione e licenza
  se si riusa codice. Per specifiche Apple: URL e versione SDK/OS applicabile.
- Semantica in Transformers del modello effettivo; Diffusers anche per
  pipeline Qwen Image, scheduler, DiT e VAE. I nomi storici `qwen35`/`qwen4exp`
  non identificano da soli il file HF corretto: verificare metadata e copertura.
- Kernel **e wrapper/dispatch/capability/test** corrispondenti in Strata e
  llama.cpp/ggml; assenza di copertura esplicita quando manca. Un kernel simile
  non certifica il supporto del modello intero.
- Ricerca locale tramite [indice](../research/INDEX.md), architecture,
  performance, diagnostici, implementazioni CUDA/Metal e casi analoghi.
- Assunzioni trasferibili e incompatibili: layout, quant, accumulo, riduzioni,
  hardware, RAM unificata, working set e concorrenza. Motivare soluzione scelta
  e alternative escluse; distinguere confronto statico da confronto eseguito.

Strata e llama.cpp restano confronti obbligatori. MLX/MLX-LM, implementazioni
Apple e altri runtime/paper diventano confronti aggiuntivi quando affrontano
il problema misurato. Nessun trapianto automatico di soglie NVIDIA, memoria
pinned, eventi CUDA, repack lossy o quantizzazioni del KV/del target.

## 5. Baseline e strumenti della prima campagna Flash

Profilo iniziale qualificato: [Flash Metal streamed](../../config/flash-metal-streamed.env.example),
capacità **8192**, prefill **64**, cache pesi inizialmente **2 GiB**, MTP/vision/suite
disattivati e reasoning none. Le misure storiche mantengono quel cap esplicito.
I profili operativi Metal ora lasciano il cap vuoto: la cache cresce/riduce
al confine dei forward sulla RAM disponibile, riservando grafo, picco restore
e il margine host configurato: **0 byte** nel profilo del6novembre,
**2 GiB** nei controlli storici e come fallback senza override.
Un cap esplicito resta un tetto. Nessun riempimento inutile
oltre i pesi utilizzabili, nessuna variazione della numerica.

L'hardware di riferimento documentato è M5 Pro 24 GiB; rilevare nuovamente
device, capacità, macOS, SDK, compilatore, memoria disponibile e stato termico.
Pianificare il working set nella RAM fisica condivisa, riserva host
configurata, guard e monitoraggio pressione/swap. Una cache maggiore è una variabile
di esperimento esplicita con il proprio picco, non un confronto a budget nascosto.

| Caso | Sequenza iniziale | Metrica/controllo |
| --- | --- | --- |
| Decode | Prefisso 64/128, prima 8–16 output per direzione; ampliare a 32–64 se ammesso | TPS e ms/token, primo passo separato dallo steady state, token realmente emessi |
| Prefill | Prompt nuovi 128/512 token, poi 2k/4k e quasi 8k solo per ipotesi pertinenti | Nuovi token elaborati/s, tempo prefill, occupazione effettiva e budget output |
| Riuso | Prefisso condiviso + suffisso nuovo; prompt diverso e cambio dominio | TTFT, righe riusate/ricalcolate, byte letti; separare hit da prefill nuovo |
| Workflow | Richiesta testuale diretta, poi turno ExMekh rappresentativo e restore se toccato | Tempo utile della richiesta e breakdown, renderer e transcript esatti |
| Speculazione | Target-only come controllo; MTP con verifica esatta in esperimento successivo | Token accettati/s, costo draft/verify/rollback e memoria sottratta alla cache |

Separare processo nuovo, cache applicativa calda/fredda, page cache del sistema
e prefix/KV hit. Non svuotare globalmente la cache del Mac e non chiamare
«cold SSD» un processo nuovo senza prova dello stato dello storage.

Riutilizzare `ds4-bench` e il qualificatore Metal esistenti. Prima di eseguire,
controllare supporto effettivo a Flash/streaming, finestre di timing, restore,
politica EOS e corretta contabilizzazione dei token nuovi tra ripetizioni.
Il benchmark teacher-forced, il generatore che esclude EOS e una risposta
libera sono workload distinti; non confrontarne i TPS senza dichiararlo.

`DS4_METAL_ENCODER_TIMELINE`, `DS4_METAL_CB_TIMES` e
`DS4_BENCH_DECODE_TIMES` sono punti d'ingresso diagnostici già presenti.
Verificare quali hook raggiungono davvero `qwen4exp`; le statistiche esperti
del vecchio streaming DeepSeek/GLM non sono automaticamente quelle di Flash.
La timeline per encoder può aggregare più dispatch e non dà necessariamente
il tempo di ogni kernel. Usare gli strumenti Metal/contatori disponibili sul
device e aggiungere soltanto l'osservabilità mancante. Misure promozionali
senza trace, capture, logging per token o sincronizzazioni diagnostiche.

Direzione breve: due casi pertinenti e due campioni, senza valore promozionale.
Conferma: warm-up esplicito e almeno cinque coppie A/B alternate sulla stessa
macchina, con campioni e dispersione conservati. Riportare mediana, p10/p90,
media, deviazione standard, CV e intervallo delle differenze appaiate. CV >5%
indica instabilità; CV <5% da solo non dimostra un guadagno dell'1%.
Definire prima budget e piano dei campioni; eventuali estensioni sono prove
separate motivate, senza selezionare soltanto i run favorevoli.

Sessioni contenute, una prova grande per volta e stima preventiva del costo.
Restano i timeout iniziali della roadmap; eventuali estensioni esplicite si
registrano prima del run. Una misura interrotta resta `NOT_VERIFIED`.
I consumer live estesi sospesi non vengono rimessi in coda dal benchmark;
i gate numerici full-model pertinenti fanno invece parte di questa campagna.

## 6. Piccoli incrementi, pacchetti e dispatch per caso

Ogni incremento positivo credibile entra nel registro, anche sotto l'1–2%.
La soglia pratica del pacchetto/workload viene dichiarata prima della misura;
**non si applica un veto automatico del 2% a ogni kernel**. Valutare anche un
guadagno minore ma riproducibile rispetto a complessità e regressioni.

Una componente corretta con segnale troppo piccolo per una decisione isolata
può restare `PENDING_BUNDLE` in un esperimento reversibile. Non equivale a
promozione in produzione. Misurare il pacchetto completo contro la stessa B0,
oltre all'incremento contro Bi; non sommare percentuali individuali e non
presumere che fusioni/cache/overlap siano indipendenti. Per accettare il
pacchetto ripetere il gate indipendente sul modello completo. Se fallisce,
isolare le interazioni prima di mantenere una sua parte.

Prefill, decode, MTP e streamed/resident possono scegliere algoritmi diversi
con la stessa semantica, tramite dispatch per fase, shape, capacità del device
e budget. Misurare il costo di preparazione, cambio percorso e workspace.
Una candidate che favorisce prefill ma peggiora decode può essere confinata
al prefill e rimisurata. La stessa regola vale per contesti corti/lunghi e
resident/streamed. Preferire dispatch interno motivato e fallback corretto;
evitare varianti semantiche permanenti o flag di tuning a carico dell'utente.

Il comparatore attuale in `tools/perf_harness.py:compare_records` applica un
KEEP automatico da +2% e contiene assunzioni CUDA. Prima di usarlo come gate
Metal verificarne raccolta, backend e policy; un suo `NEED_MORE_DATA` sotto
soglia non giustifica scartare la componente. Qualsiasi adattamento deve
avere un esperimento proprio e preservare le qualifiche storiche CUDA.

## 7. Backlog Flash iniziale: ipotesi, non classifica misurata

| ID | Area da misurare | Direzioni da confrontare dopo il profilo | Ricerca di partenza e limiti |
| --- | --- | --- | --- |
| F-P01 | I/O e attese ai confini layer/router | Buffer riutilizzabili, letture bounded anticipate, overlap con calcolo già noto | `ds4_qwen4_metal_stream.inc`; `ds4_read_queue.h`, percorso CUDA e report async; fence/lifetime e RAM unificata da riesaminare |
| F-P02 | Byte riletti e residenza dei pesi | Riuso tra layer/token, cache esperti/densi, richieste contigue e layout byte-identico | Strata streaming, policy locale, report residency; misurare hit rate, SSD effettivo e picco; non riscrivere i pesi originali |
| F-P03 | Prefill MoE e proiezioni | GEMM multi-riga, raggruppamento esperti, tile/scratch, chunk compatibile col budget | Strata `dense_mm`/`moe_mm`, ggml `mul_mm` e wrapper, MLX; chunk maggiori avevano causato drift nel precedente studio CUDA |
| F-P04 | Decode: GEMV, MoE, head e piccoli dispatch | Kernel quantizzati/fusioni compatibili, threadgroup e riduzione del costo submit | Strata `gemv`/`moe`, ggml `mul_mv` e dispatch; inventario reale dei formati prima del port |
| F-P05 | N-gram/PLE, CPU e allocazioni | Deduplicazione byte-identica, riuso/pool bounded, prefetch di righe note | Grafo Flash, ricerche PLE locali e Strata; nessuna sostituzione BF16→Q8/FP8 del target |
| F-P06 | Attention, KV e altre fasi GPU | Specializzazioni prefill/decode, layout e fusioni solo se dominanti | Transformers del modello effettivo, Strata, ggml; copertura e soglie da verificare |
| F-P07 | Richiesta e workflow | Riuso prefisso/stato, eliminazione replay e copie ridondanti, checkpoint e MTP esatto | Report replay/async, lifecycle e server; misurare costo totale e memoria del drafter |

Prima attivare soltanto l'ID giustificato dalla baseline. Le ottimizzazioni
del workflow seguono gli stessi audit e gate dei kernel, inclusi isolamento
sessioni, cancellazione, checkpoint e stato ricorrente/n-gram quando coinvolti.

## 8. Ricognizione del 2 novembre e fonti di ingresso

Checkout iniziale pulito: `1544bb91eafaefefe32862ab408c412f0beed354`.
Lettura di roadmap, `AGENTS.md`, `AGENT.md`, indice tecnico, contratto
prestazionale e ricerche Flash precedenti. Nessuna build, inferenza o misura
di velocità eseguita per preparare questo piano; nessuno speedup rivendicato.

Osservazioni statiche nel checkout:

- `ds4_gpu_qwen4_stream_layer` sincronizza e libera le entry al cambio layer;
  `qwen4_stream_read` usa `pread` sincrone a parti fino a 8 MiB;
  `qwen4_stream_selection` sincronizza prima di leggere gli ID del router.
  Sono candidati da profilare, non una prova che dominino il tempo.
- La coda C bounded esistente separa payload/buffer del caller dal lavoro
  dei worker; il report CUDA async ha gate e prestazioni propri. Si può
  studiarne il contratto, senza trasferire eventi/H2D o speedup a Metal.
- La ricerca prefill precedente registra drift aumentando i chunk. Questo
  impedisce di trattare un semplice aumento del chunk come scelta già sicura.

Hash SHA-256 dei file letti per queste osservazioni:

| Path | SHA-256 |
| --- | --- |
| `ds4_qwen4_metal_stream.inc` | `d4a302e887db766a61c33e8890d7834db74096c74cf64a6a7c05a6454563a8ff` |
| `ds4_read_queue.h` | `160d32faf6f5fb17d52fa59a79ec6e607fc156268f24e21ed78af7465399feef` |
| `ds4_metal.m` | `152b7a9a5d15521f6020da22af60e75629fe293a55a5ae41905d829ebcd11b08` |
| `ds4_bench.c` | `a9cbf60f5e0649eb70d2c03893a4629a24eae6796837fd7923b0b268d9df955e` |
| `tools/perf_harness.py` | `6ee6e56cd037c71cbec220e37b8298461e4c7b62c48a0e89442228feefef3f27` |

Ricerche interne da riaprire secondo l'operazione:

- [Piano Flash/CUDA del 4 ottobre](qwen-flash-performance-action-plan-20261004.md),
  [offload prefill e manifest](../research/guides/flash-prefill-offload-20261004.md),
  [letture anticipate](../performance/qwen38-flash-prefill-async-20261004.md)
  e [replay prefill](../performance/qwen38-flash-prefill-replay-20261004.md).
- [Audit numerici Metal](../performance/metal-apple-silicon-20261031/t06-t07/upstream-numerics.md)
  e [manifest delle revisioni](../performance/metal-apple-silicon-20261031/t06-t07/upstream-audit/sources.json):
  Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  Diffusers `86ef420b17363c28e36ebad6b58555e9aa8feda2`,
  llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`.
  Sono pin storici con copertura specifica; l'audit Gemma/vision non è un
  audit del testo Flash. Strata iniziale:
  `46c152a1310f77cd2f0ce2c7cd213962df14fe83`.

Ricerca web iniziale, solo orientativa, 2 novembre:

- [Strata, guida Mac](https://github.com/Niko1221/Strata/blob/main/docs/MAC.md):
  spunti su cache, layout degli esperti e rapporto fra costo MTP e memoria.
  Pagina mobile: fissare SHA e leggere codice/test prima di adottare le idee.
  I default con conversioni Q8 non soddisfano automaticamente il contratto
  BF16 locale. I benchmark pubblicati non sono confronti col nostro Mac/GGUF.
- [MLX, backend Metal](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal):
  punti d'ingresso `matmul.cpp`, `quantized.cpp`, kernel e runtime; contenuti
  da fissare e analizzare nell'esperimento pertinente.
- [Apple, contatori GPU](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)
  e [capability Metal](https://developer.apple.com/metal/capabilities/): ingressi
  per la profilazione; verificare API disponibili sul device/SDK locale.

Il fetch web dei singoli file Strata/ggml ai pin storici ha restituito cache
miss: questa ricognizione non viene registrata come nuovo audit dei kernel.
Prima dell'implementazione acquisire/rileggere i sorgenti esatti, usando i
checkout/artefatti locali verificati oppure il repository primario.

## 9. Registro da aggiornare a ogni iterazione

Artefatti per esperimento in
`docs/performance/metal-inference-20261102/<modello>/<id>/`: report leggibile,
`experiment.json`, manifest fonti, comandi, campioni grezzi, gate numerici,
hash e puntatori alle trace grandi locali. Non versionare pesi o dump enormi.
Il JSON registra B0/Bi, componenti del pacchetto, fase, workload, hardware,
parametri, campioni, metriche/dispersione, delta incrementale/cumulativo,
risultati numerici, limiti e verdetto. Non marcare PASS un campo non misurato.

| ID | Modello | Attività | Fonti/gate | Delta locale / cumulativo | Stato |
| --- | --- | --- | --- | --- | --- |
| M-P00 | Flash | Ricognizione, vincoli e piano persistente | Roadmap, fonti locali e ingressi web §8; nessun run | Non misurato / non misurato | `DONE`, sola pianificazione |
| M-P01 | Flash | Inventario harness, risorse e baseline B0 | 2 GiB rifiutati; baseline separata 1 GiB, 8192/64, due sweep | B0 registrata; campioni direzionali | `DONE` |
| M-P02 | Flash | Profilo e classifica hotspot | Trace separata, 64 input/4 decode; span GPU 11–14% del wall | Nessuno speedup attribuito al profiler | `DONE` |
| M-P03 / F-P01 | Flash | Letture parallele degli esperti già selezionati | Audit S1–S4, primitive/graph e 9 logits full-vocabulary/stato bit-exact; MTP e profilo 2 GiB PASS | Prefill +89,2–98,0%, decode +15,8–18,1%, incremento e cumulativo coincidono; cinque coppie | `KEEP`, subset locale |
| M-P04 | Flash | Riesame finale e chiusura ciclo rapido | Trace finale, 512 token/logits esatti, limiti e backlog nel report | Tempo fuori GPU ancora dominante; zero tuning dei kernel ereditati | `DONE` |
| M-P05 | Flash | Riesame Clean Code e proprietà del lettore host | Coda/ASan/UBSan, 49 casi host, 10 gate Metal/full-model; logits/stati bit-exact | Controllo post-refactoring instabile: `NEED_MORE_DATA` per una nuova qualifica prestazionale; dati storici conservati | `DONE`, correttezza PASS |
| M-P06 | Flash | Conferma operativa dell'utente e attivazione | Utente conferma il miglioramento e riferisce carico concorrente durante le misure; build aggiornata e test host lettore/coda PASS | Nessuna nuova misura; campioni e verdetti precedenti conservati | `KEEP`, attivo per default nello streaming Metal |
| M-P07 / Q-P00 | Qwen27 | Attivazione e baseline B0 8192/64/cache 2 GiB | Due sweep seriali 64/128 + 8 teacher-forced PASS; logits conservati | Prefill 9,62–10,06 TPS; decode 0,32–0,34; direzionale | `DONE` |
| M-P08 / Q-P01 | Qwen27 | Profilo B0 separato | 64 + 4, GPU circa 62% prefill e 16–17% decode; proiezioni UD dominanti in GPU | Nessuno speedup attribuito alla trace | `DONE` |
| M-P09 / Q-P02 | Qwen27 | Tile UD prefill 4→8 righe | 270 casi e full-model bit-exact; micro misti, nuova trace GPU non migliorata; audit S1–S4 e rollback | +3,25% locale down, regressioni −4/−16% su altre forme; nessun guadagno promosso | `REJECT`, dispatch generalizzato |
| M-P10 / Q-P03 | Qwen27 | Letture dense grandi parallele nel reader esistente | Sampling 75,1% pread; audit S1–S4, host/sanitizer/grafi, nove logits e stato bit-exact; MTP e regressione Flash PASS | Prefill +9,46–12,23%; decode +27,01–34,51%; tempo richiesta sintetica −18,15–19,85%; locale=cumulativo, cinque coppie | `KEEP`, subset locale |
| M-P11 / Q-P03 S4 | Qwen27 | Integrazione e chiusura del primo ciclo | `make all`, test finali, hash GGUF e blob upstream verificati; baseline decode CV>5% esplicito, intervalli appaiati positivi | Nessuna nuova percentuale oltre M-P10; HTTP/ExMekh e quasi8k non misurati | `DONE`, default attivo |
| M-P12 / Q-P04 | Qwen27 | Riesame Clean Code del reader e dei consumer | Batch executor privato, fixture indipendenti dalle partizioni, manifest fonti completo; 49 host, sanitizer, fondamenta e Qwen27/MTP/Flash bit-exact PASS | Prefill mediano +0,23/−0,18%; decode variabile in entrambe le versioni: `NEED_MORE_DATA` per stretta non-regressione; misure Q-P03 conservate | `DONE`, correttezza PASS |

Checklist di ripresa: leggere questa tabella, il report dell'ultimo esperimento
e il prossimo hotspot; verificare che codice/binario corrispondano a Bi;
riaprire le fonti pertinenti e aggiornare il registro prima di proseguire.

| M-P13 | 2026-11-03 | Qwen27 Q-P05 gather UD per riga | `PENDING_BUNDLE`: gate isolati e128 bit-exact PASS, screening decode0.49/0.40 variabile; profilo separato0.50; nessun commit |
| M-P14 | 2026-11-03 | Qwen27 Q-P06 residenza bounded | `IN_PROGRESS`: audit specifico prima del codice, budget2GiB poi4GiB esplicito con guard invariati |
| M-P15 | 2026-11-03 | Qwen27 Q-P06 correttezza e screening | `PENDING_BUNDLE`:128 bit-exact anche dopo reload; meno byte ma decode0.44–0.47 a2/4GiB, profilo variabile; nessun grande guadagno attribuito alla sola cache |
| M-P16 | 2026-11-03 | Qwen27 Q-P07 kernel UD decode1 | `IN_PROGRESS`: riscrittura project specializzata per righe, audit e snapshot shader prima del codice, isolati/micro/gate completi in corso |
| M-P17 | 2026-11-03 | Qwen27 Q-P07 decode1/decode16 e R8 | `PENDING_BUNDLE`:decode1 bit-exact, decode16 PASS entro soglie graph con token esatti; micro misti, full breve0.48/0.51; R8 studiato su vera attivazione senza attivare approssimazioni |
| M-P18 | 2026-11-03 | Qwen27 Q-P08 submit prima dei read | `IN_PROGRESS`:audit nuovo, isolati/lifecycle e128 bit-exact vsQ-P07b PASS; misure2/4/5GiB e profilo in corso |

**Prosecuzione richiesta dall'utente:** continuare dopo il primo1TPS fino
al plateau. Consolidare i risultati con confronti appaiati, riprofilare il
percorso migliore e provare le direzioni successive motivate dai residui.
Un plateau è un esito misurato di tentativi corretti senza ulteriori guadagni
riproducibili sulla richiesta, con incertezza e limiti dichiarati; non è
un tetto teorico di Metal né un veto ai piccoli incrementi.
| M-P19 | 2026-11-03 | Qwen27 Q-P09 MTP4 | `IN_PROGRESS`:screening1.64TPS a2GiB,16/16 ID target esatti;MTP2 0.88; full depth4 e conferma appaiata necessari; admission corrente variabile |
| M-P20 | 2026-11-03 | Qwen27 Q-P10 IQ4 constant | `IN_PROGRESS`:shader candidato separato,297/270 e full128 bit-exact; micro misti, misura4GiB rifiutata per RAM; nessuna modifica produzione promossa |
| M-P21 | 2026-11-03 | Qwen27 Q-P11 verify5 | `PLANNED`:audit specifico tile5–8, kernel FP32 e controllo contro tile4 dopo profileMTP4 |
| M-P22 | 2026-11-03 | Discrepanza MTP4 / Q-P07b | `REJECT` decode16: stessi128 input,32lane PASS/16lane FAIL oltre soglia; nessun criterio allargato;1.64TPS resta provvisorio fino alla correzione |
| M-P23 | 2026-11-03 | Qwen27 Q-P12 output pair/SIMD32 | `IN_PROGRESS`:audit nuovo, ripartizione unità/FMA/riduzione originali,297 isolati e MTP4 sul caso critico prima di promozione |

| M-P24 | 2026-11-03 | Qwen27 Q-P11/P12 corretti | `LOCAL_NUMERIC_PASS`:297 isolati bit-exact e full128/MTP4 a8192 PASS; decode breve~1.9TPS a1536MiB, ancora da consolidare |
| M-P25 | 2026-11-03 | Qwen27 Q-P13 riciclo al cap | `PENDING_BUNDLE`:grafi/EOF/retry, lifecycle e11 vettori full ordinari/spec bit-exact; profiling separato~1.99TPS |
| M-P26 | 2026-11-03 | Qwen27 Q-P14 soglia8 | `REJECT`:micro10 coppie peggiora8/16/24MiB; reader32 conservato |
| M-P27 | 2026-11-03 | Audit DS4 originale Q-P15 | `DONE`:upstream890898c, residenza/letture ordinate/PLE/sweep; applicabilità separata a27B, Flash e Image; nessun merge o speedup dedotto |
| M-P28 | 2026-11-03 | Qwen27 Q-P16 soglia64/first touch | `REJECT`:32/48MiB peggiorano anche su nuove mmap; byte esatti, reader32 conservato |
| M-P29 | 2026-11-03 | Qwen27 Q-P17 quota lazy retained | `NUMERIC_PASS`:riprodotto head tardivo fuori spazio nel piccolo test; prenotazione distinta da live, EOF/retry e11 vettori full8192/MTP4 bit-exact |
| M-P30 | 2026-11-03 | Consolidamento plateau | `IN_PROGRESS`:B0 MTP4 allo stesso cap incluso; prima matrice interrotta da rifiuto admission, fallimenti conservati e nessun guard ridotto |

| M-P31 | 2026-11-03 | Qwen27 Q-P18 retained read recovery | `KEEP`:EOF consecutivi con batch attivo riprodotti, storage non-ready resta owned, retry in-place; full8192/MTP4 bit-exact |
| M-P32 | 2026-11-03 | Qwen27 Q-P10 costante IQ4 consolidato | `REJECT`:297/full gate esatti, ma5 coppie prefill−1.55% e nessun guadagno request; outlier conservati, nessun veto2% |
| M-P33 | 2026-11-03 | Qwen27 Q-P19 costo embedding/minima riserva | `REJECT`:stessi11 vettori e−2.14% byte logici, ma prefill−4.85% e request non migliorata; ripristinataP18, API sperimentale rimossa |
| M-P34 | 2026-11-03 | Qwen27 plateau bundleP18 | `LOCAL_PERF_PASS`:warmup+5 blocchi appaiati, MTP4 su entrambe; mediana decode1.976503/prefill13.767850, request12.744541s; delta geometrici+27.00/+27.57/−21.38%; tutti16 ID e full logits esatti |

| M-P35 | 2026-11-03 | Chiusura locale e regressione | `DONE`:make all/foundations,49 host, ASan/UBSan, feature residente/fallback, Qwen target/MTP2/MTP4 e Flash completo PASS;9 ordinary/logits e payload frozen esatti; nessun commit |
| M-P36 | 2026-11-03 | Controllo CLI finale | `SCREENING_PASS`:cache2GiB,8192/64; target-only0.55TPS/prefill14.06, MTP4 1.97/prefill13.31; primo prefill CLI escluso per sleep944s, ripetizione e clock verificati, nessun timer/kernel modificato |

Chiusura della continuazione: [risultati, gate e plateau locale Qwen27](../performance/metal-inference-20261102/qwen27/continuation-20261103/README.md).
Il target1TPS è superato nel caso MTP4 misurato, non nel target-only.
Le direzioni DS4 per Flash/Image sono nell’auditQ-P15 e restano da misurare
sui loro owner e pipeline. Non sono state aperte campagne concorrenti.

| M-P37 | 2026-11-03 | Riesame kernel/streaming e integrazione autorizzata | `DONE`:corretto batch UD con ID assenti, separati orchestrazione gather/ready/transienti;49 host, sanitizer e Qwen/MTP/Flash bit-exact;5 coppie compatibili col rumore, micro residente mirato senza rallentamento confermato; commit distinti |

### Continuazione Qwen27: cache adattiva e pack per layer

| ID | Data | Attività | Esito |
| --- | --- | --- | --- |
| M-P38 | 2026-11-03 | Qwen27 Q-P20 budget adattivo | `IN_PROGRESS`:audit specifico prima del codice; profilo1536MiB rifiutato a host4.46GiB, nessun guard abbassato; usare il residuo misurato dopo costi fissi e riserva2GiB |
| M-P39 | 2026-11-03 | Qwen27 Q-P20 correctness / Q-P21 pack | `IN_PROGRESS`:automatic2.32GiB full128 bit-exact e fondamenta PASS; profilo B0 ammesso14.472991/0.548800TPS; nuovo audit pack/ring e fork9d8c1a8 prima del codice |
| M-P40 | 2026-11-03 | Qwen27 ring isolato e costo restore | `NUMERIC_PASS` target128 e4 fixture;autoMTP espone budget insufficiente per grafo candidato nel restore atomico, nuova indagine e correzione pianificata;Flash confronto iniziale non comparabile35/64input |
| M-P41 | 2026-11-03 | Qwen27 budget elastico Q-P23 | `IN_PROGRESS`:utente richiede adattamento anche durante l’uso;screening ring0.90/3.22TPS a cap fisso e target auto1.14;nuovo audit per crescita/riduzione, buffer comuni conservati e2GiB di margine |
| M-P42 | 2026-11-03 | Qwen27 telemetria dinamica | `IN_PROGRESS`:resize/grafi isolati PASS;MTP2/Flash bit-exact, target/MTP4 resume esposto a snapshot host cached;diagnostica prova release reale, audit XNUf6217f8 prima di correggere il sampler |
| M-P43 | 2026-11-03 | Qwen27 cache elastica e sampler | `NUMERIC_PASS`:resize in batch/fallback/zero/EOF e4full gate bit-exact;cache cresce/riduce durante l’uso;monitor esterno minimo2.33GiB e swap invariato,regressioni e coppie finali in corso |
| M-P44 | 2026-11-03 | Qwen27 ring+RAM elastica consolidati | `LOCAL_PERF_PASS` MTP4:5coppie2.034→3.282TPS,request−28.59%;target0.603→0.916 ma CV23%;3richieste auto target1.57–1.72/MTP4 4.11–4.39TPS,logits esatti,margine>2GiB |
| M-P45 | 2026-11-03 | Qwen27 Q-P24 worker per pack | `IN_PROGRESS`:micro2/4/8 su pack veri,8riduce25–32% su dati ripetuti;candidate8 soloQwen35,Flash4,gate/coppie full prima di decidere |
| M-P46 | 2026-11-03 | Q-P24 rigettato / Q-P25 release | `REJECT`8worker:full bit-exact maMTP4decode−3.62%;ripristinato4. Nuovi rifiuti resume automatico riaprono audit su memoria restituita al driver;discard solo regioni eliminate dopo fence,guard invariati |
| M-P47 | 2026-11-03 | Qwen27 chiusura Q-P25 | `KEEP`:cache elastica+ring4worker,discard regioni morte dopo fence;fondamenta,4full gate e2ripetizioni normaliMTP4 bit-exact,feature PASS. Ultime3richieste64+16:target1.49–1.61TPS/prefill16.96–17.12;MTP4 4.09–4.31/prefill12.96–15.83. Margine monitorato>2GiB,nessun commit;limite Numeric sin/cos preesistente documentato |
| M-P48 | 2026-11-03 | Riesame e integrazione della cache adattiva | `DONE`:open lazy a cap zero, budget prima dell’uso, factory valida prima degli effetti, layout privato con trasferimento comune. 50host+fondamenta/sanitizer,44artefatti Qwen/MTP/Flash bit-exact;5coppie resident/stream senza regressione confermata;2assert Numeric preesistenti documentati;commit per scopo |

### Qwen Image 2.1 Turbo

| ID | Data | Attività | Esito |
| --- | --- | --- | --- |
| M-P49 / I-P01 | 2026-11-03 | Profilo BF16 e proiezioni DiT | `IN_PROGRESS`. La baseline campionata attribuisce 469/655 campioni del DiT alle proiezioni. Audit specifico Diffusers/Transformers/ATen/Strata/ggml e verifica YASAKANI prima del codice. [Esperimento](../performance/metal-inference-20261102/qwen-image/ordered-bf16-20261103/README.md). |
| M-P50 / I-P01 | 2026-11-03 | Tile SIMD32 e gate immagine | `NUMERIC_PASS`. Otto forme isolate esatte contro CPU e tre pipeline indipendenti testo/RGB/RGBA PASS. Tutti i 35 artefatti nativi coincidono con la baseline. Warmup operazione204.708→125.629s, conferma appaiata e regressione BF16 condivisa in corso. |
| M-P51 | 2026-11-03 | Riesame del dispatch BF16 | `IN_PROGRESS`. Tile reso opzionale nel costruttore e selezione spostata in DS4MetalNumeric, con fallback scalare. Audit capability specifico prima del codice, build prodotto PASS, replay forzato e tre gate sul binario riesaminato in coda. |
| M-P52 / I-P01 | 2026-11-03 | Conferma e profilo successivo | `KEEP` locale. Warmup+5 coppie, 110 artefatti bit-exact, DiT138.127→61.629s e operazione207.581→129.852s. Delta appaiato operazione−36.90%[−38.46,−35.29], margine minimo4.801GiB e swap invariato. Profilo dopo il guadagno su DiT e VAE conservato. |
| M-P53 | 2026-11-03 | Integrazione del riesame I-P01 | `DONE`. Otto forme per quattro varianti, fallback di capability e 35 artefatti Image finali bit-exact; Gemma480/480 PASS, build e test host PASS. Numeric43 con due assert sin/cos preesistenti riprodotti sulla baseline. Due coppie dirette senza rallentamento riprodotto; automatico129.570s con riserva rispettata. Commit separati per scopo. |
| M-P54 | 2026-11-03 | Secondo riesame I-P01, viste e fixture | `DONE`. API Numeric.slice tramite metodo C pubblico, test di ownership e limiti, tile con offset/capacità residua/sentinelle e alias attivo. CPU45 PASS (12 skip), Metal45 con41 pass/3 skip e il solo caso sin/cos preesistente. Lifecycle PASS. Sorgenti ed eseguibili di inferenza identici a M-P53, nessuna nuova misura prestazionale. |

| M-P55 / I-P02 | 2026-11-04 | VAE Conv BF16, profilo e audit | `IN_PROGRESS`. Profilo415/418 stack Conv, circa54s VAE. Audit specifico Diffusers/ATen/Transformers/Strata/ggml prima del codice, direct2D ordinato senza im2col; YASAKANI ancora senza motore pubblico verificato. [Esperimento](../performance/metal-inference-20261102/qwen-image/ordered-conv-20261104/README.md). |

| M-P56 / I-P02 | 2026-11-04 | Conv2D ordinata, replay | `NUMERIC_PASS` isolato. Otto forme bit-exact contro CPU e baseline, regressioni batch/gruppi/sottoviste PASS. Screening1152residual318.315→59.248ms; gate completo e verifica del costo delle forme minuscole in corso. |

| M-P57 / I-P03 | 2026-11-04 | Trasporto shared dei pesi Image | `IN_PROGRESS`. Audit separato di Strata stream/llama-mmap/DS4 e Numeric prima del codice. Staging già singolo, prova attribuibile di copia parallela bounded senza buffer extra; I-P02 conferma ancora sul binario congelato. |

| M-P58 / I-P04 | 2026-11-04 | Cache Image a budget continuo | `IN_PROGRESS`. Audit budget/semantica/planner prima del codice; espansione della sola variante ammessa con stessi picchi, restore, vittime e riserva. Fonti Strata rilette da URL al pin e hash verificato; rinnovo al confine di ogni job. |

| M-P59 / I-P02 | 2026-11-04 | Conferma Conv2D ordinata | `KEEP` locale. Warmup+5coppie,110artefatti bit-exact. VAE54.666→10.511s (−80.768%), operazione129.566→84.811s (−34.918%, IC95−38.565/−31.055). Encoder+3.144% riportato e riauditato, percorso Conv non eseguito nel testo; nessuna correzione speculativa. |

| M-P60 / I-P03 | 2026-11-04 | Copia parallela, micro/full | `REJECT` candidato generalizzato: byte/guard/overlap esatti e micro128MiB2.117→1.188ms, ma encoder18.338/18.758s contro14.397s I-P02 e nessun guadagno della richiesta. Nuovo audit working set/Strata/ggml/Q-P24 prima del ritiro; ablation seriale in coda. |

| M-P61 / I-P04 | 2026-11-04 | Qualifica finale e admission | `NUMERIC_PASS`: wrapper Conv con pool ripristinato, Image testo/RGB/RGBA e Gemma480/480 bit-exact, CPU46 PASS e soli2assert sin/cos Metal preesistenti. Warmup cap10GiB rifiutato prima dell’inferenza; conservato, nuova conferma a cap comune8.5GiB senza abbassare riserva/criteri. |

| M-P62 / I-P04 | 2026-11-04 | Cache confermata / nuovo limite automatico | Cinque coppie8.5GiB: richiesta97.218→85.917s,−9.471%[−15.631,−2.860],110artefatti esatti e swap invariato. Automatico finale interrotto per+367.914MBswap con4.34GB riportati disponibili: nuova indagine XNU/Strata/ggml prima di correggere il conteggio del file cache protetto nel solo Numeric Metal. |

| M-P63 / I-P04 | 2026-11-04 | Conteggio file cache protetto | `NUMERIC_PASS` automatico: policy floor-aware solo Numeric, margine2GiB invariato, contatori XNU verificati4byte;3richieste109.342–117.455s, quote RAM diverse, output esatti e crescita swap0. Nuovi gate condivisi e conferma del bundle con stesso sampler nei due controlli in corso. |

| M-P64 / I-P02+I-P04 | 2026-11-04 | Chiusura del bundle finale | `KEEP` locale. Stesso sampler floor-aware nei due controlli, warmup+5coppie senza profiler: VAE53.960→10.406s (−80.698%), richiesta156.165→115.846s (−25.214%, IC95−26.848/−23.665).110artefatti esatti, margine minimo4.398GiB, swap growth0. DiT+4.835% con quote RAM aggiornate più basse, riportato separatamente. Tre automatici, RGB/RGBA, Gemma480/480 e test finali PASS salvo due assert sin/cos preesistenti. Nessun commit. |

| M-P65 | 2026-11-04 | Riesame finale della cache | `DONE` locale. Byte letti dal campo memoria anziché dal costo equivalente; build/test mirati e nuovo gate automatico115.458s PASS,11artefatti identici e swap growth0. Manifest distingue binari misurati e prodotto riesaminato; nessuna nuova misura appaiata e nessun commit. |

| M-P66 | 2026-11-04 | Profilo del prodotto finale | `DONE` locale. Finestra DiT8s:427/666campioni principali in memmove e217nelle proiezioni/attesaGPU. Gate indipendente119.381s PASS,11artefatti identici e swap growth0; richiesta profilata esclusa dalle cinque coppie. Trasporto residuo documentato, nessun nuovo kernel o commit. |

| M-P67 | 2026-11-04 | Riesame per integrazione | Geometria Conv CPU/Metal condivisa a 64 bit e preparazione dell'ABI nel backend. Una mappa fase/componente e fallback del planner ripetibile. Build, regressioni isolate, ASan/UBSan, fondazioni Metal, Flash SSD, bundle e HTTP locale PASS. Due assert sin/cos già presenti. Gate completi del nuovo binario `NOT_VERIFIED` per memoria sotto la riserva anche sul controllo. Commit locali in attesa dell'autorizzazione richiesta dal controllo automatico. |

| M-P68 / I-P05 | 2026-11-05 | Ripresa e prefetch per gruppo | `IN_PROGRESS`. Nuovo worktree, stessa baseline funzionale. Gate testo completo110.644s PASS con swap growth0. Profilo DiT con434campioni di copia su716. Audit specifico Diffusers/Transformers/Strata/ggml e hint locali prima del candidato bounded16MiB. Continuazione senza commit. |

| M-P69 / I-P05 | 2026-11-05 | Hint selettivo e gate | `NUMERIC_PASS` testo. Intervalli/chunk/limite16MiB e failure I/O isolati PASS, bundle12/12. Candidato automatico90.085s e11artefatti identici, swap growth0. Quota RAM diversa dal controllo, quindi nessuno speedup attribuito. Screening al medesimo ceiling2GiB in corso. |

| M-P70 / I-P05 | 2026-11-05 | Conferma appaiata | `LOCAL_PERF_PASS`, gate finali in corso. Warmup separato e5coppie AB/BA al ceiling2GiB,110artefatti esatti, stessi byte/caricamenti. DiT−11.601%, richiesta−8.838%[−16.093,−1.314], CV8–13%. Ultimo candidato+512KiB swap globale al primo secondo con processo7MB e host10.16GB. Audit XNU specifico, monitor256MiB e riserva2GiB conservati; assenza assoluta di delta globale ancora distinta. |

| M-P71 / I-P05 | 2026-11-05 | Qualifica finale e forma512 | `KEEP` locale. Eseguibile finale identico al candidato misurato. RGB/RGBA indipendenti PASS,24artefatti esatti, swap growth0. Gemma480/480 e vettori PASS, CPU47 e test host/Metal residenti e SSD PASS salvo due assert sin/cos preesistenti. Nativo512baseline236.926s/candidato228.142s,11artefatti identici, picchi ammessi e swap growth0. Oracle512aperto. Nessun commit. |

| M-P72 | 2026-11-05 | Riesame e integrazione per responsabilità | `DONE`. Limiti advisory condivisi, callback CPU nominati, costi espliciti e test del budget pubblico. Build, CPU47, fondazioni Metal/SSD, ASan/UBSan, bundle/HTTP e contratti PASS. Gemma480/480 e vettori PASS. Tre gate Image indipendenti con35artefatti esatti e swap growth0. Due assert sin/cos preesistenti riprodotti. Commit per convoluzioni, memoria/planner, prefetch e documentazione. |

| M-P73 / T08 | 2026-11-06 | Ripresa F08 Flash | `NOT_VERIFIED`. Workflow congelato8192/64/cache2GiB: file creato23byte, poi tre aperture della skill; interrotto ordinatamente820.590s, server exit0, nessun kill forzato. Minimo host5.556GiB, crescita swap3795845B. La nuova richiesta revoca la sospensione per questa campagna; nessun commit. |

| M-P74 / T08 | 2026-11-06 | Replay freddo della prima ripresa | `DIAGNOSTIC_PASS`, non F08. Stesso prompt renderizzato1422token, risposta corretta6token in90.015s senza checkpoint; crescita swap0. Differenza da indagare fra ripresa e segmentazione. Fonti e prove nel [report T08](../performance/metal-apple-silicon-20261031/t08-20261106/README.md). |

| M-P75 / T08 | 2026-11-06 | Isolamento API della frontiera | `RESTORE_EXACT`. Prefill1234+13decode, payload completo, forward interno2662token e restore:248320logits bit-exact rispetto alla continuazione diretta. Prefill freddo1422distinto: massimo delta1.37691212, stesso argmax10191. Nessun cambio aritmetico. Primo estrattore con LF rimossi interrotto; confronto valido ripetuto sui byte esatti. |

| M-P76 / T08 | 2026-11-06 | Confine del prompt di sistema | `RESTORE_EXACT`, F08 aperto. Frontiera1151+83prefill+13decode:248320logits esatti dopo restore/forward interno. Freddo con lo stesso anchor: massimo delta1.89239585 e argmax248058(`<tool_call>`) contro10191. Differenza di segmentazione riprodotta; nessun kernel cambiato o risposta forzata. Controller esteso con cache automatica esplicita e timeout per50cicli, guard invariati. Planning Image separato dai noise numerici congelati assenti. |

| M-P77 / T08 | 2026-11-06 | Stress Flash automatico | `NOT_VERIFIED`.Avviato per50cicli dopo warmup,256×256e riserva2GiB; conserva27cicli PASS più warmup, poi APIConnectionError e nessun report finale del controller. Processo assente alla ripresa, causa non determinata. La successiva prova senza riserva è distinta. |

| M-P78 / T08 | 2026-11-06 | Ripresa senza riserva fissa | `IN_PROGRESS`.Vecchio stress senza report finale:27cicli+warmup nel JSONL, poi connessione interrotta; processo assente alla ripresa, causa non determinata. Nuovo override comune engine/suite/monitor e profili streamed con riserva0. Policy/factory,49test Python e lifecycle Metal PASS; Flash32input+8output con payload/logits/restore esatti PASS11.160s e swap growth0. Nuovo stress50cicli avviato separatamente, nessun commit. |

| M-P79 / T08 | 2026-11-06 | Chiusura stress Flash senza riserva | `PASS`50cicli misurati+warmup,256px e dimensioni embedding128/256/512/768.10394.077s, server exit0 e nessun cleanup forzato. Minimo host3952607232B, picco footprint12143032776B, pressione massima2, crescita swap globale252832645B(<256MiB). Mediane Image92.449s/embedding57.748s/ciclo202.846s, CV3.92/1.53/2.03%. Nessuno speedup attribuito al solo margine: condizioni RAM e profilo distinti. F08 resta aperto. |

| M-P80 / T08 | 2026-11-06 | Gate Qwen27 senza riserva | `PASS`8.306s,32input+8output, payload/checkpoint/restore/logits esatti. Cache utile13.71GB e minimo host2019459072B: il margine2GiB non viene sottratto. Crescita swap0. F08 con cache automatica e riserva0avviato separatamente. |

| M-P81 / T08 | 2026-11-06 | F08 Qwen27 senza riserva | `NOT_VERIFIED`625.085s. Creazione reale poi tre aperture della skill dopo il risultato, interrotto con server exit0 e nessun cleanup forzato. Minimo host1457209344B e crescita swap0. La causa richiede un isolamento Qwen distinto da Flash. Stress50cicli Qwen avviato in un nuovo processo con riserva0. |

| M-P82 / T08 | 2026-11-06 | Limite risorse stress Qwen27 | `SKIP_RESOURCE_LIMIT`44.739s, durante il warmup prima del primo ciclo completo. Crescita swap globale1809254973B(1725.44MiB), soglia256MiB superata in due campioni. Riserva0, pressione massima2, server exit0 e nessun cleanup forzato. La causa specifica del picco non è attribuita a un kernel; servono più RAM/minor carico o un'indagine del picco di transizione. Gate aperto nella lista residua. |

| M-P83 / T08 | 2026-11-06 | Ammissione corpus Image senza riserva | `ADMISSION_PASS`56/56forme,8diagnostiche+48T00, in12.210s. Fixture marcate noise_zero solo per planning; geometrie1024/1536e reference resolution originali. Nessuna qualifica numerica dedotta. Noise T00 e baseline Gemma originali ancora mancanti. |

| M-P84 / T08 | 2026-11-06 | Oracle indipendente512 | `NUMERIC_PASS`profilo CPUencoder/MPSDiT/MPSVAE offloaded, fixture alpha512originale e soglie congelate. Conditioning bit-exact, max latent RMSE0.010359/relL20.007218, RGBA RMSE0.005003/SSIM0.997166/alphaMAE0.000439. Nativo224.864s,11artefatti identici al controllo512; monitor236.870s. Oracle encoder57.810s/DiT34.392s/VAE6.078s, crescita swap0in tutti. Dispositivi dichiarati separatamente; nessuna qualifica CPU pura o corpusT00dedotta. |

| M-P85 / T08 | 2026-11-06 | Regressioni finali senza riserva | `PASS`Gemma480/480frontiere esatte e vettori128entro i gate indipendenti,12.200s, crescita swap0.56test Python in13.212s, policy/factory/planner/resources/executor/HTTP/lifecycle e Ruff PASS. Nessun shader/peso/soglia cambiato. Lista residua aggiornata per F08, stress Qwen, asset, qualità e hardware; T08resta aperto, nessun commit. |

| M-P86 / T08 | 2026-11-06 | Riesame e integrazione per responsabilità | `PASS` nel perimetro del riesame: configurazione separata dal calcolo puro, policy immutabile dei qualificatori, I/O del probe verificato. 86 test Python, build CPU/Metal, fondazioni, ASan/UBSan e Ruff PASS. Flash/Qwen27 conservano 20 artefatti esatti; Image512 indipendente PASS con 11 artefatti identici, Gemma480/480 PASS. Probe di ripresa identico al controllo; Quick Chat reale PASS con riserva0, server exit0 e cleanup ordinato. Commit separati autorizzati dalla nuova richiesta. Nessuno speedup attribuito; T08 e i debiti numerici restano aperti. [Riesame](../architecture/metal-t08-clean-code-review-20261106.md). |

| M-P87 / T08 | 2026-11-06 | Indagine swap Qwen27 | `IN_PROGRESS`. Primo ciclo identico riproduce SKIP_RESOURCE_LIMIT47.697s/+1571228222Bswap. Profilo: upload mmap→Metal dell'encoder Image con cache Qwen ancora viva. Audit specifico Transformers/Diffusers/Strata/ggml/XNU prima del codice; piano e credito di reclaim in isolamento. Riserva0 e soglia256MiB conservate. [Report](../performance/metal-inference-20261102/qwen27/swap-transition-20261106/README.md). |

| M-P88 / T08 | 2026-11-06 | Isolamento costo upload e floor | `NEED_MORE_DATA`: A(costo sorgente) e A+B(floor anche SSD) non risolvono, swap+1122752267/+1416951235B. Piano A mantiene vittime0 riducendo solo la cache Image; le pagine mmap dei gruppi copiati rimangono attive. Nuovo audit del trasporto prima di C: lettura bounded F_NOCACHE direttamente nello storage shared, identità/fence/pubblicazione invariati. A ritirato, nessuna soglia modificata. |

| M-P89 / T08 | 2026-11-06 | Trasporto, contatori e rilascio GPU | C(bytes esatti, trasporto diretto) e D(limite inactive anonimi) non chiudono il picco; ritirati. Nuovo isolato E riproduce256–265MiB fisici trattenuti dopo free di un buffer usato dalla GPU, pur con Metal allocated720896B. Audit specifico di free/fence e Q-P25 prima del codice: Empty solo per storage Numeric morto, consumer completati. Nessuna nuova riserva o modifica dei gate. |

| M-P90 / T08 | 2026-11-06 | Picchi di fase e capacità dopo reclaim | E/F ritirati; isolato conferma restituzione fisica differita dopo il consumer. H conta sorgente+ritiro nel budget shared Metal e ammette l'espulsione pesi. Nuova discrepanza: allocato16.32→1.26GB ma global_free resta1.71GB, riserva successiva NOMEM. Audit specifico prima di I: refresh nel proprietario dopo il rilascio, senza credito sintetico e senza modificare CUDA o i gate. |


| M-P91 / T08 | 2026-11-06 | Reclaim e preload encoder | J/K e C+I+J+K non chiudono il gate; ritirati J/K. L svuota solo la cache opzionale prima del piano, preserva sessione/KV e ceiling dinamico: rilascio fisico verificato ma ciclo ancora SKIP46.8s. Nuova discrepanza: floor XNU128831pagine resta stale fino al pageout, poi458930; non è attesa del solo driver. |

| M-P92 / T08 | 2026-11-06 | Capacità corrente dopo cache wired | Audit XNU vm_page/vm_resident/vm_pageout/kern_descrip/vfs_cluster al pin f6217f8: speculative incluse nei pageable; floor ricalcolato solo durante scan. Candidato M deduplica speculative e usa per Numeric max(floor misurato,floor corrente10/27), senza riserva fissa. Aritmetica isolata e gate di transizione da verificare; soglie e numerica congelate. |


| M-P93 / T08 | 2026-11-06 | Trasporto realmente bounded | M non basta (SKIP44.4s). Diagnostico separato ammette14.61GB/host15.06GB ma il preload accumula sorgenti file. Isolato1GiB: lettura allineata NOCACHE senza crescita file, non allineata aggiunge~896MiB speculative. Audit XNU cluster_read_direct prima di N: staging allineato8MiB registrato in HOST e range/EOF verificati. Test BF16/FP32 byte-exact PASS; primo ciclo supera encoder Image e riprende Qwen, ancora in corso. |


| M-P94 / T08 | 2026-11-06 | Primo ciclo Qwen senza swap | C+I+L+M+N `PASS`160.186s controller/146.349s ciclo. Image57.450s, embedding58.127s, ritorni Qwen e codice ricordato corretti. Crescita swap0, pressione max2, minimo host2573680640B con riserva0; exit0/no cleanup forzato. Il trasporto allineato evita il picco dei range non allineati. Non promuove ancora stress50 o speedup; fondazioni Metal/lifecycle/HTTP PASS, gate indipendenti in corso. |


| M-P95 / T08 | 2026-11-06 | Gate del fix swap prima dello stress | `PASS`: Flash/Qwen27 20artefatti esatti anche con trim+ricrescita, Image51211artefatti esatti e gate indipendente PASS, Gemma491file esatti/480frontiere e vettori128 PASS. Fondazioni Metal/SSD/lifecycle/HTTP,86Python,48NumericCPU(14skip),12bundle,ASan/UBSan,Ruff,CPUbuild PASS. Sorgenti/binari congelati; stress Qwen50+warmup avviato senza profiler, riserva0 e guard originali. |


| M-P96 / T08 | 2026-11-06 | Audit anticipato dello stato idle | Prima prova lunga fermata dopo21PASS+warmup,2906.5s,swap0,exit0/no cleanup forzato. Il controllo finale esatto trova VAE cached675480832B solo nelle varianti residenti1/3, contro0 nella4: cache inattiva legittima, nessuna prenotazione/leak osservata. Gate invariato; audit O prima del codice per handoff simmetrico tra LLM e adapter, release_weights Image con metadata conservati. Il21non è promosso a50. |


| M-P97 / T08 | 2026-11-09 | Gate del handoff simmetrico O | Riletti i gate eseguiti il6novembre: Flash8.2s/Qwen8.3s,20artefatti esatti. Due Image512415.7s,11frontiere congelate e11reload esatte; nuovo confronto indipendente PASS. Gemma491file esatti,480frontiere indipendenti e vettori128 PASS. Stress50 successivo ancora aperto. |

| M-P98 / T08 | 2026-11-09 | Checkpoint durevoli della qualifica | Implementato journal fsync per ogni PASS, manifest di binario/GGUF/asset/configurazione/harness, lock e verifica PNG/vettori; `--resume` conserva i log di ogni tentativo. Coda troncata archiviata/riparata, avvio interrotto prima del manifest recuperabile, errori di gate non ignorabili. PASS_SEGMENTED distingue cicli funzionali cumulati dal lifetime continuo.95test Python PASS(1skip), prova reale di ripresa in corso. |

| M-P99 / T08 | 2026-11-09 | Nuovo limite Qwen e sampler SSD P | Prima prova checkpoint FAIL48.365s prima di Image: cache6.04→12.33GiB, decode0token/rifiuto221759488B, pressione4,swap+789976186B,exit0/no cleanup forzato,zero cicli validi. Riaudit separato HF/Strata/ggml/XNU prima di P: anche i buffer anonimi SSD rispettano floor misurato/corrente, riserva0 e nessuna soglia nuova; residente mmap invariato. Policy/lifecycle PASS; Flash9.2s/Qwen15.3s,20artefatti esatti. Primo ciclo reale con P completato e salvato, arresto SIGTERM ordinato177.470s; ripresa dal secondo ciclo in corso. Nessuno speedup dedotto da questi tempi. |


| M-P100 / T08 | 2026-11-09 | Ripresa reale con progresso conservato | `PASS` bounded3: SIGTERM dopo ciclo0, controller NOT_VERIFIED177.470s/client INTERRUPTED1/3; `--resume` completa solo1e2, PASS309.631s, totale3/3. PNG0SHA256+mtime invariati; owner500esatto(device0,HOST41838725B,roots10),zero prenotazioni. Pressione max1,swap globale+62915/+398459B,non zero ma sotto256MiB.95Python(1skip),build,unità native/lifecycle/HTTP/Ruff PASS; override nativi inclusi nell'identità tramite hash e rifiuto testato. Nuovo50+warmup avviato con journal durevole, zero reserve e identico corpus/guard; il vecchio21non viene importato. |


| M-P101 / T08 | 2026-11-09 | Stress Qwen27 chiuso con checkpoint | `PASS`6956.335s:50misurati+warmup nello stesso processo,6847campioni,swap growth0,pressione max2,minimo host1625800704B,exit0/no cleanup forzato. Tutti51output/continuation cached corretti,owner500device0/HOST41838725B/roots10 e nessuna prenotazione. Journal/artefatti riconvalidati dopo la chiusura;30sorgenti/5binari invariati. Ciclo medio135.648s,Image61.142s,embedding57.251s;prefill/decode separati nel report,nessuno speedup appaiato. Solo Imagevariant4 nel run; gate512indipendenti/reload separati. Restano F08/corpora T00/assert/hardware: T08 completo IN_PROGRESS. |


| M-P102 / T08 | 2026-11-09 | Regressione Flash dopo Qwen50 | `PASS` bounded1: ciclo168.030s,controller239.011s incluso fingerprint GGUF/avvio,Image65.633s,embedding54.670s,ritorno corretto/cache363/404token. Owner500identico(device0,HOST41838725B,roots10),prenotazioni vuote;pressione max1,swap growth0,exit0/no cleanup forzato. Non sostituisce Flash50storico né chiude F08. Tutti i processi della campagna terminati;hash30sorgenti/5binari invariati,nessun commit. |


| M-P103 / riesame | 2026-11-10 | Ownership e ripresa coerenti | Corretto teardown Numeric prima di un BUSY: riproduzione CPU precedente fallisce, query pubblica can_release nel registro e riuso Metal dopo il rifiuto PASS. Journal/continuità separati da SuiteReleaseHistory e opzioni possedute da QualificationClient, duplicati coerenti con argparse.105Python(1skip),CPU49(14skip),ASan/UBSan,build/fondazioni/lifecycle/trasporto PASS;20artefatti Flash/Qwen esatti,Image512+reload22artefatti esatti e gate indipendente PASS. Gemma NOMEM prima del forward e nuovo HTTP reale rinviati per RAM corrente;CUDA PENDING su istruzione utente. Due assert sin/cos e fixture budget2GiB/NOMEM restano espliciti;replay storico51record/gate/statistiche identico,nessuna nuova qualifica50 né speedup. Commit separati per scopo dopo verifica. |
| M-P104 / image progress | 2026-11-10 | Contratto comune e UI | Richiesta utente: progresso indipendente da CPU/Metal/CUDA e modelli futuri. SSE opt-in, heartbeat10s, callback SDK per invocazione e skeleton con sola percentuale/tempo. HTTP CPU, risorse/executor, bundle12, testi4, SDK118, frontend78 e browser9 PASS; wheel/lock/patch riproducibile aggiornati. Nessun kernel/dispatch cambiato. [Report](../architecture/native-image-progress-implementation-20261110.md). |
| M-P105 / image progress | 2026-11-10 | Image512 isolato e oracle | Input congelati identici, stesso binario con callback off/on:11/11artefatti esatti;8valutazioni e11eventi. Gate indipendente CPUencoder/MPSDiT/MPSVAE congelato PASS, soglie immutate. Tempi monitor230.781/229.842s con varianti adattive6/7: nessuno speedup o overhead isolato attribuito. Il vecchio binario NOMEM non è usato come controllo. CUDA NOT_RUN. [Evidenze](../performance/native-image-progress-20261110/verification.json). |
| M-P106 / image progress | 2026-11-10 | SSE durante compute nativo | Server isolato e client installato:256px in68.764s con read timeout20s,15eventi ordinati,1heartbeat e1terminale; gap massimo10.033s. Replay JSON idempotente in0.00269s con PNG identico; exit0, nessun riavvio di server utente. Discovery v1 e deadlineimage1200s verificate. Non è un benchmark LLM né qualifica CUDA. [Report](../architecture/native-image-progress-implementation-20261110.md). |
| M-P107 / image progress | 2026-11-10 | Compatibilità CUDA | Verificati il fence CUDA esistente, la selezione CUDA/cuDNN nel dry run Linux e le dipendenze incrementali dell'observer. Test HTTP host, factory CUDA con stub e sintassi C++17 PASS. Il nuovo test live di SSE e replay JSON passa su Metal in74.013s. Build `nvcc` e prova su GPU NVIDIA `NOT_RUN`. [Report](../architecture/native-image-progress-implementation-20261110.md#verifica-del-percorso-cuda). |
| M-P108 / image progress | 2026-11-10 | Revisione responsabilità | Esito SSE tipizzato, ripristino HTTP della chat, snapshot immutabile e percentuale per immagine. Nativi host e replay Metal PASS in 79,400s, exit 0. SDK 3685 pass e cinque fallimenti benchmark riprodotti sulla base. Agent 1217 pass, 49 skip live, copertura combinata 100%. Dry run Linux e factory CUDA con stub PASS. GPU NVIDIA NOT_RUN. [Report](../architecture/native-image-progress-implementation-20261110.md#revisione-delle-responsabilità). |
| M-P109 / image batch | 2026-11-10 | Baseline seriale 2×256 | 143,610 s totali, conditioning 9,048 s, DiT 113,497 s, VAE 21,050 s e H2D 228361068912 B. Congelati due PNG con seed 17/18 e replay idempotente. Audit upstream e piano per una wave a due lane. [Report](../performance/native-image-batch-20261110.md). |
| M-P110 / image batch | 2026-11-10 | Batch Metal 2×256 | 123,044 s contro 143,610 s, H2D 120381575096 B contro 228361068912 B. Due PNG bit-exact, replay e SSE PASS, exit 0. Varianti adattive 8/9, una sola coppia. CPU graph/API e dry run Linux PASS. Gate Image512 avviato. [Report](../performance/native-image-batch-20261110.md). |
| M-P111 / image batch | 2026-11-10 | Image512 con due output | 11 artefatti del primo output identici al controllo nativo, secondo PNG distinto. Oracle indipendente PASS con soglie congelate, 447,917 s monitorati con trace, exit 0. SDK 69 e frontend 5 PASS, CUDA statico e stub PASS. [Report](../performance/native-image-batch-20261110.md#gate-numerico-indipendente-a-512-px). |
| M-P112 / image batch | 2026-11-10 | Riesame ammissione e file | Budget DiT posseduto dal componente. Retry solo prima dell'executor e rollback dei PNG parziali. Test host PASS. Primo replay isolato dal server utente ma sovrapposto ai browser: `SKIP_RESOURCE_LIMIT` per crescita swap globale. Ripetizione sequenziale, soglie immutate. [Report](../performance/native-image-batch-20261110.md#riesame-del-ciclo-di-vita). |
| M-P113 / image batch | 2026-11-10 | Riesame validato | Ripetizione Metal 2×256 PASS in 124,483 s, PNG bit-exact, replay SSE e JSON PASS, 0 crescita swap ed exit 0. Agent 1218 pass, 49 skip live, copertura 100%. SDK 3686 pass, cinque fallimenti benchmark già presenti. CUDA statico e factory stub PASS, GPU NVIDIA da provare. [Report](../performance/native-image-batch-20261110.md#riesame-del-ciclo-di-vita). |
| M-P114 / Qwen27 vision | 2026-11-10 | Restore e lifetime host | Checkpoint originale integro e caricabile. Dopo vision, candidato rifiutato per memoria. Probe 64 MiB e regressione 32 MiB isolano riferimenti autoreleased nei trasferimenti host Metal; test nuovo FAIL prima della correzione. Correzione del lifetime in verifica, riserva host 0. [Diagnostica](../diagnostics/vision-session-restore-20261110.md). |
| M-P115 / Qwen27 vision | 2026-11-10 | Lifetime corretto e gate reale | Regressione, feature MTP/multimodali, lifecycle Qwen residente/SSD e Flash, primitive e grafi PASS. Dopo park Metal scende a 1769472 B, invece di 1234370560 B. Replay originale con RAM disponibile PASS, payload 2888 token ed embedding bit-exact. Due replay limitati dalle risorse conservati; nessun FPS rivendicato, riserva host 0. Avvio suite completa ancora in verifica. [Diagnostica](../diagnostics/vision-session-restore-20261110.md). |
| M-P116 / Qwen27 vision | 2026-11-10 | HTTP e tre riavvii | Suite completa avviata. Testo/vision/SSE HTTP 200, 64 poll catalogo 200. Tre riavvii con richieste vision PASS; exit 0 in 0,229–0,237 s, nessun processo/listener/checkpoint temporaneo o avviso handle residuo. RAM libera aumenta 7,90/8,09/8,83 GiB alle chiusure (contatori globali, cache adattiva). Server finale attivo a 8K, senza MTP e riserva host 0. Nessun commit. [Diagnostica](../diagnostics/vision-session-restore-20261110.md). |
| M-P117 / Metal | 2026-11-10 | Ordine del fill host | Copia GPU → fill riprodotta e corretta. Fondamenta, feature, lifecycle e fallback PASS. Flash e Qwen27 target/MTP2 streamed bit-exact sui 10/10/12 artefatti. MTP4 e replay vision a 8K limitati dall'ammissione RAM. [Diagnostica](../diagnostics/clean-code-review-20261110.md). |
