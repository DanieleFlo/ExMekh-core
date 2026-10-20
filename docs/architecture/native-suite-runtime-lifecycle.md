# Runtime e contesti della suite nativa

Implementazione T02 del 20 ottobre 2026. Il nuovo confine è in
`ds4_runtime.c/.h`, integrato con gli engine esistenti in `ds4.c`.
I nuovi kernel image/embedding e lo scheduler di memoria restano nei task
successivi. Una registrazione di descrittore non implica un modello disponibile.

## Proprietari e alternanza

Il runtime possiede mutex ricorsivo, fence e teardown condiviso. Ogni handle
modello possiede una copia immutabile della propria identità e callback di
attivazione/rilascio. Il runtime resta vivo finché esiste una reference o un
handle modello. Un rilascio selettivo non chiude stream o cuBLAS.

Le scope serializzano il compute. Scope ricorsive dello stesso modello sono
ammesse; cambiare proprietario ricorsivamente restituisce BUSY. Un altro thread
attende. Cambiare proprietario dopo una scope completata esegue la fence e
attiva il contesto destinatario. Un'attivazione fallita invalida il contesto
attivo; la successiva operazione riattiva esplicitamente il suo proprietario.
La chiusura fallita conserva l'handle. Il chiamante termina gli utilizzatori
prima di chiuderlo: non è consentito accodare lavoro su un handle distrutto.

Il bridge GGUF attuale accetta **un solo LLM legacy per runtime**. Rifiuta
un secondo open prima di cambiare shape, mapping o lock d'istanza. I nuovi
componenti nativi possono coesistere usando descrittori e tensori posseduti;
non devono entrare nella cache legacy basata su offset. Questa restrizione
confina lo stato globale senza simulare il supporto a due GGUF legacy residenti.

`ds4_engine_runtime()` espone un riferimento borrowed sul CUDA single GPU,
senza TP/distribuito; un handle componente registrato lo trattiene. Se il LLM
viene chiuso prima degli altri componenti, si liberano i suoi mapping e cache,
ma runtime GPU e lock d'istanza restano vivi fino all'ultimo componente.
`ds4_compute_runtime_open()` può inizializzare CUDA senza alcun LLM. La reference
esplicita del servizio mantiene vivi backend e lock anche senza modelli.
`ds4_engine_open_with_runtime()` carica il LLM in quel runtime: dopo aver chiuso
il vecchio engine può collegarne uno nuovo lasciando vivi gli altri componenti.
La costruzione usa `ds4_runtime_begin_update()`, che serializza anche loader,
shape e aggiornamento dei mapping prima che esista il nuovo handle modello.

La convenience API `ds4_engine_open()` conserva il runtime privato e rifiuta
l'apertura se ne esiste già uno; la condivisione deve essere esplicita.
I test verificano che stream CUDA e handle cuBLAS rimangano gli stessi durante
il rebind, che un componente possa precedere il LLM e che snapshot esistenti
continuino bit-exact su un engine ricaricato. Eviction dei soli pesi con engine
vivo e ripristino automatico della transazione/sessioni restano T04/T05.

## Inventario dello stato globale

La classificazione si applica anche alle cache statiche interne alle funzioni:
nessun nuovo adapter può usarle come stato del proprio modello. I gruppi legacy
restano esclusivi del solo handle `legacy-llm` e non vengono duplicati per copia
di struct contenenti stream, thread o mutex.

| Gruppo nel codice | Proprietario | Regola |
| --- | --- | --- |
| `g_ds4_shape`, `g_ds4_compress_ratios` | Modello legacy, vista esecutiva | Snapshot privato nell'engine; installato sotto scope. Getter identità/shape leggono lo snapshot |
| `g_qwen_file_type`, rope Qwen, `g_ds4_flash_vision_exp`, `g_ds4_qwen4_ple`, configurazione GLM | Modello legacy | Stabili durante la vita dello slot esclusivo; nessun nuovo adapter li modifica |
| vocab, pesi target/MTP/vision, graph e pool ricorrenti in `ds4_engine`/`ds4_session` | Engine/sessione legacy | Distruzione del solo proprietario; le sessioni precedono close engine |
| `g_model_*` CUDA, `g_support_*`, `g_dev_cache`, `g_cache_ranges`, derived/Q8 e cache esperti Qwen4 | Unico modello legacy, inclusi suoi sidecar | `ds4_gpu_legacy_model_release`; non namespace dei nuovi adapter |
| `g_gpu[]`, stream compute, eventi boundary, cuBLAS, bounce buffer | Runtime GPU | `ds4_gpu_cleanup` soltanto all'ultima reference |
| `g_quality_mode`, tag GLM/Qwen e math mode cuBLAS | Vista esecutiva del modello | Reinstallati dalla callback di attivazione; nessun cambio fuori scope |
| `g_cuda_tmp`, scratch per device/token-tile e workspace backend generici | Runtime/operazione seriale | Riutilizzo solo fra scope completate e dopo fence; mai memoria persistente di un adapter |
| Graph catturati, route cache, indexer e workspace specializzati Qwen/DeepSeek | Modello legacy o sua operazione | Invalidazione prima di rilasciare i mapping; non riutilizzati da un altro adapter |
| `g_pool`, `g_requested_threads` e helper CPU | Runtime | Shutdown dopo ultima reference; configurati dal solo caricatore legacy |
| `g_ds4_lock_fd` | Runtime | Lock intenzionale mantenuto durante tutta la vita del runtime |
| Stato TP, callback gate e code SSD/read/prefetch legacy | LLM legacy / operazione | Drain/unbind prima del rilascio. Nessuna nuova co-residenza TP esposta |
| Env cache, tabelle costanti e diagnostica/profiling | Configurazione runtime o diagnostica legacy | Non diventano cache di pesi o sessioni di un altro modello |

Il rilascio CUDA legacy separa range/derived/pinned staging del modello da
stream, handle e scratch del runtime. I tensori dei nuovi adapter hanno
allocazioni indipendenti: l'identità del loro handle e il relativo destructor
determinano il proprietario. Accounting, alias/view e budget sono T03.

## Compatibilità e verifiche

| Percorso | Esito locale |
| --- | --- |
| Runtime C: thread concorrenti, ricorsione, fence/activate falliti, destructor, reference | PASS, anche ASan/UBSan |
| CUDA: avvio senza LLM, LLM + due componenti, alternanza, restore snapshot, chiusura selettiva, rebind con stream/cuBLAS invariati | PASS con forward/logits bit-exact |
| Qwen27 testo/vision, MTP on/off | Logits e payload bit-exact rispetto alla baseline |
| Flash, profilo tiered del launcher esistente | Logits bit-exact rispetto al benchmark congelato |
| Rebind Qwen27 → Flash → Qwen27 nello stesso runtime | Componenti preservati, stream/cuBLAS invariati e snapshot Qwen27 bit-exact dopo il ritorno |
| MTP callback e snapshot drafts | PASS |
| CPU reference / stato sessione | Compilato, test PASS |
| Metal / ROCm / TP multi GPU / distribuito | NOT_VERIFIED su questa macchina; nessuna nuova co-residenza esposta |

Per Metal e ROCm il riferimento runtime resta interno all'engine e non si
espone ai nuovi componenti: il teardown mantiene il confine del singolo engine.
Non è una qualifica di quelle piattaforme. I test distribuiti non sono stati
avviati. Gli artefatti sono descritti nel report T00/T02 e nei manifest numerici.

Il diagnostic `ds4_qwen4_vision_dump` resta standalone: rifiuta di installare
un mapping primario quando è vivo un runtime della suite. I nuovi adapter non
modificano modalità SSD/cache del bridge legacy, che appartengono al suo
proprietario; usano tensori e cache propri.

Fence e attivazione rifiutano anche la reentrancy dei callback: non si può
chiudere o attivare un altro handle durante una transizione incompleta. Le
scope update trattengono una reference propria fino a leave, anche se il
servizio rilascia prima la sua reference. I test fault/ASan coprono questi casi.

La [revisione clean code](native-suite-clean-code-review-20261020.md) documenta
il confine pubblico dell'engine e la gestione del diagnostic vision con lo
stesso runtime esclusivo. `ds4_runtime_scope_leave()` chiude sia scope modello
sia scope update; non viene esposto il model handle interno dell'engine.
