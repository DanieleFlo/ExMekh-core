# Riesame della cache adattiva Qwen Metal — 3 novembre 2026

Base: HEAD `ab62874723c837ade442c192b90f0fba68bd2f73` più Q-P20/P21/P23/P25 non committati. La nuova richiesta autorizza commit separati dopo riesame e test. Snapshot completo prima del riesame, patch e hash: `build/metal-perf-qwen27-20261102/adaptive-review-20261103/`.

Il reload ora apre lo stream anche a cap zero; la sessione acquisisce
capacità prima di allocare o eseguire. Il core calcola il budget, Metal
sceglie e adotta il layout. I buffer mantenuti conservano quota, storage e
stato di lettura. La factory valida gli ingressi prima degli effetti e
azzera l’output in caso di errore. Gate Qwen/Flash bit-exact.

## Ispezione prima degli interventi

Riletti diff, core, engine/session creation, checkpoint/park/release/load, API GPU, cache, reader/resource registry, sampler e consumer Numeric, launcher/config/test e report della campagna. Le responsabilità restano: inventario/model semantics nel core; calcolo del budget engine tramite API pubbliche delle risorse; layout/storage/fence nel backend Metal; I/O nel reader; syscall nel sampler; sola aritmetica pura nella policy. Non convertire il bridge C globale in una gerarchia di classi né esporne storage ai consumer.

Difetti e semplificazioni da verificare:

1. `qwen_metal_stream_open` riusa `ssd_streaming_cache_bytes`, che è il risultato dell’ultimo campione, anche quando è zero. Dopo shrink a zero e release dei pesi, `weights_load` continua a passare zero all’API open che lo rifiuta; `rebalance` non può recuperare perché l’owner è chiuso/evicted. Il reload deve calcolare un nuovo piano dal campione attuale prima di aprire, mantenendo il vecchio owner in caso di failure.
2. Il calcolo `future/owned/restore` e l’applicazione del resize vivono nello stesso metodo. Estrarre la query riutilizzabile da open e rebalance; validazione del binding minimo condivisa. Per sessioni, stimare target+MTP in un metodo del core, riusato da admission e create.
3. L’entrypoint create può ricalcolare/evictare cache prima di controllare `out`. Validare puntatori/contesto prima degli effetti; inizializzare l’output della factory coerentemente con gli altri costruttori. Gli ingressi validi e il percorso CUDA numerico non cambiano.
4. `select_regions` restituisce un puntatore e tre out-param correlati; resize mescola selezione, confronto e trasferimento dello storage. Rappresentare il layout come valore privato e separare adozione/trasferimento dal metodo pubblico. Preservare ordine delle fence, quote lazy, storage failed-ready e buffer comuni; niente nuovo dispatch o politica di eviction.
5. La nuova fixture layer_stream comprende invalid metadata, binding abbandonato, EOF/retry e resize: separare gli scenari sotto lo stesso owner della fixture. Non ampliare i criteri né dipendere da slot/partizioni interne.

## Audit specifico prima del fix del workflow

Fonti rilette ai pin della campagna, con percorsi esatti:

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`, `src/transformers/models/qwen3_5/modeling_qwen3_5.py:Qwen3_5DecoderLayer.forward`, cache/stato/rmsnorm: le operazioni, dtype, maschere e rounding non cambiano. Il reload non esegue aritmetica del modello.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`, `metal/src/engine/stream.cpp:make_arena`, `metal/src/engine/window.cpp` (copia pinned nella continuation), `metal/src/mtl_rt.mm` e test `metal/tests/arena_budget_test.cpp`, dense_mm/gemv parity: query di capacità prima del layout, minimi e ownership; M5Max/percentuali/MoE e riempimento completo non trasferibili al27B denso24GiB.
- llama.cpp/ggml `dde879bc4f954b809892fd0d660515f42536394a`, `ggml/src/ggml-metal/ggml-metal.cpp:free/synchronize`, device/ops e `tests/test-backend-ops.cpp`: rilascio dopo drain e separazione owner/backend; non un oracle del nostro restore atomico. Nessun kernel diverso da importare.
- tkhr-sait/ds4 `9d8c1a8f1189dbe6403ca93097b551f8cc1e2f9c`, `ds4_metal.m:stream_nommap_routed_issue/ensure/free`: storage non pronto mai pubblicato, purge solo dopo completamento. I suoi end_commands per layer non spiegano le fence asincrone locali e non vengono copiati.
- Locale: Q-P20..P25 e i due precedenti riesami, `ds4_resources.h`, `ds4_lifecycle.inc:lifecycle_checkpoint_restore/weights_load`, `ds4_metal_memory.h`. Il cap storico di un owner rilasciato non è un cap richiesto dall’utente: conservare il ceiling originale separato dal limite effettivo. Nessuna modifica a sampler XNU o kernel Numeric, né una nuova qualifica Image/Diffusers.

Gate: riproduzione del reload dopo budget zero sul lifecycle sintetico con2GiB di riserva; invalid factory arguments senza mutazione; resize/EOF/contatori e resident/fallback Metal. Poi full Qwen128 target/MTP2/MTP4 e Flash35, confronti bit-exact con riferimenti indipendenti. Criteri primitivi2e-4+2e-5|ref|, graph2e-5+5e-5|ref| invariati. Confronto prestazionale prima/dopo a cap comune e controllo operativo auto; un processo grande per volta, guard pressione/swap/RAM invariati.

## Correzione dopo il primo gate completo

Il fixture prima del fix riproduce `weights_load` rifiutato con cap zero (riga223, `repro-zero-reload`). La prima soluzione interrogava la capacità già in `open`. Lifecycle Qwen/Flash e quattro grafi/feature isolati PASS; full target128 bit-exact, ma MTP2 rifiuta `weights_load` dopo park/release. L’owner SSD è lazy: aprire il file e copiare i descrittori non richiede già un head GPU residente. Il primo runner aveva inoltre un nome di binario MTP4 errato: MTP4/Flash di quella sequenza NOT_RUN, non una discrepanza numerica.

Nuova ispezione prima di cambiare il workflow: HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097` Qwen3_5DecoderLayer.forward consuma pesi solo nel forward; lo stato salvato si ripristina senza eseguire quei kernel. Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83` `metal/src/engine/window.cpp:stream_arena` effettua recheck per l’allocazione dell’arena, non impone un head a un lettore aperto. ggml `dde879bc4f954b809892fd0d660515f42536394a` `ggml-metal.cpp:device_init_backend/buffer_mapped` distingue apertura owner e binding storage. tkhr `9d8c1a8f1189dbe6403ca93097b551f8cc1e2f9c` `ds4_metal.m:nommap_routed_ensure` aspetta e popola la regione quando serve; il suo layout MoE resta incompatibile. Riletti locali `ds4_lifecycle.inc:weights_load/checkpoint_restore`, `qwen4_stream_alloc/resize` e P25: la riserva2GiB è verificata prima di allocare; il bootstrap a zero resta uno stream attivo e non abilita mmap fallback.

Scelta finale: `open` accetta zero come `resize`; un reload adattivo parte senza quota GPU e la sessione ricalcola prima di allocare/usare i pesi. Startup continua a usare il piano di admission già validato, senza un secondo campionamento ridondante; il ceiling utente resta separato e immutato. Il calcolo del budget rimane distinto dal resize. Nessun accesso ai pesi a cap zero e nessuna soglia modificata. Ripetere stessi casi isolati/full e misure a cap comune.

## Verifica finale di correttezza

Zero-cap lifecycle Qwen27 e Flash, constructor invalidi senza mutazione,
ceiling esplicito preservato, restore atomico e fixture resize/EOF PASS.
`make all`, `make test-metal-foundations`,50test Python host, feature residenti
con Metal4/fallback, Flash primitive/grafo, reader/coda ASan+UBSan e
compilazione CPU senza inferenza grande PASS. Il getter delle risorse
accetta un engine const e non espone nuovi internals.

Full Qwen128 target/MTP2/MTP4 e Flash35 in sequenza PASS:10/12/12/10
artefatti bit-exact contro P25, incluse riprese da checkpoint e MTP.
Nessun guard abbassato. La suite Numeric Metal (ambiente repo) riproduce
le due asserzioni sin/cos preesistenti;38test PASS,3skip, un test con2subtest
falliti. Q-P23 ha già isolato uscite prima/dopo sampler bit-exact: nessun
nuovo cambiamento del sampler o del kernel in questo riesame. Non marcare
la suite Numeric come PASS. Nessuna nuova qualifica CUDA o distribuita:
le modifiche di storage/dispatch sono sotto Metal; validazione argomenti
e getter C non cambiano percorsi validi o ABI binaria.

[Artefatti del riesame](../performance/metal-inference-20261102/qwen27/adaptive-review-20261103/):
output grezzi nei path dell’`artifacts.json`; primo tentativo/rifiuti conservati.

## Controllo prestazionale

P25 è congelato prima del riesame: il suo profilo è425CB/5997encoder,
53.628GB logici. Stessi shader,8192allocati/64input/16output greedy MTP4,
cap esplicito1536MiB. Un warmup per variante, poi5coppie alternate senza
trace, tutti i vettori/dump bit-exact. La somma prefill+decode esclude
startup/tokenizzazione/dump, non è un tempo HTTP end-to-end.

| Misura | Mediana P25 | Mediana riesame | Delta appaiato, bootstrap95% |
| --- | --- | --- | --- |
| Prefill |16.749481TPS|16.688348TPS|−0.369%[−0.850,+0.113]|
| Decode |3.360910TPS|3.365408TPS|−1.342%[−4.151,+0.570]|
| Somma delle fasi |8.566346s|8.589264s|+0.935%[−0.201,+2.682]|
| Gather residente Q6_K/5120/1 |0.244265ms|0.242295ms|−0.473%[−3.856,+2.711]|

Il gather usa stessi pesi/ID e oracle esatto indipendente,100warmup/200call
per run, poi5coppie. La matrice e le distribuzioni conservano tutti i campioni
(compreso il decode lento3.122TPS del secondo blocco). Nessuna nuova
accelerazione o regressione stabile confermata; gli intervalli non sono
una prova di equivalenza perfetta. I risultati operativi auto di P25
restano attribuiti ai loro binari.

La prima trace scende a2.397TPS, con gli stessi425CB/5997encoder,
stessi53.628GB logici e span GPU5.148s contro5.047s di P25. I cicli
allungano la parte fuori GPU; non è stato attribuito un rallentamento
a un kernel né modificato il dispatch. Riletti il wait/timeline locale
e la distinzione tra sync/async di ggml/Strata ai pin sopra. La coppia
di controllo, ancora con trace, misura3.373932→3.355168TPS, con latenza
sommata8.600414→8.590430s: il rallentamento isolato non si ripete.
Entrambe le trace e quella lenta sono conservate e restano separate
dai cinque blocchi senza profiler.

## Integrazione

Il commit `53193ef` contiene il trasporto dei layer e la gestione dei buffer
Metal. Il commit `1e05131` contiene il budget dinamico e il ciclo di vita dei
pesi. I risultati sopra verificano le due modifiche insieme.
