# Riesame Clean Code della continuazione Qwen27 Metal

3 novembre 2026. Base HEAD `fc8fb63eb62cfa993fc924e8aa8b2eb9adc11c33` più
il bundle Q-P05–P18 non committato. La nuova richiesta autorizza commit
separati dopo il riesame e le verifiche; sostituisce il precedente divieto
per questa integrazione. Le misure storiche restano riferite ai loro binari.

## Riesame prima del codice

Letti core, API GPU, lifecycle, cache/reader/coda, wrapper UD, shader, test,
qualificatore e rapporti. Il core conosce l’inventario del modello, Metal
possiede binding/cache/dispatch; il reader possiede I/O/counter e la coda i
worker/drain. Non spostare gli internals Metal nel core o nei test. Nessuna
nuova classe C++ o gerarchia per gli owner privati in C.

Risultati da correggere:

- L’ingresso pubblico batch quantizzato delega a UD prima di controllare
  `tokens != NULL`. UD interpreta il NULL come gather scalare e può scrivere
  più righe duplicate; gli altri formati rifiutano. Validare i metadati e i
  puntatori obbligatori nell’API pubblica prima del dispatch per formato.
- `qwen35_ud_tensor` mescola validazione/layout/encoding e orchestrazione
  CPU dei gather SSD. Estrarre quest’ultima in un helper del wrapper Qwen,
  usando tensor_read/view e l’API scalare esposta, senza leggere gli internals.
- Il metodo cache `clear` ora conserva le regioni retained: chiamarlo
  `clear_transient`. Separare ricerca della regione e caricamento lazy/ready,
  mantenendo la quota e lo storage fallito nello stesso owner. Nessuna nuova
  politica di eviction, fence, residenza, dimensione o aritmetica.
- La fixture `test_scalar_stream_gather` include anche batch, invalid input,
  head tardivo e recovery: separare scenari con fixture presa in prestito e
  un controllo oracle condiviso. Conservare cap, contatori e tutti i casi.
- Il rapporto della continuazione apre ancora con IN_PROGRESS nonostante
  i risultati conclusi. Distinguere stato finale, cronologia senza commit e
  qualifica dopo il riesame; aggiornare indice e provenienza.

## Audit specifico prima del refactoring e del fix

Riletti i sorgenti locali ai pin verificati della campagna:

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  src/transformers/models/qwen3_5/modeling_qwen3_5.py: embed_tokens, forward,
  Qwen3_5MLP/RMSNorm. Lookup solo con ID presenti; nessun cambio a norm,
  dtype, stato, mask o ordine delle operazioni.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  metal/src/engine/stream.cpp/h: read_part/io_main/ready/pin e
  metal/src/mtl_rt.mm: Context::wrap/free; test arena_budget/gemv_parity.
  Separazione ownership/readiness trasferibile, abort/repack e residenza
  completa densa non sono recovery/coverage del27B su questo Mac.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  ggml/src/ggml-metal/ggml-metal-ops.cpp:get_rows,
  ggml-metal-device.cpp e tests/test-backend-ops.cpp:test_get_rows,
  test_get_rows_after_mul_mat. ID e shape espliciti e dipendenze GPU;
  non importare nuovi cast/reduction o assumere che gli errori del nostro
  owner SSD siano già qualificati dal backend ggml.
- Ricerca locale: INDEX/problem-to-source, Q-P05/P06/P12/P17/P18 e riesame
  reader del2novembre; test dei buffer con retire differito e open/reload.
  Manteniamo la policy misurata, senza riproporre Q-P19 o codebook respinti.

Hash/URL delle fonti nei manifest della continuazione, snapshot prima del
riesame in `build/metal-perf-qwen27-20261102/review-20261103/initial.json`.
Il report registrerà riproduzione del NULL, isolati identici, gate indipendenti
Qwen/MTP/Flash e controllo prestazionale prima/dopo. Criteri invariati.

Il riesame non certifica tutti i principi Clean Code dell’intero repository:
restano file legacy monolitici e stato globale. Si verificano e correggono
le responsabilità coinvolte e i consumer adiacenti, senza attribuire nuovi
speedup al refactoring.

## Correzioni e verifica numerica

Il probe pubblico pre-fix fallisce sul batch UD11 con `tokens=NULL`,
17 righe vocabolario,2 righe richieste e cap1row: il vecchio metodo restituisce
successo. Con la validazione pubblica restituisce0 prima di I/O o output.
La fixture permanente verifica anche output/map assenti e vocabolario0,
per12 formati e2 larghezze, su mapping residente e streaming. L’output
precedente resta esatto e il contatore reader invariato.

`qwen35_stream_gather_rows` possiede l’orchestrazione del batch e chiama il
lookup scalare esposto; `qwen4_stream_resident_buffer` possiede quota e ready,
mentre `qwen4_stream_range` cerca la regione e pubblica l’offset solo in caso
di successo. `clear_transient` non promette più di svuotare il retained.
Nessun consumer legge campi del reader o la rappresentazione dei tensori.

Dopo il refactoring passano fondamenta,49 host, ASan/UBSan reader e coda,
feature Metal normali/fallback, grafi e primitive Flash. Gate indipendenti
Qwen128 target-only/MTP2/MTP4 e Flash completo PASS: tutti i vettori ordinari,
speculativi e payload salvati sono bit-exact rispetto al pre-riesame.
Shader e criteri numerici sono invariati. Compilazione CPU-only separata,
senza inferenza CPU grande; CUDA/ROCm/distribuito non modificati né eseguiti.

## Controllo prestazionale dopo il riesame

Warmup escluso e5 coppie alternate, cap1536MiB,8192/64,64 input e16 output
MTP4 reali. Nessun profiler o build concorrente; stessa sorgente shader,
controllo clock/uptime senza sleep. Mediane prima/dopo: prefill13.790747 /
13.761375TPS, decode1.932702 /1.990560, richiesta12.919358 /12.688637s.
Rapporti geometrici appaiati: prefill+0.309% (bootstrap95%−0.344..+1.127),
decode+2.070% (−1.392..+6.580), tempo−1.427% (−4.287..+0.986).
Nessun nuovo speedup attribuito al refactoring: differenze compatibili con
la variabilità osservata. Tutti16 ID e i vettori full-vocabulary decimali
sono esatti. I dati della campagna originale e i suoi delta restano separati.

Il micro residente usa quattro formati UD,5120 colonne e1/64 righe, senza
caricare il27B residente. Il primo lancio collideva fra nome binario e cartella
output: non avviato; rerun con path distinti. L’output iniziale stdout/stderr
si intercalava durante teardown: dati non promossi; harness privato con
stdout line-buffered e nuovi run per conservare tutte le righe. Nessuna
modifica al motore per correggere questi due problemi del solo harness.

Il primo micro completo mostra suQ6_K scalare +6.31% di tempo, intervallo
stimato+1.81..+10.94 su5 processi; circa15microsecondi a dispatch.
Indagine riaperta senza cambiare kernel/dispatch: riesaminati l’ingresso
pubblico, metadata checks, qwen35_ud_tensor, cache normale e command lifetime;
HF embedding, Strata wrapper/gemv parity e ggml get_rows/device ai pin della
scheda restano pertinenti. L’unica aggiunta scalare è la validazione iniziale;
lo shader e l’ordine delle operazioni sono identici. Non attribuire il costo
residuo al driver o al refactoring senza prova. Conferma mirata: stessa shape,
100 warmup esclusi e200 dispatch misurati senza printf/timer per chiamata,
oracle finale esatto,10 coppie processi alternate. Il nuovo harness rimane
privato; non aggiunge una variante permanente al motore.

Conferma mirata Q6 scalare:10 coppie,200 dispatch ciascuna dopo100 warmup,
oracle bit-exact; mediana per chiamata0.220943ms prima e0.217675ms dopo.
Rapporto appaiato−0.588% (bootstrap95%−6.748..+6.431). La precedente piccola
regressione non si conferma con campioni più lunghi; non si deduce uno speedup
né una causa certa della variabilità. Gli altri micro residenti1/64righe
sono conservati con tutti i2480 campioni, non selezionati per segno.
Resta non misurato il27B interamente residente, incompatibile con il budget
desktop corrente: non è sostituito da questi fixture ridotti.

## Esito e limiti

Riesame locale completato: responsabilità separate senza nuove policy,
API embedding coerente sui puntatori obbligatori, test per casi validi e
recovery; build,49 host, sanitizer e gate full-model indipendenti PASS.
Prestazioni del bundle conservate nel controllo prima/dopo, senza nuovo
speedup promosso. Nessuna modifica CUDA/ROCm/distribuita; non qualificati
nuovi hardware o pipeline Image. Restano espliciti i monoliti/stato globale
legacy, senza dichiarare che l’intero repository soddisfi ogni principio.

Evidenze compatte in
[clean-code-review-20261103](../performance/metal-inference-20261102/qwen27/clean-code-review-20261103/),
artefatti grezzi ignorati in `build/metal-perf-qwen27-20261102/review-20261103/`.
Commit distinti per bundle originale, refactoring/fix/test e documentazione.

Implementazione: bundle `6aa3333c6577`, riesame/fix `0ef84d5387c1`.
Esiti e hash nel [manifest di verifica](../performance/metal-inference-20261102/qwen27/clean-code-review-20261103/verification.json).
