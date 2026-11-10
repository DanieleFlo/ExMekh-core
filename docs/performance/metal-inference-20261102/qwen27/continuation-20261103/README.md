# Qwen27 Metal: continuazione del 3 novembre2026

Stato della campagna: **LOCAL_PASS, plateau sulle direzioni provate**.
Base `fc8fb63eb62cfa993fc924e8aa8b2eb9adc11c33`; dati e binari raccolti senza
commit, come richiesto durante le misure. Il successivo
[riesame kernel/streaming](../../../../architecture/metal-qwen27-kernel-stream-clean-code-review-20261103.md)
autorizza ora commit distinti e conserva i propri gate e confronti.
I tempi della riduzione a 16 lane ritirata non qualificano il bundle corretto.
Il risultato con MTP4 è distinto dal target-only e dalla baseline anch’essa
con MTP4.

Hardware e artefatto: M5 Pro24GiB, macOS26.6.2, SDK26.5. GGUF originale
15358213024byte, SHA25675bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017,
ricalcolato prima delle misure confermative. Un solo processo grande e guard
host2GiB/pressione/swap invariati. Capacità8192 e chunk64; occupazione64
per il benchmark breve,128 per il gate indipendente. Non qualifica velocità
con8k occupati, HTTP, ExMekh o l’alternanza con Image.

Gli artefatti grezzi restano in
`build/metal-perf-qwen27-20261102/continuation-20261103/`, inclusi tentativi
falliti/rifiuti admission, binari e shader congelati, input, logits completi,
trace separate, CSV a6decimali e report dei guard. Il manifest di ogni fase
collega codice/oggetti/shader; la libreria UD è caricata da sorgente a runtime,
quindi anche il path/hash dello shader è parte della provenienza.

Il cap1536MiB usa un harness diagnostico che imposta il campo pubblico
`engine_options.ssd_streaming_cache_bytes` dello stesso benchmark. Non cambia
admission o registry; la CLI comune accetta solo GiB interi. B0 e candidate
usano lo stesso harness, cache e16 output greedy effettivi, escludendo EOS.

Interventi e gate sono descritti nei report Q-P05..Q-P19 adiacenti:
gather per riga, residenza bounded e riapertura coerente; project UD riscritto
con due output perSIMD32; tile5 confinato al verifier; submit prima della
lettura SSD e riciclo alla pressione del cap. Nessuna attivazione Q8_1+R8
approssimata è stata attivata. La riduzione16lane è ritirata perché il gate
MTP4 supera una soglia per valore, anche quando i token greedy coincidono.
Le letture alternative8/64MiB peggiorano i micro: resta32MiB.

Le implementazioni esterne consultate sono pinnate per ogni intervento:
Transformers ca8c871..., Strata46c152a..., llama.cpp dde879bc... e DS4
originale890898c. I report indicano i percorsi e SHA completi, copertura
reale, assunzioni trasferibili/incompatibili e alternative. L’audit upstream
specificoQ-P15 separa i percorsi di Flash e Image da quello denso27B.

Riesame di responsabilità: il core descrive i pesi immutabili e centralizza
open/reload; il backend Metal possiede selezione, binding, pin, ready e quota
pending/live; il reader possiede descriptor/letture/contatori; la coda possiede
worker/drain. Nessun consumer interpreta i campi privati del reader o delle
classi Metal. Il dispatch UD esplicita il caso decode e quello verifier;
la variante project1 inutilizzata è rimossa. Non si certificano tutti i
principi Clean Code dell’intero repository: restano il runtime legacy globale
e i file monolitici, non riscritti in questa ottimizzazione mirata.

P17/P18 aggiungono due correzioni del nuovo owner: spazio per il head lazy
prima del primo accesso e storage retained non-ready posseduto durante gli
errori consecutivi. Il test usa soltanto API pubbliche, non i campi della cache.
EOF, retry, close e batch attivo sono verificati oltre al caso di successo.

Risultato locale mantenuto: **P18, plateau sulle direzioni provate**.
Warmup separato e5 blocchi appaiati con ordine ruotato/invertito, nessuna build o trace nelle finestre. Stesso cap1536MiB,8192/64, prompt64 e16 token greedy con MTP4.

| Metrica | B0 MTP4 | Bundle finale MTP4 |
| --- | ---: | ---: |
| Prefill token/s | 10.960227 | 13.767850 |
| Decode output token/s | 1.519248 | 1.976503 |
| Prefill+16 output, s | 16.401788 | 12.744541 |
| Primo blocco MTP, ms | 2640.909000 | 2058.521000 |

Differenze geometriche appaiate: prefill+27.574% (bootstrap locale95%+25.085..+31.680), decode+27.004% (+22.929..+29.915), richiesta−21.381% (−22.810..−19.501). N=5 quantifica incertezza locale, non garantisce isolamento del desktop. La seconda matrice conferma il precedente bundle~2TPS; tutti gli outlier sono inclusi.
B0 con MTP4 era già sopra1TPS: non attribuire tutto il salto rispetto al vecchio target-only ai nuovi kernel.16/16 ID e i due vettori full-vocabulary decimali per ogni run sono esatti conB0. Il gate indipendente salva invece11 vettori FP32 binari e verifica checkpoint, prefix no-op, park/reload e replay.
Il benchmark esclude EOS e ha acceptance favorevole: i4 cicli emettono4/4/5/3 token, inclusa la coda finale. Non è una promessa di2TPS su ogni prompt. gen_first_ms misura il primo blocco speculativo completo, non TTFT del singolo primo token. La richiesta è la somma delle due finestre misurate; init, snapshot e dump non sono latenza HTTP.
La dispersione decode è conservata nei JSON min/max/CV e nelle coppie, incluse le interruzioni da admission della prima matrice. Nessun purge globale, nessuna riduzione della riserva2GiB o dei criteri numerici.
Il profilo finale P18 separato misura53.542419632GB logici e53.561049088GB OS diskio per16 output,7.972920s decode sotto trace. Reader e contatore OS hanno scope diversi. I/O e GPU sovrapposti non si sommano.
Plateau: soglie SSD8/64 peggiorano il micro; codebookIQ4 constant peggiora il prefill e non migliora la richiesta; riserva minima Q-P19 legge−2.14% ma peggiora il prefill−4.85%, senza guadagno request. ROWS4 non migliora il bundleQ-P19 rispetto aROWS5. Si conservano piccoli guadagni quando sostenuti, senza veto2%. Nessuna precisione approssimata o riduzione16lane fallita rimane attiva.
Il plateau è relativo a questo workload, budget e famiglia di interventi, non un limite teorico. Un ring asincrono di layer con fence per slot, packing contiguo e staging/prefetch per Image rimane una distinta direzione architetturale. Dai soli byte GGUF, head+2 layer massimi occupano1561665536byte sotto il cap1610612736; questo non include quota extra, stato di retire o throughput misurato e non prova che una pipeline sia già sicura/veloce. Non viene dichiarata impossibile né implementata senza un nuovo ciclo di audit/gate.

Verifica finale: `make all` e `make test-metal-foundations` PASS;49 casi
host, ASan/UBSan reader/coda, feature Metal normali e fallback senzaMetal4,
primitive e grafo Flash streamed. Modello realeQwen128 target-only/MTP2/MTP4
e Flash completo PASS in sequenza. Nove logits ordinari full-vocabulary
bit-exact con i riferimenti storici per ogni caso; payload prefill target-only
e Flash bit-exact. I due replay8 MTP restano entro i criteri originali
`2e-5+5e-5*abs(ref)`; non si pretende uguaglianza ordinary/spec bit-per-bit.
Le fixture verificano anche cap, alias, errori EOF, retry, OOM e reload.
Il test reader finale conserva l’assert pool-lazy sotto32MiB oltre alla
nuova copertura medium/EOF; ricompilato e verificato anche con sanitizer.

`make test` globale non eseguito: include altre famiglie e prove live grandi
non appartenenti alla modifica. Nessun runtimeCUDA/ROCm/distribuito cambiato;
non sono qualificati su questo Mac. Il27B interamente residente non è una
nuova misura di questa campagna: percorso residente coperto dalle fixture
ridotte, speedup completo dichiarato soltanto per lo streaming misurato.
I file monolitici legacy restano debito architetturale, senza certificazione
globale Clean Code. Nessun commit creato durante la campagna prestazionale; l’integrazione è
registrata nel successivo riesame autorizzato dall’utente.

Controllo CLI finale (screening, cache2GiB, distinto dal confronto1536MiB):
target-only4 eval teacher-forced a64 occupati, prefill14.06/decode0.55TPS;
MTP4 con16 output greedy, prefill13.31/decode1.97TPS. Il target-only resta
sotto1TPS: il traguardo misurato richiede MTP4 e acceptance favorevole.
Il comando nativo usa le normali opzioni `--mtp-model` e `--mtp-draft 4`;
non abilita MTP automaticamente nel prodotto o nei profili target-only.

L’iniziale screening CLI target riportava prefill0.07TPS pur con soli18.7s
di uptime processo: campione escluso, file originali conservati. Indagine
prima di cambiare codice: il manuale SDK26.5 clock_gettime.3 distingue
CLOCK_MONOTONIC (avanza in sospensione) da CLOCK_UPTIME_RAW/mach_absolute_time
(non avanza); Python monotonic usa il secondo su macOS. Confrontati bench
locale e DS4 originale: nessuna aritmetica kernel coinvolta. Il log pmset
conferma sospensioni nella finestra del campione. Ripetizione senza modifiche:
prefill14.06, decode0.55, clockwall/continuous/uptime concordi a~13.217s.
Tutte le finestre della matrice appaiata sono <=tempo processo con margine:
nessuna anomalia simile nel set promozionale, outlier ordinari conservati.
Documentazione Python: https://docs.python.org/3/library/time.html#time.monotonic .
Nessuna correzione del timer o del kernel introdotta da un campione con sleep.
