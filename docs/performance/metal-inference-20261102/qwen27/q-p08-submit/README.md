# Q-P08: inviare il lavoro GPU prima delle letture SSD

3 novembre2026, S1 prima del codice, da eseguire dopo screening Q-P07.

Profilo P06:3435 sample main,802 in fence layer, resto dominato da wait coda
I/O/pread; GPU circa0.35–0.66s contro~2s/token. L'encoder accumula lavoro,
poi il prossimo binding legge i pesi con CPU bloccata: quel lavoro GPU può
restare non inviato durante la lettura. Ipotesi diversa dalla sola cache.

Audit specifico riletto prima dell'intervento:
- HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py`, proiezioni/GDN/residui:
  stesso ordine dipendenze sul command queue, nessuna aritmetica da cambiare.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp:submit/io_main/layer`, `stream.h`,
  `metal/src/mtl_rt.mm:Context::free/run_graph`, test arena/gemv. Nuovo fetch
  `metal/src/engine/window.cpp:Window::run`, cache cuts e stream_arena:
  segmenti per layer, densi separati da arena, buffer non liberati fino al
  consumo. Il profilo con hot routed e prefetch esperti non è denso27B.
  Nessuna semantica MTP lazy-fold/CPU PLE importata.
- ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/ggml-metal.cpp`, ops/device e `src/llama-mmap.cpp`:
  risorse shared e encoding mantengono dipendenze; non offre qui un oracle
  completo dello streaming denso. Kernel/test matmul confermano algebra,
  non sovrapposizione del reader host.
- Locale: `ds4_gpu_flush_commands` chiude encoder, commit senza wait, trattiene
  pending CB e ricrea batch; synchronize/retire attendono prima del riciclo.
  Owner cache/read e test errori, CUDA ring/transfer e report async precedenti.
  Reader continua a completare/drain tutte le letture prima di esporre dati.

Intervento: solo owner con retention configurata (Qwen denso), flush del
batch prima delle letture grandi>=8MiB (taglia chunk reader). Nessun worker
GPU, buffer o budget aggiuntivo; lo stesso thread legge nel buffer ancora
non consumato. Ai confini senza entry transiente né compact non c'è storage
da riciclare: si può conservare il batch. Fence sempre prima di liberare o
sovrascrivere storage transiente; close/park attendono pending CB.

Alternativa ring2 completo richiede più metadati/riserva e non è il primo
esperimento necessario; aumento worker non elimina lavoro GPU non inviato.
Nessuna modifica a Flash (nessuna retention), CUDA, CPU o distribuito.
Gate isolati grafi resident/streamed/lifecycle/errori poi Qwen128 indipendente,
MTP e Flash seriali. Criteri frozen e memoria invariati. Screening/profilo e
poi5 coppie del bundle; confrontare2GiB a2GiB e4GiB esplicito separatamente.

Prima dell'implementazione: nuova trace Q-P07b64+4 PASS, decode0.57 sotto
profiler (~1.75s/token), quindi la riduzione del solo costo kernel non chiude
il residuo host. B0/Bi eseguibili e shader separati; nessuna trace promozionale.

S2: grafi ridotti, feature (compreso MTP), lifecycle Qwen residente/streamed
e Flash PASS. Gate Qwen128 indipendente: nove full-vocabulary e payload
bit-exact con Q-P07b (stessa numerica); checkpoint/park/reload/continuazione
PASS. Nessun nuovo buffer. Screening a2/4GiB, prova5GiB soggetta alla stessa
admission, poi trace separata4GiB prima del confronto appaiato finale.

## S3 direzione e profilo

Gate invariati PASS. Screening64/128+4, non promozione:
-2GiB: prefill14.48/14.40, decode0.57/0.56;
-4GiB: prefill14.54/14.74, decode0.62/0.62;
-5GiB: prefill15.04/14.92, decode0.70/0.71; steady0.78/0.74.
Trace separata4GiB: prefill15.16, decode0.69. Non usare timing profiler come
speedup promozionale. La cache più grande è una variabile esplicita.

Estensione definita prima dei run: segnale5GiB promettente ma ancora sotto1TPS;
provare6GiB e solo se ammessa7GiB (massimo due processi120s), passando sempre
attraverso admission e guard invariati. Nessun aumento della riserva disponibile
inventato e nessun aggiramento se l'host rifiuta. Poi MTP come esperimento
separato e conferma appaiata del bundle scelto.

6GiB rifiutati dall'admission: need7.42GiB, budget6.90, host8.90 e riserva2.
Tentativo terminato al caricamento, FAIL conservato;7GiB non eseguiti dopo il
rifiuto. Le prove2/4/5GiB sono PASS, nessuna crescita swap; min host
5.78/4.46/3.80GiB, footprint campionato3.19/5.18/6.18GiB rispettivamente.
Questi limiti escludono di cercare1TPS semplicemente forzando più RAM.
