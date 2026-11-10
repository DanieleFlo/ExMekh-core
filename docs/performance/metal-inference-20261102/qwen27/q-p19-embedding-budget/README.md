# Q-P19: costo del binding embedding posseduto dal backend

S1 prima del codice, 3 novembre2026. Q-P06 mantiene come riserva transiente
l’intera tabella embedding Q3_K da546304000byte, anche dopo che Q-P05 ha
qualificato batch e scalar lookup su una sola riga alla volta. Nel GGUF
corrente il layer più grande richiede meno spazio della tabella: riservarla
ancora riduce inutilmente i pesi che restano residenti. Una stima maggiore
non altera i risultati ma può causare riletture evitabili.

Riesame specifico: HFca8c871ede62b521a75a2ddfc5e1968ac9c8e097
models/qwen3_5/modeling_qwen3_5.py:embed_tokens/MLP/DecoderLayer (righe lookup
indipendenti, ordine invariato); Strata46c152a1310f77cd2f0ce2c7cd213962df14fe83
metal/src/engine/stream.h/cpp:load_dense/make_arena/hold_region e testarena,
gemv.metal/parity e mtl_rt.mm (costo di working set distinto da residenza,
densi interamente residenti non applicabili); ggml dde879bc4f954b809892fd0d660515f42536394a
src/models/qwen35.cpp embedding/head, ggml-metal-ops.cpp:get_rows,
device e test-backend-ops (lookup quantizzato, non oracle SSD bounded).
DS4 originale890898c static weights headroom e separazione weight/runtime
budget, già verificati inQ-P15. Locale: scalar/batch Q-P05 al cap1row,
loader/gather/capability e il plannerP06/P17; pin/fence e failed-readP18.

Scelta: query pubblica del backend per il massimo binding embedding SSD.
Per UD, dopo validazione, è una riga; per altri formati conserva la tabella
intera richiesta dal batch generico. Il core usa quella query e la massima
somma di layer per la riserva, senza duplicare elenco quant o introspezionare
la cache. Non è un nuovo kernel, non cambia cap, precisione o admission.
Alternative dimezzare una costante o conoscere gli enum UD nel core rompono
il confine di responsabilità; ridurre la riserva per tutti i formati è scorretto.

ProfiloP18 separato necessario prima dell’implementazione; poi isolati
query/gather, grafi+cap+recovery, gatefull128/MTP4 e confronto conP18 allo
stesso1536MiB/8192/64. ConfrontareROWS4/5 direttamente nel bundle finale:
il vantaggio GPU del tile5 non ha finora migliorato la richiesta in modo certo.
Nessuna soglia minima2% e nessun commit.

Profilo prima del codice, P18 corretto:53542419632byte logici,
53561049088byte OS diskio_read; decode7.972920s per16 output,
cicli4/4/5/3. Lo streaming resta rilevante dopo il kernel corretto.
Solo l’OS counter è process-wide; non è un oracle di SSD isolato. Ora
implementare la query, conservando gli stessi cap/worker/kernel/criteri.

S2 PASS: query per12 tipi e due larghezze, invalid shape, gatherUD a1row,
head tardivo/EOF/retry; graph/cap e modello128 a8192/MTP4.11 vettori
bit-exact conP18. Profilo successivo separato: logical52394646704byte
(−2.14% vs53542419632), OSdiskio52408115200, live1577189248 nel cap1610612736.
Decode8.413517s/16 (1.901702TPS sotto trace), rispetto a7.972920s del profilo
precedente: meno byte da solo non certifica lo speedup. Attendere5 blocchi
senza trace; non scartare il componente per un piccolo effetto o un solo run.

S3, indagine riaperta dopo lo screening negativo (prima di cambiare ancora):
meno byte ma più tempo, anche nelle prime coppie fuori profiler. Riletti
Strata stream.cpp make_arena/hold_region e cap minimo per evitare stalli,
ggml Metal buffer lifetime, DS4 originale ds4.c static-budget headroom,
e local load/synchronize/available. La riserva minima per un layer non è
necessariamente la riserva più veloce per overlap: con una regione transiente
più piccola cambiano frequenza di eviction/fence e assegnazioni. Le trace
separate mostrano più command buffer nel processoP19; è un indizio, non
un’attribuzione causale completa di ogni millisecondo. Nessun kernel o peso
è cambiato e il gate è bit-exact. Completare le5 coppie e ripristinare la
policyP18 se il bundle non riduce la richiesta; nessun tuning di costanti
senza una nuova ipotesi/profilo.

S4,5 blocchi appaiati dopo warmup, tutti16 ID e full logits decimali esatti:
P18 prefill13.767850/decode1.976503/request12.744541s mediani;
P19 13.127004/1.919512/13.240292. Prefill appaiato−4.848%
(CI locale−6.414..−3.657), decode−0.680% (−3.229..+4.052),
request+2.245% (−0.265..+3.768). Il request non migliora e il prefill
peggiora in modo riproducibile. **REJECT del bundle**, non per una soglia2%.
Ripristinata la riservaP18; query e relativa API/test sperimentali rimossi,
nessuna variante inutile in produzione. La quota più ampia è intenzionale
per il tradeoff cache/overlap misurato, non l’asserzione che il gather legga
ancora tutta la tabella. Due trace process-wide:743→813 command batch,
tensor copy597 invariati; non sommare GPU/I/O sovrapposti.

A riservaP19 uguale, anche ROWS4 rispetto aROWS5 non migliora la richiesta:
decode−3.518% (CI−8.364..+0.579), request+2.421%
(−0.261..+5.359). Si conservaROWS5 nel bundle finaleP18; il solo guadagno
GPU non è presentato come speedup request indipendente.
