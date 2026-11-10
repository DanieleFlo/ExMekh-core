# Q-P17: prenotare la quota lazy dei pesi retained

Riesame del 3 novembre2026, prima della correzione. Nel nuovo owner Q-P06,
la selezione delle regioni immutabili lascia il massimo layer libero, ma
la cache transiente può occupare anche i byte selezionati e ancora non
allocati. Un head richiesto per ultimo può allora fallire pur essendo incluso
nel piano. Statico: tre righe transienti riempiono cap3row; head lazy1row non
entra, benché il piano avesse riservato2row transienti. Il full27B finora passa,
ma la policy pubblica deve essere indipendente dall’ordine degli accessi.

Nuovo audit pertinente prima dell’implementazione: HF
ca8c871ede62b521a75a2ddfc5e1968ac9c8e097, Qwen3_5MLP/DecoderLayer e norm:
trasporto immutabile, nessun cambio cast/reduction/state. Strata
46c152a1310f77cd2f0ce2c7cd213962df14fe83, metal/src/engine/stream.cpp/h:
arena/make_arena/victim/pin e arena_budget_test.cpp; capacità riservata prima
di pubblicare slot, geometrie esperti e abort non trasferiti. ggml
dde879bc4f954b809892fd0d660515f42536394a, ggml-metal-device.cpp/ops.cpp,
llama-mmap.cpp e test-backend-ops.cpp: ownership fino al termine del consumo,
nessun budget lazy equivalente. DS4 originale890898cfb437ed256be58ebcf7c58a6dfa8f2f53,
ds4.c static weights headroom e22e4172: non spendere due volte gli stessi byte.
Locale: alloc/load/range/clear, risorse e test alias/deferred-retire, P06/P13.

Scelta da verificare: contatore privato dei byte retained non ancora allocati;
la disponibilità transiente sottrae sia live sia quella prenotazione. Una
regione lazy converte la propria prenotazione in allocazione, la ripristina
su fallimento. Il contatore pubblico resta storage realmente posseduto.
Nessuna nuova opzione, staging o crescita del cap. Flash senza piano ha
prenotazione0 e conserva la policy. Alternativa evict-all al primo head più
complessa e dipendente dall’ordine; non scelta.

Prima: riprodurre con soli gather pubblici, stessi tre byte-row e head tardivo.
Dopo: stesso input e oracle, cap/error/recovery, full8192 ordinary/MTP4,
poi confronto appaiato della correzione con il bundle precedente.

Riproduzione prima della correzione: **FAIL** sul primo UD11, offset1760,
head110byte, cap330byte. Tre miss avevano speso l’intero cap. Nessun mismatch
numerico: binding rifiutato. Il nuovo test mantiene lo stesso ordine e aggiunge
EOF del head tardivo, ripristino backing, retry e hit successiva:5row letti
in totale, storage <=3row. La correzione usa quota privata pending→live;
la API pubblica continua a riportare solo storage posseduto.

Il successivoP18 conserva storage non-ready dopo erroreI/O invece di
liberarlo/riprenotarlo: evita sottocontabilità durante il retire differito.
Questa è la policy finale; il rollback della sola allocazione fallita continua
a ripristinare la prenotazione, senza storage posseduto.
