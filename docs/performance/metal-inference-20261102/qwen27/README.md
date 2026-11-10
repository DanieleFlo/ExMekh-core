# Qwen27 Metal streamed: campagna del 2 novembre 2026

Stato storico del 2 novembre: **KEEP, prima campagna conclusa nel subset M5 Pro misurato**.
La [continuazione del 3 novembre](continuation-20261103/README.md) registra
le riscritture dei kernel, MTP4 e il successivo plateau locale; modalità e
budget sono distinti dai numeri target-only conservati qui sotto.
Passaggio da Flash autorizzato dall’utente. Q-P03 attivo per default; Q-P02 ritirato.
B0: HEAD `089255a`, checkout inizialmente pulito. Artefatti grezzi ignorati:
`build/metal-perf-qwen27-20261102/`. I risultati Flash non si trasferiscono.

## Q-P00: baseline e piano congelati

Qwen3.8-27B UD-Q4_K_S originale, 15358213024 byte, backend Metal,
contesto allocato 8192, chunk 64, cache pesi 2 GiB, target-only, senza vision,
MTP o suite. Due processi nuovi, frontiere 64/128 dal prompt raw
`tests/test-vectors/flash-0731/prompts/long_code_audit.txt`, otto token
teacher-forced per frontiera. La seconda frontiera elabora 64 token nuovi;
snapshot/ripristino fuori dalle finestre prefill/decode. Il benchmark conserva
logits full-vocabulary prima e dopo decode. Non è generazione libera né HTTP.

Budget iniziale: due sweep direzionali, ciascuno limitato a 120 s; poi una
trace separata 64 input/quattro decode con gli hook Metal esistenti.
Guard: una sola grande istanza, riserva host 2 GiB, pressione e swap monitorati.
Nessuna pulizia globale della page cache, nessuna affermazione di SSD freddo.
La cache applicativa parte vuota. Binario e oggetto Metal B0 copiati prima
di ogni modifica. Hash e piano in `initial.json` negli artefatti grezzi.

Un primo comando usa l’opzione inesistente `--dump-frontier-logits`: termina
al parser in circa un secondo, senza caricare il modello. Il rerun usa
`--dump-frontier-logits-dir`; il tentativo fallito resta conservato.
Il rilevamento hardware dentro sandbox è negato; ripetuto fuori sandbox:
Mac17,9/M5 Pro 24 GiB, macOS 26.6.2 (25G83), SDK 26.5, clang 21.0.0.
Nessun avviso termico o di performance in `pmset -g therm`; nessun altro
processo LLM rilevato. Non si inferisce da questo l’assenza di ogni carico host.

## Q-P01: profilo e primo intervento

Due sweep B0 PASS: prefill 9,62/9,87 e 9,79/10,06 token/s per le frontiere
64 e 128; decode 0,33/0,34 e 0,33/0,32. Richiesta sintetica (prefill + otto
passi eval) circa 30–31 s per frontiera, distinta dal processo totale con
snapshot, dump e ripristino. Sono dati direzionali, non promozionali.

Trace separata: prefill 6056,5 ms wall, span GPU 3726,5 ms; decode
2893,8–3013,1 ms wall, GPU 468,2–514,6 ms. Le proiezioni UD dominano il
prefill GPU; il decode è soprattutto fuori GPU, senza attribuire tutta la
quota residua a I/O fisico. I buffer concorrenti non si sommano al wall.

[Q-P02, tile UD da otto righe](q-p02-ud8/README.md) è stato ritirato:
270 casi e full-model bit-exact, ma regressioni micro su tre delle quattro
forme rappresentative e nessuna riduzione dello span GPU prefill. Il +3,25%
locale sulla down resta un’alternativa specifica da confermare, senza veto
ai piccoli incrementi. Nessuna modifica ai kernel resta attiva.

[Q-P03, letture dense parallele](q-p03-dense-reads/README.md) segue un nuovo
profilo host: 4844/6447 campioni del thread d’inferenza in pread. Il reader
suddivide regioni >= 32 MiB tra i suoi worker senza nuovi buffer. Gate host,
sanitizer, grafi e Qwen27 completo bit-exact PASS. Cinque coppie alternate
con warm-up separato confermano il guadagno:

| Metrica | Frontiera 64 | Frontiera 128 (64 nuovi token) |
| --- | ---: | ---: |
| Prefill, incremento TPS | +12,23% | +9,46% |
| Decode, incremento TPS | +27,01% | +34,51% |
| Prefill + quattro eval, riduzione tempo | −18,15% | −19,85% |

Stesso profilo 8192/64/cache 2 GiB e stessi pesi originali. La baseline decode ha CV>5%:
instabilità conservata, tutti i delta appaiati favorevoli e intervalli stimati
positivi; limiti e dispersione nel report Q-P03. Il controllo completo usa
un prompt indipendente: nove logits e stato target bit-exact. MTP PASS con
token esatti e stesso piccolo roundoff entro soglia su B0 e candidate;
regressione Flash bit-exact. Binari `make all` aggiornati, nessun nuovo flag.

La richiesta sintetica non misura HTTP/ExMekh. La capacità allocata 8192 non
qualifica nuove misure a 8k occupati. CUDA/ROCm/distribuito e altri modelli
restano distinti. I residui principali sono residenza/riuso dei pesi e overlap,
che cambiano budget/lifetime e richiedono il ciclo completo di un nuovo audit.
Il tile down con piccolo segnale positivo rimane una direzione specifica,
non promossa dal microbenchmark. Nessun commit durante le misure originali.
Ogni intervento ha seguito la scheda fonti della roadmap prima del codice,
con Transformers del modello effettivo, Strata, llama.cpp e ricerca locale.
La promozione Q-P03 include correttezza isolata e gate full-model indipendente,
warm-up e cinque coppie alternate fuori dal profiler. Le misure direzionali
non sono state usate da sole per KEEP.


## Riesame Clean Code successivo

Il [riesame architetturale](../../../architecture/metal-qwen27-stream-clean-code-review-20261102.md)
separa pianificazione contigua/indicizzata ed esecuzione del batch nel reader;
cache Metal e coda mantengono le proprie responsabilità e API. I test di EOF
non dipendono più dalla formula di suddivisione dei worker. Coperti anche
rifiuto submit, drain e priorità dell’errore I/O. Il qualificatore include ora
header e fixture host nel proprio manifest di provenienza.

Gate numerici successivi e confronto prestazionale prima/dopo sono conservati
separatamente dalle misure storiche Q-P03. I commit distinguono ottimizzazione,
refactoring, copertura UD, provenienza ed evidenze.

Il [report del riesame](clean-code-review.json) registra correttezza PASS.
Il controllo a cinque coppie ha mediane prefill simili e CV decode elevato
in entrambe le versioni: `NEED_MORE_DATA` per una stretta non-regressione
prestazionale, senza attribuire nuovi speedup al refactoring. La qualifica
Q-P03 e i suoi campioni originali restano distinti.
