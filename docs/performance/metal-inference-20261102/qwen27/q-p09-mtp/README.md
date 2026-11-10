# Q-P09: MTP esatto nello streaming Metal

3 novembre2026, S1 prima della prova workflow. Mandato1TPS, nessun commit.
Non sostituisce la misura target-only. Verificare token effettivamente emessi,
numero di cicli/acceptance, prefill e decode separati, costi di sidecar/snapshot.

Riesame specifico prima di usare MTP come ottimizzazione:
- HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`, Qwen3_5: lookup, proiezioni,
  causalità/GDN/residui target. Non contiene da solo l'oracle completo del
  protocollo NextN di questo sidecar; tale assenza non si trasforma in qualifica.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/models/qwen35.cpp:load_block_mtp` e grafo head/h_nextn: embedding/head
  shared opzionali e blocco distinto. Riaperta guida locale
  `qwen36-mtp-llamacpp-ds4-control-flow.md` con audit upstream storico
  `1a064ab0921238c1daa397d6f4a900ef33884de2`, common/speculative.cpp e sampling,
  state ricorrente e rollback. Il nome Qwen3.6 nel report è storico; non è
  una misura Qwen3.8/Metal né un permesso di cambiare arithmetic/sampling.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/window.cpp:Window::run/commit`, pending fold e verifier
  state, stream.cpp e wrapper/free: riuso dei pesi fra colonne e frontiera
  committed distinti dai draft. Flash dense/routed e lazy fold non sono
  automaticamente il nostro target27B Metal; non importati.
- Locale: qwen graph MTP verify/catch-up/rollback e
  `ds4_session_eval_speculative_constrained_sample`, sidecar5120 validato,
  feature/lifecycle/probe, Q-P03 MTP gate e report compatibilità Qwen3.8.
  `ds4_bench.c` usa lo stesso greedy con esclusione EOS per target e NextN;
  confrontare gli ID salvati e dichiarare che non è latenza HTTP o EOS libero.

Ipotesi: verificare più token per scansione SSD può ammortizzare le letture;
non garantisce guadagno se draft/verify/rollback o acceptance scarsa dominano.
Prima gate MTP128 due replay8 con soglie congelate; poi tre screening16 output,
contesto64, cap2GiB: target-only, draft2, draft4. Profondità/cap dichiarati,
nessun widening adattivo nuovo. Guard e admission invariati. Dump/logits fuori
finestre; tracing cicli solo screening diagnostico, promozione senza trace.
Una candidate promettente avrà warmup e5 coppie con token identici; eventuale
risorsa negata resta conservata senza abbassare riserva o bypassare guard.

S2: gate128 PASS, due replay8 greedy esatti, maxdelta ordinary/spec2.57492065e−5
entro lo stesso criterio per valore. Nove vettori rispetto al riferimento
storico restano entro2e−5+5e−5|ref|; max assoluto2.74181366e−5. Nessuna
crescita swap/guard bypassata. Ora screening16 a2GiB con timing cicli.

S3 screening16: target-only0.57, MTP2 0.88, MTP4 **1.64 token/s**. Prefill
14.39/13.32/12.57 rispettivamente. Tutti16 ID esattamente identici; MTP4
produce4/4/5/3 token per ciclo, MTP2 3/1/3/1/3/3/2. Cache2GiB, output
realmente emessi, non teacher-forced. Acceptance favorevole su questo prompt:
non generalizzare ad altri testi o HTTP. Un primo tentativo target fallisce
per directory dump mancante, poi corretto nello script e conservato FAIL.
Prima di KEEP: gate depth4 e5 coppie senza trace, controllo baseline MTP4,
profilo del verify e prove successive fino al plateau richiesto dall'utente.

La capacità host è cambiata durante la sessione: nuovo sample7.21GiB,
pressione1, swap in calo rispetto all'inizio. Admission MTP4 a2GiB rifiuta
need5.50 controbudget5.25; niente inferenza né timing valido nel tentativo.
Estensione prima dei run: usare **1536MiB** espliciti per profile/gate depth4
se il cap2GiB non entra, conservando8192/64 e riserva2GiB. Ripetizioni e
baseline a quel budget saranno separate, senza confronti a memoria nascosta.
La stessa variabilità rende la prova5GiB target una misura locale precedente,
non un default sempre ammissibile sul desktop condiviso.

Il parser CLI ammette solo GB interi: tentativi `1536MB` e `1.5GB` FAIL al
parser, senza inferenza, conservati. Non si modifica il parser condiviso
CUDA/Metal per questa prova. Harness diagnostico locale: stesso ds4_bench,
CSV6decimali e solo `engine_options.ssd_streaming_cache_bytes=1610612736`.
API già supportata e sottoposta agli stessi admission/registry/guard.
Budget effettivo1.5GiB esplicito nei record e nei log: non è un aggiramento
né una variazione nascosta. Le misure2GiB precedenti restano distinte.

## Nuova discrepanza, audit riaperto prima di cambiare kernel/dispatch

`p09-mtp4-full` FAIL al check logits ordinary/spec per valore (line304),
cache1536MiB e prompt128 indipendente. Il successo16 ID del benchmark non
qualifica la numerica completa. Nessuna soglia modificata;1.64TPS resta
provvisorio fino a risoluzione. Profondità2 passava con la nuova geometria.

Riletti per questa differenza: HF Qwen3_5 GDN/decoder/MLP al pin
`ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`, Strata gemv/LPR/NTOK e test
al pin `46c152a1310f77cd2f0ce2c7cd213962df14fe83`, ggml mul_mm e qwen35.cpp
al pin `dde879bc4f954b809892fd0d660515f42536394a`; locale ordinary decode,
verify5, snapshot/rollback e confronti Q-P03/Q-P07. LPR cambia l'associazione
delle somme: ordinary16 e verifier32 non sono bit-equivalenti. Il precedente
PASS128 target-only non spiegava questo caso4draft automaticamente.

Piano diagnostico controllato: stesso core/cache/submit/input e probe depth4,
ma wrapper Metal precedente decode1/32lane (Q-P07A bit-exact conB0), senza
modificare il codice produzione. Compilare un oggetto locale cambiando solo
l'include del wrapper; stesso shader con entrambi i kernel presenti. Questa
ablation isola la riduzione, non è un confronto performanceB0 né una correzione
già accettata. Se persiste, investigare GDN/verifier5 prima di altre modifiche.

Il primo controllo32 non arriva all'inferenza: admission need5.00GiB contro
budget4.53, poi host disponibile5.79GiB. Rifiuto conservato separato dalla
discrepanza. Per isolare la riduzione senza aspettare RAM del desktop,
confrontare entrambi i wrapper sullo stesso input128 e capacità256/cache1GiB,
tramite solo probe diagnostico locale; stesso errore/soglia/logits e nessun
claim prestazionale/qualifica8192 da questa prova. Il gate8192 resta da ripetere
sul candidato finale quando ammesso. Non si abbassa la riserva2GiB.

Controllo concluso: ordinary32/verifier32 PASS a256 con maxdelta7.15255737e−6;
ordinary16/verifier32 FAIL sullo stesso input, index10525 got0.0408163071,
ref0.0408395231,delta2.32160091e−5,bound2.20419752e−5. Causa isolata nella
nuova riduzione, non in reader/SSD/approssimazioneR8. Q-P07b ritirato;
nuova riscrittura che conservaSIMD32 nel reportQ-P12 prima del codice.

Provenienza: i sorgenti nuovi window.cpp e i pin comuni sono elencati nel
[manifest Q-P08](../q-p08-submit/sources.json). L’audit storico del controllo
MTP è nella [guida di ricerca](../../../../research/guides/qwen36-mtp-llamacpp-ds4-control-flow.md):
il pin storico non sostituisce il confronto Qwen3.8/Metal di questa campagna.
