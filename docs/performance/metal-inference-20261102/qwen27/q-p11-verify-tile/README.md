# Q-P11: tile per la verifica MTP a cinque righe

3 novembre2026, S1 prima del codice; scelta definitiva dopo il profilo MTP4.
Bi Q-P08, Q-P10 distinto fino a decisione, nessun commit.

Audit ripetuto specificamente per5 righe:
- HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`, Qwen3_5MLP/GDN/attention:
  le righe delle sole proiezioni sono indipendenti; conv/GDN e stato causale
  restano negli operatori originali, senza cambiare maschere o cast.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/kernels/gemv.metal` NTOK e riuso di unità dequantizzate per colonne,
  `metal/tests/gemv_parity.cpp:Cfg/cfg_ok/xkind`, wrapper/runtime, dense_mm:
  specializzare il tile al verifier è pertinente; s16/preparazione attivazioni
  e GEMM half non sono equivalenti e non vengono importate. Copertura completa
  del27B Metal non attestata da questi kernel.
- ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/kernels/mul_mm.metal`, ops/device e test_mul_mat:
  dispatch per forma e riuso dei pesi; non trasferire tipi half o nuovo ordine
  di accumulo. Qwen35.cpp identifica MTP distinto dal target.
- Locale: Q-P02 aveva respinto tile8 generalizzato soprattutto su T64;
  non misura il casoT5 del verifier. Q-P09 produce4/4/5/3 token per ciclo,
  massimo target seed+4draft=5 righe, limitehard4 invariato. Riaperti
  project, wrapper, snapshot/rollback MTP, ds4_bench e report CUDA verifier
  multi-colonna. CUDA DP4A/R8 non trasferito; attivazioni FP32 originali.

Ipotesi da confrontare: tile8 solo per5–8 righe (quindi verifier5 e suffissi
prefill compatibili), tile4 per le altre forme e decode16 per una riga.
Si decodifica l'unità una volta per le cinque righe invece di due gruppi4+1;
stesso ordine FMA e riduzione per riga. Aumento registri/occupancy può
annullare il riuso: misurare, senza presumere guadagni dal tile64 precedente.

Gate:297 input isolati uguali includendoT5, oracle originale, full Qwen128
bit-exact con Bi e MTP depth4 due replay8 entro criteri frozen. MicroT3/4/5/8/64
poi richiestaMTP16 output reali e profilo distinto, senza includere le misure
profiler nella promozione. Budget2GiB o1536MiB dichiarato a seconda admission,
sempre stesso budget nel confronto appaiato e riserva2GiB intatta.

Scelta dopo profiloMTP4 a1536MiB: PASS, sotto profiler1.803367TPS, cicli
2391/2263/2205/2013ms. I soli encoder IQ4_XS con grid.y2 (tile4 per5 righe)
aggregano838.986ms contro261.130ms delle formegrid.y1; Q5_K329.397ms pergrid.y2.
Sono somme di encoder per classificare lavoro, non wall/speedup. Il costo
ripetuto su5 righe giustifica la prova tile8 confinata, senza cambiaredepth4.

S1 precisazione prima del codice: il limite reale del verifier è5 righe;
la specializzazione **ROWS=5 solo quando rows==5** evita sia la duplicazione4+1
sia tre accumulatori inutilizzati diROWS8. Strata NTOK è una costante1..8,
non richiede potenze di due. Il vecchio esperimentoT64/tile8 non misura
questa forma. ScegliereROWS5 per il primo tentativo; altre forme conservano4.
Aspettare il profilo sulla versione correttaQ-P12 prima di implementare.

Profilo aggiornato sulla versione correttaQ-P12:1.926159TPS sotto profiler
(non promozione), IQ4_XS grid.y2 aggrega831.721ms e Q5_K343.139ms: il costo
che giustificaROWS5 persiste. La riduzione16 ritirata non è parte del nuovo
confronto. Implementare oraROWS5 con lo stesso ordine per-riga32lane.

S2:297 input bit-exact con32 originale, oracoli/GDN PASS, gate128 completo
a8192 con cache1536MiB eMTP4 PASS. Tutti11 ordinary/spec vettori bit-exact
conQ-P12, stesso roundoff ordinary/spec7.15255737e−6. Nessun cambiamento
mask/ordine/precisione. Screening e profiloI/O successivi separati.
