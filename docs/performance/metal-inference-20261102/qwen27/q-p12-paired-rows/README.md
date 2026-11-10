# Q-P12: due output per SIMD32, ordine originale

3 novembre2026, S1 prima del codice, dopo nuova discrepanza Q-P09.

Controllo identico input128,ctx256/cache1GiB, stesso core/cache/submit:
32lane PASS due replay8 maxdelta7.15255737e−6;
16lane FAIL index10525: got0.0408163071/ref0.0408395231,
delta2.32160091e−5 > bound2.20419752e−5. Criteri invariati.
Il caso8192 precedente falliva già. **Q-P07b REJECT**: non mantenere una
geometria che supera il solo target-only e fallisce MTP4.1.64TPS precedente
resta provvisorio finché la versione corretta non completa gate/misure.

Riesame specifico ripetuto prima della nuova riscrittura:
- HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`, Qwen3_5MLP/GDN/residui,
  frontiere di norm/softmax e cache: nessun cambio al target o agli snapshot.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/kernels/gemv.metal:gemv` e `metal/tests/gemv_parity.cpp:Cfg`
  includono LPR32/NR2/SG4/KS1. Più output perSIMD conserva la ripartizione
  delle unità, riusa l'attivazione e ammortizza la tabella. I suoi combine,
  attivazioni preparate e riduzioni diverse non vengono importati.
  Wrapper/runtime e test riesaminati, copertura modello27B Metal non presunta.
- ggml `dde879bc4f954b809892fd0d660515f42536394a`, mul_mm/dequantize,
  ops/device/test_mul_mat e qwen35.cpp: tileoutput e ownership pertinenti,
  half/tensor/altre associazioni incompatibili con la nuova parity richiesta.
- Locale: project32, deq/unità, riduzione simd_sum, confronto270 e nuovo
  controllo depth4, report drift CUDA e Q-P02/07. Non si usa la vecchia
  qualifica CUDA per giustificare un nuovo errore FP32 su Metal.

Scelta: ciascunSIMD32 calcola due output separati con due accumulatori;
`u=lane;u+=32`, FMA seriali per unità, `scale*sum` e simd_sum originali
per ogni riga. Le x si leggono una volta per due output; tabella condivisa
fra8 output/128thread. Gestire coda dispari senza leggere fuori dai pesi.
Registro aggiuntivo può peggiorare alcuni formati: micro e full request.
Q-P11 tile5–8 resta un intervento successivo, Q-P10 non attivo in produzione.

Gate prima di promozione:297 input identici conT5/coda17 bit-exact con32lane,
oracle originale e Qwen128/MTP4 indipendente entro criteri frozen. Qualifica
8192 distinta dai diagnostici256. Nessun nuovo flag, quant o commit.

S2:297 input bit-exact contro32 originale, oracoli scalari/GDN PASS.
Sul caso MTP4 critico a capacità256, due replay8 PASS con maxdelta
7.15255737e−6, e tutti11 vettori ordinary/spec bit-exact con il controllo32.
Anche i nove ordinary del controllo256 sono bit-exact con il riferimento
8192 storico. Il gate8192 reale è stato avviato appena ammesso, cache1GiB
esplicita: resta distinto dalla prova256 e dalle misure a2GiB.

S3 numerico completo: **PASS a capacità8192**, cache1GiB dichiarata,
128 input indipendenti, checkpoint/park/reload e due replay8 MTP4; maxdelta
7.15255737e−6. Nove full-vocabulary ordinari bit-exact conB0 storico,
undici ordinary/spec a256 bit-exact con controllo32. La prova8192 non viene
sostituita dal diagnostico256. Non attribuire alla versione corretta i tempi
raccolti prima con16lane: nuove misure e profilo sono separati.
