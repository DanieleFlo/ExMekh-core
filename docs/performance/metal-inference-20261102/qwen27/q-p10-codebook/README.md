# Q-P10: codebook IQ4 costante, kernel prefill e decode

3 novembre2026, S1 prima del codice. Bi Q-P08, shader congelato.
Il profilo per formato mostra IQ4_XS dominante; la trace Q-P07b riduce il
suo tempo GPU rispetto a Q-P07. Le grandi differenze di alcuni micro wall
non si riproducono per formato nella trace del modello: non scegliere il
dispatch da soli campioni host non appaiati.

Riesame specifico prima di riscrivere il decoder:
- HF `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`, Qwen3_5MLP/GDN/proiezioni:
  nessun cambio a frontiere dtype o ordine di dot. Il codebook è nel formato
  GGUF, non una nuova quantizzazione del modello.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`, nuovo fetch
  `metal/kernels/quant.metal:QUANT_CTAB/load_tab/deq`,
  `metal/src/engine/window.cpp:metal_ctab` e wrapper/runtime/test gemv:
  legge griglie/sign da constant invece di staging threadgroup, con gli
  stessi valori. Il claim M5 Max non si trasferisce come misura al nostro Mac.
  Coverage completa Qwen27 Metal assente; non importare attivazioni preparate.
- ggml `dde879bc4f954b809892fd0d660515f42536394a`, nuovo fetch
  `ggml/src/ggml-metal/kernels/dequantize.h:dequantize_iq4_nl/iq4_xs`,
  mul_mm, ops/device e test_mul_mat: lookup costante di16 valori, scale e
  nibble layout. Il suo ordine di multiply e i tipi reg non vengono copiati.
- Locale: deq20/deq23, fmt.TAB, load_tab, kernel project/gather, codebook
  condiviso e oracle scalare. Tutti i16 valori interi sono rappresentabili
  esattamente in half e FP32. Conversione half->float della tabella staged
  e int8->float della stessa tabella devono dare gli stessi bit; verificare.

Ipotesi: rimuovere staging256 coppie half2 e relativa barriera per IQ4_NL/XS,
leggendo16 valori costanti nei decoder. Conservare scale/unità/FMA/riduzioni.
Solo shader candidato negli artefatti fino al gate; nessun flag prodotto.
Altri IQ conservano staging e barrier. Confronti isolati270 bit-exact, scalar
oracle, micro poi Qwen128 indipendente bit-exact con Bi, e nuovo profilo.
Se peggiora, ritirare lo shader candidato senza accumulare varianti inutili.

S2:270 input e nove logits/payload128 bit-exact conBi; scalar/gdn PASS.
Micro wall misti, prefill IQ4_XS+3.1%tempo sulla up, +1.3%down, +4.8%qkv,
IQ4_NL+6.4%tempo; alcuni decode hanno variazioni molto grandi non causali.
Misura4GiB non ammessa (need5.42,budget5.25), nessuna qualifica performance.
La scoperta numerica16lane riguardaBi, non viene attribuita al lookup
costante bit-exact. La prova full request deve usare la versione corretta32
prima di decidere; shaderQ-P10 resta esclusivamente diagnostico e non attivo.

Screening sulla versione correttaQ-P12, stesso binario1536MiB/8192/64 e16
output: baseline prefill13.994011/decode1.940442, constant13.804174/1.972207.
Richiesta sintetica64+16 circa12.82->12.75s (−0.5%), segnale non consolidato.
**PENDING_BUNDLE**, non promuovere né scartare per soglia2%; isolare il costo
prefill e ripetere appaiato se resta candidato. Confronto controllato senza
profiler, dati precedenti con16lane esclusi dalla qualifica finale.

S4 nuovo confronto bundle dopo correzioneSIMD32 e quotaP17: riletti
Strata46c152a... quant.metal QUANT_CTAB e gemv.metal/test gemv_parity,
ggml dde879bc... dequantize.h IQ4 e wrapper/device, HFca8c871... norm/MLP.
I16 valori int8 IQ4 sono rappresentabili esattamente inhalf/F32: cambia solo
l’accesso al codebook. Il candidate P17ctab combina la stessa trasformazione
privata già auditata conROWS5 e quota lazy corretta; non viene attivato prima
di297 isolati, full128/MTP4 e confronto appaiato. La cache del modello e la
struttura del verifier restano identiche aP17.

Consolidamento:297 e11 vettori PASS anche nel bundleP17. Warmup escluso e
5 blocchi appaiati, no profiler: prefill geometricamente−1.55%
(CI bootstrap locale−2.23..−0.58), decode mediano1.985740 vs2.036763,
request mediano12.737075 vs12.509941s. Due outlier decode1.405/1.307 sono
conservati; niente attribuzione causale di tutta la dispersione al codebook.
**REJECT nel bundle corrente**, senza soglia minima2%: il costo prefill è
riproducibilmente peggiore e nessun guadagno request è confermato. Shader
constant resta nei soli artefatti diagnostici; niente flag/dead variant prodotto.
