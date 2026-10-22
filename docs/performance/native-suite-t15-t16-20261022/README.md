# T15–T16: audio e composizione multimodale Gemma

Qualifica locale del 22 ottobre 2026: Ubuntu 24.04/WSL, RTX 3090 `sm_86`,
CUDA 13, EmbeddingGemma 2 revisione
`914f7f89142e33e77833254d9c9b90c3cef7303b`, profilo `bf16-native-v1`.
T15/T16 sono completati nei gate C/CUDA descritti qui. HTTP e client restano T17/T18.
Python, Transformers, PyAV ed eSpeak sono esclusivamente strumenti di sviluppo;
l'inferenza usa C, CUDA, cuBLAS e cuDNN con i decoder nativi T13.

Riesame successivo: [responsabilità, coerenza e limiti](../../architecture/native-suite-t15-t16-clean-code-review-20261022.md),
[test e hash della revisione](review-verification.json). I manifesti originali
conservano gli hash della prima qualifica; la revisione registra separatamente
le API di residenza ora esposte dai proprietari e il refactoring successivo.

## Implementazione e contratto

`ds4_gemma_audio.c/.h` possiede la torre audio: due convoluzioni 3×3 stride 2,
12 blocchi Conformer, FFN, attenzione relativa a blocchi, convoluzione causale
depthwise di ampiezza 5, output 1536 e proiezione 512. Il binding verifica tutti
i tensori, comprese le soglie di clipping. [Inventario](audio-inventory.json).
Maschere e sottocampionamento selezionano soltanto i soft token validi; il limite
è 30 secondi per traccia, fino a 750 soft token. Rimosso il precedente cap 280:
quel valore appartiene a un helper upstream, mentre `replace_audio_token`
effettivamente usa la maschera sottocampionata senza quel cap.

`ds4_gemma_multimodal.h` espone parti tipizzate ordinate: ogni parte contiene
testo oppure un handle media immutabile. Il builder inserisce separatori,
placeholder e posizioni. Image/video/audio passano al medesimo backbone T12;
un solo mean pooling, seguito da Matryoshka e L2, produce un vettore per elemento.
I batch ragged restano isolati. Il wrapper visual esistente condivide questa
implementazione e mantiene i precedenti simboli C.

La preparazione è host-only e rifiuta oltre 8192 token operativi. Il contratto C
di composizione v2 accetta segmenti espliciti con ID e intervallo temporale:
il chiamante decodifica ciascun segmento con `ds4_media_decode`, prepara un
elemento per segmento e recupera la corrispondenza mediante
`ds4_gemma_multimodal_segment`. Nessuna segmentazione o media di vettori implicita.
I timestamp dei frame restano separati dalla durata della traccia audio.
L'identità SHA-256 comprende modello/revisione/profilo, recipe di composizione,
token ordinati, chiavi di preprocess v2, task, dimensioni e segmento.

Il piano tenta prima la co-residenza. Se il budget non ammette il bound completo,
preverifica il massimo delle singole fasi e il recupero dei propri pesi residenti.
Solo dopo l'ammissione rilascia i pesi, conserva feature host registrate e carica
un encoder alla volta; il backbone viene eseguito dopo le torri. Le reservation
coprono l'intera operazione e il lock runtime impedisce compute concorrente.
Il rilascio non modifica sessioni o stato LLM. La query pubblica dei byte residenti
per ramo rende verificabile il caricamento selettivo.

## Riferimento numerico e diagnosi T00

Le soglie [T00](../native-suite-t00-20261020/tolerances.json) sono invariate.
La baseline storica `native-suite-gemma-corpus-20261020-r2` ereditava
`allow_bf16_reduced_precision_reduction=True` da PyTorch; il profilo nativo e
gli oracle di componente T12/T14 usano accumulo FP32 e quel flag disabilitato.
Il replay diagnostico con il flag storico riproduce i 16 vettori audio T00
entro 2.98e-8. Il confronto del profilo FP32 con r2 fallisce realmente:
cosine minimo 0.998347 e max-abs 0.010863 a 128 dimensioni; anche un gruppo
retrieval held-out/256 fallisce. Questi FAIL sono conservati nella qualifica v4.

Il riferimento separato `native-suite-gemma-corpus-20261022-fp32-reduction`
è prodotto dall'oracle completo ufficiale sugli stessi 104 elementi, 240 input
unici e 48 gruppi retrieval, con flag di precisione espliciti. Non sostituisce
r2 e non modifica corpus, pesi o tolleranze. [Diagnosi e hash](reference-audit.json).
Il gate finale usa questo riferimento coerente con `bf16-native-v1` e, in modo
indipendente, il replay per componenti e `AutoProcessor` + modello completo.

Correzioni numeriche attribuibili:

| Confine | Causa e correzione |
| --- | --- |
| FFT → BF16 | Magnitudine complessa SIMD NumPy: rapporto FP32 scalato e FMA esplicita; `hypotf` attraversava rari tie BF16 |
| Attenzione audio | Packing K trasposto, shift relativo, divisione per 50 e finestra causale `distance < 12` |
| Posizioni relative | Frequenze, fasi e trigonometria ai confini BF16 del modello fissato |
| GLU e RMS depthwise | Sigmoid/moltiplicazione FP32 con un cast; riduzione nel layout channel-major originale |
| Linear dopo SiLU | Preservati gli stride della matrice trasposta nell'upstream, evitando diverse somme cuBLAS nei tie |

Il backend estende il GEMM a layout esplicito già esistente al BF16 e riusa
la riduzione channel-major per RMS con scala. Nessun nuovo percorso FP16,
TF32, riduzione BF16, fast math, trascrizione o modello ausiliario.

## Gate e limiti

La [verifica completa](verification.json) registra **287 casi**, **1450 frontiere
audio** e 501 confronti di input. Le frontiere audio hanno errore numerico zero;
gli eventuali bit diversi sono il segno dello zero. Passano 128/256/512/768
dimensioni: contro il riferimento FP32 max-abs ≤5.97e-8; contro il modello
ufficiale completo max-abs ≤0.001069 e cosine ≥0.9999899, entro T00.
Le differenze del modello completo nei casi con più media dipendono dalle
GEMM raggruppate del suo piano, già qualificate in T14.

Passano **64 gruppi split/link/dimensione**: i 48 del corpus T00 e 16 nuovi
testo↔interleaved, con ground truth delle caption congelate e riferimento
indipendente dell'oracle. Batch 1/2/3 sono bit-exact; ordine e ripetizione delle
clip sono coperti. Cambiare separatamente testo, immagine, video o audio nella
composizione a quattro modalità cambia il vettore. Coperti audio sintetici in
inglese, italiano, spagnolo e tedesco, non-speech, 161/16017/480000 campioni di
silenzio, padding, e un container video con traccia audio esplicitamente scelta.
È una qualifica di regressione sintetica, non un benchmark di qualità linguistica.

Le trace dimostrano zero pesi audio per testo/image/video e zero pesi vision
per testo/audio. Nel caso capiente non viene rilasciata alcuna torre. Nel caso
con budget 2 GB il bound scende da 2,684,239,960 a 1,405,979,648 byte: prima
del backbone le torri sono a zero e l'output è bit-exact. Bound e picco misurato
sono distinti; il picco device posseduto sul corpus completo è 1,751,685,530 byte.
Il passaggio da una richiesta composta a solo testo con budget ridotto a
500 MB recupera anche le torri rimaste calde: il piano restituisce gli stessi
vettori del caso capiente e conserva soltanto il backbone. Il fallback viene
quindi valutato anche quando la nuova richiesta non contiene media.
Cancellazione a nove frontiere, budget negato, dimensioni invalide e reload
passano con tutte le risorse rilasciate alla chiusura.

[Accettazione e regressioni](acceptance.json) includono test host, primitive
CPU/CUDA, ownership/runtime/executor, sanitizer, build e campioni temporali.
I tempi escludono decode/preprocess e HTTP; non viene dichiarato uno speedup
da questi campioni. Metal/ROCm/distribuito e stress integrato con alternanza LLM
restano `NOT_VERIFIED` da questo run. Il percorso HTTP non è abilitato da T16.

## Riproduzione

Riutilizzare bundle, asset e dipendenze T06/T12/T13, senza scaricare altri pesi:

```sh
make -j4 tests/probe_gemma_visual tests/libds4_suite_cpu.so \
  tests/libds4_numeric_cuda.so CUDA_ARCH=sm_86 \
  SUITE_MEDIA_PREFIX=performance-results/native-suite-t13-t14-20261021/dependencies/installed \
  SUITE_CUDA_LIBDIR=agent/.venv/lib/python3.13/site-packages/nvidia/cu13/lib \
  CUDNN_HOME=agent/.venv/lib/python3.13/site-packages/nvidia/cudnn
agent/.venv/bin/python tools/score_suite_gemma_corpus.py \
  --prepared performance-results/native-suite-oracles-20261020 \
  --corpus tests/fixtures/native-suite-v1/corpus.json \
  --output /tmp/gemma-reference-fp32
agent/.venv/bin/python tools/qualify_gemma_multimodal.py \
  --prepared performance-results/native-suite-oracles-20261020 \
  --bundle performance-results/native-suite-t05-t06-20261020/bundles/embeddinggemma-2-bf16/bundle.ds4b \
  --text-asset performance-results/native-suite-t11-t12-20261021/gemma-f32-rope.ds4g \
  --processor-asset performance-results/native-suite-t13-t14-20261021/gemma-media.ds4p \
  --baseline /tmp/gemma-reference-fp32 --output /tmp/gemma-multimodal-qualification
```

Le tre fixture linguistiche sono già versionate, con provenienza e SHA; non
richiedono eSpeak al replay. Gli output diagnostici finali sono compressi gzip
senza perdita; JSON, vettori e manifest sono conservati. La pulizia dei vecchi
dump T13 autorizzata durante la sessione è registrata separatamente nella
receipt locale; il run T15/T16 non cancella modelli né artefatti T00.
