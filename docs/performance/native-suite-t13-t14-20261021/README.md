# T13–T14: media nativi ed embedding vision/video

Qualifica locale: Ubuntu 24.04/WSL, RTX 3090 `sm_86`, CUDA 13,
`google/embeddinggemma-2` revisione
`914f7f89142e33e77833254d9c9b90c3cef7303b`, profilo `bf16-native-v1`.
Le API C gestiscono preprocessing di immagini/video/audio e inferenza
image/video con il backbone T12. L'encoder audio resta T15; HTTP resta T17.
Python, Transformers e PyAV servono esclusivamente per compilazione offline,
fixture e oracle. Nessun processo esterno esegue l'inferenza nativa.

Revisione del 22 ottobre: [responsabilità e correzioni](../../architecture/native-suite-t13-t14-clean-code-review-20261022.md),
[regressioni e hash](review-verification.json). Il campionamento raw/container
ha un piano comune, il processor verifica il totale visual/audio e il backbone
possiede la validazione dei placeholder. Il testo espanso usa memoria misurata.
I risultati originali qui sotto restano la baseline; il replay della revisione
conserva input, hidden e vettori byte per byte.

## Responsabilità e workflow

1. `ds4_media` copia l'input, separa originali e dati decodificati e applica la
   recipe esplicita di segmento/campionamento/traccia audio. JPEG/PNG riusano
   Iris e la gestione EXIF; i container usano libav.
2. `ds4_gemma_processor` stima token e memoria e produce `ds4_gemma_media`
   immutabili: patch FP32, posizioni, maschere, timestamp e feature audio.
   La cache LRU è opzionale, limitata in byte e a 64 elementi. La chiave include
   asset/configurazione del modello, profilo, hash originali/decodificati,
   dimensioni, budget, campionamento e segmento.
3. `ds4_gemma_visual_prepare` mantiene l'ordine dei media, espande i placeholder,
   applica il task e controlla conteggi e contesto prima di caricare pesi.
   Gli input trattengono i media: decoder, cache e processor possono essere
   rilasciati prima dell'inferenza.
4. `ds4_gemma_visual_embed` prenota l'intera operazione e visita un media alla
   volta. I frame dello stesso video conservano il batching eager; ogni elemento
   ragged ha il proprio backbone. Vision restituisce feature BF16 detached;
   il backbone le inserisce dopo la scalatura del testo e prima della PLE,
   tramite una propria API privata, senza esporre il layout dell'input.
5. Output host e feature hanno ownership esplicita. I pesi text/vision sono
   caricati per gruppi, riutilizzabili e rilasciabili separatamente. Il ramo audio
   non viene caricato. L'adapter testuale continua a caricare solo testo.

Il decode/preprocess non entra nel runtime dei modelli: errori e budget host
insufficienti non mutano sessioni LLM. Le chiamate sullo stesso handle vanno
serializzate; le callback durante un run ricevono `BUSY` se tentano di
rilasciare input, adapter o pesi attivi.

## Recipe, limiti e dipendenze

Il processor congelato usa patch 16, pooling 3, bicubic uint8, RGB con alpha
scartato e rescale FP32 1/255; non applica crop o normalizzazione aggiuntivi.
Budget image/video di default: 280/140 soft token; budget espliciti supportati:
70, 140, 280, 560, 1120. Il contesto operativo completo è 8192 token;
superarlo produce errore, senza troncamento implicito.

La recipe video del processor è 1 fps, massimo 32 frame, overflow uniforme.
Va richiesta con `ds4_gemma_processor_video_recipe` e passata al decoder:
il decoder generico non la applica di nascosto. Il campionamento replica gli
indici del processor e il suo fps medio. I timestamp di presentazione reali
restano distinti da quelli del processor, anche nei container VFR. Il decode
esegue una passata di censimento e una di conversione dei soli frame scelti;
mantiene un packet e un frame applicativi, senza coda dell'intero filmato.

Default: 64 MiB encoded, 512 MiB di storage trattenuto totale, 4 megapixel per
frame, 16384 frame sorgente, 30 secondi audio. Le allocazioni applicative sono
registrate; i workspace delle librerie hanno una prenotazione conservativa
separata, con codec a un thread, dimensioni e input limitati. La prenotazione
non è una misura delle singole allocazioni interne di libav.

Audio: prima traccia oppure esclusione esplicita per i video; media uniforme
dei canali FP32, libsoxr HQ mono 16 kHz, lunghezza realmente prodotta da soxr
come in T00. Nessuno zero aggiunto per arrotondare a `ceil`. I segmenti usano
il clock della traccia; discontinuità oltre la precisione del timestamp sono
rifiutate. Le feature riproducono Hann periodico, FFT 512, hop 160, finestra
320, banca mel FP64 e log con floor 0.001. Input oltre il limite del processor
sono rifiutati prima del backbone.

Dipendenze scelte: FFmpeg **7.1.1**, librerie C condivise LGPL, e libsoxr
**0.1.3-4build3** Ubuntu amd64. La build esclude programmi, rete, GPL/nonfree,
filtri e autodetect di librerie esterne. AVIO accetta soltanto memoria e nega
apertura di risorse esterne. Codec e hash sono in `provenance.json`; licenze e
attribuzioni in `third_party/ffmpeg`, `third_party/soxr` e
`third_party/transformers`. La selezione deriva dalle API native di
[FFmpeg](https://ffmpeg.org/about.html) e
[libsoxr](https://github.com/chirlu/soxr/blob/master/src/soxr.h).

Le fixture codec qualificano JPEG/PNG, FFV1 RGB/YUV e PCM WAV/Matroska.
Le conversioni YUV rispettano matrice BT.601/709/2020 NCL e range full/limited,
con il profilo swscale congelato. PyAV 16 crea i container; il suo FFmpeg 8 ha
arrotondamenti SIMD differenti e non è il golden dei pixel YUV. Il riferimento
di quel confine usa l'API pubblica di swscale 7.1.1, separata dal wrapper DS4.
HDR PQ/HLG e BT.2020 CL sono rifiutati: manca una recipe di tone mapping.
Gli altri codec compilati non hanno ancora fixture pixel dedicate.

## Evidenze e correzioni numeriche

`verification.json` conserva il rapporto completo dei 29 casi: 16 immagini,
8 video e 5 diagnostici di ordine, composizione e budget. **93 input** pixel,
posizioni e token sono esatti; **3479 frontiere** vision/backbone hanno errore
numerico zero nell'oracle per componenti. Le differenze di bit residue in
alcune norm riguardano il segno dello zero.
Il gate vision comprende i budget predefiniti e il diagnostico a 70; il gate
host di patch/posizioni comprende tutti e cinque i budget e sei aspect ratio.
Una qualifica esaustiva degli encoder su budget/aspect estremi resta da estendere.

I vettori passano tutte le dimensioni 128/256/512/768, contro il replay per
componenti, `AutoProcessor` + modello ufficiale completo e le baseline T00.
Nel modello ufficiale i due casi con più immagini usano una GEMM diversa dal
piano seriale DS4: il confronto resta entro T00, con errore massimo 0.001102
e cosine minimo 0.999977. Le **32 combinazioni** split/link/dimensione del
retrieval testo↔immagine e testo↔video passano. Cambi di contenuto e ordine
di immagini/frame modificano i vettori; batch e singolo sono bit-exact.
Cancellazione a sei frontiere, budget negato, dimensioni invalide e
rilascio/reload dei pesi passano senza risorse residue.

Le prove intermedie fallite restano in
`performance-results/native-suite-t13-t14-20261021/`. Le soglie T00 non sono
state allargate. Le cause isolate e corrette sono:

| Confine | Correzione |
| --- | --- |
| RMS delle head vision 64 | Somma lane/lane+32 prima della riduzione warp, come `Reduce.cuh` fissato |
| Pooling | Budget completo e matrice one-hot trasposta nella GEMM FP32; BF16 prima e dopo la scalatura |
| Attention video | Batch e head uniti nelle GEMM dense ammesse; separare `P@V` cambiava tie BF16 |
| RoPE di sequenze lunghe | Fasi FP32 e sin/cos nel backend numerico; libm host differiva in rari tie BF16 |

Il piano rimane BF16/FP32, senza FP16, quantizzazione, riduzione dei frame o
approssimazione dell'attenzione. Cache e residenza separata seguono i principi
di budget già adottati in T11; nessuna policy per esperti MoE è stata importata
in questo encoder denso.

`acceptance.json` riporta regressioni, picchi, campioni di latenza ed esiti
dei replay finali. Le misure calde comprendono vision + backbone su media già
preprocessati, dopo warmup; non sono tempi HTTP o decode. Non viene promosso
uno speedup da tempi di processo, dump o campioni instabili.
Immagine singola: mediana calda 115.13 ms, cinque campioni, CV 0.81%, primo
forward 549.86 ms. Video di quattro frame: mediane diagnostiche 233.21 e
228.51 ms in due run; CV 6.12% e 5.86%, quindi `NEED_MORE_DATA` per una
promozione prestazionale. Il picco device posseduto sul corpus è 1,140,448,450
byte; decode e pesi mmap non sono sommati arbitrariamente a questa misura.

## Riproduzione

Preparare oracle e bundle T00/T06, poi usare directory di output nuove:

```sh
python3 tools/build_suite_media_dependencies.py
uv pip install --python agent/.venv/bin/python --no-deps \
  --target performance-results/native-suite-t13-t14-20261021/development av==16.0.1
agent/.venv/bin/python tools/compile_gemma_media.py \
  --prepared performance-results/native-suite-oracles-20261020 \
  --output performance-results/native-suite-t13-t14-20261021/gemma-media.ds4p
DS4_MEDIA_CODEC_PREFIX=build/suite-media \
make -j4 CUDA_ARCH=sm_86 SUITE_MEDIA_PREFIX=build/suite-media \
  CUDNN_HOME=agent/.venv/lib/python3.13/site-packages/nvidia/cudnn \
  SUITE_CUDA_LIBDIR=agent/.venv/lib/python3.13/site-packages/nvidia/cu13/lib \
  SUITE_PYTHON=agent/.venv/bin/python \
  tests/probe_gemma_visual tests/probe_native_suite \
  tests/libds4_suite_cpu.so test-gemma-media test-suite-numeric-cuda

agent/.venv/bin/python tools/qualify_gemma_visual.py \
  --prepared performance-results/native-suite-oracles-20261020 \
  --bundle /path/to/embeddinggemma-2-bf16/bundle.ds4b \
  --text-asset /path/to/gemma.ds4g \
  --processor-asset performance-results/native-suite-t13-t14-20261021/gemma-media.ds4p \
  --output /tmp/gemma-visual-qualification
```

Il builder supporta `--offline` con archivi già presenti e verificati. La build
isolata è stata ripetuta con un secondo prefix. Le dipendenze media non entrano
nel link dei quattro eseguibili LLM legacy, ricompilati nel controllo finale.
Metal e TP/distribuito non sono qualificati da questa esecuzione Linux.

Sviluppi successivi: encoder audio T15, composizione completa T16 e routing
T17; per T19, streaming/prefetch per gruppi e accorpamento GPU dei soft token
senza il passaggio host del pooling. Codec aggiuntivi, HDR, rotazioni/SAR dei
container e altri backend richiedono fixture dedicate; non si estende la
qualifica dei dati RGB congelati a quei formati.
