# I-P05 — Anticipo limitato dei gruppi DiT

Il 5 novembre riparte il worktree `1340`, HEAD `4fef55f`, con le modifiche
I-P02/I-P04 e il successivo riesame. Gli artefatti pesanti sono letti dal
checkout originale. Build, binari e risultati di questa campagna sono locali
in `build/metal-perf-image-20261105`. Modello attivo Qwen Image 2.1 Turbo BF16,
revisione `fc13ba1986fcb905854dc8de2f52eee395f49b5b`, M5 Pro 24 GiB.

## Baseline e confronto prima del codice

La richiesta automatica testo 256×256, 8 passi, CFG1, seed20261020 supera
il gate indipendente in 110.643773 s, encoder12.608993/DiT87.656022/VAE10.378569.
Minimo host monitorato7478198272B e crescita swap0. La quota DiT è
2760798176B. La finestra diagnostica di 8 s, ogni 10 ms, contiene716campioni
del thread principale, con434nella copia del gruppo. Questo campione è
separato dalle successive coppie prestazionali.

Fonti verificate nuovamente tramite hash nel manifesto `sources.json`.

- Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
  `src/diffusers/models/transformers/transformer_qwenimage21.py`,
  `_qwenimage21_prepare_qkv` e `QwenImage21Transformer2DModel.forward`.
  Q/K/V e i 32 blocchi hanno ordine deterministico. Il prossimo gruppo
  si conosce durante il calcolo del blocco corrente. Cache prefix, dtype,
  maschere e ordine delle operazioni restano quelli congelati.
- Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
  `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`.
  Encoder testo/vision e patch Conv3d sono fasi separate. Il candidato
  anticipa soltanto i dati dei blocchi DiT.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp`, `stream.h`, `window.cpp`, `arena_budget.h`,
  `metal/src/mtl_rt.mm`, `metal/tests/dense_mm_parity.cpp`.
  Il prompt anticipa gli esperti del layer successivo, mentre il decode
  usa predizioni e slot bounded. Si trasferisce la conoscenza del prossimo
  gruppo. Router, FP16/MMA e arena degli esperti hanno contratti diversi dal
  DiT BF16 e dalla sua cache. Window usa anche consigli `MADV_WILLNEED`.
- llama.cpp/ggml `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/llama-mmap.cpp`, kernel/wrapper/capability/test elencati nel manifesto.
  Il wrapper mmap distingue gli intervalli lazy e anticipa soltanto il
  complemento tramite `POSIX_MADV_WILLNEED`. Il consiglio sulle pagine lascia
  al kernel la gestione del file cache e non garantisce residenza.
- Ricerca locale `q-p15-upstream-ssd`, `weight-transport-20261104`,
  `continuous-cache-20261104`, `ds4_bundle.c:hint_worker`,
  `ds4_metal.m:ds4_gpu_stream_expert_readahead_range`,
  `ds4.c:metal_graph_stream_readahead_range_impl`.
  Il bundle usa già `F_RDADVISE`/`posix_fadvise` con chunk1MiB e limite16MiB.
  La copia parallela peggiorava il modello completo. Un hint per gruppo
  mantiene distinti la preparazione delle pagine e la copia dei pesi.

## Candidato e piano delle prove

Un hint sincrono leggero presenta al sistema al massimo16MiB dei tensori
selezionati del prossimo gruppo dopo la proiezione Q del blocco corrente.
Il sistema può leggere le pagine mentre proseguono K/V, attention e MLP.
Il gruppo anticipato non acquisisce storage Numeric né modifica LRU, visite,
hit o residenza. Il limite riusa quello già presente nel bundle. Il percorso
DiT Apple abilita il tentativo. L'upload autorevole continua a verificare il
file e pubblica i pesi soltanto dopo la barriera di disponibilità.

Prima test isolati su intervalli selezionati, limite cumulativo, gruppo già
residente, errore del consiglio I/O e assenza di allocazioni. Poi gate completi
testo/RGB/RGBA sugli oracle congelati. Screening a ceiling comune2GiB,
una grande istanza per volta, riserva2GiB e monitor swap. Una conferma usa
warmup0 e cinque coppie alternate1..5, se lo screening giustifica la prova.
Le quote e i caricamenti devono coincidere per attribuire un delta all'hint.
Un risultato indistinguibile dal rumore richiede una decisione esplicita.

## Prima verifica del candidato

Test degli intervalli con syscall simulata PASS. Il test verifica anche che
il consiglio fallito lasci invariati storage, stato residente e contatori LRU.
La suite bundle passa12/12 con il server locale di fixture. Il primo tentativo
nel sandbox impediva il bind locale ed è conservato come errore ambientale.

Il gate automatico del candidato passa90.084671s con11artefatti identici alla
baseline e crescita swap0, minimo host5779750912B. In quella richiesta la
quota DiT è6850752480B e i caricamenti159, contro2760798176B/229della
baseline automatica. I tempi automatici descrivono condizioni RAM diverse.
Lo screening a ceiling2GiB deve quindi fissare lo stesso lavoro di trasporto
prima di attribuire un guadagno all'anticipo delle pagine.

## Screening e conferma predefinita

Il warmup cap2 misura87.199755/92.931843s ed è escluso dallo screening.
La coppia1 misura98.247922/91.438093s, con DiT74.881519/68.442633s.
Plan1879048192B, quota DiT1496875008B,250caricamenti e109GBdi payload DiT
coincidono. Tutti22artefatti della coppia sono esatti e swap growth0.
Il segnale giustifica una conferma separata. Prima del suo avvio vengono
fissati nuovo warmup0, cinque coppie1..5 e ordine AB/BA alternato fra coppie.
Ceiling2GiB, fixture, compilazione, riserva e criteri restano quelli descritti
sopra. La coppia di screening è conservata separatamente dalla conferma.

## Estensione della forma prevista dopo il confronto

La ricetta `alpha-512` già presente in `tools/qualify_qwen_image_pipeline.py`
viene preparata a512×512, seed20261021, con il suo prompt e otto passi.
Prima si confrontano baseline e candidato nativi sul medesimo file fixture.
Il timeout massimo è600s per richiesta, fissato prima del run per il maggiore
costo spaziale. Restano riserva2GiB, monitor e una sola istanza.
Il confronto binario verifica il percorso sulle forme più grandi.
L'oracle indipendente completo a512 è una qualifica distinta, perché quelli
già congelati coprono256. La CPU resta riservata ai replay piccoli.

## Cinque coppie e contatore globale dello swap

Tutti110artefatti delle cinque coppie sono bit-exact e tutti i gate
indipendenti passano. Quote, caricamenti, hit, eviction e byte trasferiti
coincidono fra i due binari in ogni coppia. Bootstrap appaiato20000resample,
seed20261105. Distribuzioni complete e dati grezzi in `confirmation.json`.

| Fase | Mediana baseline s | Mediana candidato s | Delta appaiato geometrico, IC95 |
| --- | ---: | ---: | --- |
| Encoder | 12.765 | 12.870 | −1.379% [−4.956,+1.852] |
| DiT | 72.228 | 56.877 | −11.601% [−21.361,−1.416] |
| VAE | 10.471 | 10.473 | −0.032% [−0.111,+0.052] |
| Operazione | 95.620 | 80.356 | −8.838% [−16.093,−1.314] |
| Processo | 108.779 | 93.570 | −7.520% [−13.899,−1.134] |

CV operazione8.145/9.065%, DiT10.857/12.594%. La variabilità supera il5%
e limita la precisione del guadagno locale. Sono conservate anche la coppia
sfavorevole e le richieste calde. Il guadagno riguarda il ceiling2GiB misurato.

La prima analisi segnala un incremento globale di swap524288B nella richiesta
candidato5. Gli altri nove campioni misurati hanno crescita0. Il monitor
congelato conserva riserva2GiB e soglia256MiB per crescita swap. Tutti i run
rispettano quel monitor. `strict_zero_swap=false` conserva separatamente
l'assenza di una qualifica completamente priva di incrementi globali.

Prima della decisione riletto XNU `f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`,
`bsd/kern/kern_sysctl.c:sysctl_swapusage`, e SDK25F70 `sys/sysctl.h`.
La statistica deriva da `macx_swapinfo` globale e da totale meno disponibile.
Al primo incremento il processo occupa7193152B, resident17072128B e la
disponibilità host è10159177728B, pressione1. L'incremento precede il
caricamento dei gruppi DiT. Strata arena_budget/window e ggml llama-mmap ai
pin del manifesto usano capacità host complessive, senza attribuzione dello
swap al singolo gruppo. La policy Numeric mantiene il file-cache floor XNU.

La decisione mantiene tutti i campioni e prosegue con i gate finali già
previsti, con riserva e monitor esistenti. L'attribuzione di quel piccolo
incremento globale rimane aperta. I nuovi gate registrano ogni contatore.

## Esito finale

`KEEP` locale per I-P05. Il codice presenta al kernel i primi16MiB selezionati
del prossimo gruppo DiT. L'eseguibile finale coincide byte per byte con il
candidato delle cinque coppie. La cache normale continua a usare il budget
dinamico per richiesta. Il ceiling2GiB appartiene al confronto controllato.

Test del bundle12/12 e hint con ASan/UBSan PASS. Build prodotto, fondazioni
Metal, Qwen27 residente/streamed, Flash SSD, executor, risorse, lifetime,
input/output/graph Image e HTTP locale PASS. Numeric CPU47 con13skip PASS.
Numeric Metal47 con3skip conserva le due asserzioni sin/cos già riprodotte
sul controllo. La regressione del padding largo e il recupero del cap Metal
passano. Gemma480/480frontiere sono bit-exact e i vettori128passano i gate.

I gate finali RGB100.595604s e RGBA91.393763s passano gli oracle congelati.
Tutti24artefatti nativi coincidono con i controlli precedenti. Minimo host
monitorato5275107328B e crescita swap0 in entrambi. La finestra successiva
del DiT contiene658campioni principali,395nelle copie,237nelle proiezioni
comprensive di attesa GPU e1nell'hint. È una richiesta RGB profilata con
quota automatica, separata dalle cinque coppie senza profiler.

A512×512 la baseline termina236.926321s e il candidato228.141556s.
Gli11artefatti nativi, tutti gli stati latenti e il PNG, sono identici.
Dimensioni PNG512×512 e dimensione RGBA verificate. I picchi rientrano nei
piani ammessi. Minimi host5602344960/5851037696B e crescita swap0.
Queste due richieste usano RAM automatica e costituiscono un controllo
funzionale della forma più grande. L'oracle indipendente512rimane aperto.
I tempi512restano diagnostici, distinti dalla conferma appaiata256.

Il profilo [GitHub indicato per YASAKANI](https://github.com/morphicode-jp)
è stato ricontrollato il5novembre tramite ricerca dei repository pubblici.
La risposta contiene soltanto `f32-patch-gemma`. Restano disponibili gli
spunti delle implementazioni pubbliche fissate nel manifesto.

`final-validation.json`, `large512.json` e `final-artifacts.json` conservano
esiti e hash. Rimangono da qualificare l'oracle512, altre recipe e rapporti
di forma e la distinta campagna prestazionale EmbeddingGemma. Le future
prove di preparazione diretta dei buffer richiedono un nuovo ciclo sul
lifetime e sul picco di due gruppi. F08 conserva la sospensione precedente.
Continuazione completata senza commit.

Il successivo riesame richiesto dall'utente autorizza l'integrazione in
commit distinti. Limiti condivisi, tabella CPU con callback nominati e costi
del planner espliciti conservano il comportamento qualificato. Il
[riesame](../../../../architecture/metal-image-prefetch-clean-code-review-20261105.md)
e `review-verification.json` distinguono i nuovi gate dai binari misurati.
