# T08 — ripresa della qualifica, 6 novembre 2026

Stato: `IN_PROGRESS`. Avvio della campagna autorizzato senza commit, su
`1cae644` nel worktree1340. Apple M5 Pro, 24GiB; una sola istanza grande
per volta. Il monitor conserva riserva host2GiB, pressione critica4 e
soglia di crescita swap256MiB. Un rifiuto di memoria non qualifica il caso.

## Aggiornamento: riserva zero su richiesta dell'utente

La nuova istruzione del6novembre elimina il margine host2GiB per le
successive prove. `DS4_METAL_HOST_RESERVE_BYTES=0` seleziona la capacità
dinamica senza sottrazione fissa; in assenza dell'override resta disponibile
il profilo storico2GiB. LLM, suite e monitor devono leggere la stessa scelta.
Restano i costi reali di grafo/restore, il limite Metal, il floor XNU per
gli upload mmap, la pressione critica e gli errori di allocazione.
Non si alloca RAM inutilizzata quando tutti i pesi utili sono già residenti.

Confronto specifico prima del codice, sugli snapshot locali fissati:
Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
`src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`, mantiene stato
conv/PLE/attention indipendente dalla quota della cache pesi. Diffusers
`aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`,
`models/transformers/transformer_qwenimage21.py`, conserva ordine/dtype
dei32blocchi; il margine appartiene all'ammissione, non all'aritmetica.
Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
`metal/src/engine/arena_budget.h:arena_budget/refill` e `stream.cpp`,
separa capacità corrente e costi futuri; le sue percentuali e il reclaim
purgeable non sono trasferiti. llama.cpp
`dde879bc4f954b809892fd0d660515f42536394a`, `src/llama-mmap.cpp`,
kernel/wrapper Metal del manifest I-P05 e `tests/test-backend-ops.cpp`,
non fornisce equivalenza dei payload DS4 né una quota da copiare.
Riletti policy condivisa `ds4_metal_memory.h`, sampler `ds4_metal_device.h`,
ammissione/rebalance Qwen, suite Numeric e test pubblici di budget.
I report Q-P25 e I-P04 documentano già costi di restore e floor file-cache.
Scelta: override di risorsa unico, senza cambio di kernel, chunk, maschere,
sampling o numerica. Prima policy CPU e lifecycle, poi modelli in sequenza.

Il vecchio stress Flash termina senza report del controller:27cicli
misurati PASS più warmup nel JSONL, poi `APIConnectionError` nel client.
Al controllo della ripresa nessun server/controller risulta attivo.
Causa dell'uscita non determinata; non attribuita a RAM né promossa a PASS.
La campagna senza riserva riparte come prova distinta.

Policy CPU e factory PASS anche a margine0; il primo nuovo test della
factory era stato inserito prima del reset del guasto simulato e viene
corretto senza modificare il prodotto. Build engine/server/suite/probe,
planner Image, resources/executor e49test Python PASS. I25test dei
launcher/fondazioni passano separatamente e in parte si sovrappongono.
Il primo lifecycle sandbox non accede a Metal; la ripetizione esterna
con riserva0 passa entrambe le famiglie, inclusi fault di restore e park.

Flash32input+8output PASS11.159597s, tutti i logits e payload conservati
esattamente dopo restore, minimo host9831940096B e crescita swap0.
Il nuovo stress automatico parte con cache9.41GiB e log esplicito della
riserva0. Il binario iniziale è conservato come
`no-reserve/server-stress-before-discovery`: la discovery di quel binario
dichiara ancora il vecchio valore2GiB, mentre budget e monitor usano0.
La discovery viene corretta nel prodotto successivo senza cambiare
alcun dispatch o costo; le prossime prove devono verificare anche quel
campo reale. Non si modifica la catalog.json già raccolta.

### Stress Flash senza riserva: PASS

50cicli misurati dopo un warmup,256×256/8passi, cinque input embedding
testo/image/audio/video/interleaved e dimensioni128/256/512/768 alternate.
Cache del LLM riusata dopo gli adapter, codice della conversazione
conservato, PNG e vettori deterministici sui casi ripetuti, nessuna
reservation residua e owner500stabile dopo ogni operazione.
Durata10394.076880s, server exit0, cleanup_forced vuoto.

Minimo host3952607232B, massimo footprint12143032776B, pressione massima2.
Crescita swap globale252832645B(241.12MiB), sotto la soglia256MiB congelata.
Il monitor ha riserva0; il risultato non qualifica assenza assoluta di swap.
Mediane Image92.449s, embedding57.748s, ciclo202.846s;
CV3.92%,1.53%,2.03%. Sono misure di questa prova di lifetime, senza
confronto appaiato per attribuire uno speedup al margine eliminato.
Restano distinti F08 e corpus qualità estesi.

### Qwen27 senza riserva

Gate32input+8output PASS8.306s, payload e logits esatti dopo restore,
minimo host2019459072B e crescita swap0. La nuova F08 conserva gli stessi
assert e prompt del test, cache automatica/riserva0: crea il file reale,
esce e riapre la skill tre volte. Il controller viene fermato dopo625.085s,
server exit0, nessun cleanup forzato, minimo host1457209344B e crescita
swap0. Esito `NOT_VERIFIED`. Non è un rifiuto di ammissione; la diagnosi
numerica Flash non identifica automaticamente la causa Qwen.

Segue un nuovo stress Qwen50cicli+warmup,256px e riserva0, con il binario
che dichiara anche nella discovery il margine effettivo. Il budget globale
rimane14400s e i criteri di stato/determinismo restano quelli Flash.

Lo stress Qwen si ferma dopo44.739303s con `SKIP_RESOURCE_LIMIT` durante
il warmup, prima di un ciclo completo. La crescita swap globale è
1809254973B(1725.44MiB), in due campioni successivi oltre256MiB.
Pressione massima2, minimo host2060713984B, server exit0, cleanup_forced
vuoto. Il LLM aveva cache14.28GiB e la richiesta Image era iniziata.
Il contatore è globale: non attribuisce da solo tutto lo swap al processo
o a un kernel. La qualifica resta aperta e non si alza la soglia per
ottenere PASS. Prima di una modifica del budget/park serve indagare quel
picco, con nuovo confronto delle fonti; questa campagna passa alla
verifica Image in un processo separato.

### Image: ammissione e profilo indipendente512

Il planner ammette56/56forme:8diagnostiche e48T00, senza inferenza e
con `noise_kind=admission_zero`. Le geometrie originali e il reference
resolution1024T00 sono conservati. Il massimo costo ammesso non prova
che una successiva inferenza completi senza pressione/swap.

Per il nuovo caso indipendente512si congela prima del run il profilo:
fixture `alpha-512` già provata nativamente, encoder Transformers CPU
offloaded (solo testo), DiT Diffusers MPS offloaded e VAE MPS offloaded.
Tutti gli input e le soglie originali restano invariati; i dispositivi
saranno dichiarati nel risultato, senza promuoverli a un oracle CPU puro.
Il DiT CPU grande non viene avviato sul Mac. MPS usa frazione massima1
nel proprio processo isolato; il monitor esterno mantiene riserva0,
pressione critica4 e soglia swap256MiB. Deadline600s per componente,
senza retry automatici. Native600s, una sola istanza grande per volta.

Riletti `run_qwen_image_pipeline_oracle.py` e `suite_offloaded_oracle.py`:
pesi caricati per modulo, barriera/svuotamento cache dopo i moduli MPS,
SDPA math con asse completo e tiling query bounded. Transformer e VAE
Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e` e encoder Qwen3VL
Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7` sono quelli del
bundle e del manifest I-P05; nessun aggiornamento delle dipendenze.
È un nuovo profilo di qualifica indipendente a device espliciti, senza
cambio di kernel del prodotto né sostituzione dei golden T00.

Il qualificatore espone anche `--oracle-encoder-device`, simmetrico
all'override VAE esistente, per dichiarare correttamente il profilo misto
CPU/MPS/MPS. In assenza dell'opzione eredita `--oracle-device`, come prima.
Non cambia i confronti né le soglie; il test di rigetto del conditioning
verifica anche che i dispositivi effettivi siano registrati separatamente.

Il caso512termina `NUMERIC_PASS` con il profilo dichiarato:

| Verifica | Esito |
| --- | --- |
| Conditioning CPU | Bit-exact,81920elementi |
| Otto latenti | Massimo RMSE0.010358629, relativoL20.007217926 |
| RGBA | RMSE0.005003053, SSIM RGB0.997165813, alpha MAE0.000439032 |
| Confronto nativo precedente |11/11artefatti identici, inclusi PNG e latenti |
| Risorse | Crescita swap0nelle quattro esecuzioni |

Tempo nativo224.864140s; monitor completo236.870472s, minimo host5395857408B.
Oracle encoder CPU57.810251s, DiT MPS34.392039s, VAE MPS6.077977s.
Il PNG nativo è stato ispezionato: cerchio rosso, bordi con alone; il canale
alpha è presente, estremi0/255e angoli trasparenti. Questa ispezione di un
caso non chiude la revisione umana di tutte le recipe. Il confronto numerico
non è un oracle CPU puro e non sostituisce i golden del corpus T00.

## Chiusura della campagna locale senza commit

Il controllo Gemma senza riserva passa in12.199511s:480/480frontiere
BF16/probabilità bit-exact e vettori128entro le soglie indipendenti
(massimo errore5.96e-8). Minimo host16380723200B, crescita swap0.
I test finali Python passano56/56in13.212s; i25test dei launcher e delle
fondazioni sono separati e parzialmente sovrapposti. Build, Ruff, policy,
factory, planner, risorse, executor, HTTP e lifecycle Metal passano.
Non sono stati modificati shader, aritmetica, pesi o soglie numeriche.
Le due asserzioni sin/cos preesistenti restano nella lista residua; non
sono nascoste dal successo dei test mirati di questa campagna.

La qualifica T08 completa resta aperta. Il risultato locale chiude lo
stress Flash e il caso numerico512con device dichiarati; conserva F08,
stress Qwen fermato per risorse, corpus/asset e hardware non disponibili
come verifiche distinte. [Lista residua](remaining.md).
Nessun commit eseguito.

La richiesta successiva dell'utente autorizza il riesame e l'integrazione
con commit distinti. Il [riesame architetturale](../../../architecture/metal-t08-clean-code-review-20261106.md)
separa configurazione e aritmetica della capacità, introduce la policy
immutabile dei controller e verifica l'I/O del probe. I nuovi test sono
registrati in `review-verification.json`; i manifest della campagna qui
sopra conservano hash, configurazioni ed esiti originari.

## Ordine delle verifiche

1. Flash: F08 Custom letterale/greedy completo, poi alternanza/stress.
2. Qwen27: gli stessi gate, in un processo distinto.
3. Image: ammissione del corpus congelato e oracle indipendente512,
   quando le risorse consentono l'esecuzione senza cambiare le soglie.
4. Matrice finale con prove locali, limiti hardware e lavoro residuo.

F08 ripete `test_live_e2e_custom_literal_greedy_file_workflow` senza
cambiare prompt, effetti attesi, controlli receipt/history o sampling.
Il profilo storico conserva8192/64 e ceiling cache2GiB per rendere il
nuovo esito confrontabile. I profili di prodotto conservano invece la
cache dinamica. Timeout complessivo7200s, come nei tentativi precedenti.
La nuova istruzione revoca la precedente sospensione per questa campagna;
non riscrive gli esiti del2novembre.

Gli artefatti pesanti rimangono in `build/metal-t08-20261106/`; GGUF,
bundle e ambiente Python provengono dal checkout originale tramite link
locali ignorati. Il codice frontend in prova è quello di questo worktree.

## Verifiche senza modello

Build `ds4-server` PASS. I38test dei qualificatori, JUnit e confronti
pipeline passano; il primo tentativo sandbox conserva un errore di bind
locale, risolto ripetendo la stessa suite fuori dal sandbox.
`make test-server-interactive` PASS, inclusi i controlli letterali.

## Esiti live

In raccolta. La pressione iniziale è normale e lo swap iniziale è zero.
Il contatore globale cresce successivamente di256KiB: non si dichiara
assenza assoluta di swap. La soglia del monitor rimane invariata.

Gli11PASS funzionali locali precedenti e i sei casi L4 a2048/4096/8176
posizioni restano nel report T06–T07. Non sono nuovi esiti di questa
campagna né prove degli ambiti estesi.

### Asset del corpus Image

Ricerca locale anche negli altri checkout sotto `Desktop/Progetti/Training`:
i sei NPZ congelati non sono presenti. Eseguito lo script originale
`tools/freeze_suite_noise.py` senza modelli: PyTorch2.15.0su questo Mac,
contro2.14.1+cu130nel manifest. Nessuno dei sei hash BF16 o NPZ coincide.
`noise-recovery.json` conserva il confronto. I nuovi file restano separati
e non vengono usati al posto degli input congelati. La causa del diverso
RNG/ambiente non è attribuita a un kernel del prodotto.

Trovato anche il wheel ARM locale di PyTorch2.14.1, git
`5c4886908584029761b579af026dcfb627c84070`, con estensione Python3.12.
Il primo import con3.13fallisce senza generare file; la ripetizione con
Python3.12e dipendenze locali produce sei NPZ, ma nessun hash congelato
coincide neppure qui. Conservati log e `noise-recovery-2141.json`.
Il solo pin della versione non recupera questi asset.
I sei file rigenerati dalle due versioni ARM coincidono fra loro, anche
come NPZ; entrambi rimangono diversi dal manifest T00. Non si presume
portabilità del noise né si rigenerano i golden.

### F08: differenza fra ripresa e prefill completo

Flash crea realmente il file23byte, esce, poi riapre la skill tre volte.
Interrotto il controller dopo820.590s: `NOT_VERIFIED`, server exit0,
nessun cleanup forzato, minimo host5965299712B, crescita swap3795845B.
Il replay a freddo della prima risposta al caller termina in90.015s e
risponde `FRONTEND_INTERACTIVE_OK`,6token, invece di riaprire la skill.
I prompt renderizzati coincidono:1422token,6072caratteri,
SHA256 `587e119ee988e1c0dfe5434cad5207446de5ecbaaf5b05f92aa5a0ae78b00dd6`.
Questo diagnostico non promuove F08 e non dimostra da solo un errore nel
checkpoint: la frontiera viva include13token valutati in decode dopo
1234di prefill, mentre il caso freddo usa solo blocchi di prefill64.

Prima di cambiare kernel, dispatch o ripresa, si separano questi due fattori
con le API pubbliche di sessione: continuazione diretta dalla stessa
frontiera; payload restore dopo un forward interno; prefill completo freddo.
Si confrontano tutti i logits e si conserva distinta la diversa segmentazione.

Riesame specifico prima del probe:

- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`,
  `Qwen4ExpTextPLEEmbedding.forward` e `Qwen4ExpTextPLE.forward`:
  cronologia token/conv e stato attention appartengono al caller. Il modello
  HF pesante non viene eseguito sul Mac; il diverso accumulo batch/decode
  non è un oracle bit-exact dei kernel Metal.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/models/qwen4exp.cpp`: PLE richiede anche cronologia conv ricorrente.
  Il layout ggml non è quello dei payload DS4 e non ne prova il restore.
- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp`, slot/pin/wait: stato del trasporto distinto
  dallo stato del modello; nessuna prova del protocollo skill DS4.
- Locale `ds4_server.c:skill_frame_save/restore`,
  `skill_frame_matches_request`, `ds4_kvstore.c` e
  `tests/probe_qwen_metal_stream.c`: il server salva il payload completo
  prima della canonicalizzazione della risposta. Le prove precedenti
  bit-exact di restore non confrontano questa specifica segmentazione
  mista1234+13decode+175prefill con un prefill freddo1422.
  Consultati il report vision/checkpoint del1ottobre e il report Flash
  del2novembre, che conserva già il drift al cambio chunk.

Scelta: probe isolato, nessuna modifica aritmetica e nessun cold replay
forzato nel prodotto per ottenere una risposta preferita.

Un primo estrattore del trace rimuoveva i newline finali; il relativo probe
è stato interrotto e conservato senza conclusioni numeriche. La ripetizione
`flash-return-isolation-exact` conserva i byte esatti. Il server interrompe
inoltre il primo prefill alla frontiera del system1151prima di completare
1234token: questa segmentazione va dichiarata separatamente dal probe API
che esegue direttamente1234token iniziali.

Entrambi i probe terminano PASS, con restore bit-exact su248320logits.
Il caso con anchor1151 riproduce anche la scelta osservata: diretto e
restored hanno argmax248058(`<tool_call>`), freddo argmax10191; massimo
delta freddo1.89239585. Durata276.007s, minimo host6884016128B,
crescita swap2097152B. Il caso senza anchor dura279.901s, massimo delta
freddo1.37691212, argmax10191in entrambi, minimo host7318011904B,
crescita swap2296381B. Nessuno dei confronti cambia la numerica del prodotto.

La divergenza riguarda la segmentazione prefill/decode; non è una prova di
corruzione del payload. F08 rimane `NOT_VERIFIED`. Per intervenire serve
localizzare la prima operazione divergente sugli stessi input e ripetere
l'audit pertinente; non si applica il cold replay come scorciatoia per
ottenere la risposta desiderata. I vettori completi sono conservati nei
due `data/` sotto `build/metal-t08-20261106/`.

I due prompt esatti sono archiviati in `fixtures/`, compresi i newline
finali. Per riprodurre il confronto con la frontiera del sistema, dalla
root del repository e con un solo modello attivo:

```sh
make tests/probe_qwen_skill_return
agent/.venv/bin/python tools/qualify_metal_command.py \
  --output build/metal-t08-20261106/reproduce-return --timeout 600 \
  --timeout-multiplier 1 -- tests/probe_qwen_skill_return \
  gguf/Qwen3.8-Flash-Next-Q2.gguf \
  docs/performance/metal-apple-silicon-20261031/t08-20261106/fixtures/return.prompt.txt \
  docs/performance/metal-apple-silicon-20261031/t08-20261106/fixtures/inner.prompt.txt \
  build/metal-t08-20261106/reproduce-return/data 1234 1247 1151
```

Omettendo l'ultimo argomento si riproduce il controllo senza frontiera
intermedia. I pesi rimangono quelli originali, non distribuiti nel report.

## Preparazione della qualifica estesa

Il qualificatore server imponeva sempre il ceiling storico2GiB e limitava
il timeout base a1800s. Per provare il profilo di prodotto e50alternanze
si espone `--automatic-cache` (omissione del ceiling CLI, come nei profili
streamed già qualificati) e si ammette un timeout base fino a21600s.
Il default rimane2GiB/120s; monitor, soglie numeriche, timeout delle singole
operazioni e motore non cambiano. È configurazione del controller di prova,
senza nuova ottimizzazione dei kernel o del workflow di inferenza.
Il report salva la scelta della cache; i run automatici sono distinti dal
controllo F08 congelato. Per lo stress si dichiara prima dell'avvio un
budget complessivo di4ore per famiglia,50cicli dopo il warmup,256×256.

Il corpus Image T00 ha24recipe/48seed a1024×1024,1024×1536 o1536×1024.
I file noise congelati non sono presenti nel checkout originale. Il planner
dipende da geometrie e conteggi, non dai valori del noise. `--plan-only`
usa pertanto fixture zero con identica forma, marcate `admission_zero` nel
manifest. Non esegue inferenza né rivendica parity del corpus. La normale
qualifica continua a richiedere i file e gli hash congelati; nessun seed,
rapporto di forma, reference, soglia o input numerico è sostituito nei gate.


## Aggiornamento9novembre: swap Qwen e ripresa delle prove

La [campagna della transizione](../../metal-inference-20261102/qwen27/swap-transition-20261106/README.md)
conserva i candidati falliti, i pin upstream/XNU e i gate indipendenti.
Il trasporto allineato evita l'accumulo della sorgente file nell'upload,
l'handoff cede le cache ricostruibili senza cambiare lo stato di sessione e
il sampler SSD esclude il file cache protetto da XNU anche con riserva0.
Nuovo stress Qwen27 PASS6956.335s:50misurati+warmup,un processo,swap growth0,
pressione max2,server exit0,nessun cleanup forzato. Output deterministici,
continuation cached, owner idle esatto e prenotazioni vuote su tutti i51cicli.
I precedenti SKIP e21cicli fermati non sono promossi o importati.

Il qualificatore salva adesso ogni ciclo PASS con fsync e può riprendere
con `qualify_metal_server.py --resume`, verificando compatibilità e integrità.
Prova reale: stop dopo1ciclo,ripresa dei2rimanenti,PASS3/3 senza riscrivere
il primo PNG. Un50cumulativo fra processi rimane distinto dal lifetime
continuativo; il PASS50Qwen nuovo soddisfa anche quest'ultimo.95test Python
PASS(1skip),build/unità native/lifecycle/HTTP PASS. Nessun commit nuovo.
La [lista residua](remaining.md) conserva gli altri gate aperti: nessuna
qualifica completa del prodotto né cambiamento a ready:false.
