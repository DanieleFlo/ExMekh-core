# Riesame di trasporto pesi, handoff e ripresa delle prove

Base60e9400 più la campagna swap/checkpoint del6–9novembre. L'utente
richiede riesame, correzioni, test successivi e commit separati per scopo.
Le modifiche iniziali, le prove fallite e il PASS Qwen50continuativo restano
conservati nel report swap-transition-20261106 e nei suoi manifest.

## Responsabilità e difetti individuati prima dell'intervento

- Il registro risorse possiede l'ammissione al rilascio. Numeric deve poter
  verificarla tramite API prima di distruggere lo stato del backend: oggi
  prepare_close può liberare lo staging HOST/event0, mentre il metadata del
  contesto eredita un evento dispositivo ancora attivo e release restituisce
  BUSY. Il contesto sopravvive con stagingNULL. Stesso ordine problematico
  per il transfer CUDA. Il chiamante possiede la serializzazione del contesto;
  la query è una verifica, non un lease o un credito sintetico.
- Il journal deve possedere atomicità, integrità, sequenza e continuità.
  Le regole numeriche/native della suite, i digest deterministici e i riepiloghi
  delle misure appartengono al qualificatore della suite. Non deve essere
  necessario conoscere campi privati del checkpoint per eseguire i cicli.
- L'invocazione immutabile del client deve possedere entrypoint e opzioni.
  Il controller legge oggi la prima opzione duplicata, mentre argparse usa
  l'ultima: può verificare la directory o l'URL sbagliati. Le parole contenute
  nei valori di altre opzioni non devono classificare il programma eseguito.
- Le interruzioni durante gli upload iniziali devono avere lo stesso report
  durevole delle interruzioni dentro un ciclo. La fixture generata, se troncata
  prima del manifest, deve essere ripubblicata atomicamente senza falsare il
  progresso. Errori di gate e continuità rimangono distinti dai retry di I/O.

Nessun kernel, dtype, maschera, ordine delle operazioni o criterio viene
modificato. Nessuna nuova ottimizzazione prestazionale è rivendicata.

## Riaudit dei confini prima della correzione lifetime

Pin/hash rilette e verificate tutte le19fonti di
`docs/performance/metal-inference-20261102/qwen27/swap-transition-20261106/sources.json`.
Transformers ca8c871ede62b521a75a2ddfc5e1968ac9c8e097,
`src/transformers/models/qwen3_5/modeling_qwen3_5.py:574,641` conserva lo stato
conv/ricorrente distinto dai pesi; Diffusers aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e,
`pipelines/qwenimage21/pipeline_qwenimage21.py:183,856` separa offload e fine
pipeline. Non autorizzano a mutare lo stato di un contesto la cui chiusura è
rifiutata. Strata46c152a1310f77cd2f0ce2c7cd213962df14fe83,
`metal/src/engine/stream.cpp` e `arena_budget.h` separa reader/arena e
ownership; ggml dde879bc4f954b809892fd0d660515f42536394a,
`ggml/src/ggml-metal/ggml-metal.cpp:31` possiede la distruzione dei buffer.
Non coprono il protocollo di lease DS4: non si importano layout, percentuali,
lock o semantiche diverse. Riletti localmente ds4_resources.c:register_host/
release, ds4_numeric.c:close, Metal closeUpload e CUDA gpu_prepare_close,
il precedente riesame T08 e i diagnostici della campagna swap. Il difetto
è nel lifecycle, non in un confronto numerico o in una regressione kernel.

## Verifica prevista

Riprodurre il rilascio anticipato prima della correzione; test CPU del
registro/hook e prova Metal pubblica di closeBUSY seguito da upload/riuso.
Test dei qualificatori per parsing duplicato, confini, ripresa e gate.
Build CPU/Metal e regressioni delle fondazioni; gate Flash poi Qwen su
identici input, Image512/reload e Gemma confrontati ai medesimi oracle.
Prova HTTP breve di interruzione/ripresa sulla versione riveduta. Il PASS50
storico resta attribuito al suo binario/harness: un refactoring equivalente
non viene presentato come una ripetizione della qualifica di quasi due ore.
CUDA e distribuito non sono disponibili localmente; richiesto accesso CUDA
per il percorso di teardown condiviso secondo AGENT.md. Nessuna esecuzione
distribuita senza la specifica autorizzazione prevista.


## Esiti e limiti,10novembre

Il test CPU fallisce sul codice precedente perché prepare_close viene
chiamato prima del rifiuto dell'evento. Ora il registro espone can_release;
Numeric lo usa dopo il fence e prima del teardown, senza leggere il layout
privato del registro. La prova Metal pubblica mantiene i medesimi root/byte
su BUSY, poi stage_file/read byte-exact e chiusura dopo complete. ASan/UBSan
del lifecycle Numeric e del registro PASS. Nessun ordine aritmetico cambia.

SuiteReleaseHistory possiede geometria, finitezza/norma dei vettori, digest,
idle e statistiche. ReleaseCheckpoint espone pending_cycles/completion e
snapshot dei record che non espongono lo stato interno; impedisce anche la ricomparsa intercalata
di un server ID che falserebbe la continuità. Le nuove classi rileggono tutti
i51record storici con gli stessi gate e statistiche esattamente identiche.
QualificationClient possiede entrypoint/opzioni, applica l'ultima occorrenza
come argparse e rifiuta anche un secondo URL che punti a un server estraneo.
Le abbreviazioni CLI sono disabilitate per evitare che un secondo URL eluda il controllo.
Le fixture video sono pubblicate atomicamente; interruzioni degli upload
producono un riepilogo durevole. Gate numerici falliti rimangono non riprendibili.
I binding Numeric espongono dtype PyTorch coerenti, con validazione dei range
prima di apertura/allocazione. Nessun chiamante deve usare i codici ABI0/30.

105test Python PASS(1skip); Numeric CPU49PASS(14skip); build CPU isolata,
Metal/prodotto, risorse/executor/bundle/reader/planner/HTTP e lifecycle Metal
Qwen27/residente/SSD/Flash PASS. Trasporto Metal BF16/F32 con offset17,
più di8MiB,coda/EOF PASS; Ruff e diff-check PASS. Flash10.2s/Qwen11.2s:
20artefatti completi esatti sul controllo. Image512con reload423.4s:
22artefatti(tensori e PNG)esatti; confronto indipendente di conditioning,
8latenti e RGBA PASS anche dopo reload,SSIM0.9971658133.

Il primo lancio Image ometteva CASES_ONLY: i due casi erano passati, poi la
matrice aggiuntiva di fault è scaduta a600.9s. La ripetizione con l'ambiente
congelato corretto passa; non è stata cambiata una soglia. Un secondo errore
solo del verificatore contava22artefatti contro20perché escludeva i2PNG:
aggiunta la loro verifica, il gate già PASS è stato riutilizzato senza nuova
inferenza. Log e report di entrambi gli errori sono conservati. I primi test
bundle senza permessi socket falliscono nel sandbox; la ripetizione con
accesso localhost passa, senza disabilitare casi.

Restano due assert sin/cos Metal già riprodotti sul controllo precedente,
con shader invariati. La suite Numeric Metal esegue49test: questi2assert,
un NOMEM nella fixture con riserva fissa2GiB e3skip. La capacità sopra il
floor attuale è inferiore alla riserva della fixture; anche un tensore di
16float viene correttamente rifiutato. Nessun gate viene allargato.

Il tentativo nuovo Gemma termina prima del forward con plan:NOMEM6,senza
crescita swap. Dopo il riavvio del Mac il campione corrente ha~20kpagine
free,~496kfile e~816kanonime; il floor protegge~490kpagine di file. La
capacità effettiva è di poche centinaia diMiB, distinta dagli~8GiB del
contatore generico free+inactive-file. Occorrono più risorse disponibili
per ripetere Gemma e la prova HTTP completa di ripresa sulla versione
riveduta. Le prove storiche Gemma491/480, Qwen50e arresto/ripresa1+2rimangono
attribuite ai rispettivi binari/harness e non sono promosse come nuove.
La regressione ripresa/HTTP simulata e il replay completo sono PASS.

SSH a192.168.1.29:22va in timeout prima dell'autenticazione. Su istruzione
successiva dell'utente, validazione CUDA lasciata PENDING; nessun ulteriore
tentativo remoto o esecuzione distribuita. Questi limiti e F08/corpora T00/
assert già aperti non vengono presentati come chiusura completa T08.


## Integrazione per responsabilità

- `e46c4ad`: trasporto Numeric/bundle, staging contabilizzato, rilascio
  rifiutabile senza perdere il backend e dtype pubblici coerenti.
- `6b29099`: floor SSD e handoff attraverso i proprietari di modello/cache,
  con stato di sessione preservato e fallback degli adapter non qualificati.
- `6721055`: checkpoint durevoli, regole numeriche nel proprietario della
  suite, invocazione del client e interruzioni coerenti.

Documentazione, pin e manifest integrati in un commit separato. Il manifest
`review-verification-20261110.json` registra gli hash effettivamente verificati,
i tentativi non conclusivi e i controlli rimasti PENDING. La qualifica50del
9novembre rimane associata al suo snapshot: nessuna nuova misura equivalente
viene inventata per i commit del riesame.
