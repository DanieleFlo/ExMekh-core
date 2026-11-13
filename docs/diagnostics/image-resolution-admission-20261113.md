# Risoluzione suggerita quando Qwen Image supera il budget

Il turno Flash `49cc7839-c903-4c0f-bdcc-7b8eb71d85f1` ha ricevuto
`insufficient_resources` a 2048×1152 prima del compute. Il secondo tentativo
a 1344×768 ha usato la variante streamed 6 con 3.463.961.580 byte device
recuperati dalla sessione. Il server conosce il piano ammesso per la geometria
provata. Il limite statico del catalogo è invece il limite del protocollo.

## Scelta prima dell'implementazione

Base DS4 `a420a1398ae67dda8386c3475365cf2314f591a2` con l'integrazione Flash
del [report precedente](flash-native-suite-20261113.md). Riletti
`ds4_qwen_image_pipeline.c:configure_plan`, `ds4_qwen_image_dit.c:qi_dit_memory`,
`ds4_qwen_image_vae.c:va_decode_bytes` e `ds4_executor.c:ds4_execution_plan_build`.
Il working set cambia con token, geometria, cache e ripristino. Il calcolo
riusa lo stesso planner con geometrie allineate più piccole, lo stesso prompt,
gli stessi riferimenti e le stesse risorse recuperabili. Una stima a pixel
fissi per modello o GPU ignorerebbe questi dati. Le verifiche restano in fase
di pianificazione e la risposta conserva il codice di errore.

Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`, pipeline
`src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` e VAE
`src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py`, conferma
che forma dei latenti e decodifica dipendono dalla geometria. L'encoder Qwen
in Transformers `d4fcbccd0966ace9505b9cbd3041d28ad822d8a7`,
`src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`, mantiene il costo
del prompt e dei riferimenti anche quando cambia la dimensione dell'output.
Strata `6674a0065fb96bacde33e3eb10f91a1df86f95f2`, `src/prefill/prefill.cpp`,
conta cache e scratch per fase. llama.cpp `1a064ab0921238c1daa397d6f4a900ef33884de2`,
`ggml/src/ggml-cuda/ggml-cuda.cu` e `tests/test-backend-ops.cpp`, verifica
le forme di singole operazioni. Nessuno dei due calcola il budget del
pipeline Image DS4. Riferimenti fissati nel [manifest T00](../performance/native-suite-t00-20261020/sources.json).

## Verifica CUDA e margine della raccomandazione

Su RTX 3090 con Flash a 65.536 token e limite device di 19 GiB, la prima
stima ha indicato 1728×960 sia in JSON sia in SSE. Il retry al margine esatto
del planner ha restituito ancora `insufficient_resources` prima della
generazione. Il planner usa una fotografia del registro e contabilizza in
anticipo le risorse recuperabili. La prenotazione effettiva segue il rilascio
e usa la disponibilità osservata in quel momento. `ds4_executor.c` separa
questi passaggi. Il report [T19–T22](../performance/native-suite-t19-t22-20261022/README.md)
distingue il picco stimato, quello misurato e il budget software. Il pipeline
Diffusers fissato sopra sceglie la forma dei buffer a partire dalla risoluzione,
mentre il budget di ripristino e il campione CUDA sono specifici di DS4.

Il messaggio distingue quindi il massimo stimato dal planner da una misura
consigliata con un margine di risoluzione. Il margine riguarda il suggerimento,
senza cambiare l'ammissione o la precisione della richiesta effettiva.

Il replay su RTX 3090 ha rifiutato 1504×832 e 1280×736 prima della fase
`conditioning`, con evento SSE di ripristino Flash e codice
`insufficient_resources`. Il ripristino è quindi avvenuto, mentre il piano
che richiedeva il rilascio della sessione non è arrivato al compute. A
1152×640, il piano senza vittime (variante streamed 8) ha prodotto il PNG in
62,846 s. Conditioning 5,464 s, DiT 55,577 s, decode VAE 1,767 s.

`ds4_executor.c:ds4_execution_plan_build` preferisce già un piano senza
vittime a parità di richiesta. `ds4_executor.c:ds4_execution_run` prenota il
picco dopo il rilascio delle vittime. Anche con margine geometrico, il solo
limite teorico non rappresenta questa differenza tra i piani. La stima
continua a includere le risorse recuperabili, mentre il suggerimento cerca
prima una dimensione con margine che il planner ammette senza liberare
altri modelli. Se serve comunque liberare risorse, usa il margine sul limite
completo. Questa decisione riusa la preferenza del planner DS4. I riferimenti
upstream fissati sopra si applicano al costo della forma, mentre Strata e
ggml non espongono questo ciclo di rilascio e ripristino della suite.

Se un'esecuzione restituisce NOMEM e termina il ripristino, può fornire
lo stesso suggerimento. La decisione di riprovare rimane al chiamante.
I fallimenti di ripristino mantengono il proprio codice e la precedenza.

## Verifica finale

- Build CUDA SM 8.6 e test `test-suite-http`,
  `test-qwen-image-cache-budget` e `test-executor` riusciti. Il test
  del budget comprende RAM unificata e limite device simulati su host.
  L'esecuzione su hardware Metal resta da verificare.
- Stesso prompt della chat, Flash 65.536 token, seed 17 e limite CUDA di
  19 GiB. Il rifiuto a 2048×1152 riporta 1.658.880 pixel stimati e
  1024×576 consigliati, sia in HTTP JSON sia con il client ExMekh su SSE.
  Il rifiuto a 1280×736 restituisce a sua volta 1024×576 dopo il ripristino.
- Il client ha riprovato a 1024×576 e scaricato il PNG. Firma e dimensioni
  confermate. Variante streamed 8 senza vittime, 58,935 s totali.
  Conditioning 5,415 s, DiT 52,080 s e decode VAE 1,408 s. Gate isolato
  terminato con exit 0.

## Revisione di responsabilità

Il server usa un solo mapping degli errori nativi. Il worker HTTP passa
lo stato e l'eventuale suggerimento dopo avere chiuso l'operazione o
trasferito la proprietà al recupero. Il planner di Image riusa i descrittori
della cache senza creare un executor per le risoluzioni di prova.
La fase SSE rispecchia l'ultima fase effettiva, incluso `preparing` per
un rifiuto prima dell'executor.

Il catalogo pubblica `accepting_requests` indipendentemente dalla
qualifica del backend. Core ricava le funzionalità configurate dal modello
scoperto. Nel menu Agent, nel wizard e nella pagina dei modelli opzionali
la compatibilità viene dal catalogo del profilo in uso. Il profilo esporta
la propria configurazione senza richiedere una copia dei metadati runtime.

Test nativi generali e mirati riusciti su Linux/CUDA. Agent e Playwright
1.313 passati e 50 saltati, copertura configurata al 100%. Ruff riuscito.

Replay finale su RTX 3090 e budget device di 19 GiB. Il catalogo indica
`accepting_requests`, il rifiuto a 2048×1152 termina in fase `preparing`
e il suggerimento 1024×576 produce un PNG verificato in 58,690 s, variante
8 senza vittime. Conditioning 5,449 s, DiT 51,788 s e decode VAE 1,421 s.
Il turno Flash successivo riusa 27 token e restituisce il codice ricordato.
Exit 0. Metal hardware resta da verificare.
