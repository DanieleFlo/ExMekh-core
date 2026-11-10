# Riesame offline dell'inferenza, 2 novembre 2026

Checkout `9805acbbd65a833a5e2ad53dfae49b33616dbd05` più modifiche non
committate, Apple M5 Pro 24 GiB. L'utente ha sospeso i test live rimanenti:
prima una sessione dedicata ai TPS, poi la ripresa della qualifica live.
Al controllo delle 07:08 UTC tutte le code precedenti erano già terminate;
nessun nuovo server, GGUF o modello completo è stato avviato nel riesame.

## Difetti confermati e correzioni

- `tools/bench_qwen_flash_agent.py`: il provider greedy ricostruiva una
  `ProviderRequest` dai messaggi già espansi. Un `ToolMessage` con allegato
  poteva quindi espanderlo due volte e l'handler interattivo non veniva copiato.
  L'adattatore ora usa il metodo pubblico `request.with_extra(...)`, che
  conserva messaggi, handler, streaming e metadati agentici. La regressione
  controlla identità dell'handler e una sola occorrenza dell'allegato, senza
  modello. Rimosso anche l'import di `apply_ds4_scope`, non più esposto dal
  frontend; i capture conservano i metadati originali e il trace nativo resta
  autorevole per il rendering sul server.
- `tools/qwen_flash_launcher.py`: i consumer del benchmark e della qualifica
  CUDA leggevano i default automatici del Mac, ricevendo il profilo Metal.
  Il dry-run ora accetta un backend esplicito; entrambi i consumer CUDA
  richiedono `backend="cuda"`. Il default generale resta `auto`; nessuna
  modifica ai default del launcher di produzione. Test separati verificano
  isolamento dagli override ereditati e assenza di variabili CUDA su Metal.
- `ds4_server_suite.inc`: la discovery dichiarava ancora `FAIL` per i vecchi
  mismatch interni Gemma e conditioning Qwen Image, già corretti e verificati.
  Ora dichiara `NOT_VERIFIED` con il motivo corrente in `ds4_suite_text.h`:
  subset numerici locali superati, qualifica completa/qualità estesa/stress
  ancora aperti. Conserva `ready:false` e `qualified_backends:[]`.

## Riesame numerico e di ownership

Riletti validazione pubblica in `ds4_numeric.c`, dispatch e ownership in
`ds4_numeric_metal.m`, trasferimenti host e buffer di comando in `ds4_metal.m`,
streaming in `ds4_qwen4_metal_stream.inc` e checkpoint in `ds4_lifecycle.inc`.
La selezione Conv BF16 basata sull'input è coerente: `ds4_num_conv` rifiuta
pesi, bias o output di dtype incompatibile prima di raggiungere il backend.
Le fixture coprono alias/capacità, code delle riduzioni, trasferimenti,
state/replay, cache ridotta ed errori di lettura/allocazione.

Non è emerso un nuovo mismatch numerico riproducibile in questo controllo.
Non sono cambiati kernel, dispatch aritmetico, recipe, soglie o oracle. Gli
[audit upstream fissati](upstream-numerics.md) e i confronti full-model già
eseguiti restano evidenze separate. Una futura differenza richiede una nuova
indagine delle implementazioni upstream prima di intervenire sul kernel;
questa revisione non sostituisce tale indagine e non qualifica input estesi.

## Verifica senza modelli

| Gruppo | Esito |
| --- | --- |
| Build suite/server e fixture native | PASS |
| Host nel virtualenv oracle predisposto | 77 PASS in 12,749 s |
| Primitive e lifetime Metal | 39 PASS, 3 skip CUDA in 4,068 s |
| Grafi sintetici Qwen27, MTP/vision, Flash resident/SSD e tre lifecycle | 6 PASS |
| Provider greedy e dry-run launcher | 13 PASS |
| Frontend agent design, compaction e raccolta file workflow | 85 PASS, 13 live skip in 8,01 s |
| HTTP host: interi, receipt, multipart, limiti e deadline | PASS |
| Test configurazione testi di prodotto | 4 PASS |
| Ruff sui Python modificati | 40 file PASS |

La prima raccolta host con il virtualenv frontend si è fermata per assenza
di `accelerate`; il tentativo è conservato. La ripetizione usa il virtualenv
oracle già predisposto, senza installare dipendenze o caricare pesi.
I log delle fixture includono errori deliberati di allocazione/lettura per
verificarne recovery e atomicità. Tutti i sei processi escono zero. I monitor
conservano riserva host di 2 GiB e non registrano cleanup forzati.

I report sono in `evidence/offline-nov02-{host-oracle-env,numeric,native}/`,
la lista degli eseguibili sintetici e relativi hash in
`evidence/run-offline-nov02-native.json`, il JUnit frontend in
`evidence/offline-nov02-frontend.xml`. La precedente suite frontend completa
resta separata: 1213 PASS, 49 live skip, copertura statement/branch 100%;
non è stata ripetuta integralmente dopo la sola variante greedy della fixture.

## Esiti live già conclusi e lavoro sospeso

La lettura offline di report/JUnit conferma i nuovi PASS Qwen27: tool image
registrato con continuità della conversazione in 1066,38 s, browser con PNG,
checksum, preview e persistenza in 1603,08 s, rifiuto della seconda risorsa
con template letterale in 251,85 s. Server exit zero e nessun kill forzato.
Le schermate `generated-images.png` e `last-page.png` del browser sono state
ispezionate: immagine originale visibile e codice richiamato dopo reload
senza rigenerazione. F10/F11 passano quindi su entrambe le famiglie nel
subset locale. I precedenti timeout rimangono nell'archivio.

Il campo descrittivo `reason` di questi report riusa il testo del controller
file condiviso; comandi, hash sorgente client e JUnit identificano i test
realmente eseguiti. Le misure originali non sono state riscritte.

F08 resta `NOT_VERIFIED`: i casi Custom letterali/greedy Flash e Qwen27
raggiungono il deadline client dopo 7204,17 e 7201,06 s, con cleanup pulito.
Non esiste un JUnit finale che dimostri l'intero workflow create/edit/delete.
Effetti parziali o skip non diventano PASS. Nessun retry è accodato. T07 e le
qualifiche estese restano aperti; l'ottimizzazione TPS non è iniziata qui.

Nessun commit eseguito.
