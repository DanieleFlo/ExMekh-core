# Flash: doppio prefill del prefisso agentico — 4 ottobre 2026

## Diagnosi del caso segnalato

Il server lanciato con `agent/start-ds4-server-flash.sh` riceve un prompt
Responses con tool di 4431 token per il saluto `ciao`. Il prefisso di sistema,
comprensivo delle istruzioni e del registry, occupa 4419 token. Il log mostra
una prima passata fino a 4419 token, la cattura del frontier residente e poi
una nuova passata da 256 token nella stessa richiesta.

La policy residente usa il confine di sistema esatto. Flash non ha il flag
`managed_cache_subdir` del 27B e seguiva quindi la policy generica del checkpoint
cold: trim 32 e allineamento 2048 selezionavano 4096 token. Chiedere quel prefisso
più corto dopo aver elaborato 4419 token azzera lo stato ricorrente Flash e
ricalcola il prompt. Il problema è nell'ordine delle frontiere del server.

La media riportata dopo la ripartenza è ingannevole: il tempo trascorso include
la prima passata, mentre il numeratore dei token riparte da 256. I chunk della
singola passata sono prevalentemente intorno a 20–33 token/s; la prima passata
nel log impiega circa 175 s. Eliminare la duplicazione non dimostra da solo
un aumento del throughput del singolo prefill.

## Correzione e compatibilità

Cache residente e checkpoint cold Qwen usano lo stesso prefisso di sistema
validato. Nel caso segnalato il target disco diventa 4419, poi vengono elaborati
solo i 12 token rimanenti. La policy usa il prefisso del contesto agentico già
calcolato; per richieste normali senza quel contesto conserva la tokenizzazione
del testo renderizzato. Gli altri protocolli mantengono la selezione generica.

Non cambiano modello, pesi, kernel, aritmetica o formato dei checkpoint.

## Verifiche

- PASS: unit test con le lunghezze reali 4419/4431, sia Flash sia 27B, limite
  cold inferiore al prefisso e cache disabilitata.
- PASS: gate CUDA sul Flash Q2 e sul 27B Q4_K_S, ctx 1024. Il fixture usa un
  prefisso di 150 token e un prompt di 180; trim 16 e allineamento 64 riproducono
  il target generico più corto. Vengono elaborati esattamente 180 token, senza
  ripartenze, con tutti i 248320 logits identici al percorso di riferimento.
- PASS: fallback del confine renderizzato per richieste senza contesto agentico,
  restore di sistema/skill e rifiuto della corruzione sui due modelli.
- Il rollback locale compatto resta verificato sul 27B. Il fixture Flash verifica
  invece il payload completo portabile, che può sopravvivere all'invalidazione:
  il primo tentativo del fixture assumeva erroneamente il contratto compatto
  per entrambi i modelli, ed è conservato nei log.
- PASS: HTTP Responses con tool e contesto allocato 131072, chunk 256 e
  `--mtp`, usando un prefisso sintetico di 2321 token e prompt di 2335.
  Il primo prefill avanza una sola volta fino a 2335; il checkpoint cold è
  scritto a 2321. Risposta `Ciao!`, prima richiesta 54,44 s complessivi;
  una nuova conversazione con lo stesso sistema riusa 2321 token e richiede
  2,73 s complessivi, elaborando solo i 14 token del suffisso. È una singola
  prova funzionale, senza confronto A/B né equivalenza al prompt originale.
- PASS: suite Python repository 199 passati, 41 saltati; suite agente con
  browser/mock 739 passati, 40 saltati, copertura configurata 100%; sintassi
  Bash e contratti dei launcher; documentazione 10 test.
- Gate statici server, trie dei vincoli, pattern layer 27B e build CUDA: PASS.

Manifest e log locali: `performance-results/flash-prefill-replay-20261004/`.
Questi sono gate di correttezza e conteggio del lavoro, non benchmark A/B.
Il primo smoke HTTP ha risposto correttamente ma il driver assumeva un messaggio
`MTP enabled` che il server non emette. Il tentativo è conservato; il controllo
finale verifica comando prodotto, contesto dichiarato, avanzamento monotono
sopra la frontiera generica 2048 e riuso del sistema.
Il prompt originale completo di 4431 token non è stato ricostruito dal log;
la qualifica prestazionale precedente resta NEED_MORE_DATA.

## Lentezza residua e hardware

Nella rilevazione del processo originale: circa 22799 MiB di VRAM usata,
GPU 44% in un singolo campione, PCIe Gen4 x2 e swap del processo pari a zero.
Il log dichiara 41,73 GiB di pesi del modello, circa 14,17 GiB di cache VRAM
per esperti, 16 GiB di RAM pinned e 95,37 GiB di n-gram BF16 letti da disco.
Il modello non è interamente residente sulla GPU; copie e letture restano
necessarie. Il collegamento x2 e i miss delle cache possono limitare il
throughput, ma questo log non attribuisce i tempi a ciascuno stadio.

Per quantificare i colli di bottiglia della singola passata servono contatori
cache e profiling sul prompt lungo dopo questa correzione. Il solo utilizzo
GPU campionato non certifica né esclude un kernel inefficiente.

## Default dei launcher confermati

Su richiesta dell'utente, Flash conserva il contesto predefinito 131072 e
il launcher comune abilita MTP per entrambi i modelli. Il wrapper 27B non
aggiunge più un secondo `--mtp`. I test verificano un solo flag e gli override.

Nel nuovo avvio Flash: KV 4,17 GiB, contesto/buffer 4887,91 MiB, cache totale
pesi automatica 16,04 GiB e budget esperti 10,33 GiB. Nel log originale a 16K
il budget esperti era 14,17 GiB. Il contesto allocato più grande lascia meno
esperti in VRAM anche con un prompt breve; ciò può aumentare i miss, ma non è
stato misurato qui un confronto controllato 16K/128K. MTP riguarda il decode
e non risolve la duplicazione del prefill. Queste prove non qualificano prompt
pieni a 128K o il throughput MTP su generazioni lunghe.
