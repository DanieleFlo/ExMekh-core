# T06–T07: streaming SSD e suite Metal

Aggiornamento del6novembre: l'utente ha ripreso la qualifica e richiesto
riserva host0. Il [seguito T08](../t08-20261106/README.md) registra50cicli
Flash PASS, alpha512indipendente CPUencoder/MPSDiT/MPSVAE PASS, Gemma
480/480e i nuovi esiti F08/stress Qwen. La sospensione e i timeout del
2novembre qui sotto restano evidenze storiche. La [lista residua](../t08-20261106/remaining.md)
conserva i gate ancora aperti; T07/T08non sono dichiarati completi.

MacBook Pro M5 Pro, 24 GiB, macOS 26.6.2. Esecuzioni dal 31 ottobre al 2 novembre 2026,
checkout successivo a `9805acb`, senza nuovi commit. Directory originale delle
prove: `build/metal-qualification/t06-t07-20261031/`.

Stato corrente: T03/T05/T06 locali PASS; T07 ha superato numerica indipendente,
pipeline image complete e recovery su entrambe le famiglie. Anche F10/F11
passano sul subset locale. F08 resta `NOT_VERIFIED`: i test live sono sospesi
su richiesta dell'utente del 2 novembre, fino a una successiva sessione TPS.
Il [riesame offline](offline-inference-review-20261102.md) documenta i difetti
corretti e le regressioni senza modelli. La matrice sotto distingue
questi gate dalle prove storiche fallite e dai subset estesi non qualificati.

Il [riesame Clean Code successivo](../../../architecture/metal-t06-t07-clean-code-review-20261102.md)
separa il client dal runner, usa l'API metadata di Safetensors e rimuove lo
shader softmax inutilizzato. Le sue nuove regressioni senza modelli sono
archiviate separatamente; non riaprono le code live e non modificano le soglie.

## LLM reali: T03 locale e T06 PASS

Entrambi i modelli usano contesto 8192, prefill 64, pesi originali, target-only,
cache SSD 2 GiB e riserva host 2 GiB. Nessun fallback LLM su CPU. Il contesto
è riempito realmente: non si deduce la capacità dalla sola allocazione.

| Prova | Flash | Qwen27 |
| --- | ---: | ---: |
| Prompt breve, 8 token, prefix/checkpoint/restore | PASS 16,2 s | PASS 116,5 s |
| Quick Chat del frontend, circa 2500 token di input | PASS | PASS |
| 2048 posizioni occupate | PASS 199,5 s | PASS 233,8 s |
| 4096 posizioni, ripresa del payload precedente | PASS 208,7 s | PASS 247,4 s |
| 8176 posizioni, budget iniziale di 16 output | PASS 430,9 s | PASS 484,1 s |
| LLM → Gemma reale → LLM, continuazione identica | PASS 20,4 s | PASS 35,5 s |
| LLM → immagine nativa reale → LLM, payload/logits identici | PASS 133,56 s | PASS 142,25 s |

Ogni caso L4 salva logits completi e stato prefill, confronta il payload SHA256
prima/dopo il riuso del prefisso e il restore, verifica metadati e continuazione
byte per byte e salva otto posizioni di logits. I casi progressivi ripristinano
il payload nello stesso contesto 8192 in un processo nuovo. Picchi footprint
L4: Flash 1,44–1,58 GiB; Qwen27 4,97–5,32 GiB. Nessun incremento di swap nei sei
casi L4. Le misure sono footprint del processo, distinte da buffer Metal e cache.

La prova immagine usa la pipeline nativa originale 256×256, otto passi, CFG1,
seed 20261020 e un solo modello attivo per fase. Il LLM viene parcheggiato e
i pesi rilasciati tramite API pubbliche; adapter e contesto numerico image
vengono chiusi prima della ripresa. Il payload SHA256 completo e i logits di
ogni elemento del vocabolario coincidono bit per bit con la baseline senza
tool. Il PNG diventa disponibile soltanto dopo il restore verificato e sono
salvati otto token greedy. Questa prova funzionale non promuove il gate
indipendente del conditioning image, verificato separatamente nei run descritti sotto.

L'input reale di Quick Chat comprende sei tool e istruzioni del frontend;
l'output `CONFERMATO` arriva tramite Responses al vero `run_agent_turn` e coincide
con lo stream. Il primo budget Flash di 120 s era insufficiente; la ripetizione
motivata passa entro 300 s. Qwen27 usa prefill multi-riga anche sotto 512 token:
la soglia CUDA token-exact lasciata sul percorso SSD rileggeva tutti i pesi per
ogni token. La correzione è limitata a Metal+SSD.

## Storage e numerica

Flash locale: 147207127040 byte, SHA256
`b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`.
Qwen27: 15358213024 byte, SHA256
`75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017`.
Sono stati verificati hash interi, directory tensori, offset, dimensioni e file
non sparsi. I 95,37 GiB della tabella n-gram Flash restano su SSD.

Il trasporto esegue `pread` bounded verso buffer posseduti. Gli argomenti del
dispatch restano pinned; eviction/riuso richiedono fence e riconciliazione del
ledger con la memoria fisica. L'unione MoE viene suddivisa per token se supera
il cap, senza cambiare routing, ordine o kernel aritmetici. I command buffer e
le risorse Objective-C temporanee vengono drenati per batch/layer.

Le fixture confrontano lo stesso grafo residente e SSD: Qwen27 ibrido con
attenzione/GDN, Flash con PLE su disco, GDN, attenzione e MoE. Flash passa anche
con cache 500 KiB e unione esperti suddivisa; payload KV/ricorrente/PLE e logits
sono identici. Entrambe coprono letture fallite e ripresa. Le primitive mantengono
gli oracle e le tolleranze precedentemente congelate. Non sono cambiati shader,
formati quantizzati o arrotondamenti per far passare i casi.

## MTP e vision reali

Flash embedded MTP e Qwen27 con sidecar corrispondente eseguono decode reale.
Il sidecar 27B è fissato alla revisione
`4ca720788d1e01f1bff70c033e0d0028fd02e502`, 1369590656 byte, SHA256
`50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e`.
Il mapping secondario del sidecar ora viene creato su Metal e ricreato dopo
il rilascio dei pesi; la precedente esclusione era specifica del target CUDA.
Un semplice exit code zero non dimostra MTP: il primo caso 27B aveva ripiegato
su target-only ed è mantenuto come tentativo non qualificante.

Il test pubblico multimodale usa il costruttore di messaggi della famiglia,
non un marcatore `user` preso da un'altra tokenizzazione. Flash passa vision
target-only e vision+MTP. Qwen27 passa le stesse modalità dopo la correzione
dei trasferimenti host Metal: una CPU write non può precedere un produttore
GPU ancora codificato che poi la sovrascrive. Read/write ora rispettano
l'ordine, drenano quando necessario e conservano il batch attivo del chiamante.
La regressione ridotta prova esplicitamente l'effetto dell'immagine sui logits.
La query pubblica `has_mtp` ora include il predictor embedded: i chiamanti non
devono conoscere la differenza interna tra flag Flash e sidecar 27B.
La verifica Metal Flash procede per righe causali e conserva gli stream per
NextN, evitando che il block-key universe venga completato da righe future.
Sul GGUF completo otto token speculativi coincidono con il decode ordinario,
max delta logits **0**; due replay del checkpoint hanno payload identico.
Il percorso CUDA multi-riga rimane separato. Nessuno speedup MTP è rivendicato.

Ogni prova reale usa `earth.jpg`, encoder GGUF corrispondente, cap esplicito
128 image token (121 effettivi), contesto 8192/prefill64. Verifica riuso del
prefisso senza prefill, replay, immagine contro embedding nulla, fingerprint
modificato, ripristino dell'immagine e otto token di generazione. I confronti
di replay/restore mantengono soglia assoluta 1e-6 e rifiutano NaN/Inf.

## Suite e qualità

Gemma testo: tre stringhe, ranking correlato/non correlato e dimensioni
128/256/512/768 PASS contro Transformers CPU indipendente. Gemma media: immagine,
audio, due frame video e interleaving ordinato, tutte le dimensioni PASS con le
stesse tolleranze congelate. Questo subset di vettori finali non promuove il
corpus completo o le frontiere BF16 interne. Il precedente report testuale
dichiara 2/70 frontiere esatte, mentre il JSON pubblicato ne contiene 54,
due esatte; il report media contiene 400 frontiere, quattro esatte.
Entrambi i gate rigidi erano falliti; questa incongruenza storica nel
denominatore non cambia il verdetto.
Il riesame dei dump correnti del 31 ottobre conferma il fallimento: **3/81**
frontiere testuali e **6/480** frontiere media/probabilità coincidono bit per
bit. Tutte le dimensioni dei vettori finali passano. `--analyze-only` ricalcola
questi verdetti dai file esistenti senza eseguire un modello; i nuovi run
completi del qualificatore includono automaticamente il gate interno e non
possono più restituire PASS basandosi soltanto sui vettori finali.

Image BF16: generazione nativa 256×256, seed 20261020, otto passi, CFG1 PASS;
PNG ispezionato (quadrato rosso e cerchio blu su bianco). Il riferimento
indipendente usa i sorgenti pinned Transformers/Diffusers e PyTorch MATH,
BF16 e rumore identici: tutti gli otto latenti e RGBA PASS, SSIM RGB 0,998477.
Il controllo obbligatorio del conditioning di quel run era **FAIL**: RMSE 0,49835 /
relative L2 0,04011 contro MPS, 0,44954 / 0,03610 contro CPU, rispetto ai
limiti 0,02 / 0,01. Anche l’editing RGB passa latenti, VAE reference e RGBA
(SSIM 0,999246), ma fallisce conditioning. Il qualificatore ora include queste
frontiere: il verdetto complessivo è FAIL e non viene promosso dal PNG corretto.
La prima lettura eager dell'oracle ha raggiunto pressione memoria elevata ed è
stata fermata. Il trasporto lazy safetensors con fence per modulo passa encoder,
DiT e VAE in 30,5 / 91,1 / 9,1 secondi, mantenendo modelli e aritmetica upstream.
La traccia nativa completa dell'encoder, con cache pubblica dei pesi 2 GiB,
passa in **86,97 s**. Token, posizioni ed embedding iniziali sono esatti;
il primo layer ha RMSE 0,000102886 contro CPU e il conditioning finale
0,449544 / relative L2 0,0361033. Il trasporto bounded risolve il precedente
rifiuto di allocazione del probe al layer 24, senza risolvere il gate numerico.
Il replay delle primitive su input CPU identici localizza differenze nei GEMM
e in pochi tie BF16: Q projection del layer zero, 58/139264 elementi diversi,
max assoluto 0,001953125. RMSNorm iniziale, RoPE e SiLU sono esatti sui casi
esaminati; l'attention isolata è esatta al layer zero. Il calcolo manuale CPU
riproduce l'oracle upstream ai layer 0 e 6. Questa diagnosi non dimostra che
ogni divergenza successiva sia spiegata e non cambia il verdetto della pipeline.

Il primo novembre una prova separata 2×8, senza pesi modello, conferma che
`MPSMatrixMultiplication` rifiuta `MPSDataTypeBFloat16` sul device locale;
l'enum presente nell'SDK non garantisce il supporto dell'operazione.
Il log di capability conserva l'assert del framework. Questo esclude la
sostituzione diretta delle matrici FP32 con BF16 nella stessa API MPS;
non qualifica altre API Apple e non altera i gate falliti.

Il [riesame HF/ggml del primo novembre](upstream-numerics.md) registra
revisioni, operazioni e dtype di Transformers, Diffusers, llama.cpp e
stable-diffusion.cpp. Il GEMM SIMD-group BF16 di ispirazione ggml conserva
le differenze MPS nei due esempi reali esaminati. La successiva riproduzione
Metal dell'ordine FP32 usato dalla routine BF16 CPU di PyTorch le elimina
in entrambi; anche dieci RMSNorm isolate coincidono con l'oracle usando la
riduzione a cascata e `pow`. Cinquanta verifiche di primitive, code, bias,
alias e cleanup passano su librerie sperimentali separate. I replay successivi conservano anche GELU e attention esatte su input
identici, usando il percorso SLEEF fissato e gli ordini di somma ATen.
I forward completi del candidato restano distinti: queste prove isolate
non promuovono i gate di qualità del prodotto.

Il nuovo percorso integrato del primo novembre supera **81/81 frontiere
Gemma testo**, per tutte le dimensioni 128/256/512/768, e rende bit-exact
l'encoder Qwen testo completo contro CPU, conditioning compreso. Il precedente
risultato 3/81 resta storico. Dopo le ulteriori correzioni Conv, attention video,
normalizzazioni audio e softmax/trigonometria, il forward multimodale completo
passa **480/480 frontiere** BF16/probabilità e i vettori finali. Il precedente
125/480 con sola LayerNorm resta conservato come fallimento intermedio.
CPU e MPS upstream differiscono già di RMSE 0,619411 sul conditioning testo
identico: le nuove prove immagini scelgono esplicitamente CPU come riferimento,
con soglie congelate e conservando separatamente i precedenti fallimenti MPS.

La discovery distingue modello configurato e qualifica completa: Metal non
annuncia backend qualificati finché i gate rigidi restano aperti e riporta il
motivo e la riserva di memoria. Le richieste funzionali restano verificabili.

Il primo ciclo HTTP completo Qwen27 passa in 370,18 s (455,2 s inclusa apertura
del server): tool richiesto, immagine nativa 256×256, continuazione, embedding
multimodali reali e ulteriore continuazione, con memoria posseduta osservabile.
La generazione richiede 109,62 s e l'embedding 31,65 s. Una sola ripetizione
non dimostra stabilità statistica o lo stress esteso; il report mantiene
`NEED_MORE_DATA` sulle distribuzioni di latenza.
Passano anche i due casi HTTP 27B di generazione/retry concorrente, editing,
Files con checksum, continuazione con cache e disconnessione recuperabile:
438,82 s di test, **520,7 s** con avvio del server. Il test della cancellazione
su una frontiera DiT richiede il transcript completo per continuare una chat
stateless; il tentativo senza history fallisce e resta archiviato.

La ripetizione Flash esegue **due cicli completi** in 472,1 s inclusa apertura.
Il primo novembre passa anche la selezione HTTP Flash di generazione,
retry concorrente, editing, Files/checksum, ripresa della chat e recupero
dopo disconnessione: **311,22 s**, con entrambi i casi eseguiti.
Il flusso tipizzato ExMekh con tool immagini registrato e continuazione passa
in **369,97 s**; le preview native di entrambe le famiglie passano conservando
il PNG originale 256×256 e fornendo l'allegato 128×128.
Passano inoltre la cancellazione sulla frontiera DiT con riprova e checksum
(167,0 s), ranking/tool/embedding/continuazione (167,9 s) e il prefill vision
lungo dopo tutti gli encoder embedding (249,9 s). Entrambe le famiglie passano
la selezione SDK embedding: dieci test e 21 subtest per famiglia, dimensioni,
task, float/base64 e media reali. La selezione Flash agentica passa cinque
test API in 328,4 s; il sesto caso interattivo è saltato per un limite del
modello e non qualifica l'editing. Il browser e i casi interattivi 27B restano
registrati separatamente.

I test agentici grezzi ammettono `DS4_AGENTIC_REASONING_EFFORT`: il qualificatore
seleziona esplicitamente `none`, come i client. Senza questa impostazione la
prima selezione 27B scade durante il ragionamento; quel risultato rimane
`NOT_VERIFIED`. Il default dei test usati fuori da questo profilo non cambia.
`--kv-seed` copia soltanto i prefissi SSD completi di un'istanza precedente,
registrandone dimensioni e SHA256; non trasferisce checkpoint agentici attivi.
Questo distingue le prove dopo restart da quelle con cache inizialmente vuota.
La discovery/debug response non viene allargata per i normali flussi browser:
il qualificatore abilita il ledger dettagliato soltanto per phase-control e
`qualify_suite_release.py`, oppure su richiesta `--resource-metrics`.
`--no-resource-metrics` mantiene la risposta di produzione. Il vecchio profilo
browser aggiungeva 3044 token di continuazione, in gran parte diagnostica,
e scadeva: il nuovo profilo non cambia image recipe, input utente o assert.
Il budget di attesa iniziale del tool vision è dichiarato a 480 s dopo i costi
SSD osservati; il processo completo resta limitato a 600 s e ogni immagine a
300 s. Nel riavvio Flash sono stati recuperati 1670 token in 22,7 ms.

Il browser reale Flash passa in **553,87 s** (537,61 s di test): una sola
chiamata registrata produce il PNG originale 256×256, la preview 128×128 arriva
come pixel al modello, il file mantiene SHA256 e la UI resta responsiva
(tre heartbeat circa 1–1,5 ms). Dopo il reload ritrova messaggi e media e
richiama il codice senza un'altra generazione. Il server riusa 1916 token al
ritorno dal tool e 3030 nella chat riaperta, con fingerprint immagine identico.
PNG originale e screenshot sono stati ispezionati: quadrato rosso su bianco,
preview corretta e codice mantenuto. Questo gate usa il prefisso SSD dopo
restart; non viene presentato come un avvio con cache vuota.

Il primo novembre la selezione di sette protocolli/metadata/compaction/skill
annidate Flash passa in **435,83 s**. Qwen passa sei casi ma fallisce il riuso
Responses di una risposta testuale senza tool: il renderer iniziale include
un blocco thinking vuoto che il transcript storico non conserva. Il server
ora costruisce la chiave visibile con lo stesso renderer storico, lasciando
lo stato KV sotto il proprietario esistente. La regressione C e la riprova
reale passano: 28 token cached su 46 di prompt. Nello stesso run passano
nome tool esatto e ranking/embedding/continuazione; tre casi in **558,37 s**.
Il primo fallimento rimane nell'evidenza.

## Policy RAM aggiornata su richiesta dell'utente

Nella stessa sessione l'utente ha chiesto di usare tutta la RAM disponibile,
conservando soltanto lo spazio per l'agente. I nuovi run mantengono **2 GiB di
riserva host**, timeout invariati e controlli su crescita sostenuta dello swap
e pressione critica. Un warning di pressione isolato e il superamento del
precedente cap prudenziale di footprint 10 GiB non fermano più il processo.
Il cap di processo è opzionale e dichiarato nei report; zero significa budget
dinamico. Il profilo suite usa il planner con la memoria realmente disponibile.
La stessa distinzione XNU (warning 2, critica 4) vale per l'ammissione nel
runtime, non soltanto per il monitor esterno. Il test CPU della policy verifica
che il warning ammetta la capacità misurata, che la riserva resti indisponibile
e che la pressione critica azzeri la capacità aggiuntiva senza perdere lease.
`device_working_set_bytes` resta un limite opzionale esplicito per diagnosi;
non è impostato nel profilo automatico. Le tolleranze numeriche restano immutate.

I precedenti `SKIP_RESOURCE_LIMIT` conservano la policy allora applicata e
non vengono riscritti in PASS. La cache SSD da 2 GiB dei profili T06 non è
il limite RAM dell'intero processo.

La regressione host finale passa **55 test**: policy e monitor RAM, timeout,
inventario sorgenti, launcher, schemi API, trasporto lazy PyTorch e rifiuto
del gate image quando fallisce il conditioning. Non carica modelli né
sostituisce le prove native e frontend.

La regressione host del primo novembre passa **57 test**, inclusi due nuovi
controlli sul budget temporale del server. Il limite standard resta quello
precedente; una prova separata può richiedere fino a 1800 s e un timeout di
avvio tool esplicito, entrambi convalidati prima di aprire un modello.
Il browser 27B viene ripetuto con 720 s per la chiamata iniziale e 1200 s
complessivi: le prove precedenti a 480 s conservano FAIL. Sono estensioni
motivate dalla velocità osservata di circa 0,45 token/s, senza cambiare
recipe, schema, allegati, verifiche di persistenza o soglie di accettazione.
La verifica successiva alle correzioni di cleanup e analisi passa **64 test**
in 8,31 s (22,2 s includendo avvio e monitor), senza test saltati.
Anche la ripetizione estesa resta FAIL: 237 token prodotti prima del limite
iniziale di 720 s, ancora senza richiesta image. Il run registra pressione
normale e nessuno swap. Il qualificatore ora salva anche la telemetria
esistente delle fasi decode/vincoli (`--no-phase-profile` per disabilitarla),
per distinguere costo del modello, verifica speculativa e parser.
Il campione reale di una chiamata tool 27B misura 39 token in 172,35 s:
0,64 s nei vincoli CPU, 33 cicli MTP, sette draft accettati e 37 fallback.
La traccia di processo mostra attesa Metal e `pread`. Non è una prova di
speedup MTP: la verifica browser conclusiva usa esplicitamente target-only,
come il profilo iniziale T06, senza cambiare i controlli funzionali.

La suite frontend del primo novembre passa **1210 test e 46 skip** in 273,75 s.
L'invocazione dalla radice ha raccolto un sovrainsieme dei file configurati:
il report coverage è stato ricalcolato sugli stessi dati applicando soltanto
le esclusioni `run.omit` già presenti in `agent/pyproject.toml`, ottenendo
statement e branch **100%**. Configurazione, dati e comando sono identificati
nel report di verifica della copertura; nessuna esclusione è stata aggiunta.

Le fondamenta finali passano **27 gruppi**, incluse build Metal e CPU isolata,
runtime, ledger, lifecycle residente/SSD, grafo Flash con cache ridotta e
MTP/vision sintetici. La suite frontend completa passa **1210 test, 46 skip**,
copertura statement e branch **100%**, in 243,86 s. Un controllo separato delle
primitive SSD passa MoE da 17 token su dominio 512, eviction densa ripetuta,
isolamento della cache e sostituzione fallita senza perdere i pesi precedenti.

La verifica SHA-256 su macOS usa ora CommonCrypto mantenendo lo stesso
contratto streaming, digest integrali e backend portabile sugli altri sistemi.
Passano i vettori NIST (input vuoto, `abc`, 56 byte e un milione di `a`),
frammentazione 1–65 byte e **12 test dei bundle**, inclusi corruzione e file
modificati. Sul medesimo sidecar da 1369590656 byte entrambi i backend
restituiscono il digest fissato: 0,539 s sistema, 3,333 s portabile. È un
campione di sviluppo con cache OS non controllata, non uno speedup statistico.
La modifica riduce il costo della convalida durante l'apertura dei server;
non salta la verifica dei pesi e non cambia l'aritmetica dei modelli.

La ripetizione reale `qwen27-agentic-available-ram` passa **351,82 s**, con
picco footprint **11,34 GiB**, disponibilità host minima **8,55 GiB**, pressione
massima normale e incremento swap **0**. Copre tre ritorni annidati con logits
bit-exact, isolamento, checkpoint corrotti/troncati, cancellazione, sessione
portabile e reload visibile da SSD, continuazione e limite di contesto. Il
checkpoint compatto è 159852624 byte; la sessione portabile 195898524 byte.
Il processo usa capacità 8192/64 e transcript breve; il caso finale prova una
sessione esplicita di 512 posizioni, senza dedurne la capacità 8k già verificata
in T06. Il precedente stop al solo cap 10 GiB rimane nell'archivio.

## Matrice F01–F12 in corso

Le prove funzionali locali e i gate di qualità hanno verdetti distinti.
`NOT_VERIFIED` comprende i casi sospesi, incompleti o saltati per limiti
del modello. F08 e le qualifiche estese aperte impediscono di dichiarare T07
completato; al 2 novembre non rimangono consumer live in esecuzione.

| ID | Flash | Qwen27 | Evidenza e limite |
| --- | --- | --- | --- |
| F01 | PASS | PASS | Testo 8k, prefix/checkpoint, MTP e vision reali; profili distinti |
| F02 | PASS locale | PASS locale | Generazione completa: conditioning esatto contro CPU, otto latenti e RGBA entro le soglie; VAE reference MPS dichiarata |
| F03 | PASS locale | PASS locale | RGB/RGBA completi: conditioning bit-exact, reference VAE/latenti/pixel entro le soglie; preservation estesa distinta |
| F04 | PASS locale | PASS locale | Dopo integrazione BF16: tutte le dimensioni e 81/81 frontiere testuali identiche alla CPU |
| F05 | PASS locale | PASS locale | Forward media completo: 480/480 frontiere BF16/probabilità esatte e vettori finali entro le soglie |
| F06 | PASS locale | PASS locale | Payload/logits esatti su entrambe; generazione/editing/embedding HTTP e ripresa passano, due cicli Flash |
| F07 | PASS locale | PASS locale | Protocolli, metadata tipizzati, compaction e skill annidate passano; corretto e riverificato il prefisso Responses Qwen |
| F08 | NOT_VERIFIED | NOT_VERIFIED | I casi Custom greedy completi raggiungono il timeout (7204,17/7201,06 s); nessun JUnit finale. Prove live sospese; i precedenti fallimenti Generic restano archiviati |
| F09 | PASS locale | PASS locale | Ranking testuale/visivo con embedding nativi e cache del client; corpus locale fissato |
| F10 | PASS locale | PASS locale | SDK embedding/gen/edit/Files e preview passano entrambi; conversazione Qwen con tool registrato PASS 1066,38 s, rifiuto seconda risorsa con template PASS 251,85 s |
| F11 | PASS locale | PASS locale | Browser Flash e Qwen con pixels/checksum/reload e richiamo del codice senza rigenerazione; Qwen target-only PASS 1603,08 s |
| F12 | PASS locale | PASS locale | Cancellazione al latente/retry/restore: Flash 290,51 s e Qwen 300,36 s; guasti e disconnect verificati |

Il qualificatore conserva anche i risultati JUnit per singolo test: uno skip
per `MODEL_LIMITATION` resta `NOT_VERIFIED`, anche quando altri casi della
selezione passano. XML interrotto o errori di cleanup non eliminano il report
di memoria e non possono produrre PASS. Un rifiuto di segnale macOS a un
gruppo terminato viene ignorato soltanto dopo aver verificato l'assenza di
membri ancora attivi; sono ammesse esclusivamente le entry zombie o assenti.
Il cleanup attende anche i worker dopo l'uscita del processo principale e
registra l'eventuale uso di SIGKILL e l'exit code del server. La necessità di
una terminazione forzata o un'uscita anomala non può promuovere una prova PASS.
Le regressioni host aggiornate passano **72 test**; Ruff passa sui 28 Python
modificati. Il wrapper numerico passa sei test, incluso il budget Metal reale.

La creazione file dal frontend Qwen del primo novembre fallisce dopo 1013 s:
il client interattivo impone 300 s per singola chiamata logica, mentre una
continuazione richiede circa 380 s. Il tentativo e l'errore di cancellazione
sono conservati; il timeout complessivo di 1200 s non sostituisce quel limite
interno della dipendenza. Nessuna modifica ai suoi attributi privati o alle
assertion del test viene usata per certificare il flusso.

## Riproduzione e ambiti

Il primo novembre l'utente ha richiesto timeout temporanei ×2/×4 su questa
macchina, per entrambi i LLM. Le nuove prove usano **×4** tramite
`--timeout-multiplier 4` oppure il profilo esplicito
`config/metal-local-qualification.env.example`. I report conservano fattore,
timeout base ed effettivo, compresi HTTP, attesa tool e chiamata interattiva.
I default senza profilo restano invariati; le soglie numeriche non cambiano.
La prima dipendenza installata localmente aveva un override di configurazione
`EXMEKH_INTERACTIVE_TIMEOUT_SECONDS`, default 300 s e profilo locale 1200 s.
Originale, patch e SHA256 sono salvati nella directory di evidenza
`local-interactive-timeout`; quel primo tentativo lasciava wheel e lockfile
invariati. La correzione riproducibile successiva è ora inclusa nella wheel
con hash fissato e nella patch `agent/vendor-patches/exmekh-lib-budget.patch`;
la reinstallazione conserva il supporto all'override, con default sempre 300 s.
Il nuovo handler SIGTERM consente al
qualificatore di salvare il report e chiudere i propri processi figli.

Con gli artifact già preparati e una directory nuova per ciascuna prova:

```sh
make -j2 ds4-server-suite tests/probe_qwen_metal_stream
python3 tools/qualify_metal_command.py \
  --output build/metal-qualification/new-flash --timeout 120 -- \
  ./tests/probe_qwen_metal_stream gguf/Qwen3.8-Flash-Next-Q2.gguf \
  build/metal-qualification/new-flash-artifacts

# Lo stesso probe accetta occupazione e payload prefill precedente.
python3 tools/qualify_metal_command.py \
  --output build/metal-qualification/new-qwen-2k --timeout 300 \
  --reason 'Occupazione reale 2k dopo smoke breve' -- \
  ./tests/probe_qwen_metal_stream gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  build/metal-qualification/new-qwen-2k-artifacts 2048

python3 tools/qualify_metal_server.py --family flash \
  --suite-config config/native-suite-metal.example.json \
  --output build/metal-qualification/new-http --timeout 120 -- \
  agent/.venv/bin/python -m pytest -q -s \
  tests/test_suite_http_live.py::SuiteHttpLive::test_01_sdk_text_tokens_and_identity
```

Il probe MTP è esplicito: `DS4_PROBE_MTP=1` per Flash oppure
`DS4_PROBE_MTP_MODEL=gguf/mtp-Qwen3.8-27B-Q4_0.gguf` per Qwen27.
Il server di qualifica abilita l'encoder corrispondente soltanto con `--vision`.

Le responsabilità restano nei rispettivi proprietari: il backend ordina le
copie host rispetto ai produttori GPU; lo storage SSD possiede slot, pinning,
eviction e remapping; il motore espone capacità MTP uniformi. I probe reali
usano checkpoint, lifecycle e messaggi multimodali pubblici. Il riferimento
lazy cambia soltanto il trasporto dei pesi tramite `Mapping`/`cpu_offload`,
mentre Transformers e Diffusers mantengono l'aritmetica del modello.


I profili `config/flash-metal-streamed.env.example` e
`config/qwen27-metal-streamed.env.example` allineano server e frontend a 8192/64.
`config/native-suite-metal.example.json` indica bundle BF16 e asset preparati
localmente; non distribuisce i pesi. Avviare una sola istanza di modello grande.

`tools/qualify_metal_command.py` esegue casi con timeout, footprint, pressione,
swap e riserva host monitorati. `tools/qualify_metal_server.py` gestisce server,
client reale e cleanup. Timeout e rifiuti di risorse non diventano PASS. Gli
artifact binari grandi restano nella directory ignorata; le evidenze compatte,
comandi e hash vengono salvati qui a conclusione delle verifiche.

Restano separati gli altri chip Apple, CUDA/ROCm, il corpus qualità esteso e lo
stress di almeno 50 alternanze. Le prove funzionali locali non certificano tali
ambiti. Il registro finale distingue esplicitamente ciascun gate ancora aperto.

La riprova frontend Qwen con timeout ×4 termina dopo 2461,17 s: il file
corretto è creato e il ritorno dalla skill è verificato, ma il modello riapre
la skill e supera il budget del prompt (7218 stimati contro 7168, capacità
8192). È un FAIL distinto dal vecchio timeout, con uscita server zero e nessun
cleanup forzato. Le assertion finali non sono state eliminate.


La verifica finale image locale passa i tre casi testo, RGB e RGBA contro un
riferimento indipendente con encoder/DiT CPU e VAE MPS dichiarata. I due encoder
con reference sono bit-exact su tutte le **69/69 frontiere**; pixel, token e
posizioni coincidono. RMSE massimo degli otto latenti: testo 0,0045133,
RGB 0,0059462, RGBA 0,0052515. SSIM RGB finale rispettivamente 0,999323,
0,999181 e 0,999375; alpha MAE al massimo 5,4777e-5. I PNG nativi sono stati
ispezionati: quadrato rosso e cerchio blu su bianco, senza artefatti visibili.
Gli oracle sono riutilizzati solo dopo verifica SHA del fixture serializzato,
rumore incluso. Le prove finali RGB/RGBA durano 225,32/220,18 secondi misurati;
le sospensioni del Mac rendono diverso il tempo a parete. Il tentativo VAE solo
CPU resta NOT_VERIFIED e i precedenti fallimenti restano separati.

Le regressioni numeriche aggiornate passano **42 test**, con tre skip solo CUDA.
I nuovi controlli verificano sin/cos sui valori vision e nell'intervallo firmato,
GELU erf su tutti i 65280 BF16 finiti ordinati e rimescolati, code e zero con segno.
L'observer decoder è allineato al forward hook HF prima della fusione deepstack;
questa correzione dei trace non cambia il conditioning né l'aritmetica del modello.

Il moltiplicatore locale ×4 ora copre anche il timeout dell'operazione suite:
300→1200 s nella copia temporanea della configurazione, lasciando intatta la
configurazione sorgente. Prima di questa correzione i timeout dei client erano
già scalati, ma la fase immagine poteva essere annullata dal limite servizio
rimasto a 300 s. Il runner propaga ora SIGTERM/Ctrl-C al proprietario dopo aver
salvato report e cleanup, evitando di avviare il caso successivo dopo l'interruzione.


La verifica host dopo il completamento numerico e la correzione dei timeout
passa **75 test** in 12,865 s. Tre regressioni aggiunte coprono il deadline
suite scalato senza modificare il profilo sorgente, il rifiuto dei valori
invalidi prima della preparazione e la propagazione dell'interruzione dopo
il cleanup. I test frontend mantengono il caso originale; una prova live
aggiuntiva, distinta, esplicita una sola invocazione skill e il ritorno a root,
con gli stessi controlli su contenuto, history e tool completati. L'originale
fallimento Qwen per riapertura della skill resta nel proprio report.


Il confronto dei timer locale conferma due clock distinti: Python usa
`mach_absolute_time()`, mentre il server C usa `CLOCK_MONOTONIC` POSIX.
Il [sorgente Apple](https://github.com/apple-oss-distributions/Libc/blob/63976b830a836a22649b806fe62e8614fe3e5555/gen/clock_gettime.c)
deriva quest'ultimo dal tempo civile meno il boot time; la
[documentazione Python](https://docs.python.org/3.13/library/time.html#time.monotonic)
indica il timer Mach usato su macOS. Revisione Apple, SHA e misure sono
in `mac-clock-audit.json`. Le sospensioni incidono diversamente: durate del
monitor e tempi a parete non costituiscono lo stesso benchmark. I nuovi
controller mantengono una lease temporanea `caffeinate -i` fino al cleanup,
senza modificare le impostazioni di alimentazione della macchina.


La selezione interattiva Qwen completa termina con quattro PASS e due
NOT_VERIFIED, più due subtest PASS, in 1997,78 s di client (2031,74 s con
avvio/cleanup). Passano create-reply rifiutata, file esistente vuoto/replay,
framing letterale con streaming e senza, e due modifiche allo stesso path
con ID distinti. Il caso con separatore senza ID non emette esattamente
`hello`; il prompt del rifiuto seconda risorsa produce tag deformati e non
raggiunge quella frontiera. Il protocollo non viene qualificato dai due skip.
Una prova distinta mantiene gli stessi assert di rifiuto e fornisce il template
letterale completo; non sostituisce né promuove il risultato originale.


Con i deadline suite/client esplicitamente ×4, la prova Flash di cancellazione
al primo latente reale e retry passa in **290,51 s** (274,00 s di test), con
uscita server zero e cleanup senza terminazioni forzate. Verifica continuità
Responses con gli output originali, riuso KV dopo restore, assenza di file
pubblicati dalla richiesta cancellata, embedding di recovery, riserve rilasciate,
retry con la stessa chiave e un solo PNG disponibile tramite Files.


La stessa prova di recovery passa su Qwen27 in **300,36 s** (286,20 s di test),
con cache riutilizzata, PNG pubblicato una sola volta e shutdown pulito.


Il controllo host finale passa **76 test** in 12,711 s, aggiungendo il rifiuto
di un moltiplicatore d'ambiente invalido prima dell'avvio del server. Il parser
CLI e la lettura programmatica condividono lo stesso validatore. Passano anche
i quattro controlli `make test-product-text` e i test C input/output image.

## Compaction del client e baseline numerica locale

Il frontend Flash generico crea correttamente `probe.txt` e restituisce la
skill, poi riapre ripetutamente `mock-skill`. Il tentativo è stato interrotto
ordinatamente dopo 1328,19 s: `NOT_VERIFIED`, uscita server zero, nessuna
terminazione forzata. Non è un timeout e non viene promosso dalla sola
presenza del file. La prova distinta con istruzioni di una sola invocazione
mantiene gli stessi controlli di effetti, history e lifecycle.

Durante questa verifica è stato isolato un difetto SDK: il wrapper sintetico
della compaction può ingrandire un output breve conservato integralmente.
`NodeRuntime`, proprietario della history e degli scope, ora confronta il
request completo prima di applicare la sostituzione. Una compaction automatica
budgeted deve ridurre i token e rispettare il limite; altrimenti conserva gli
originali, non ribasa gli scope e non emette un archivio di sostituzione.
Il limite di capacità continua a fermare richieste che non possono entrare.
La compaction manuale mantiene il proprio contratto.

La [patch riproducibile SDK](../../../../agent/vendor-patches/README.md) è
inclusa nella wheel con hash `97e0119e…`, fissato in `agent/uv.lock`.
Sono modificati soltanto runtime e configurazione del timeout nel pacchetto;
metadata e versioni dipendenti coincidono col precedente artefatto. Passano
**1510 test SDK** dei componenti coinvolti, **159 test sul pacchetto installato**
e **50 test frontend della compaction**, compreso il caso di risposta breve
che non deve crescere. Il test di ripristino dell'usage usa una risposta
realmente riducibile per verificare anche l'applicazione della sostituzione.
La suite frontend completa è verificata separatamente dopo questa modifica.
La ripetizione passa **1211 test**, 47 skip e copertura statement/branch al
100%. Un successivo riesame del percorso Generic ha isolato un altro difetto:
il coding-agent bundled non dichiarava file-processing, pur richiedendo nel
proprio prompt l'ispezione e la modifica dei file. Il trace reale mostrava
`allowed_skills: []` e attese ripetute su un job `noop` inesistente.
Il suo HDS ora dichiara la skill opzionale con guida condizionale. La
regressione sull'inventario pubblico falliva prima della correzione con
capability sia predefinite sia esterne; dopo la correzione passano 98 test
dei componenti coinvolti. La suite completa successiva passa **1213 test**,
48 skip e copertura statement/branch al **100%** in 623,99 s. La verifica
host finale passa 77 test in 12,83 s; Ruff passa sui 36 Python modificati
e passano i quattro controlli sui testi di prodotto.

Il secondo tentativo Flash Generic, interrotto per correggere questo difetto,
resta `NOT_VERIFIED` dopo 1258,54 s, con shutdown zero e nessun SIGKILL.
Il nuovo profilo file Custom viene costruito tramite `AgentDesign`, come il
profilo browser image: espone file-processing e i cinque tool originali,
con gli stessi controlli di effetti, receipt, autorizzazioni e history dei
test Generic. Crea un file reale, modifica quel medesimo file dopo inspect/read
e lo cancella. Il contesto resta 8192/64; il prompt root è 1234 token.
Non modifica i profili Generic né promuove i loro esiti precedenti; questa
selezione reale Custom è un gate distinto, poi concluso senza qualifica.
Flash crea e modifica realmente il file, ma il parametro edit emesso include
un LF finale: 32 byte anziché i 31 richiesti. Il contratto interattivo conserva
questi byte letteralmente; il parser non viene modificato per assorbire la
differenza. Dopo la conferma il modello riapre la stessa skill: il tentativo
è fermato e registrato `NOT_VERIFIED` dopo 2158,24 s, con shutdown zero e
nessuna terminazione forzata. Anche il tentativo originale Qwen rimane
`NOT_VERIFIED`, interrotto ordinatamente dopo 2203,76 s.
Un caso distinto esplicita nell'input la chiusura del parametro nativo senza
newline e conserva tutti gli assert e il risultato atteso. È accodato una
sola volta per le famiglie il cui workflow originale non passa, senza retry
ricorsivi. L'ultima regressione completa dopo questa fixture passa 1213 test,
49 skip live e copertura statement/branch al 100% in 634,37 s. Gli skip non
qualificano automaticamente i consumer reali. Le successive prove Custom
con sampling greedy e framing esplicito raggiungono entrambe il deadline
client: Flash 7204,17 s, Qwen27 7201,06 s, uscita server zero e nessuna
terminazione forzata. Non è disponibile un JUnit finale: gli effetti parziali
non promuovono il workflow. Ulteriori ripetizioni sono sospese dall'utente.

La baseline `numeric-local-baseline` esegue **14 casi Metal**, tutti PASS con
gli stessi oracle e limiti del benchmark esistente; **11 bit-exact**.
Gli altri tre casi FP32 (MQA, LayerNorm, L2Norm) hanno massimo errore assoluto
rispettivamente 1,68e-8, 7,15e-7 e 2,24e-8. Operandi/output integrali sono
salvati in NPZ. Un warmup e due campioni, verdetto `BOUNDED_SAMPLE`: nessuna
qualifica statistica o misura di speedup contro Strata. Le tre GEMM BF16 hanno
mediane 47,12/9,21/35,92 ms; patch Conv3D BF16 65,47 ms. Il monitor registra
3,1 s complessivi, pressione normale e swap invariato; il picco numerico è
quello cumulativo del contesto, distinto dal footprint del processo.
