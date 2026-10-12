# Test completi XML v2 del 11 ottobre 2026

Le suite locali passano. Le prove live conservano fallimenti: il server è
passato a v2, il client applicativo richiede ancora v1 e alcuni output del
modello non rispettano il contenuto richiesto. La qualifica non è tutta verde.

Il [report JSON completo](interactive-edit-xml-v2-full-tests-20261011.json)
contiene esiti per caso, traceback, skip, configurazioni, hash e profili.
Il [wire sintetico](interactive-edit-xml-v2-full-wire-20261011.json) conserva
richieste e risposte prima/dopo la correzione. La
[qualifica precedente](interactive-edit-xml-v2-qualification-20261011.json)
rimane disponibile come evidenza storica.

## Correzione server trovata durante i test

Uno stream batch terminava con `incomplete_generation`: il parser ammetteva
`<&lt;/｜DSML｜parameter`, un opener XML incompleto seguito da un escape
parziale del delimitatore nativo. Da quel prefisso non esisteva una
prosecuzione valida. Il server ora impedisce il fallback dell'escape quando
sono incompleti markup XML, entità o scalar UTF-8, anche nei path. Le regressioni
C coprono il prefisso reale DSML e il trasporto Qwen. Il difetto è distinto da
un documento valido con contenuto diverso dal testo richiesto.

Lo stream di regressione passa sul server corretto in target, MTP e batch.
I confronti
maschera/oracolo verificano l'accordo tra i filtri; il difetto iniziale mostra
perché zero divergenze da solo non dimostra la vitalità dei prefissi. Snapshot
ripetitivi e confini CR/LF hanno anche un oracolo indipendente su 400 fixture.

## Suite locali

| Suite | Risultato |
| --- | --- |
| `make test`, gate disponibili CUDA/motore/server | PASS, ripetuto dopo la correzione |
| Python server, gate Qwen CLI/MTP/performance/SHA inclusi | 274 eseguiti: 222 PASS, 52 SKIP |
| ExMekh-Lib unit | 2482 PASS, 47 SKIP |
| Agent unit | 931 PASS, 40 SKIP; copertura 100% |
| Parser C ASan/UBSan, leak detection | PASS dopo la correzione; oggetti del motore collegati non strumentati |
| Verifica numerica MTP nativa | PASS: 231 token speculativi, chunk massimo 3, gap argmax peggiore 0 |
| Checkpoint agentici extreme | PASS: 10000 token parent, restore/cancel/nesting e confronti bit-exact |
| Ruff lint, formato Python, client Pyright | PASS |
| Formatter esteso a tutta ExMekh-Lib, inclusi esempi Markdown | FAIL: sei documenti preesistenti, dettagli nel JSON |
| Indici e collegamenti della documentazione | PASS, 10 controlli |

I controlli saltati restano elencati nel JSON. Sei fixture benchmark agent
sono state ripetute con entrambi i package sul PYTHONPATH e passano. La raccolta
Python finale include queste fixture e il nuovo caso vision opt-in.

## Suite live

Le prove usano CUDA, RTX 3090, Qwen3.8-27B Q4_K_S, mmproj F16 e sidecar MTP Q4_0.
Contesto principale 22593; batch quattro sessioni con contesto 3072. È avviato
un solo modello grande alla volta. MTP reale omette completamente
`--batched-session`; anche il valore 1 disabilita la speculazione.

| Suite | Metodi/casi | Asserzioni fallite | Errori | Skip |
| --- | ---: | ---: | ---: | ---: |
| XML target dopo la correzione | 16 | 3 | 0 | 0 |
| XML MTP dopo la correzione | 16 | 3 | 1 | 0 |
| XML batch dopo la correzione | 16 | 3 | 0 | 0 |
| HTTP target completo prima della correzione, XML/agentico/JSON/sampling | 59 | 3 | 0 | 0 |
| HTTP XML + sampling MTP reale prima della correzione | 22 | 3 | 0 | 0 |
| HTTP XML batch prima della correzione | 16 | 5 | 0 | 0 |
| ExMekh-Lib live | 47 | 16 | 0 | 1 |
| Agent live | 40 | 3 | 0 | 22 |
| SDK typed Chat/Responses, Qwen27, matrice portabile | 20 | 0 | 0 | 0 |
| Browser bootstrap separato | 1 | 0 | 0 | 0 |

Le asserzioni includono sottocasi: due fallimenti delete appartengono allo
stesso metodo. Non sommare queste colonne come un numero di metodi passati.
Il bootstrap browser saltato nella suite completa è eseguito separatamente
con successo. Dei 22 skip agent, 20 richiedevano Flash: la matrice portabile
è poi eseguita su Qwen27, che non qualifica il modello Flash effettivo.
Gli altri due casi non verificano il payload avversario previsto
(timeout / modello che non produce quel payload).

JSON Schema e protocollo agentico ordinario passano tutti i casi. Le matrici
sampling target/MTP hanno 12/12 scenari ciascuna; tutti i sei hash semantici
greedy coincidono. Questi confronti precedono la correzione limitata al parser
XML e restano separati dalle ripetizioni XML.

Nei profili delle ripetizioni XML si contano 601 confronti maschera/oracolo
target, 919 MTP e 598 batch, con zero divergenze. Il run MTP registra 59 cicli
speculativi vincolati e 58 draft accettati. Non compaiono errori «nessun token
valido» nei tre run. MTP conserva comunque il timeout di creazione a 300 s:
il modello ripete chiusure bare ammesse come contenuto senza completare la call.

Le prove CLI client includono interruzione/riavvio e completamento spontaneo.
Restano FAIL: un payload usa `tool_choice=none` insieme a capability agentiche;
il run combinato esauriva inoltre il budget dei checkpoint. La ripetizione
su un server fresco conserva entrambi i fallimenti: uno raggiunge
l'incompatibilità file-processing v1, l'altro completa ma non soddisfa le
asserzioni sulle capability richieste. Il problema di quota non si ripete.

La prova vision `tool-409` passa: una continuazione con immagine non ricostruibile
riceve 409, mentre quella testuale riusa il checkpoint. La prova eviction
fallisce prima delle eviction forzate: A2 riusa 2613 token, ma `alien1-again`
riusa 0 invece degli almeno 21 attesi. Il trace indica uno slot senza checkpoint;
la causa routing/canonicalizzazione rispetto a un miss sicuro resta aperta.

## Responsabilità e controlli ancora aperti

| Esito | Responsabile |
| --- | --- |
| Prefisso XML senza prossimi token | Server: corretto, con regressioni e sanitizer |
| `code` generato solo dal server, prefill e chiusura corrispondente | Server: verificato; reply client con code rifiutate 400, stato preservato |
| Delete produce riscrittura valida; CDATA viene emesso come opener escapato | Modello/inferenza: contenuto non conforme; maschera non deve dedurre l'intento dal prompt. Causa modello rispetto a prefill/KV non stabilita |
| MTP ripete bare `</new_text>` e non chiude entro il timeout diagnostico di creazione | Modello/inferenza/framing: prefisso testuale ammesso, richiesta non completata; timeout 300 s conservato |
| Nove fallimenti lib chiedono v1 | Client: migrare capabilities, wire, parser e dispatcher a v2 |
| Sei fallimenti lib inviano file nativi | Client: convertire allegati in testo/immagini supportati |
| Prompt deep-thinking 61533 token con contesto 22593 | Client: dimensionare cronologia/output; server: rifiutare capacità superata |
| Tre test agent senza feedback atteso per saved output ID/placeholder | Client: validazione semantica e feedback; diagnosi live ancora necessaria |
| `tool_choice=none` con capability agentiche | Client: costruzione del payload secondo il contratto server |
| Mancato riuso cache / eviction non raggiunte | Server: routing e replay ancora da qualificare; nessun bypass XML dimostrato |
| Path, workroot, inspect/read, hash, intervalli start/end, alias, overlap, hook e scrittura | Client: restano obbligatori; le prove server XML non applicano edit al filesystem |

Una riscrittura valida o il mancato adempimento di una richiesta delete non
diventano un bypass della maschera. I test di intento rimangono falliti.
Il caso CDATA con la chiusura codificata effettiva passa nel parser nativo;
il modello live non produce il documento richiesto, quindi quella generazione
non è qualificata.

## Tentativi e limiti conservati

Il primo tentativo MTP usava `--batched-session 1`: il runtime disabilitava MTP.
Le asserzioni sui contatori rilevano la configurazione errata. I risultati
restano un tentativo non qualificante; MTP è poi ripetuto con speculazione attiva.

Due configurazioni a quattro sessioni raggiungevano circa 24200 MiB dei
24576 MiB disponibili: contesto 22593 con vision, e contesto 4096 con vision e
sidecar MTP caricato ma inattivo. Sono interrotte e non ricevono un PASS.
Il run batch target finale usa 3072, senza sidecar, con margine di memoria.

Metal, ROCm, distribuito, streaming SSD degli esperti, Flash reale e integrazione
filesystem client XML v2 restano NOT_VERIFIED. Timeout, casi non raggiunti e skip
non sono successi. Le istanze temporanee sono fermate. Le sole cache v2 di test
sono rimosse dopo l'archiviazione dei risultati.
