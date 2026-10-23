# T19–T22: scheduling, qualifica e consegna

Target locale: RTX 3090 sm_86, Ubuntu 24.04 WSL, GPU singola e compute
serializzato. Baseline sorgenti `fc5ad4a`; worktree iniziale pulito.
Implementazione e comandi nella [guida di rilascio](../../architecture/native-suite-release.md).

Questo report conserva separatamente implementazione, gate locali e rilascio.
La qualità image T11 resta aperta; non viene trasformata in PASS da prove di
residenza bit-exact, da immagini plausibili o dal completamento dello stress.
I nuovi backend/famiglie non disponibili restano `NOT_VERIFIED`.

## Implementazione

T19 aggiunge cache C per gruppi e budget in byte, mantenimento adattivo in base
alle visite, coda mmap→host bounded, due slot pinned contabilizzati, stream ed
eventi CUDA per DMA e riuso. Il planner image offre varianti residenti e streamed
prima di alterare le sessioni. Il medesimo contratto è collegato ai tre rami
Gemma; le reservation nidificate pianificano contro il proprio lease.
Gli hint CPU sono espliciti o derivano da operazioni API completamente validate.
Nessun testo parziale del LLM avvia caricamenti o tool.

T20 aggiunge un registro e un valutatore che impedisce di pubblicare candidati
senza tutti i gate. Il percorso pubblico resta BF16. I profili non implementati
sono rifiutati prima del binding; mix ultimi layer, wavefront e grafi con
indirizzi mobili restano candidati senza promozione. Le soglie T00 non cambiano.

T21 aggiunge stress HTTP riproducibile con 50 cicli dopo warm-up, tool call ID,
quattro modalità più composizione, continuità del LLM, hash deterministici,
campioni e contatori per proprietario dopo il cleanup. T22 aggiunge ricette,
manifest e packaging con un gate di rilascio obbligatorio.

## Equivalenza e residenza

La fixture `text.ds4p` T11, 256×256, seed 20261020, usa gli stessi pesi,
rumore, schedule e kernel in entrambi i percorsi. Il primo confronto ha
RGBA FP32 bit-exact, SHA-256
`ac5320532fcd6120a6e0a8dffcfdf25ea40fcf63881f87d1f43f9a7ade183ba2a`.

| Percorso | Picco device posseduto | Tempo operazione | Piano |
| --- | ---: | ---: | --- |
| Baseline seriale, DiT residente | 14.334.216.192 byte | 12,803 s | variante 1, nessuna vittima |
| Gruppi streamed, budget controllato 4 GiB | 3.488.326.928 byte | 40,464 s | variante 6, nessuna vittima |

Questi due tempi sono campioni diagnostici singoli, non una promozione
prestazionale. Tutti gli otto latenti e RGBA sono confrontati byte per byte;
il fallback con pinned budget zero produce gli stessi byte. Una capacità
software ridotta non qualifica altro hardware.

Gemma sul batch multimodale 3 T15 restituisce gli stessi vettori FP32 nel
percorso residente e con budget controllato di 1.250.000.000 byte: picco
posseduto 630.264.792 byte, bound del piano 1.241.480.320 byte. Le tre torri
sono rilasciate a fine richiesta streamed. Il tentativo a 768.000.000 byte è
rifiutato in ammissione: lo scratch dichiarato non entra. Non si modifica
workload, precisione o soglia per farlo entrare.

## Accettazione funzionale

Passano 50 cicli dopo un warm-up, usando un server e un client SDK: LLM →
image RGBA 64×96 → LLM → embedding testo/image/video/audio/composizione →
LLM. Le immagini usano seed ripetuti e nuovi; gli embedding alternano
128/256/512/768 dimensioni. Restano positivi i cached token dopo entrambi i
tool. Le prenotazioni tornano a zero; dopo warm-up owner 500 resta a 14 root,
zero view, 8.519.680 byte device, 33.450.797 byte host e 4.194.304 byte pinned.

I tempi dello stress hanno CV oltre il 5% e restano `NEED_MORE_DATA`:
nessuno speedup dichiarato. Gli studi T19.5/T19.8, T20.1–T20.5 e T21.6
restano aperti su richiesta dell'utente. Non sono eseguite nuove ablation di
quantizzazione o ricerche wavefront.

Passano i gate core di risorse/executor/transfer/numeric lifetime, dodici test
bundle e sei scenari HTTP SDK reali, inclusi editing, retry, quattro modalità,
continuazione con cached token e disconnessione. Le regressioni host di API,
processor, contratti e profili sono conservate nei log locali. Passano anche
25 test numerici CUDA, executor nelle modalità 0/1/2 con logits bit-exact,
runtime ownership/riapertura legacy e lifecycle con MTP esterno/vision,
checkpoint corrotti e restore fallito.

Passano 64 iniezioni native: cancellazione su conditioning, 32 layer DiT,
otto uscite latenti, frontiere encoder/decoder VAE, posterior ed encoded;
errore di inferenza ai capi delle stesse fasi. Nessun risultato viene
pubblicato e le prenotazioni tornano a zero. Dopo l'intera sequenza il nuovo
RGBA è bit-exact al primo. Il trasferimento supera inoltre ASan/UBSan con
leak detection e cento alternanze; GCC `-fanalyzer` non emette diagnostica
per i due nuovi moduli C.

Il pacchetto `package-functional/` contiene eseguibile, codec, indici/asset,
launcher diretto C e albero sorgenti. L'albero è compilato da pulito; dopo
le correzioni della diagnostica e del campione globale vengono ricompilate
le unità HTTP e CUDA interessate.
Sul binario risultante passano tutti e otto gli scenari: sette SDK e il tool
registrato ExMekh, con continuazione cached e anteprima reale 128×128.
Contesto 2048 invariato. Prima dello shutdown: 17 file, 2.300.504 byte;
dopo lo shutdown: zero file. Il pacchetto resta etichettato `development`,
`release_qualified=false`. Anche la prova reale di rilascio senza
`--development` è rifiutata prima di creare la directory.
Il controllo definitivo verifica tutti i 287 hash del manifest: zero file
non elencati, oggetti di compilazione o sorgenti dello staging non necessario.

La prosecuzione chiude la verifica HTTP per fase con 17 socket reali:
conditioning; primo/ultimo blocco DiT; primo/ultimo passo latente;
encoder/decoder VAE ai layer 0, 1, 2, 3 e 5; posterior ed encoded.
Una barriera compilata nel solo server di qualifica segnala la frontiera
GPU fenced; il client chiude il socket e il percorso ordinario gestisce
cancellazione, cleanup e restore. Dopo ogni caso il LLM ricorda BETA21 con
cached token positivi, le prenotazioni sono zero e restano solo i due file
previsti. Il PNG finale coincide con quello iniziale:
`a6cffaa536c27c49f0b18bede72682bf1b33cd0a067b28d6924fbccc412dd27e`.
Ogni layer e passo DiT resta inoltre coperto dalle 64 iniezioni native;
il distacco socket distinto di tutte le ripetizioni non è dichiarato.

## Esperimenti e limiti

Il primo driver stress ricostruiva i messaggi assistant come testo: il server
ha segnalato correttamente `token-mismatch` nel prefisso canonico. Il driver
finale usa eventi tool registrati e `function_call_output`, conservando il gate
`cached_tokens > 0`. La prova incompleta e i log restano separati.

La matrice HTTP successiva parte da un editing prima del primo prefill LLM.
Questo ordine ha scoperto un falso OOM: il registro condiviso conservava un
campione globale preso mentre i tensori nativi erano vivi. Liberarli non
aggiornava quel campione; l'allocator legacy poteva quindi rifiutare anche
un transitorio di 0,01 MiB. Il backend nativo aggiorna ora il campione dopo
`cudaFree`, come il backend legacy, preservando limite e margine configurati.
La regressione avvelena il campione a zero, libera un tensore e verifica sia
il nuovo campione sia una nuova allocazione. Passano tutti i 25 test CUDA.
Non servono eviction aggiuntive o modifiche alla precisione.

Il binario dello stress precede gli ultimi interventi che conservano il
layout ABI pubblico, aggiungono guardie dei profili e rendono opt-in la
telemetria estesa, con i contatori `live`.
Gli hash dei binari restano distinti; l'accettazione del pacchetto usa
l'eseguibile ricompilato dai sorgenti finali. Le equazioni non cambiano.
Lo stress da 50 cicli precede anche il refresh del campione globale CUDA;
resta riferito al binario registrato. La build finale ha i 25 test CUDA,
17 distacchi HTTP e otto scenari live del pacchetto, compreso editing prima
del primo prefill. Non viene dichiarata una seconda esecuzione dei 50 cicli.

Il primo passaggio dal pacchetto ha superato i sei scenari SDK, ma la
continuazione ExMekh ha ecceduto il contesto di 2048 token: i nuovi ledger,
serializzati nel risultato del tool, portavano il prompt a 3796 token.
La correzione conserva i ledger soltanto con
`DS4_SUITE_RESOURCE_METRICS=1`; il protocollo host verifica default, `0`, `1`
e JSON valido. Il driver stress richiede la diagnostica e non accetta
qualifiche della memoria senza contatori. Il contesto ExMekh resta 2048;
non si riduce il workload o si amplia il contesto per superare la prova.
La ripresa corretta usa 1785 token e restituisce il codice ricordato;
generazione e anteprima restano quelle richieste.

Le fonti per le nuove responsabilità e le esclusioni sono in
[provenance.json](provenance.json). Stato di rilascio e candidati sono in
[qualification.json](qualification.json) e [candidates.json](candidates.json).
Un campo mancante resta `NOT_VERIFIED` o `NEED_MORE_DATA`; il packager non emette
un pacchetto di rilascio in presenza di gate obbligatori incompleti.

Hash, output completi confrontati e log sono congelati in
[verification.json](verification.json); campioni, statistiche campionarie e
stato caldo osservato sono in [stress.json](stress.json). I distacchi HTTP
per fase sono in [disconnects.json](disconnects.json).

I log e gli output numerici integrali sono in
`performance-results/native-suite-t19-t22-20261022/`. I processi modello sono
eseguiti in sequenza. Nessun test TP/distribuito è stato avviato.
