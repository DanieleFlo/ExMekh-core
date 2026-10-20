# Registro risorse e lifecycle T03–T04

Sorgenti di partenza: `821baef5e6451ffbe6cde4dbf2c7f15738c5623f`.
Modifiche preesistenti escluse: `agent/uv.lock`, wheel ExMekh,
`docs/experiments/_INDEX.md`. Modelli, hash e fixture derivano dai manifest T00.
Comandi, hash dei sorgenti/binari e risultati effettivi sono in `verification.json`.
Log e output completi: `performance-results/native-suite-t03-t04-20261020`.
Il riesame successivo è documentato in `review-verification.json` e nella
[revisione delle responsabilità](../../architecture/native-suite-clean-code-review-20261020.md).
I log del riesame sono in `performance-results/native-suite-review-t03-t04-20261020`.

## Accounting

`ds4_resources.c/.h` registra richiesto/allocato/vivo, owner, location, formato,
device, refcount, evento e autorità/ricostruibilità. Formato `0` significa
storage opaco. Le view trattengono anche l'allocazione CUDA reale e non
aggiungono byte. Destructor falliti conservano il riferimento del chiamante.
Il trimming richiede anche un destructor: eliminare soltanto una riga del
registro non equivale a rilasciare lo storage. I free CUDA heap/in-place usano
lo stesso cleanup, propagano il fallimento al registro e conservano l'handle.
Nell'adapter Qwen27 qualificato, pesi/staging hanno owner stabile `2`, il runtime `1`;
sessioni e nuovi componenti hanno owner distinti. Il reload non cambia owner.

I ledger mmap, residente, anonimo, pinned e checkpoint sono distinti. Il budget
host comprende anonimo/residente/pinned, con un ulteriore limite pinned.
Mmap non viene sommato a RSS. Prenotazioni atomiche comprendono copie in volo,
staging doppio e conversioni; commit trasferisce i byte allo storage, cancel
attende la timeline. La copia D2D CUDA marca entrambi gli operandi e il free
anticipato drena gli eventi. Gli altri percorsi mantengono le fence legacy.

La capacità virtuale managed ha un ledger proprio: le pagine possono migrare
e non vengono addebitate come se fossero tutte residenti in VRAM o RAM. Questo
mantiene la semantica dei percorsi legacy con KV managed/streamed.

L'ammissione prende il minimo fra capacità residua posseduta e free globale,
poi sottrae prenotazioni e margine: non sottrae due volte memoria già inclusa
in `cudaMemGetInfo`. `set_budget/get_budget` espongono limiti e conoscenza del
campione globale; `reconcile` aggiorna il campione. Le somme per owner includono
le reservation. Il dump contiene righe `B` (location, limit, margin, global_free,
known, allocated, reserved) e `R` (owner, location, device, format, requested,
allocated, live, refs, event, authoritative, rebuildable, view).

Il confine CUDA intercetta allocazioni device/pinned e registra le mappe GGUF.
Driver/cuBLAS/MMQ restano osservati attraverso il free globale. La memoria host
legacy e le pagine residenti mmap non sono ancora inventariate automaticamente
per owner: `host_known=false`. La riconciliazione fisica nuova è single GPU;
non applica la capacità della GPU zero alla somma dei device TP.

## Checkpoint e residenza

L'API engine separa `trim`, `park/resume` e `weights_release/load`. L'eviction
dei soli pesi mantiene engine, tokenizer, mmap, sessioni e capacità GPU dello
stato. Eval/sync/speculative decode rifiutano risorse non residenti prima di
avanzare token o RNG. Il reload invalida i grafi CUDA contenenti vecchi indirizzi;
runtime, stream, cuBLAS e lock d'istanza restano vivi.

L'envelope `DS4CKP` v1 contiene ABI, fingerprint SHA-256 delle directory GGUF e
dell'identità dei file target/support/vision aperti, digest di configurazione e
checksum del payload/metadata. Device/inode, dimensione e mtime/ctime con
nanosecondi legano il checkpoint al file aperto; il corpo dei pesi non viene
riletto durante una richiesta vision. Gli SHA-256 completi dei pesi restano
nel manifest T00. Cambiare artefatto o usare una copia su un altro inode richiede
un nuovo checkpoint. I test rifiutano anche una modifica del solo corpo pesi.
Il payload interno e l'ABI delle cache server esistenti rimangono invariati.

File e directory vengono sincronizzati prima di distruggere il grafo autorevole.
Il restore valida tutto prima di allocare un candidato; ogni fallimento conserva
stato vivo e file. Solo un candidato validato sostituisce il grafo. Il checkpoint
esplicito resta fino a `checkpoint_free`; quello del parcheggio si elimina dopo
un restore riuscito. I checkpoint multipli non si sovrascrivono.

Identità della sessione, callback, statistiche e stato applicativo host rimangono
negli oggetti originali. RNG/sampler/request/context rimangono del server durante
vision; il checkpoint accetta anche metadata opachi del chiamante e li restituisce
solo dopo successo. Non serializza puntatori del server. Il wrapper vision
riprende soltanto le sessioni che ha parcheggiato esso stesso.
Invalidare una sessione parcheggiata elimina il checkpoint interno: il resume
ricrea uno stato vuoto senza ripristinare la conversazione invalidata.
`set_logits` rifiuta lo stato parcheggiato, il cui checkpoint è autorevole.

Le mutazioni lifecycle pubbliche entrano in uno scope esclusivo tra operazioni.
Una callback di progresso/cancellazione riceve `BUSY` se tenta di parcheggiare
stato o scaricare pesi mentre il chiamante sta usando il grafo. Le query memoria
rimangono consentite. Il wrapper vision possiede già lo scope e usa gli helper
privati dell'adapter, con test CPU dedicati per entrambe le condizioni.

## Inventario degli adapter

| Famiglia/percorso | Stato pertinente | Nuova API lifecycle |
| --- | --- | --- |
| Qwen27 `qwen35` Q4_K_S, CUDA single GPU | KV, recurrent/conv, logits | Gate locale nel manifest |
| Qwen27 MTP esterno | Anche KV MTP, hidden pendente, logits/draft e modalità | Gate locale nel manifest |
| Qwen27 multimodale | Anche fingerprint immagini, posizioni/delta mRoPE | Gate locale nel manifest |
| Flash Next/Mini `qwen4exp` | KV/IK/block keys, recurrent/history, PLE, pos3, nextn; cache/ranking esperti condivisi | `UNSUPPORTED`; regressione legacy distinta |
| DeepSeek V4 | KV raw/compresse/indexer, frontiere, DSpark target/support e pending state | `UNSUPPORTED` |
| DeepSeek V4.1 | Window, compressed/index e coppie di compressione pendenti | `UNSUPPORTED` |
| GLM | KV dense/compatte, recurrent/KDA, MTP/rollback | `UNSUPPORTED` |
| Metal/ROCm/TP/distribuito/SSD | Pool/workspace condivisi, ownership distribuita e code I/O | `NOT_VERIFIED`; nuove API rifiutate senza mutazioni |

Il MTP embedded Flash non è qualificato dal test del sidecar Qwen. Il percorso
vision legacy delle altre piattaforme rimane separato. I processi modello vanno
eseguiti in sequenza; il test host non richiede modelli.

```sh
make tests/test_resources tests/test_runtime tests/test_session_state \
  tests/test_lifecycle_cuda tests/test_runtime_cuda tools/probe_suite_baseline \
  ds4-server ds4-bench CUDA_ARCH=sm_86 -j2
./tests/test_resources
./tests/test_runtime
./tests/test_session_state
./tests/test_lifecycle_cuda gguf/Qwen3.8-27B-UD-Q4_K_S.gguf
./tests/test_lifecycle_cuda gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  gguf/mtp-Qwen3.8-27B-Q4_0.gguf
./tests/test_lifecycle_cuda gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  - gguf/mmproj-Qwen3.8-27B-F16.gguf
./tests/test_lifecycle_cuda gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  gguf/mtp-Qwen3.8-27B-Q4_0.gguf gguf/mmproj-Qwen3.8-27B-F16.gguf
python3 tools/run_suite_baseline.py --output <directory-nuova> \
  --compare performance-results/native-suite-t00-20261020-r2
./tests/test_runtime_cuda gguf/Qwen3.8-27B-UD-Q4_K_S.gguf
```

Tutte e quattro le combinazioni testo/vision e MTP spento/acceso passano le tre
modalità lifecycle: solo pesi, solo sessione e sessione+pesi. La continuazione
speculativa conserva token e logits bit-exact; ogni esecuzione termina con zero
root/view. I test host verificano anche la restituzione dei budget prenotati.
I gate comprendono sessioni multiple, un
checkpoint corrotto indipendente, budget SSD insufficiente e restore fallito
per budget device seguito da retry riuscito.

T03/T04 restano `IN_PROGRESS` per l'inventario automatico host legacy e la
copertura delle famiglie/backend esclusi. T05 conserva ammissione automatica,
transazione server e journal. I gate locali non certificano inferenza nativa
image/embedding o piattaforme non verificate.

Le prove Flash confrontano il binario T00 e quello corrente con il profilo del
launcher e con backing KV streamed (`ctx_alloc=4096`, resident=2052, prefisso
32 token). Questo verifica allocazione/teardown e regressione numerica; non
qualifica una continuazione oltre 2052 token. Il tentativo resident=16 è
conservato come fixture rifiutata anche dalla baseline.

Il probe di errore nativo richiede `UINT64_MAX/2` byte e verifica il rimborso
della contabilità, senza occupare la GPU. Una prima fixture che richiedeva
soltanto total-VRAM+1MiB è stata scartata: WSL/WDDM ne consentiva l'allocazione
virtuale. Anche quel fallimento di fixture è conservato, senza cambiare golden.
