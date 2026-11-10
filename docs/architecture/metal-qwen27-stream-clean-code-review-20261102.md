# Riesame Clean Code Qwen27 streamed, 2 novembre 2026

Base HEAD `089255a1d3b6162e59c61a6bede9ece5e672330f` più Q-P03 non
committato. La richiesta comprende riesame dei consumer adiacenti, test e
commit distinti. Evidenze precedenti Q-P02/Q-P03 restano storiche e non si
attribuiscono automaticamente a un refactoring successivo.

## Riesame prima del codice

Letti reader, coda, cache Metal, consumer CUDA/PLE/transfer, test host e UD,
Makefile, qualificatore e rapporti. Cache/binding/fence appartengono a Metal;
descriptor, partizioni, I/O e contatori appartengono al reader; mutex, task e
worker appartengono alla coda. I consumer usano già i metodi esposti senza
accedere ai campi interni di reader/queue: questo confine va conservato.
Non servono classi C++ o una gerarchia di oggetti per il codice host C privato.

Il metodo completo usa invece indici artificiali per richiamare la lettura
indicizzata: non viola ownership, ma mescola due modalità del dominio.
L'esecuzione di un batch (avvio lazy, submit, wait di tutti, pubblicazione di
counter e primo errore) deve essere un helper privato del reader, riutilizzato
dai due pianificatori. Conservare esattamente partizioni, coda bounded,
ordine dei submit/wait, chunk 8 MiB, soglia 32 MiB e coda dispari sincrona:
questo riesame non è un nuovo esperimento di scheduling.

La fixture grande mescola verifica del pool lazy, letture, EOF, sentinelle e
recovery. Separare gli assert del contratto di lettura completa dall'impostazione
della fixture e verificare il numero di byte restituiti/contati su confini di
file indipendenti dalla formula di partizione dei worker. Non far conoscere
ai test l'implementazione dei job quando basta l'API del reader.

Le dipendenze Makefile di header/consumer sono presenti. I percorsi CUDA,
PLE e transfer hanno ownership di staging/lifetime diversi: non spostarli
nel reader Metal solo per eliminare codice somigliante. Il core e il runtime
Metal conservano stato globale e responsabilità legacy: una certificazione
di tutti i principi Clean Code dell'intero repository non è sostenibile con
questo riesame; si correggono i confini coinvolti e si documenta il residuo.

## Riesame delle fonti pertinente a questo refactoring (prima del codice)

Riletti i file locali verificati contro i blob Git dei pin seguenti:

- Strata `46c152a1310f77cd2f0ce2c7cd213962df14fe83`,
  `metal/src/engine/stream.cpp:read_part,submit,io_main`: pezzi distinti e
  pubblicazione soltanto dopo l'ultimo. Non trasferire arena/predizione,
  conversioni o abort sugli errori; nessuna copertura Qwen27 Metal completa.
- llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`,
  `src/llama-mmap.cpp:read_raw_unsafe` e wrapper Metal/GGUF dell'audit Q-P03:
  EINTR, short read e ownership; non fornisce l'oracle di drain del nostro pool.
- Transformers `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097`,
  `models/qwen3_5/modeling_qwen3_5.py:Qwen3_5MLP,RMSNorm` e Flash già
  verificato: proiezioni, dtypes e frontiere FP32/BF16 non cambiano.
- Ricerca locale: [Q-P03](../performance/metal-inference-20261102/qwen27/q-p03-dense-reads/README.md),
  [precedente riesame Flash](metal-flash-stream-clean-code-review-20261102.md),
  coda e relativi test. Il profilo Q-P03 è ancora il riferimento; non è emersa
  una nuova discrepanza numerica o una regressione misurata prima del riesame.

URL/SHA-256 dei pin sono nel [manifest Q-P03](../performance/metal-inference-20261102/qwen27/q-p03-dense-reads/sources.json).
Snapshot di sorgenti/binario prima del riesame:
`build/metal-perf-qwen27-20261102/clean-review/initial.json`.

## Piano di verifica

Test host e ASan/UBSan, casi originali inclusi; fixture Metal resident/streamed,
lifecycle e oracle scalari; Qwen27 target/MTP e regressione Flash in sequenza
con input indipendente e criteri congelati. Confrontare file di logits/stato
col Q-P03 precedente. Controllo prestazionale limitato contro il binario
precedente, con parametri identici e senza build concorrenti; i suoi campioni
non riscrivono la qualifica statistica Q-P03.

Commit separati per ottimizzazione originaria, refactoring/test, copertura UD
e documentazione/evidenze. Nessun dump o peso grande viene aggiunto a Git.


## Modifiche applicate

`ds4_file_reader_run_batch` possiede avvio, submit, drain e pubblicazione.
`read` prepara regioni contigue, `read_indexed` prepara selezioni; nessuno
costruisce più l'altra modalità per ottenere concorrenza. Parametri e ordine
I/O invariati, con costanti private per chunk/soglia. La coda condivisa e i
consumer GPU non cambiano. Il diagnostico EBUSY identifica ora la richiesta
rifiutata; un precedente errore I/O continua ad avere priorità, dopo il drain.

I test controllano il rifiuto del submit tramite l'API della coda, senza
ispezionarne lo stato. Soglia/laziness, EOF a zero, a un terzo del file e
nell'ultimo byte, contatori e recovery sono separati dal setup e indipendenti
dalle partizioni dei worker. I casi UD sono ordinati e distinti dai casi
piccoli generici: sparisce il numero magico del prefisso dell'array, soglie
numeriche e copertura precedente conservate.

Fuori dall'area modificata, il manifest `SOURCES` del qualificatore Metal
ometteva i due header host e i relativi test, pur includendo il consumer Metal.
Aggiunti gli input mancanti nel proprietario della provenienza, senza farli
conoscere ai modelli o duplicare il relativo algoritmo nel qualificatore.
Le dipendenze Makefile erano già corrette e non richiedono riscrittura.

Test host e ASan/UBSan iniziali PASS. Dei 49 test Python, nella prima prova
sandbox uno non può fare bind localhost (48 PASS, un errore di permesso);
la ripetizione esterna è registrata separatamente. Nessun risultato fallito
viene trasformato in PASS. I gate successivi partono soltanto dopo la build
completa riuscita.

Prima della raccolta il controllo viene fissato a una coppia di warm-up
scartata e cinque coppie alternate, frontiere 64/128 (64 nuovi token ciascuna)
e quattro teacher-forced, 8192/64/cache 2 GiB. Harness identico con sei decimali
nel CSV, oggetti Metal prima/dopo separati, nessun profiler/build concorrente.
Si riportano dispersione e differenze senza rivendicare un nuovo speedup;
un eventuale segnale negativo riapre l'indagine specifica prima di cambiare
policy, kernel o dispatch.

Dopo la prima parity host rilette pubblicazione/countdown di `io_main` Strata,
EOF/EINTR ggml e invarianti borrowed della coda C. Il reader mantiene un solo
chiamante proprietario; i job scrivono soltanto destinazioni disgiunte e il
caller pubblica i risultati in ordine. Errori del pool non diventano errori
parziali non drenati. Nessun nuovo mismatch numerico osservato.

## Risultati finali

Tutti i dieci gruppi di verifica successivi alla build passano: 49 test host
(la ripetizione esterna include il bind), `make test-metal-foundations`,
ASan/UBSan, fixture feature Metal4/general, primitive/grafo Flash streamed,
Qwen27 target/MTP e Flash completo. Le fondamenta comprendono anche 24 test
host già inclusi nei 49, oracle quant/GDN, mRoPE, grafo hybrid e lifecycle
Qwen27 residente/streamed e Flash. Non sommare i 24 come casi nuovi.

I tre gate reali conservano ciascuno nove vettori full-vocabulary e il payload
prefill **bit-exact** col Q-P03 precedente. MTP conserva token esatti e delta
massimo1,1920929e-5 rispetto al decode ordinario, identico a prima e dentro
la soglia congelata. Il codice e i test che eseguono aritmetica mantengono
precisioni, ordine e coefficienti numerici; restano i due warning preesistenti
del core incluso dalle fixture. Il `make test` globale comprende ogni workload
GGUF, anche grande e fuori profilo; qui si esegue la suite bounded pertinente
con i gate completi espliciti sopra, senza riavviare i consumer live sospesi.

Controllo fuori profiler, un warm-up e cinque coppie alternate:

| Fase / frontiera | Prima mediana | Dopo mediana | Delta TPS / riduzione tempo |
| --- | ---: | ---: | ---: |
| Prefill64, TPS | 11,322227 | 11,348771 | +0,23% |
| Prefill128 (64 nuovi), TPS | 11,256729 | 11,236042 | −0,18% |
| Decode64, TPS | 0,421886 | 0,431742 | +2,34% |
| Decode128, TPS | 0,424979 | 0,450296 | +5,96% |
| Richiesta64 +quattro eval, secondi | 15,283006 | 14,952732 | −2,16% |
| Richiesta128 +quattro eval, secondi | 15,305157 | 14,718173 | −3,84% |

Sono controlli di revisione, non speedup promossi. CV decode11,69/12,07% prima
e12,72/8,04% dopo; request8,27–10,04% nel caso64. Anche il prefill dopo supera
il5% di CV. Tutti gli intervalli appaiati stimati attraversano zero.
**NEED_MORE_DATA** per certificare una stretta non-regressione o una nuova
qualifica prestazionale. Nessun campione eliminato, nessuna estensione selettiva;
i dodici processi terminano PASS con i guard originali. I dati Q-P03 restano
riferiti al codice allora misurato, senza riscriverli come nuovi risultati.

Il primo campione lento compare nel decode64 di entrambe le versioni; un
altro nel decode128 precedente al refactoring. Questa osservazione non prova
la causa. Per il nuovo segnale sono stati riletti `read_part/io_main` Strata,
retry/EOF ggml, callback pread prima/dopo, executor e pianificatori locali:
stessi quattro worker, byte, offset, ordine dei job e fence. I test isolati
e gli stati completi coincidono. Il precedente caso Flash di variabilità è
un precedente documentale, non una spiegazione automatica di questi campioni.
Non si cambia kernel/dispatch/policy sulla base del rumore e non si attribuisce
il segnale al carico concorrente senza una sua misura indipendente.

Il refactoring richiesto è mantenuto per chiarezza delle responsabilità e
correttezza verificata, conservando esplicitamente il limite prestazionale.
Dettagli, campioni, min/p10/mediana/media/p90/max/SD/CV, intervalli stimati,
comandi e hash nel [report verificabile](../performance/metal-inference-20261102/qwen27/clean-code-review.json).


## Commit

- `2ede410`: perf(metal): parallelize bounded dense weight reads.
- `614980f`: refactor(io): separate read planning from batch execution.
- `eff9f9c`: test(metal): cover UD projection token boundaries.
- `87cf102`: fix(tools): fingerprint Metal file reader sources.

Il commit finale separa report, roadmap, indice, profilo documentato e ignore
degli artefatti locali. Il checkout di produzione viene lasciato aggiornato.
