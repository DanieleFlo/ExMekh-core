# Revisione clean code di T15/T16

22 ottobre 2026. Riesame di audio, composizione multimodale, vision/backbone,
primitive numeriche, gruppi pesi, runtime/executor, decoder/processor e confini
della pipeline Qwen Image. I moduli C usano handle opachi: le responsabilità
appartengono ai componenti proprietari e i chiamanti ne usano i metodi.
La revisione non certifica ogni file della repository o backend non disponibili.

## Correzioni e responsabilità

| Problema | Intervento |
| --- | --- |
| Un header privato accoppiava i tre encoder e calcolava anche un totale mai usato | Ogni encoder espone la propria query di byte residenti; rimosso `ds4_gemma_weights_internal.h`. Il planner non conosce gruppi, indici o struttura dei pesi |
| Estimate e planner attraversavano nuovamente gli stessi media e ricostruivano le stime di fase | Una raccolta unica delle stime fornite dai componenti, condivisa da co-residenza e fallback; somme controllate e massimo delle fasi distinte |
| L'esecuzione della richiesta gestiva anche tutti i buffer di ogni elemento | `encode_item` possiede staging e cleanup delle feature; l'operazione possiede transazione, reservation e pubblicazione del risultato |
| La proiezione audio sceglieva il layout tramite il nome del tensore | Il grafo dichiara esplicitamente `AUDIO_ROWS` o `AUDIO_TRANSPOSED_ROWS`; il metodo di proiezione gestisce clipping e layout senza convenzioni sui nomi |
| Le posizioni relative avevano un ciclo di allocazione/cleanup separato dai temporanei audio | Metodo dedicato e workspace unico con capacità verificata dalla dimensione dell'array; eliminati i tre buffer esterni e il cleanup duplicato |
| I limiti scalari duplicavano il lookup dei pesi dentro il metodo di proiezione | Lookup e lettura dello scalare sono metodi locali dell'encoder, separati dall'equazione della proiezione |
| Creazione numerica CPU/CUDA, allocazione e query dei token lasciavano valori precedenti dopo errori | Output azzerati prima della validazione, anche quando manca uno degli output indipendenti; il wrapper visual delega al proprietario dei token |
| RMS rifiutava una scala collocata nella capacità inutilizzata del risultato | Verifica dell'aliasing sui soli prefissi letti/scritti, come le altre primitive. Le code rimangono invariate e le vere sovrapposizioni restano rifiutate |
| L2 e RMS duplicavano il calcolo della geometria CUDA | Un solo dispatcher host; kernel, ordine aritmetico e geometria qualificata restano quelli precedenti |
| L'header vision non aveva la protezione C ABI degli altri encoder | Uniformato `extern "C"` per i chiamanti CUDA/C++ esistenti |

I nuovi test riproducono gli output non azzerati CPU/CUDA e il rifiuto
erroneo per capacità inutilizzata prima delle correzioni. Sono conservati
i log `api-before.log`, `cuda-api-before.log` e `capacity-before.log`, distinti
dagli esiti finali. La stessa regressione del costruttore gira su CPU e CUDA.

## Principi applicati e confini

| Principio | Applicazione verificabile |
| --- | --- |
| Responsabilità singola | Decoder: bytes/frame/waveform. Processor: feature/maschere/cache. Encoder: binding e grafo. Backbone: token e scatter. Orchestratore: ordine, staging e piano. Registro: accounting. Runtime: serializzazione |
| Incapsulamento e dipendenze esplicite | Handle opachi, viste immutabili e query del proprietario. Nessun cast dell'handle di un altro componente; il ponte interno per lo scatter resta posseduto dal backbone |
| Interfacce ristrette | Query di residenza espongono solo byte; nessun elenco di tensori/gruppi entra nel planner o nell'API client |
| Coerenza e gestione degli errori | Output puliti su input invalidi; cleanup locale unico; stesse regole di capacità/aliasing per le primitive; rifiuto esplicito di budget e dimensioni invalide |
| Riduzione delle duplicazioni | Una raccolta di costi, un lookup audio, un workspace audio e una geometria host per L2/RMS. Le equazioni specifiche dei modelli restano nei rispettivi adapter |
| Nomi e leggibilità | Layout espliciti al call site; separati piano, esecuzione, feature e posizioni relative; commenti spiegano le scelte numeriche e di lifetime |
| Testabilità e semplicità | Test di regressione con difetto riprodotto, oracle già congelato, replay bit-exact e sanitizer. Nessun framework di classi, nuova gerarchia, flag numerico o astrazione di modelli incompatibili |

Nei moduli adiacenti il registro e l'executor possiedono già le scelte di budget
e il lifecycle; il bridge numerico usa l'API dei gruppi pesi senza interpretarne
lo storage. La pipeline Qwen interroga il piano del VAE e quello eseguito
dall'executor. Questi confini sono stati mantenuti, senza spostare policy nel
backend o applicare una riscrittura estesa al monolite LLM.

Il codice matematico audio resta esplicito dove l'ordine delle operazioni è
parte della qualifica BF16. Estrarre una classe generica per ogni singola
operazione o modificare riduzioni e layout per ragioni estetiche renderebbe
più difficile verificare la corrispondenza con il checkpoint.

## Verifica e limiti

Il [report della revisione](../performance/native-suite-t15-t16-20261022/review-verification.json)
conserva comandi, hash, test e confronti con la qualifica precedente.
Il replay completo confronta 1952 file di input/output del corpus a 287 casi,
inclusi i 1450 confini audio. Tutti i vettori a 128/256/512/768 dimensioni,
batch 1/2/3, nove cancellazioni, piani capiente/limitato e caricamento selettivo
devono restare bit-exact. Non si modificano corpus, pesi, recipe o soglie T00.

Le regressioni CPU/CUDA, i test host di media/tokenizer/API, i gate C di
risorse/runtime/executor e i sanitizer coprono i confini modificati. La build
LLM verifica anche l'assenza di librerie media nei quattro eseguibili legacy.
Le misure temporali sono descrittive, senza promuovere uno speedup.

L'analisi GCC è stata eseguita con `-fanalyzer -c` su audio, orchestratore,
backbone, vision e primitive numeriche. Il vecchio comando `-fsyntax-only`
registrato nella prima qualifica verificava la sintassi, non equivaleva a
questa analisi. Restano quattro segnalazioni di possibile puntatore nullo,
riesaminate senza disabilitare i warning:

- `requests.data`: il percorso ipotizza zero byte, ma `count` è compreso fra
  1 e 1024 e viene moltiplicato per una struttura di dimensione costante.
- `media.data`: il percorso conta zero media nel primo passaggio e incontra
  un media nel secondo, sugli stessi input immutati. Ogni inserimento ha
  incrementato il conteggio nel primo passaggio, limitato a 1024² elementi.
- Buffer embedding: l'analizzatore assume `n == 0` per l'allocazione, poi entra
  nel ciclo `i < n`. Il tokenizer produce inoltre sempre BOS/content/EOS.
- Conteggi vision: `estimate` ha già rifiutato zero frame nella stessa vista
  immutabile del media; l'analizzatore perde questa condizione fra le chiamate.

Sono falsi positivi rispetto alle invarianti controllate dei rispettivi
proprietari, non quattro correzioni omesse. Log e comandi integrali restano nel
report. ASan/UBSan e il replay verificano anche questi percorsi con input reali.

Metal/ROCm, TP/distribuito e stress integrato LLM richiedono hardware o
qualifiche dedicate e non ricevono un `PASS` da questa revisione. Restano i
limiti già dichiarati: contesto operativo 8192, segmentazione esplicita,
workspace conservativi e routing HTTP previsto da T17.
