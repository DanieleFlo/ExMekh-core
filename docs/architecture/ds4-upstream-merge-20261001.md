# Merge DS4 upstream — 1 ottobre 2026

Verifiche residue concluse il 2 ottobre: [accettazione finale e integrazione exmekh-lib](ds4-merge-acceptance-20261002.md).
La matrice seguente conserva gli esiti alla data originaria.

## Provenienza e perimetro

Branch `codex/merge-ds4-upstream-2026-10-01`, creato da
`5bf2b96ee5e888a2b09212e84dbc29751ca4a194`. Upstream fissato a
`0aaea5a238fb41a35106a551e73c8409dfb751ac`, con 277 commit dalla base comune
`c1d4597`. La storia del fork viene conservata tramite un merge a due genitori.
`main` rimane il riferimento precedente. Il lavoro è eseguito da un solo agente Codex.

exmekh-agent / exmekh-lib rimane l'unico frontend agentico del prodotto. `ds4_agent.c`,
`ds4_web.c` e il relativo eseguibile non vengono ripristinati. I test standalone
upstream degli agenti nativi sono conservati come materiale di riferimento;
non qualificano exmekh-agent / exmekh-lib e non diventano target di build del prodotto.

Sono integrati loader e grafi DeepSeek 4.1, GLM 5.3, Qwen Flash Next/Mini,
CUDA, Metal, ROCm, vision, memoria/SSD streaming, server, strumenti GGUF,
quality testing e fixture upstream. Le guide originali sono raggiungibili
nell'[indice upstream](../upstream/_INDEX.md).

## Contratti preservati

- Qwen3.8-27B mantiene architettura `qwen35`, ID pubblici, flag, sidecar MTP,
  tokenizer, prompt, kernel specializzati, GDN, grafi CUDA e launcher locali.
- Flash Next/Mini usa `qwen4exp`, con tokenizer, template, MTP integrato,
  hyper-connections, PLE/n-gram e cache specifici. I protocolli server sono
  selezionati per famiglia; le due implementazioni Qwen non sono intercambiabili.
- I valori già pubblici degli enum non cambiano; le famiglie upstream sono aggiunte.
- Restano accettati gli alias `--glm-mtp`/`--glm-mtp-timing` di CLI/server
  e `ds4-eval --mtp FILE`, oltre alle nuove opzioni upstream.
- Qwen3.6 resta escluso da catalogo, loader, downloader e API.
- Tool calling, output strutturati, cancellazione, frontiere residenti e
  checkpoint SSD di skill/subagent restano nel percorso exmekh-agent / exmekh-lib.
- Il formato KVC esterno rimane versione 1; il payload ABI passa da 2 a 3.
  I checkpoint pre-merge vengono rifiutati prima del ripristino. I test coprono
  sia ABI 0 sia ABI 2, evitando di leggere un vecchio layout con il nuovo grafo.

Il ripristino dei checkpoint agentici durante la vita del server conserva
logit e continuazione esatti, inclusi tre livelli annidati. Le skill private
sospese sono associate al processo/sessione: come nel fork precedente, un nuovo
server pulisce i frame orfani. Il recupero dopo riavvio usa history exmekh-agent / exmekh-lib e cache
KV durevoli; non è introdotta una nuova persistenza di stack privati incompleti.

## Sviluppo vision successivo al merge

Il [report Qwen27 vision/checkpoint](qwen27-vision-checkpoint-20261001.md)
qualifica il matching encoder del 27B, mRoPE interleaved, parcheggio completo
delle sessioni su SSD e ripristino di skill/subagent. La matrice qui sotto
resta la fotografia del commit di merge; le verifiche vision successive
sono registrate nel report dedicato, senza promuovere i vecchi risultati.

## Immagini e presentazione del codice

Sono importati integralmente decoder PNG/JPEG, fixture, encoder e percorsi vision
upstream per le famiglie che li supportano, inclusi gli input multimodali HTTP,
le associazioni immagini/prefissi e il loro riuso KV. Il modello e l'encoder
vision devono appartenere alla famiglia corretta e l'encoder va configurato
con `--vision`; una build riuscita non equivale a inferenza vision verificata.

exmekh-agent / exmekh-lib conserva allegati, streaming Markdown, blocchi di codice, syntax highlighting,
copia del testo originale, history e callback di tool/skill. La formattazione
non deve alterare il codice copiato o interpretare marker di protocollo letterali
negli argomenti dei tool. Le prove exmekh-agent esistenti coprono questi contratti.

## Kernel e risoluzioni rilevanti

Il MMQ Q4_K generico upstream non supera il gate numerico esistente del fork su
`sm_86`; il dispatch Q4_K locale resta attivo. Non sono allargate tolleranze né
aggiornati golden. I nuovi formati Q2_K, IQ2_XXS e MXFP4 restano disponibili nei
percorsi upstream. Le ottimizzazioni specifiche del 27B sono preservate.

L'arena CUDA conserva il budget locale di 256 MiB ma alloca almeno la dimensione
richiesta per oggetti più grandi: il sidecar MTP richiede span superiori a quel
budget. Sono mantenute la lookup dei range mmap upstream e la cache locale.
Il server conserva il ripristino della frontiera di ritorno skill prima del
normale riuso dei prefissi e mantiene distinti DSML locale e XML Qwen4 upstream.

Il kernel Qwen4 attention raggruppato upstream provoca un accesso non valido
riproducibile su sm_86/CUDA 12.4. Per questa architettura il dispatch usa il
percorso scalar/split già presente; la suite completa passa anche contro
l'oracle double. Il kernel originale resta importato per gli altri dispositivi.
Il test DeepSeek 4.1 richiede una sincronizzazione prima dell'accesso host alla
memoria managed su WSL: vengono conservati tutti i confronti e tutte le soglie.
Compute Sanitizer è stato provato, ma l'instrumentazione device è indisponibile
con l'interfaccia WDDM corrente; non è stato modificato il driver Windows.

È corretto anche il rewind al confine dei blocchi MTP: il server copia il
prefisso conservato prima che il rewind Qwen35 invalidi e svuoti la timeline,
quindi lo ricostruisce attraverso il sync esistente. Il batch dei token JSON
forzati continua anche dopo un token di confine anticipato dal verifier. Il
gate sul prefill strutturale falliva anche nel binario baseline con MTP e ora
passa senza cambiare il test; questa correzione preserva l'RNG dei token forzati.

I sorgenti e gli script hanno regole Git LF esplicite per checkout Windows/WSL.
Le regole dei fixture e dei metadata con checksum non sono modificate.
I fixture/golden già tracciati nel fork non cambiano. Le 1804 nuove fixture
quality/vision sono identiche ai blob upstream, inclusi gli spazi generati
dai modelli; il controllo whitespace dei sorgenti e della documentazione passa.

## Hardware, RAM e artefatti

RTX 3090, 24 GiB, `sm_86`, driver 610.62, CUDA toolkit 12.4; i9-14900K,
circa 64 GiB fisici. Ubuntu 24.04 su WSL2. Il limite WSL è passato da 16 a 48 GB,
con le altre impostazioni conservate e backup
`C:\Users\danie\.wslconfig.ds4-merge-20261001.bak`.
WSL è stato riavviato prima della baseline e dei confronti.

Il volume Windows aveva circa 438 GiB fisici liberi prima del download Flash
Next; non è stato usato come riferimento lo spazio virtuale del VHDX.
I launcher continuano a selezionare Qwen27. Il downloader offre `--no-link`
per scaricare altri modelli senza modificare `ds4flash.gguf`.

| Artefatto | Byte | SHA-256 |
| --- | ---: | --- |
| Qwen3.8-27B-UD-Q4_K_S.gguf | 15358213024 | `75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017` |
| mtp-Qwen3.8-27B-Q4_0.gguf | 1369590656 | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |
| Qwen3.8-Flash-Next-Q2.gguf | 147207127040 | `b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9` |

Baseline binari: CLI `bfbce4a6b1294bc9a3fc9b4ed1ea0eb1d536de301b94bb217f0988c6847cc31d`,
server `29b5b53645b8bdb0f1979b530d5061d040d5d33cc475f1b015fcfec325239d1e`,
bench `7cf17356077bbcb4a10698c64bf57c14d42d698a76a8431689348f4e77ae1f1b`.
Gli artefatti estesi sono conservati localmente in
`performance-results/merge-upstream-20261001/`, esclusi da Git.

## Confronto Qwen27

Cinque ripetizioni misurate dopo warm-up, contesto allocato 22593, prefill chunk
2048, decode deterministico di 64 token e nessun MTP. Mediane in token/s:

| Contesto | Prefill prima | Prefill dopo | Variazione | Decode prima | Decode dopo | Variazione |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 787.78 | 766.82 | −2.66% | 28.74 | 28.31 | −1.50% |
| 8192 | 717.49 | 711.18 | −0.88% | 25.70 | 25.30 | −1.56% |
| 16384 | 662.54 | 653.70 | −1.33% | 22.53 | 22.36 | −0.75% |
| 22528 | 626.46 | 618.07 | −1.34% | 20.61 | 20.52 | −0.44% |

Prefill e decode hanno CV sotto il 5%; nessuna regressione supera il limite
concordato del 5%. Tutti i 64 token coincidono con la baseline in ciascun contesto.
I gate dei logit su 248320 elementi passano senza modifiche; i logit completi
fra i due binari non sono bit-exact. Il primo passo di decode a 2K ha CV 13.65%
e non è una misura valida del TTFT HTTP. Le misure TTFT sono registrate separatamente.
Il verdetto automatico del perf harness rimane `NEED_MORE_DATA`: i quattro
campi della build (`compiler`, `cuda_arch`, `cflags`, `nvccflags`) non erano
registrati nel manifest della baseline, che congela binari già esistenti.
Hash, hardware, parametri, output e misure sono disponibili; le percentuali
riportate sono confronti osservati e non una promozione automatica del harness.
Non viene dichiarato un miglioramento prestazionale complessivo.


TTFT HTTP misurato fino al primo frammento di testo SSE, con 32 token di
output, warm-up e cinque richieste per contesto. Ogni richiesta ha un prefisso
diverso per evitare il riuso KV. Tutti gli output corrispondenti coincidono.

| Contesto | TTFT prima (ms) | TTFT dopo (ms) | Variazione | CV prima / dopo |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 2712.35 | 2825.83 | +4.18% | 1.34% / 1.03% |
| 8192 | 11498.06 | 11554.95 | +0.49% | 1.27% / 1.52% |
| 16384 | 25067.70 | 24785.55 | -1.13% | 0.29% / 0.63% |
| 22528 | 35884.31 | 35754.86 | -0.36% | 0.13% / 0.36% |

Il TTFT rispetta il limite del 5% con CV sotto il 5% in tutte le prove.
Il gate MTP CLI/server esistente passa con la soglia originale di 20 token/s.
Il riavvio del server, con e senza MTP, recupera il checkpoint del prefisso
sistema (10 token), produce gli stessi 32 token e consente la continuazione.
Questo test non certifica il recupero dell’intero prompt o di stack privati
sospesi attraverso un riavvio.

Il [riepilogo JSON](../performance/ds4-upstream-merge-20261001.json) conserva
hash, misure aggregate e campioni, gate numerici e risultati dei checkpoint.

## Matrice di verifica

Il 27B supera i gate locali di accettazione. Gli stati sotto indicano le prove
effettivamente eseguite; il supporto importato non qualifica automaticamente
modelli o backend non disponibili.

| Controllo | Stato | Evidenza |
| --- | --- | --- |
| Build CPU e CUDA sm_86 | PASS | Binari CLI/server/bench/eval |
| Gate numerici Qwen27 e UD quant | PASS | Tolleranze correnti, nessuna promozione golden |
| CLI Qwen27, MTP esplicito e automatico | PASS | 19 test live, 2 prove opt-in escluse da quel comando |
| Confronto 2K/8K/16K/22K | PASS | 4 contesti × 5 misure, token deterministici identici |
| Server thinking/sampling target e MTP | PASS | Matrice delle due varianti |
| JSON/tool/schema/input ostili target | PASS | Matrice live completa |
| JSON/tool/schema/input ostili MTP | PASS | 18 test live, soglie originali |
| Checkpoint SSD senza e con MTP | PASS | Logit, tre livelli, continuazione e confine contesto bit-exact |
| Corruzione, isolamento sessioni, cancellazione | PASS | Rifiuto prima di modificare lo stato; frontiera server bit-exact |
| Regressioni Python engine/strumenti/wiki | PASS | 236 test, 41 opt-in o dipendenti da modelli non eseguiti |
| CUDA Q8/reductions sintetici | PASS | Shape e buffer con guardie esatte |
| CUDA shared-expert Spark | NOT_VERIFIED | Il test restituisce skip 77 sulla 3090 |
| Trasporto TCP sintetico | PASS | Buffer piccoli, EOF e peer disconnessi |
| RDMA reale | NOT_VERIFIED | Verbs assenti, test escluso |
| Socket timeout WSL | FAIL | test_tp_commands richiede esattamente 50 ms; il kernel restituisce 52 ms, soglia non modificata |
| exmekh-agent regressioni e formattazione | PASS | 612 pass, 16 live esclusi; ruff e copertura 100% |
| exmekh-agent / exmekh-lib provider reale | PASS | Target: 12 pass/4 non esercitati; MTP: 7 pass, incluse skill e history multi-turn |
| Immagini: decoder e prompt/prefissi sintetici | PASS | Test senza modello |
| Inferenza vision con encoder reale | NOT_VERIFIED | Encoder non disponibile e architettura da qualificare |
| TTFT HTTP | PASS | 4 contesti × 5 misure, CV < 5%, output identici |
| Cache dopo riavvio | PASS | Prefisso sistema recuperato, output identico e continuazione, senza/con MTP |
| Gate prestazionale MTP | PASS | Soglia originale CLI/server di 20 token/s |
| CLI/help/placement/sampling senza modello | PASS | 72 controlli shell, test C e parser/rendering server |
| Flash Next Q2 download/checksum | PASS | Download originale verificato, selezione launcher conservata |
| Flash Next inferenza RAM/GPU | PASS | Caricamento, risposta breve, continuazione e streaming da 16 token, senza/con MTP; qualificazione limitata a queste prove |
| Kernel Qwen4 e DeepSeek 4.1 sintetici | PASS | Suite complete sm_86, fallback attention e sincronizzazione host esplicita |
| Metal, ROCm, multi-GPU e distribuito reale | NOT_VERIFIED | Hardware non disponibile |
| DeepSeek 4/4.1, GLM 5.2/5.3, Qwen4 Mini reali | NOT_VERIFIED | Modelli non disponibili/superiori alle risorse del PC |
| Oracle llama.cpp esterno completo | NOT_VERIFIED | Baseline locale e gate correnti non sostituiscono l'oracle esterno |

## Tentativo Flash Next

È usato il Q2 originale, circa 137.10 GiB su disco: circa 41.73 GiB di pesi
principali/MTP e 95.37 GiB di n-gram BF16 da leggere su SSD. Upstream offre Q2 e
Q4; non è introdotto un formato Q1.

Il tentativo usa il percorso esistente, `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=16`,
contesto 2048 e prefill chunk 256. Non viene aggiunto un gestore di offload o
uno scheduler. Lo streaming SSD degli esperti Flash Next non è implementato
upstream; lo streaming degli n-gram è un meccanismo distinto.


Il caricamento e le due richieste brevi producono “Ciao” sia senza sia con
MTP. La continuazione riusa 22 token. Una seconda prova SSE genera gli stessi
16 token (“1, 2, 3, 4, 5, 6”) nelle due varianti:

| Variante | TTFT HTTP (s) | Prefill (token/s) | Decode (token/s) | Durata totale (s) |
| --- | ---: | ---: | ---: | ---: |
| Target | 17.54 | 2.05 | 0.41 | 54.06 |
| MTP | 17.49 | 2.09 | 0.52 | 45.55 |

Sono prove singole su prompt brevi, non un benchmark stabile su contesto 2K
pieno. Il picco RSS/HWM osservato è circa 42.04 GiB; il picco GPU complessivo
è 2.46 GiB secondo `nvidia-smi`. I 16 GiB sono un limite di cache, non memoria
GPU effettivamente consumata. Non è osservato swap del processo. La latenza
e la velocità rendono il modello poco pratico su questo PC; il 27B resta il
modello selezionato. Tool/JSON, vision e contesti lunghi di Flash Next restano
`NOT_VERIFIED`. Non sono scritti nuovi kernel o scheduler per questo tentativo.

Avvio esplicito riproducibile, dopo aver fermato altri processi motore:

```sh
DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=16 ./ds4-server --cuda \
  -m gguf/Qwen3.8-Flash-Next-Q2.gguf --ctx 2048 --prefill-chunk 256
```

Aggiungere `--mtp` per la variante con MTP integrato.

## Riproduzione

```sh
make -j4 ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=sm_86
make -j4 cpu
python3 -m unittest tests.test_model_download tests.test_docs_wiki -v
DS4_TEST_QWEN38_LIVE=1 python3 -m unittest tests.test_qwen38_compatibility -v
DS4_AGENTIC_TEST_DIR="$PWD/performance-results/checkpoint-audit" bash tests/run_agentic_checkpoint.sh
DS4_AGENTIC_TEST_DIR="$PWD/performance-results/checkpoint-audit-mtp" bash tests/run_agentic_checkpoint.sh --mtp
cd agent
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

I test live richiedono un server reale con `--ctx 22593` e il 27B esplicitamente
selezionato. Non eseguire altri processi motore durante i gate: il lock della
repository protegge il modello e impedisce carichi GPU concorrenti.