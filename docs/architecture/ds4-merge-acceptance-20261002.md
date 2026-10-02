# Accettazione merge DS4 e QS — 2 ottobre 2026

## Esito e provenienza

**PASS per Qwen3.8-27B sulla RTX 3090.** Sono concluse le prove residue:
CLI vision, append di immagini, client QS reale, memoria tra immagini e tool,
checkpoint skill/subagent e launcher FE con MTP. Il branch è
`codex/merge-ds4-upstream-2026-10-01`; `main` resta `5bf2b96ee5e888a2b09212e84dbc29751ca4a194`.
Il merge completo `0edbded` integra upstream `0aaea5a` (277 commit).

I risultati iniziali restano nel [report del merge](ds4-upstream-merge-20261001.md)
e nel [report vision/checkpoint](qwen27-vision-checkpoint-20261001.md).
Quelle matrici fotografano il 1 ottobre; questa verifica conclude i casi
allora rimandati. Modelli e backend assenti conservano le qualificazioni dichiarate.

QS è al commit `a665e5a85bd00c36031f3fb0c380e61e4ec4f6dd`, working tree pulita e sorgenti non modificati
in questa chiusura. Il wheel FE è sincronizzato: `1863bfa6f9f866f819ca06c8fc1065fc35190ca70a2bcf47008fa1b1a66f8ea1`.
Il lockfile cambia soltanto quell’hash. I test usano la nuova API di lista
ordinata di testo e `ImagesUserMessage`. FE/QS resta l’unico agente del prodotto.

## Foto e memoria tra tool

La fixture JPEG reale `earth.jpg` proviene dal corpus upstream. Sono usati
originale e pixel specchiati della stessa fotografia: fingerprint differenti,
non due scene distinte. Nella CLI i file copiati hanno nomi neutri.

Un tool richiesto dal modello legge un codice casuale da un file e lo elimina
prima delle immagini. Dopo ciascuna immagine, un altro tool registra l’oggetto
e produce una ricevuta casuale. Le risposte intermedie non espongono questi
valori. La chiamata finale, senza tool e con schema JSON, deve restituire
esattamente codice e ricevute. Il codice non compare nei prompt utente.
Il test verifica anche pixel nella history, ordine, tool call e fine degli stream.

La CLI nativa esegue due `/read` e ricorda oggetti e codice. Verifica vision
e memoria conversazionale; i tool reali intercalati sono eseguiti dal test
CLI QS installato in FE, senza ripristinare un agente nativo DS4.

| Controllo | Target | MTP | Evidenza |
| --- | --- | --- | --- |
| CLI foto e ricordo del codice | PASS | PASS | Due encoder, output finale esatto |
| HTTP append/JSON/tool/SSE/thinking/Anthropic | PASS | PASS | 10 casi per variante |
| QS Chat Completions e Responses | PASS | PASS | 6 casi per variante, foto e tool reali |
| QS layout misti e continuation | PASS | PASS | 8 test per variante; 6 documenti nativi esclusi |
| FE history multi-turn, tool e skill | PASS | PASS | Test live esistente |
| VRAM liberata durante vision | PASS | PASS | Target 3.82–3.90 GiB; MTP 4.79–4.97 GiB |
| Checkpoint su prefisso 20K | PASS | PASS | Token/logit bit-exact, tre livelli annidati |

Il launcher target usa contesto 22593, quello FE/MTP 28000. Il gate checkpoint
usa 22593 e popola 20.000 token prima dell’immagine; verifica frontiere,
continuazione, rifiuto atomico di corruzione e sessione ricreata.

## Regressioni e prestazioni

QS: **1.958 PASS**, copertura 100%, Ruff e Pyright puliti. FE: **612 PASS,
16 live esclusi**, copertura 100%, Ruff pulito; UI/Playwright offline e mock
inclusi, con due prove aggiuntive sul modello reale. I gruppi C senza modello (layer pattern, trie e server) passano serialmente;
i contratti Python hanno 109 test, 4 opt-in esclusi. I gate numerici Qwen
e l’oracle mRoPE sono stati ripetuti e passano con le soglie originali.

I SHA-256 CLI/server/bench coincidono con i binari già qualificati con build
CPU/CUDA sm_86, gate numerici e confronti 2K/8K/16K/22K. In questa chiusura
il motore non cambia. Nessun golden promosso, tolleranza allargata o speedup
dichiarato. Il [JSON conclusivo](../performance/ds4-merge-acceptance-20261002.json)
conserva hash, risultati e checksum dei log; i dati estesi restano in
`performance-results/qwen27-vision-20261002/`, ignorati da Git.

## Limiti conservati

OCR `RED7` resta FAIL (`R7` anche nell’oracle llama.cpp con gli stessi GGUF).
La foto non certifica OCR universale. Resta FAIL l’uguaglianza esatta del
timeout socket WSL (50 ms richiesti, 52 restituiti); il gate non cambia.
Questi esiti rimangono separati dall’accettazione Qwen27.

Metal, ROCm, distribuito, modelli assenti e checkpoint vision Flash Next
restano NOT_VERIFIED. Flash Next Q2 mantiene le prove precedenti di inferenza
breve e checkpoint testuali senza/con MTP embedded; non viene introdotto
streaming SSD degli esperti o un nuovo scheduler.

Gli allegati FE restano riferimenti a file; l’invio pixel diretto dalla UI
non è aggiunto. QS e HTTP possono inviare pixel. I dump con hash non
ricostruiscono immagini: gli allegati devono restare disponibili al chiamante.
La qualificazione del riavvio riguarda cache sistema e replay della history,
senza nuova persistenza degli stack privati sospesi.

## Riproduzione

Con gli altri processi motore arrestati:

```sh
./agent/sync_exmekh_lib.sh
python3 tests/run_qwen27_vision_acceptance.py \
  --out-dir performance-results/qwen27-vision-recheck
```

L’orchestratore esegue serialmente le varianti e arresta solo i propri server.
Per ogni verifica serve una nuova directory: il runner rifiuta di sovrascrivere
risultati già presenti. I test non modificano conversazioni FE dell’utente.
