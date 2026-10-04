# Flash Next: pipeline CUDA e protezione Qwen27 — 4 ottobre 2026

## Perimetro

Implementazione del [piano di azione](../roadmaps/qwen-flash-performance-action-plan-20261004.md) sul working tree `codex/qwen-flash-3090`, HEAD `bd89b13f0c839d3d9e17c7afab1f8a4dcf1b98e6`. Le modifiche preesistenti a cache, server/schema e agente sono conservate. La baseline è il codice effettivamente presente all’inizio di questa implementazione, congelato con sorgenti/binari/hash in `performance-results/flash-pipeline-20261004/`.

Hardware locale: RTX 3090 sm_86, 24 GiB, i9-14900K, WSL Ubuntu 24.04 con 16 CPU logiche e circa 47 GiB di RAM. I campioni sotto carico continuano a mostrare PCIe Gen4 x2. La causa fisica non è risolta dal codice. Il GGUF Flash locale mantiene quantizzazioni originali e PLE BF16; non è sostituito con W4A16/FP8PLE esterno.

## Modifiche

- Capability dell’engine per la cache tiered, attiva solo su Flash CUDA single-device. Un flag Flash ereditato non abilita il budget automatico o la riserva degli allocator del 27B. Guard Apple/ROCm e cleanup espliciti.
- Un readback pinned degli ID `T × NS` per coppia mid/down, invece di due tabelle `NE × cap`. Il builder invalida la generazione; scritture/liberazioni dei tensori invalidano il piano. Chiamate pubbliche con liste precompilate o ripetute conservano il readback sincronizzato.
- Stream di copia separato, gruppi di esperti, eventi copia→calcolo e consumo→riuso. Due gruppi di staging limitano VRAM; i puntatori agli hit residenti rimangono diretti. Tutti gli esperti selezionati sono eseguiti negli slot originali e la riduzione conserva l’ordine precedente.
- RAM pinned in slab da 64 MiB, con riuso degli slot. Il budget conta anche padding e tail inutilizzati. `auto` mantiene il tetto di 16 GiB e lascia almeno 6 GiB o il 25% della memoria disponibile; budget numerici, incluso zero, restano espliciti.
- Ranking delle vittime costruito una volta per proiezione, con stesso pinning, frequenza, recency e tie-break della scansione. Le copie GPU sono vittime RAM preferite; le demotion aggiornano la classificazione. Import dei checkpoint invalida tutti i ranking.
- Contatori di readback, gruppi, I/O, selezione CPU, RAM fisicamente allocata e traffico trasferito, attivati dal flag diagnostico cache stats.
- Launcher con `DS4_MODEL_PROFILE=flash`, modello/proiettore corrispondenti, contesto e chunk configurabili. Qwen27 rimane il default. `CUDA_ARCH` usa `native` quando non esplicitato.

I kernel aritmetici, il router, top-k/tie-break, BF16 PLE, KV e il formato dei checkpoint non sono cambiati. Il percorso full-payload di Flash conserva la parità funzionale dei checkpoint senza introdurre ora uno snapshot compatto.

## Avvio

```sh
# Default Qwen27, comando revisionabile senza build/inferenza:
./start-ds4-server.sh --print-command

# Launcher dedicati dell’agente, uno per modello:
./agent/start-ds4-server.sh        # Qwen27B, MTP, ctx 28000
./agent/start-ds4-server-flash.sh  # Flash Next, ctx 16K / chunk 256

# Le opzioni esplicite restano configurabili; MTP è opzionale:
DS4_MODEL_PROFILE=flash DS4_CTX=16384 DS4_PREFILL_CHUNK=256 \
  ./start-ds4-server.sh --print-command --mtp
```

`DS4_MODEL`, `DS4_VISION_MODEL` (vuoto disabilita vision), `DS4_CTX`, `DS4_PREFILL_CHUNK`, porta e directory/budget KV sono configurabili. `DS4_KV_DISK_DIR` vuoto omette gli argomenti KV del launcher. Il wrapper `agent/start-ds4-server.sh` conserva i precedenti default Qwen27: MTP, ctx 28000 e nessun argomento KV esplicito; `agent/start-ds4-server-flash.sh` seleziona Flash con MTP opzionale. Entrambi i wrapper fissano il proprio profilo; il launcher radice conserva la scelta tramite `DS4_MODEL_PROFILE`. Il contesto richiesto non è ridotto silenziosamente. Il motore rifiuta configurazioni tiered incompatibili. `DS4_CUDA_QWEN4_NO_OVERLAP`, `DS4_CUDA_QWEN4_COPY_GROUP` e `DS4_CUDA_QWEN4_LINEAR_VICTIMS` sono confronti diagnostici, non profili di prodotto alternativi.

## Evidenze e protocollo

Artefatti locali: `performance-results/flash-pipeline-20261004/`. Ogni tentativo conserva log, comando, telemetria e risultato; gli output voluminosi non diventano fixture pubbliche. Un processo modello alla volta e timeout per i gate. Il primo tentativo baseline privo di directory logits è INVALID_SETUP, non un fallimento numerico. Il primo gate HTTP con reasoning cambiato durante il restore e l’interprete fuori virtualenv è conservato come tentativo invalido.

I benchmark usano `ds4-bench`, GGUF locale, stessi prompt/shape/contesto per baseline/candidate, fuori profiler. Warm-up e campioni sono espliciti nei manifest. I logit completi alle frontiere e dopo decode teacher-forced sono confrontati byte per byte. Il tempo del primo prefill del benchmark esclude startup e preload denso e non equivale da solo al TTFT HTTP a processo freddo.

La prova chunk 1024 su frontiere 1K/2K non ha mostrato un vantaggio stabile sufficiente: non è scelto come default del profilo locale. I/O worker dedicati, PLE/prefetch, snapshot compatti e graph CUDA restano subordinati a nuove misure. Nessuna matrice 32K–262K o promozione universale su hardware non disponibile.

## Risultati qualificati localmente

Il [record JSON](qwen38-flash-pipeline-20261004.json) conserva campioni, statistiche complete, comandi, hash di sorgenti/binari e risultati dei gate. Flash: ctx allocato 16384, chunk 256, 16 token di decode teacher-forced, quattro sweep di warm-up e cinque misurati. Ogni frontiera prefilla **256 token nuovi**: la seconda aggiunge il suffisso a un prefisso di 256 token; non è un prompt nuovo da 512 token.

| Frontiera | Fase | Baseline media token/s | Candidate media token/s | CV baseline | CV candidate |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 | Prefill 256 nuovi | 43,58 | 81,22 | 16,80% | 0,54% |
| 512 | Prefill 256 nuovi | 67,09 | 113,93 | 31,86% | 0,68% |
| 256 | Decode 16 | 20,64 | 21,93 | 6,29% | 1,05% |
| 512 | Decode 16 | 17,88 | 18,88 | 8,81% | 1,00% |

**Verdetto prestazionale: NEED_MORE_DATA.** La candidate è stabile in questi campioni; la baseline supera il limite CV 5% del contratto anche dopo ulteriore warm-up. Le differenze indicano una direzione favorevole, senza autorizzare una promozione universale KEEP o una percentuale di speedup certificata. Il percorso resta subordinato alla scelta esplicita Flash/tiered; Qwen27 rimane il default.

Qwen27: ctx allocato 2048, stesso chunk/frontiere/decode, due warm-up e cinque misure, con il flag Flash ereditato ma il normale budget residente del 27B. Prefill baseline/candidate: 29,75/29,61 e 380,09/376,77 token/s; decode: 28,38/28,24 e 27,96/27,79. Tutte le variazioni sono inferiori all’1% e tutti i CV inferiori all’1,2%: **REGRESSION_PASS**. Nel benchmark il 27B usa il fallback di replay del prefisso quando lo snapshot non è supportato: il primo valore prefill non rappresenta da solo la velocità di prefill bulk del modello.

Telemetria dell’intera esecuzione, incluse startup e fasi diverse: picco VRAM 23131 MiB baseline / 22865 MiB candidate; RSS 20268168 / 17805240 KiB; swap zero in entrambi. Utilizzazione GPU massima campionata 84% / 92%, con intervallo circa un secondo: non misura occupancy o saturazione sostenuta e non attribuisce il miglioramento a un singolo kernel. PCIe Gen4 x2 persiste. Nsight non è disponibile: le dipendenze eventi sono implementate e verificate numericamente, la sovrapposizione effettiva non è ancora quantificata con una timeline.

## Correttezza e checkpoint

- **PASS:** 200 posizioni teacher-forced × 248320 logit, 198656000 byte per variante, identici byte per byte alla baseline. Anche logit di frontiera e dopo decode dei benchmark Flash e Qwen27 sono identici.
- **PASS:** primitive CUDA contro oracle CPU, gruppi piccoli/parziali, numero di esperti dispari, crescita indipendente dei buffer, liste esterne aggiornate, cache piccole con eviction e RAM disabilitata. Confronto heap/scansione: stessi hit/miss/eviction nei due record del fixture, senza cambiare il ranking.
- **PASS:** checkpoint Flash target, MTP, vision e vision+MTP; checkpoint 27B target e vision+MTP con flag Flash ereditato. MTP Flash plain/speculative coincide per chunk 1/2 e verify unfused 128.
- **PASS:** composito Flash popolato (512 token di testo + immagine, parent 587 token, ctx 16K, MTP): restore skill, subagent/sessione ricreata, continuation, tre livelli annidati, rollback pre-thinking, parking vision e header legacy; logit e cronologia esperti identici. Corruzione respinta senza mutazione. Il payload pieno è 141315164 byte; capture circa 81–100 ms e scrittura circa 123–157 ms in questo caso. Non è un nuovo formato compatto.
- **PASS:** quattro casi HTTP agentic: registry padre/figlio distinto, skill consentita e ritorno, nesting con consumo dei file nell’ordine corretto, file SSD corrotto conservato dopo il rifiuto. Il fixture parent/child/restore ora dichiara lo stesso reasoning `none` in tutte le richieste: evita troncamenti dovuti al default Flash entro il budget 64 già esistente; vincoli su completamento, restore e cache hit restano invariati. Le prove reasoning sono separate.
- **PASS:** nove turni con reasoning omesso/eco, tool registry, eviction SSD e riavvio server; prompt reali 3081–3422 token, cache hit 3115–3395 token. In questi casi richiesta iniziale 46–48 s, continuazione 4,0–4,9 s, dopo riavvio 9,3–10,5 s (tempi HTTP complessivi, non confronti A/B).
- **PASS:** client tipizzato dell’agente su chat/responses, stream/non-stream, temperature 0 e thinking disabilitato, target e MTP (otto casi). Gli altri incroci di sampling/thinking sono NOT_VERIFIED, non certificati dal filtro di questa esecuzione.
- **PASS:** suite agente 715 test, copertura 100% (40 saltati); statici server, pattern layer 27B, memoria Linux, PLE/n-gram BF16 e policy cache. Compatibilità/launcher/sampling fixture: 22 passati, tre saltati. Build CUDA sm_86 e CPU, smoke `--help` dei quattro binari CPU. Una chiamata del test C è stata allineata alla firma già corrente del parser; il gate launcher ora verifica il comando prodotto e gli override anziché la vecchia stringa hardcoded.

I checkpoint del 27B restano utilizzabili con il 27B; i loro scenari/contratti vengono esercitati sul Flash, non i file KV di un modello diverso. Fingerprint e famiglia continuano a impedire restore incrociati.

## Riproduzione e lavoro rimanente

```sh
make -j2 ds4 ds4-server ds4-bench CUDA_ARCH=sm_86
make test-qwen4-cuda CUDA_ARCH=sm_86
make test-linux-memory test-qwen4-ngrams test-qwen4-expert-cache
./ds4_server_test
python3 -m unittest tests.test_qwen38_compatibility \
  tests.test_server_sampling_matrix.SamplingMatrixFixtureTests -v
(cd agent && .venv/bin/python -m pytest)
make docs-check
```

I comandi precisi per baseline/candidate sono nel JSON. I manifest locali `qualified-performance-gates.json`, `qualified-integration-gates.json`, `final-candidate-gates.json` e i runner `run-final-numerical.py`, `run-qualified-integration.py` conservano anche timeout, ambienti e selezioni. Al termine di quella qualifica, i binari CUDA CLI/server/bench installati erano quelli congelati `.qualified`; la verifica CPU era stata seguita da ricompilazione CUDA e ripristino di questi binari.

La nuova pipeline costituisce il primo blocco del piano, con parte dei gate applicativi già chiusa. Rimangono: stabilizzare la baseline per la decisione KEEP; misurare con profiler disponibile le attese residue SSD/PLE/CPU; ampliare progressivamente prompt/contesti e sampling; qualificare gli altri hardware. Pool I/O, PLE/prefetch, snapshot compatti e graph verranno scelti soltanto se i profili ne giustificano il costo. Metal, ROCm, distribuito/multi-GPU, browser UI live e matrice 32K–262K sono **NOT_VERIFIED** in questa esecuzione.


Aggiornamento dei launcher: i due entry point dell’agente sono separati dopo la qualifica numerica. Gli hash dei launcher nel JSON documentano lo stato al momento dei benchmark; in quel passaggio il motore e i binari qualificati non erano cambiati. I wrapper separati sono verificati con sintassi Bash e comandi prodotti, inclusi profilo ambientale opposto e override espliciti.

La successiva [revisione del codice](../architecture/clean-code-review-20261004.md)
riorganizza i confini dell'agente, unifica l'interpretazione dei parametri nativi
Flash e centralizza il confronto delle vittime della cache senza modificare
l'aritmetica del modello. I suoi binari e gate sono registrati separatamente:
gli hash e i campioni del JSON precedente restano quelli della qualifica originale.
Il verdetto prestazionale **NEED_MORE_DATA** resta invariato.
