# Indice: architecture

Indice generato; non modificare a mano.

## [Revisione del client e dell'agente — 15 ottobre 2026](clean-code-client-review-20261015.md)

`ProviderResponse.history_messages()` possiede la costruzione della history

- righe 9–49: **Responsabilità e comportamento** — `ProviderResponse.history_messages()` possiede la costruzione della history
- righe 50–76: **Principi controllati** — Lo screening dei richiami privati ha trovato soltanto accessi fra istanze della
- righe 77–91: **Verifiche** — Le suite complete e i controlli statici sono registrati nel JSON collegato.

## [Revisione del codice e qualifica live — 14 ottobre 2026](clean-code-live-review-20261014.md)

`FileWriteTarget` raggruppa workspace, risorsa, path canonico e snapshot.

- righe 9–38: **Responsabilità, incapsulamento e argomenti** — `FileWriteTarget` raggruppa workspace, risorsa, path canonico e snapshot.
- righe 39–54: **Principi controllati** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 55–73: **Configurazione e testo** — `config/interactive.py` è il proprietario della versione wire, del tipo di
- righe 74–91: **Difetto aggiuntivo trovato dai test live** — Dopo cancel, il checkpoint dell’ultimo turno completato era pubblicato, mentre
- righe 92–111: **Verifiche e distribuzione** — Passano 2767 test della libreria, 927 test paralleli dell’agente e 3 test UI
- righe 112–125: **Limiti e lavoro strutturale restante** — Il server C, alcuni moduli di inferenza e alcune funzioni di orchestrazione UI
- righe 126–140: **Verifica successiva della collocazione dei testi** — Il controllo è stato esteso oltre le sole eccezioni: restavano istruzioni del

## [Revisione del codice Flash e dei confini dell'agente — 4 ottobre 2026](clean-code-review-20261004.md)

Revisione delle modifiche pendenti sul branch `codex/qwen-flash-3090`, a partire

- righe 3–16: **Perimetro ed esito** — Revisione delle modifiche pendenti sul branch `codex/qwen-flash-3090`, a partire
- righe 17–46: **Responsabilità e contratti corretti** — Gli import pubblici precedenti del runner sono mantenuti per compatibilità;
- righe 47–82: **Verifiche della revisione** — Le prove HTTP usano il server compilato con la correzione degli schemi nativi.
- righe 83–98: **Riproduzione dei gate statici** — (cd agent && .venv/bin/ruff check src tests && .venv/bin/python -m pytest)

## [Revisione del codice e delle responsabilità — 8 ottobre 2026](clean-code-review-20261008.md)

Revisione delle modifiche Flash pendenti e dei confini fra engine, backend,

- righe 3–10: **Perimetro** — Revisione delle modifiche Flash pendenti e dei confini fra engine, backend,
- righe 11–28: **Responsabilità corrette** — Il test del launcher è allineato al default attuale di 64K, senza modificare
- righe 29–63: **Verifiche** — Risultati e comandi sono conservati in `performance-results/clean-code-20261008/`.
- righe 64–71: **Limiti** — La revisione applica principi di responsabilità unica, incapsulamento, coerenza

## [Revisione del server XML e dei contratti tool](clean-code-review-20261012.md)

I quattro callback di sampling sono riuniti in un'unica policy.

- righe 7–27: **Responsabilità e interfacce** — I quattro callback di sampling sono riuniti in un'unica policy.
- righe 28–65: **Coerenza dei comportamenti** — Il formato predefinito è v2 senza code: pausa dopo il path, autorizzazione,
- righe 66–77: **Verifica** — Gli esiti e i limiti delle verifiche sono conservati nel

## [Revisione del client XML e delle responsabilità — 12 ottobre 2026](client-xml-clean-review-20261012.md)

Il bug sullo scope si manifestava quando una skill veniva chiusa e riaperta

- righe 7–30: **Responsabilità e refactoring** — Il bug sullo scope si manifestava quando una skill veniva chiusa e riaperta
- righe 31–49: **Criteri di leggibilità** — Le operazioni di copia mantengono sottoclasse e metadati senza modificare il tool
- righe 50–90: **Verifiche** — La suite completa ExMekh-Lib passa **2.611 test**, con **55 SKIP live opt-in**
- righe 91–104: **Contratto e limiti** — Il protocollo resta XML v2 senza attributo `code`.

## [Client XML edit v2 — 12 ottobre 2026](client-xml-edit-v2-20261012.md)

Il documento dopo gli hook deve contenere gli stessi path nello stesso ordine

- righe 7–30: **Responsabilità** — Il documento dopo gli hook deve contenere gli stessi path nello stesso ordine
- righe 31–69: **Verifiche** — La suite completa ExMekh-Lib sul client aggiornato passa: **2.588 PASS,
- righe 70–90: **Limiti** — Il masking garantisce vincoli del documento, non l'intento dell'utente:

## [Revisione del budget di compattazione — 16 ottobre 2026](compaction-budget-clean-review-20261016.md)

espansa: gli allegati dei risultati dei tool comparivano due volte.

- righe 9–27: **Difetti e correzioni** — espansa: gli allegati dei risultati dei tool comparivano due volte.
- righe 28–46: **Proprietari delle responsabilità** — I chiamanti usano le API pubbliche invece di ricostruire dettagli dei dati di
- righe 47–61: **Verifica e limiti** — Le suite complete delle due repository, i test browser, lint, tipi della

## [Accettazione merge DS4 e exmekh-lib — 2 ottobre 2026](ds4-merge-acceptance-20261002.md)

**PASS per Qwen3.8-27B sulla RTX 3090.** Sono concluse le prove residue:

- righe 3–20: **Esito e provenienza** — **PASS per Qwen3.8-27B sulla RTX 3090.** Sono concluse le prove residue:
- righe 21–51: **Foto e memoria tra tool** — La fixture JPEG reale `earth.jpg` proviene dal corpus upstream.
- righe 52–66: **Regressioni e prestazioni** — exmekh-lib: **1.958 PASS**, copertura 100%, Ruff e Pyright puliti.
- righe 67–84: **Limiti conservati** — OCR `RED7` resta FAIL (`R7` anche nell’oracle llama.cpp con gli stessi GGUF).
- righe 85–97: **Riproduzione** — Con gli altri processi motore arrestati:

## [Merge DS4 upstream — 1 ottobre 2026](ds4-upstream-merge-20261001.md)

Branch `codex/merge-ds4-upstream-2026-10-01`, creato da

- righe 6–23: **Provenienza e perimetro** — Branch `codex/merge-ds4-upstream-2026-10-01`, creato da
- righe 24–46: **Contratti preservati** — tokenizer, prompt, kernel specializzati, GDN, grafi CUDA e launcher locali.
- righe 47–54: **Sviluppo vision successivo al merge** — Il report Qwen27 vision/checkpoint
- righe 55–67: **Immagini e presentazione del codice** — Sono importati integralmente decoder PNG/JPEG, fixture, encoder e percorsi vision
- righe 68–102: **Kernel e risoluzioni rilevanti** — Il MMQ Q4_K generico upstream non supera il gate numerico esistente del fork su
- righe 103–127: **Hardware, RAM e artefatti** — RTX 3090, 24 GiB, `sm_86`, driver 610.62, CUDA toolkit 12.4; i9-14900K,
- righe 128–173: **Confronto Qwen27** — Cinque ripetizioni misurate dopo warm-up, contesto allocato 22593, prefill chunk
- righe 174–211: **Matrice di verifica** — Il 27B supera i gate locali di accettazione.
- righe 212–249: **Tentativo Flash Next** — È usato il Q2 originale, circa 137.10 GiB su disco: circa 41.73 GiB di pesi
- righe 250–267: **Riproduzione** — make -j4 ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=sm_86

## [Semplificare i tool di scrittura, modifica e cancellazione — 13 ottobre 2026](file-tool-simplification-research-20261013.md)

Il punto fragile è l'accoppiamento fra operazione, selezione e rappresentazione del contenuto.

- righe 5–39: **1. Risultato della lettura** — Il punto fragile è l'accoppiamento fra operazione, selezione e rappresentazione del contenuto.
  - righe 13–21: **Codice e documentazione esaminati** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 22–39: **Evidenze locali** — La revisione client più recente riporta **13 PASS e 2 FAIL su 15 casi target-only**, senza skip.
- righe 40–448: **2. Cosa fanno altri strumenti** — Questa sezione segue il processamento dei file per agente: scoperta, ispezione, lettura, creazione, modifica, inserimento, rimozione, rinomina e verifica.
  - righe 46–111: **2.1 Codex: ricerca e lettura tramite strumenti disponibili, modifiche tramite patch** — Il catalogo di tool dipende dall'harness, non dal solo nome del modello.
    - righe 50–74: **Scoperta, ispezione e lettura** — Nel set di questa chat ricerca e lettura testuale passano soprattutto per `exec_command`: il modello genera un comando, il runtime avvia il processo e restituisce output, exit code o session ID se continua.
    - righe 75–103: **Creazione, edit, insert, rimozione e rename** — `apply_patch` separa l'operazione nell'intestazione dal corpo orientato alle righe.
    - righe 104–111: **Contratto API, risultati e garanzie** — Il tool **Responses API** `apply_patch` è distinto dalla forma freeform locale.
  - righe 112–164: **2.2 Claude Code: Read, Edit e Write, ricerca e shell** — Claude Code cerca, legge, modifica ed esegue verifiche nel proprio loop.
    - righe 116–132: **Tool e argomenti** — Le firme sono nella reference degli input degli hook, che documenta le chiamate reali.
    - righe 133–144: **Edit: sostituzione, inserimento e rimozione di testo** — {"file_path":"/work/config.py","old_string":"enabled = True\n","new_string":"enabled = True\nverbose = False\n","replace_all":false}
    - righe 145–150: **Write: creazione, riscrittura e svuotamento** — Write crea o sovrascrive con il contenuto intero; non appende o fonde.
    - righe 151–158: **Delete, rename, notebook e ispezione strutturale** — Il catalogo pubblico non elenca un tool Delete filesystem: rimozioni e spostamenti passano dalla shell.
    - righe 159–164: **Controlli e risultati** — Permessi su file e shell, normalizzazione dei path e hook sono responsabilità del runtime.
  - righe 165–183: **2.3 Claude via Messages API: text editor con comandi espliciti** — L'editor API `str_replace_based_edit_tool` ha schema definito da Anthropic.
  - righe 184–239: **2.4 Aider: file nel contesto, repo map e coders di editing** — Il percorso ordinario di Aider organizza i file nel contesto e interpreta la risposta del modello.
    - righe 188–207: **Scoperta, ispezione e lettura** — La repo map presenta percorsi, simboli e firme pertinenti entro un budget.
    - righe 208–227: **SEARCH/REPLACE: edit, insert, append e creazione** — Il formato diff è percorso più fence e blocchi SEARCH/REPLACE.
    - righe 228–233: **Whole e udiff, delete e rename** — Il coder whole riceve il contenuto completo del file e lo riscrive.
    - righe 234–239: **Risultati e verifica** — L'integrazione Git offre diff, commit e undo, con auto-commit configurabile.
  - righe 240–285: **2.5 Pi: set piccolo ed edit strutturati sul file originale** — Pi è distinto dal fork oh-my-pi.
    - righe 244–261: **Lettura e ispezione** — `read(path, offset?, limit?)` accetta path relativo/assoluto e righe 1-based.
    - righe 262–277: **Edit, inserimento e rimozione di testo** — La firma corrente è `edit(path, edits=[{oldText,newText}, ...])`.
    - righe 278–285: **Write, delete, rename e verifica** — `write(path, content)` crea/sovrascrive, crea parent mancanti e scrive UTF-8.
  - righe 286–344: **2.6 Oh-my-pi: lettura unificata, snapshot e linguaggio hashline** — Il lavoro originale hashline proponeva brevi riferimenti per riga; l'API corrente usa anche un tag di snapshot per file e un linguaggio più ampio.
    - righe 290–295: **Read: ispezione e lettura** — `read(path)` può indirizzare file, directory, URL o risorse interne.
    - righe 296–303: **Grep e Find** — `grep(pattern, path?, case?, gitignore?, skip?)` cerca contenuti con regex, contesto e gruppi per file.
    - righe 304–328: **Edit hashline: replace, insert, cut, delete e move** — `edit(input)` usa sezioni `[PATH#TAG]` e numeri dello snapshot originale.
    - righe 329–334: **Write e risorse speciali** — `write(path, content)` crea/sovrascrive file e può delegare a handler per risorse interne, archivi, SQLite e conflitti.
    - righe 335–344: **AST, LSP e trasformazioni strutturali** — `ast_grep(pat, path?, lang?, skip?)` cerca nodi e catture sintattiche, usando linguaggio inferito o esplicito.
  - righe 345–390: **2.7 Qwen Code: lettori dedicati, write/edit e stato di lettura** — Qwen Code è un harness completo, distinto dall'utilizzo di un modello Qwen dentro DS4.
    - righe 349–366: **Scoperta e lettura** — Le prime quattro firme sono nella reference filesystem; la lettura multipla ha una reference propria.
    - righe 367–378: **Edit e inserimento** — `edit(file_path, old_string, new_string, replace_all?)` richiede un match univoco salvo sostituzione globale.
    - righe 379–386: **Write, notebook, delete e rename** — `write_file(file_path, content)` crea o sostituisce il contenuto completo.
    - righe 387–390: **Utilità per DS4** — Write/edit separati e stato di lettura sono precedenti pertinenti.
  - righe 391–412: **2.8 DeepSeek Tool Calls: protocollo di funzioni, non catalogo filesystem** — La guida DeepSeek definisce funzioni del chiamante con nome, descrizione e JSON Schema.
    - righe 395–406: **Cosa è definito e cosa no** — Non è corretto inventare un read_file universale di DeepSeek a partire da questo protocollo.
    - righe 407–412: **Schema e controllo dell'inferenza** — La modalità strict beta vincola la forma alla funzione definita dal client; richiede gli oggetti con campi required e additionalProperties false, con limitazioni sui tipi supportati.
  - righe 413–418: **2.9 Llguidance: controllo della generazione, non agente di file** — Llguidance non definisce Read/Edit/Write e non decide come leggere o cancellare.
  - righe 419–448: **2.10 Mappa comparativa delle operazioni** — La tabella sintetizza i paragrafi precedenti nel medesimo perimetro/versione; non introduce garanzie ulteriori.
- righe 449–463: **3. Le protezioni da conservare** — La maggior parte della complessità del filesystem ha una responsabilità reale e può restare interna al runtime:
- righe 464–517: **4. Contratto candidato: write, edit, delete** — Il primo candidato da confrontare con v2 usa tre strumenti con argomenti ordinari:
  - righe 482–493: **Se new deve essere distinto da write** — Esiste un compromesso reale.
  - righe 494–509: **Esempi delle operazioni** — Questi sono oggetti logici, non una nuova sintassi da copiare dentro una stringa XML:
  - righe 510–517: **Un file per chiamata e gestione dei batch** — Una chiamata con tre campi piatti evita il documento di frammenti, i path ripetuti e le pause a metà XML.
- righe 518–533: **5. Eliminare CDATA: distinguere dati e trasporto** — CDATA non cambia il valore logico dopo il parsing, ma cambia ciò che il modello deve generare.
  - righe 528–533: **Perché non partire da un formato raw nuovo** — Un payload raw sarebbe più vicino al file, ma ha comunque bisogno di distinguere contenuto e confini.
- righe 534–560: **6. Usare il controllo dell'inferenza senza ampliare l'API** — Il motore può seguire un flusso piccolo e distinto per operazione:
- righe 561–580: **7. Una restrizione da riesaminare: unicità globale** — Il contratto impone contemporaneamente partenza da inizio riga e unicità globale, comprese occorrenze a metà riga.
- righe 581–605: **8. Blocchi lunghi, inserimenti e file vuoti** — Per un file non vuoto una sostituzione che conserva l'ancora rappresenta già l'inserimento prima/dopo di essa.
  - righe 583–588: **Inserimenti** — Per un file non vuoto una sostituzione che conserva l'ancora rappresenta già l'inserimento prima/dopo di essa.
  - righe 589–594: **Intervalli numerici attuali** — Il ramo `start_line/end_line` viene controllato solo lessicalmente dal server: positività, ordine e limiti sono verificati dal client.
  - righe 595–605: **Alternative da valutare solo se emerge il bisogno** — Fra queste, due ancore letterali potrebbero riutilizzare meglio il masking esistente; riferimenti a blocchi possono essere migliori se la lettura espone già unità adeguate.
- righe 606–623: **9. Cosa togliere, cosa spostare, cosa rinviare** — La separazione di delete consente anche di riesaminare la dipendenza dalla lettura del contenuto: cancellare un file non richiede decodificarne il testo.
- righe 624–635: **10. Alternative di formato e priorità** — L'XML standard dei **risultati** e dei metadati del runtime è un confine diverso.
- righe 636–652: **11. Skill candidata più corta** — Una descrizione candidata dell'editing, da adattare alla semantica batch scelta, potrebbe essere:
- righe 653–672: **12. Confronto sperimentale e migrazione** — Il prossimo lavoro dovrebbe confrontare v2 e un solo candidato minimale con richieste naturali, senza dettare nel prompt i tag o gli argomenti da emettere.
- righe 673–675: **13. Stato delle verifiche di questa ricerca** — Sono stati letti codice, skill e report locali e confrontate fonti primarie esterne.

## [Tool file letterali v3 — 13 ottobre 2026](file-tools-literal-v3-20261013.md)

Edit usa soltanto due marcatori:

- righe 8–35: **Contratto implementato** — Edit usa soltanto due marcatori:
- righe 36–44: **Distribuzione** — La wheel `agent/wheels/exmekh_lib-0.1.0-py3-none-any.whl` è stata ricostruita
- righe 45–76: **Verifiche e limiti osservati** — I tre fallimenti live sono svuotamento dell'intero file, inserimento con CRLF

## [Revisione della compattazione e dei confini del frontend — 16 ottobre 2026](history-compaction-clean-review-20261016.md)

Il frontend non interpreta più gli elementi XML degli header della libreria.

- righe 8–26: **Responsabilità e interfacce** — Il frontend non interpreta più gli elementi XML degli header della libreria.
- righe 27–58: **Difetti riprodotti e corretti** — risultato più recente, anche quando nella history era visibile uno snapshot
- righe 59–78: **Principi controllati e limiti** — La revisione ha verificato responsabilità, API pubbliche, dipendenze,
- righe 79–82: **Verifiche** — Gli esiti, i conteggi e la provenienza della wheel sono registrati nel

## [Audit RAM e VRAM durante l'inferenza — 13 ottobre 2026](inference-memory-audit-20261013.md)

1.

- righe 8–28: **Correzioni** — 1.
- righe 29–51: **Memoria che rimane intenzionalmente disponibile** — non più visibili possono restare nello spazio prenotato dal contesto e vengono
- righe 52–94: **Verifica e limiti delle misure** — Qwen3.8-27B UD-Q4_K_S, sidecar MTP, RTX 3090, CUDA/Linux su WSL.

## [Suite multimodale nativa DS4: analisi e decisioni](native-multimodal-suite-analysis-20261020.md)

Le istruzioni della conversazione corrente definiscono il perimetro del lavoro.

- righe 7–22: **1. Requisiti acquisiti e confini** — Le istruzioni della conversazione corrente definiscono il perimetro del lavoro.
- righe 23–43: **2. Cosa recuperare dalla chat condivisa** — La chat condivisa è stata letta nel browser.
- righe 44–75: **3. Stato reale di DS4 e ostacoli prioritari** — L'audit in lettura ha usato il checkout con HEAD `5e2e99cf3249a5a28efed038bfd4fb2fbda6ce21`, contenente anche modifiche locali preesistenti.
  - righe 48–59: **3.1 Base già disponibile** — Il report Qwen27 vision, `ds4.h` e le funzioni `engine_vision_suspend_sessions`, `engine_vision_resume_sessions`, `ds4_engine_vision_encode_image` in `ds4.c` documentano e implementano:
  - righe 60–67: **3.2 Stato globale da separare** — `g_ds4_shape` alimenta macro come `DS4_MODEL_FAMILY`, `DS4_N_LAYER`, `DS4_N_EMBD`.
  - righe 68–75: **3.3 Il consumo KV non coincide con la chat condivisa** — `qwen_graph_alloc` usa `DS4_QWEN_ALLOC` con `sizeof(float)` anche per key/value cache.
- righe 76–105: **4. Qwen-Image-2.1-Turbo: semantica da preservare** — Il checkpoint ufficiale comprende encoder Qwen3-VL, DiT single-stream block-causal e VAE dedicato.
  - righe 82–89: **4.1 Conditioning e scheduler** — La pipeline Diffusers costituisce un oracle semantico: template e padding, esclusione del prefisso di sistema, ordine delle reference e hidden state prima della RMSNorm finale devono coincidere.
  - righe 90–97: **4.2 Attention e prefix cache** — Nel transformer Diffusers testo e blocchi immagine hanno una maschera block-causal.
  - righe 98–105: **4.3 Memoria e ordine delle fasi** — Il piano base è preprocessing → encoder/encoding reference → DiT ripetuto → VAE decode → ripristino.
- righe 106–129: **5. EmbeddingGemma 2: quattro modalità nello stesso spazio** — La model card Google descrive backbone testuale ed encoder vision/audio caricabili selettivamente, spazio comune a 768 dimensioni, contesto operativo 8192 e troncamenti supportati 512/256/128.
  - righe 114–121: **5.1 Pooling, precisione e task** — I moduli del checkpoint includono pooling e normalizzazione.
  - righe 122–129: **5.2 Immagini, video e audio** — L'immagine richiede preprocessing, posizioni patch, encoder, riduzione/proiezione e inserimento dei soft token nel backbone.
- righe 130–151: **6. Cosa trasferire dai motori esaminati** — Le fonti seguenti sono state aperte e confrontate con il codice locale.
- righe 152–189: **7. Gestore comune di memoria ed esecuzione** — Servono descrittori per modello, componente, gruppo di pesi, sessione, risorsa e operazione.
  - righe 154–161: **7.1 Oggetti minimi** — Servono descrittori per modello, componente, gruppo di pesi, sessione, risorsa e operazione.
  - righe 162–173: **7.2 Ammissione e scelta del piano** — La condizione di ammissione di ciascuna fase è:
  - righe 174–181: **7.3 Transazione, cancellazione e recupero** — Sequenza concettuale: validate → reserve → quiesce → checkpoint se necessario → execute → release/retain → restore → commit result.
  - righe 182–189: **7.4 Prefetch e riuso** — Il prefetch CPU può precedere l'esecuzione solo per modelli già configurati e con budget host.
- righe 190–201: **8. Contratto client e integrazione agentica** — Un solo server e una sola configurazione client, con selezione `model`.
- righe 202–210: **9. Decisioni numeriche e sperimentali** — 1.
- righe 211–225: **10. Rischi, evidenze mancanti e scelta dell'ordine** — Il percorso critico parte dai confini runtime e dal gestore di risorse, poi sviluppa i percorsi numerici separati e li integra.

## [Revisione di responsabilità e coerenza della suite nativa](native-suite-clean-code-review-20261020.md)

l'engine conosce shape e modello; il backend possiede stream e allocazioni;

- righe 8–23: **Correzioni applicate** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 24–57: **Principi verificati e limiti** — l'engine conosce shape e modello; il backend possiede stream e allocazioni;
- righe 58–66: **Evidenze** — Comandi, hash e log sono nel manifest di verifica
- righe 67–104: **Riesame di T03–T04** — Il secondo passaggio esamina registro, accounting CUDA, checkpoint, adapter
- righe 105–141: **Riesame di T05–T06** — Il terzo passaggio comprende planner, executor, registro, loader, tool offline e
- righe 142–186: **Riesame di T07–T08, 21 ottobre 2026** — Riletti i nuovi moduli numerici, encoder/tokenizer, preprocessing, bridge dei
- righe 187–226: **Riesame di T09–T10, 21 ottobre 2026** — Riletti DiT, VAE, primitive e helper, risultati, risorse/runtime, bridge dei

## [Runtime e contesti della suite nativa](native-suite-runtime-lifecycle.md)

Il runtime possiede mutex ricorsivo, fence e teardown condiviso.

- righe 8–48: **Proprietari e alternanza** — Il runtime possiede mutex ricorsivo, fence e teardown condiviso.
- righe 49–78: **Inventario dello stato globale** — La classificazione si applica anche alle cache statiche interne alle funzioni:
- righe 79–110: **Compatibilità e verifiche** — Per Metal e ROCm il riferimento runtime resta interno all'engine e non si

## [Revisione clean code di T11/T12](native-suite-t11-t12-clean-code-review-20261021.md)

Runtime ed executor possiedono serializzazione, admission, recupero e commit.

- righe 9–24: **Correzioni e responsabilità** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 25–38: **Contratti conservati** — Runtime ed executor possiedono serializzazione, admission, recupero e commit.
- righe 39–60: **Verifica** — Il report T11/T12
- righe 61–76: **Debito e limiti** — completa di workspace nel backend numerico potrebbe eliminare la duplicazione

## [Revisione clean code di T13/T14](native-suite-t13-t14-clean-code-review-20261022.md)

Il decoder possiede byte originali, frame e waveform; il processor possiede

- righe 9–20: **Correzioni** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 21–34: **Confini conservati** — Il decoder possiede byte originali, frame e waveform; il processor possiede
- righe 35–63: **Verifica e limiti** — Il report della revisione

## [Contratti della suite nativa DS4, versione 1](native-suite-v1-contract.md)

Il catalogo distingue `model_id`, `revision`, `profile` e `component`.

- righe 8–31: **Identità e discovery** — Il catalogo distingue `model_id`, `revision`, `profile` e `component`.
- righe 32–77: **Operazioni e input standard** — Le immagini usano `model`, `prompt`, `n` (default 1, intervallo 1–8), `size` (default
- righe 78–126: **Estensione embedding multimodali** — La versione è dichiarata da `ds4.version=1`; il campo `input` contiene un
- righe 127–185: **Risultati, idempotenza e completamento** — Embedding mantiene `{object:"list",data:[{object:"embedding",index,embedding}],
- righe 186–226: **Esempi client e matrice delle capability** — Le chiamate seguenti descrivono il contratto futuro T17: non sono test live
- righe 227–256: **Confine C e lifecycle** — `ds4_runtime` possiede serializzazione e teardown condiviso.

## [Qwen Flash: esperti ibridi, PCIe e KV paginato](qwen-flash-hybrid-kv-20261006.md)

Richiedono `DS4_CUDA_QWEN4_TIERED_CACHE=1`; il modello deve essere Flash.

- righe 8–80: **Controlli** — Richiedono `DS4_CUDA_QWEN4_TIERED_CACHE=1`; il modello deve essere Flash.
- righe 81–186: **Piano di placement e lifetime** — La policy `stable` riprende le quote per layer di Strata.
- righe 187–240: **KV per sessione** — Per il prefill con policy stabile, il workspace identico al layout originale
- righe 241–305: **Profili e misure** — Si conta ogni scelta una volta per layer/riga, al gate, e si salva un payload
- righe 306–360: **Provenienza e 4 BIS** — Riferimento architetturale: Strata commit
- righe 361–382: **Liste compatte del prefill — incremento successivo al ritiro 4 BIS** — `DS4_CUDA_QWEN4_COMPACT_LISTS=1` è una variante sperimentale, disattivata
- righe 383–404: **Copie anticipate bounded nel prefill** — `DS4_CUDA_QWEN4_PREFILL_AHEAD=1`, con policy `stable` e ring attivo,
- righe 405–476: **NextN e selezione per fase** — La v68 ha provato NextN embedded di default per Flash CUDA.

## [Qwen27 vision e checkpoint completi — 1 ottobre 2026](qwen27-vision-checkpoint-20261001.md)

Sviluppo sul branch `codex/merge-ds4-upstream-2026-10-01`, dopo il merge

- righe 6–23: **Implementazione e perimetro** — Sviluppo sul branch `codex/merge-ds4-upstream-2026-10-01`, dopo il merge
- righe 24–38: **Ordine delle immagini e del thinking** — Il forward ufficiale Transformers
- righe 39–62: **Ciclo VRAM effettivo** — Per ogni immagine non presente nella cache embedding, sotto il lock d’inferenza:
- righe 63–88: **Payload, skill e subagent** — Un payload multimodale contiene un envelope `VMCP`, versione 1, con:
- righe 89–108: **Artefatti riproducibili** — Il mmproj Unsloth
- righe 109–145: **Verifica** — I test numerici non cambiano tolleranze né golden.
- righe 146–183: **Prestazioni del percorso testo** — Baseline vision: merge `0edbded`, con binari congelati; implementazione verificata
- righe 184–215: **Limiti osservati e verifiche mancanti** — La prima fixture OCR con scritta `RED7` ha prodotto `R7`; l’oracle llama.cpp
- righe 216–226: **Riproduzione** — ./download_model.sh qwen38-27b-vision
- righe 227–228: **In un secondo terminale, con server fermo rispetto ad altri client:** — python3 tests/test_qwen35_vision_api.py --url http://127.0.0.1:8080
- righe 229–232: **Ripetere con ./agent/start-ds4-server.sh per MTP.** — DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=16 DS4_TEST_CTX=256 \

## [Qwen3.6 MTP implementation](qwen36-mtp-design.md)

DS4 supports native Qwen3.6 Multi-Token Prediction on the single-GPU CUDA

- righe 7–29: **Scope and contract** — DS4 supports native Qwen3.6 Multi-Token Prediction on the single-GPU CUDA
- righe 30–58: **GGUF and NextN semantics** — The support model is selected structurally, not by filename:
- righe 59–74: **Session state** — Each enabled Qwen session owns:
- righe 75–111: **Production adaptive-depth cycle** — Below 2K, the configured depth-two cycle is:
- righe 112–141: **Sampling and logits masks** — DS4 follows llama.cpp's current sample-and-match contract.
- righe 142–170: **CUDA paths** — Q4_K, Q5_K, and Q6_K verifier matmuls use Q8_1 plus a second Q8_1 residual.
  - righe 144–163: **Target verifier** — Q4_K, Q5_K, and Q6_K verifier matmuls use Q8_1 plus a second Q8_1 residual.
  - righe 164–170: **Support network** — The Q4_0 shared head uses Q8_1+residual DP4A and schedules eight output rows per
- righe 171–182: **Failure policy** — diagnostic.
- righe 183–253: **Validation gate** — The production model-backed command is:

## [Qwen3.8 27B UD-Q4_K_S compatibility audit](qwen38-compatibility.md)

The 2026-10-01 merge audit qualifies this

- righe 7–12: **Full upstream merge** — The 2026-10-01 merge audit qualifies this
- righe 13–27: **Native vision on CUDA** — The 27B is a vision-language model; its GGUF text weights need the matching
- righe 28–40: **Pinned artifacts** — DS4 recognizes only the audited Unsloth pair at repository revision
- righe 41–69: **Architecture and GGUF contract** — The runtime architecture is the existing `qwen35` implementation: 64 target
- righe 70–92: **Frontend behavior** — Both `ds4` and `ds4-server` recognize the stable model ID
- righe 93–118: **Test and oracle isolation** — `make test-qwen38` validates the pinned metadata snapshots, embedded and
- righe 119–129: **CUDA runtime status** — The Qwen3.6 graph is reused, while Q3_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,

## [Qwen3.6-27B Q4_K_S — Architettura completa, memoria ricorrente e quantizzazione](qwen_architecture.md)

Modello:

- righe 29–59: **1. Identità del modello** — Modello:
- righe 60–87: **2. Parametri principali** — ---
- righe 88–150: **3. Visione globale del modello** — Per una sequenza di testo:
- righe 151–209: **4. Ordine esatto dei 64 layer** — Il pattern è:
- righe 210–270: **5. Struttura generale di ogni decoder layer** — Ogni decoder layer mantiene la dimensione:
- righe 271–332: **6. Token embedding** — Il vocabolario contiene:
- righe 333–380: **7. RMSNorm** — La normalizzazione principale usa:
- righe 381–481: **8. FFN: tipo e dimensioni** — Tutti i 64 decoder layer contengono lo stesso tipo di FFN.
  - righe 439–481: **Matrici** — In convenzione PyTorch `[out_features, in_features]`:
- righe 482–529: **9. Gated DeltaNet: panoramica** — 48 dei 64 layer usano un **Gated DeltaNet** al posto della
- righe 530–611: **10. Proiezioni del Gated DeltaNet** — Da un hidden state da 5120:
- righe 541–561: **Q/K/V** — in_proj_qkv:
- righe 562–575: **Gate z** — in_proj_z:
- righe 576–585: **Gate b** — in_proj_b:
- righe 586–595: **Gate a** — in_proj_a:
- righe 596–611: **Output** — Dopo il DeltaNet:
- righe 612–689: **11. Gated DeltaNet completo** — h_t
- righe 690–743: **12. Depthwise causal Conv1D** — Prima della Delta Rule, Q/K/V attraversano una convolution temporale.
- righe 744–819: **13. Primo stato persistente: Conv state** — Durante il decoding token-per-token non avrebbe senso ricalcolare
- righe 820–896: **14. Secondo stato persistente: matrice ricorrente DeltaNet** — La memoria più importante è:
- righe 897–949: **15. Come funziona matematicamente S_t** — Consideriamo una singola testa.
- righe 916–949: **Passaggio 1 — decay della memoria** — S̄_t = exp(g_t) · S_(t-1)
- righe 950–978: **16. Lettura predittiva dalla memoria** — La key corrente interroga lo stato:
- righe 979–1016: **17. Calcolo dell'errore / Delta** — Si confronta ciò che la memoria prevedeva con il vero `v_t`:
- righe 1017–1071: **18. Scrittura nella memoria** — La memoria viene aggiornata mediante un prodotto esterno:
- righe 1072–1092: **19. Lettura finale dalla nuova memoria** — Dopo aver scritto il nuovo dato, la query legge lo stato aggiornato:
- righe 1093–1163: **20. DeltaNet completo in un singolo diagramma didattico** — STATO DAL PASSATO
- righe 1164–1251: **21. Da dove arrivano g_t e β_t?** — Il modello produce due vettori da 48 elementi:
- righe 1252–1303: **22. Q/K vengono replicati 3 volte nel DeltaNet** — Originariamente:
- righe 1304–1349: **23. RMSNormGated all'uscita del DeltaNet** — L'output ricorrente non viene inviato direttamente a `out_proj`.
- righe 1350–1391: **24. Full Attention: i restanti 16 layer** — Ogni quarto layer usa una normale causal self-attention,
- righe 1392–1473: **25. Proiezioni della Full Attention** — L'input è:
- righe 1401–1445: **Query + output gate** — Qwen usa una particolarità:
- righe 1446–1453: **K** — k_proj:
- righe 1454–1461: **V** — v_proj:
- righe 1462–1473: **Output** — Dopo l'attention:
- righe 1474–1540: **26. Full Attention completa** — h_t
- righe 1541–1584: **27. Partial RoPE** — La dimensione di ciascuna attention head è:
- righe 1585–1650: **28. Terzo stato persistente: KV cache** — I 16 Full Attention layer NON usano la matrice ricorrente `S_t`.
- righe 1651–1702: **29. Differenza fondamentale: S_t contro KV cache** — Non conserva tutti i token.
- righe 1653–1673: **Gated DeltaNet** — Non conserva tutti i token.
- righe 1674–1702: **Full Attention** — Conserva K e V di ogni posizione:
- righe 1703–1716: **30. Le tre memorie temporali di Qwen3.6-27B** — Nel decoding esistono quindi tre tipi distinti di stato.
- righe 1717–1802: **31. È quindi autoregressivo oppure ricorrente?** — **Entrambi**, ma le parole descrivono aspetti differenti.
- righe 1721–1755: **Autoregressivo** — Significa che la distribuzione del token successivo è:
- righe 1756–1802: **Ricorrente** — Significa che internamente alcuni layer hanno uno stato:
- righe 1803–1837: **32. Cosa attraversa realmente il confine fra due token** — Alla fine dell'iterazione `t` abbiamo:
- righe 1838–1941: **33. Vista completa token-per-token** — ╔══════════════════════════════════════════════════════════════╗
- righe 1942–1986: **34. Un errore concettuale comune** — Una descrizione troppo semplice di un Transformer dice:
- righe 1987–2047: **35. Perché combinare DeltaNet e Full Attention?** — Le due memorie hanno proprietà complementari.
- righe 1991–2009: **DeltaNet** — Vantaggi:
- righe 2010–2027: **Full Attention** — Vantaggio:
- righe 2028–2047: **Architettura ibrida** — Qwen usa:
- righe 2048–2084: **36. Dimensione dello stato ricorrente DeltaNet** — Per un Gated DeltaNet:
- righe 2085–2109: **37. Dimensione del Conv state** — Per un GDN:
- righe 2110–2153: **38. Crescita della KV cache** — Per un Full Attention layer e un token:
- righe 2154–2219: **39. Final RMSNorm e LM Head** — Dopo il decoder layer 64:
- righe 2220–2250: **40. Vision Encoder** — Qwen3.6-27B è multimodale.
- righe 2251–2301: **41. Patch embedding visuale** — Il patch embedding è realizzato con una Conv3D:
- righe 2302–2339: **42. Vision Transformer** — Sono presenti:
- righe 2340–2404: **43. Vision merger** — Il merger spaziale usa:
- righe 2405–2447: **44. Pipeline multimodale semplificata** — IMMAGINE / VIDEO
- righe 2448–2483: **45. Quantizzazione Q4_K_S: cosa significa** — Ora separiamo completamente due concetti:
- righe 2484–2559: **46. Pesi statici contro stato dinamico** — Questa distinzione è fondamentale.
- righe 2488–2521: **Pesi statici** — Esempi:
- righe 2522–2559: **Stato dinamico** — Esempi:
- righe 2560–2605: **47. Grafico: dove agisce Q4_K_S** — MODELLO SU DISCO
- righe 2606–2641: **48. Cos'è il primitive Q4_K** — Il tipo di base del preset Q4_K_S è:
- righe 2642–2729: **49. Struttura di block_q4_K** — Un super-block contiene:
- righe 2730–2783: **50. Perché Q4_K_S non equivale esattamente a 4.5 bpw** — Il nome completo interno è:
- righe 2784–2828: **51. Alcune promozioni del preset Q4_K_S** — Nel codice corrente di llama.cpp,
- righe 2829–2870: **52. `S` in Q4_K_S** — È importante non interpretare `S` come:
- righe 2871–2913: **53. Q4_K_S non modifica lo stato DeltaNet** — Questa distinzione merita di essere visualizzata.
- righe 2914–2965: **54. Q4_K_S non modifica le equazioni del modello** — BF16 e Q4_K_S rappresentano lo stesso modello matematico
- righe 2919–2930: **BF16** — h
- righe 2931–2965: **Q4_K_S** — h
- righe 2966–3017: **55. Riassunto completo delle matrici principali** — ---
- righe 2968–2977: **Embedding/output** — ---
- righe 2978–2987: **FFN — tutti i 64 layer** — ---
- righe 2988–3003: **Gated DeltaNet — 48 layer** — ---
- righe 3004–3017: **Full Attention — 16 layer** — ---
- righe 3018–3107: **56. Mappa mentale finale** — Qwen3.6-27B
- righe 3108–3180: **57. Il modello in una sola immagine concettuale** — PASSATO
- righe 3181–3270: **58. Frase conclusiva tecnicamente precisa** — Qwen3.6-27B può essere descritto così:
- righe 3271–3312: **59. Fonti primarie di riferimento** — I dati architetturali vanno verificati principalmente in:
- righe 3290–3312: **Nota importante sulla certezza di Q4_K_S** — `Q4_K_S` definisce un preset di quantizzazione, non la lista

## [Istruzioni agentiche e diagnosi dei test live — 14 ottobre 2026](server-agent-guidance-removal-20261014.md)

Eliminati il blocco `agent_runtime` aggiunto a user/tool message, le istruzioni

- righe 7–19: **Rimozioni dal server** — Eliminati il blocco `agent_runtime` aggiunto a user/tool message, le istruzioni
- righe 20–48: **Criteri delle prove** — Un test sul software confronta i byte scritti con gli argomenti realmente
- righe 49–74: **Esecuzione e risultati** — Passano build CUDA, test nativi del server e del protocollo interattivo,

## [Revisione delle responsabilità dei draft OLD — 12 ottobre 2026](snapshot-draft-clean-review-20261012.md)

La policy dei batch ha operazioni distinte per entrare in una fase, leggere

- righe 10–27: **Responsabilità e interfacce** — La policy dei batch ha operazioni distinte per entrare in una fase, leggere
- righe 28–54: **Correzioni e uniformità** — la selezione completa con un delta.
- righe 55–69: **Verifiche e limiti** — Le evidenze finali, gli hash e gli esiti delle prove sono nel

## [Correzione del riuso KV dopo steering — 14 ottobre 2026](steering-cache-fix-20261014.md)

`responses_prepare_live_continuation` cercava soltanto risultati di tool in

- righe 10–37: **Cause e comportamento corretto** — `responses_prepare_live_continuation` cercava soltanto risultati di tool in
- righe 38–56: **Contratto delle prove** — Il test di cancellazione confronta la ripresa con i token già riusati prima
- righe 57–76: **Risultati verificati** — Tutte le 13 prove live della cache passano, senza errori HTTP nelle fixture.
- righe 77–93: **Ripetibilità** — DS4_STEERING_CACHE_BASE_URL=http://127.0.0.1:8080 \
- righe 94–101: **Verifica successiva: cancel seguito da richiesta ausiliaria** — La revisione con test live ha aggiunto un

## [Riproduzione del prefill dopo steering — 14 ottobre 2026](steering-cache-reproduction-20261014.md)

Ogni caso apre una skill sintetica, scalda una conversazione con un tool,

- righe 12–54: **Risultati** — Ogni caso apre una skill sintetica, scalda una conversazione con un tool,
- righe 55–71: **Conseguenze per la diagnosi** — La diagnosi iniziale limitata alla cancellazione interattiva era incompleta:
- righe 72–94: **Comandi ripetibili** — Su una sola istanza server inattiva, dalla radice del repository:

## [Steering dopo una risposta completata — 15 ottobre 2026](steering-completed-response-20261015.md)

I call ID uniscono le chiamate generate nel log, le chiamate e risposte nella

- righe 17–42: **Corrispondenza fra chiamate ed esecuzioni** — I call ID uniscono le chiamate generate nel log, le chiamate e risposte nella
- righe 43–66: **Correzione** — `InteractiveResponseOpenAI` distingue il risultato completato dal server e
- righe 67–120: **Verifica** — Le regressioni con SDK e runtime reali, trasporto simulato e file temporanei
