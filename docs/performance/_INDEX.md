# Indice: performance

Indice generato; non modificare a mano.

## [Contratto operativo per kernel engineering](agent_performance_contract.md)

Una modifica è utile soltanto se migliora il workload dichiarato di DS4 senza

- righe 7–19: **Obiettivo** — Una modifica è utile soltanto se migliora il workload dichiarato di DS4 senza
- righe 20–39: **Ciclo obbligatorio** — 1.
- righe 40–55: **Significatività minima** — Il default è cinque ripetizioni dopo un warm-up per workload.
- righe 56–66: **Correttezza** — Il harness non sostituisce gli scorer Qwen già presenti in
- righe 67–74: **Misurazione hardware** — I tempi finali vengono dal percorso sincronizzato di ds4-bench.

## [Test completi XML v2 del 11 ottobre 2026](interactive-edit-xml-v2-full-tests-20261011.md)

Uno stream batch terminava con `incomplete_generation`: il parser ammetteva

- righe 14–29: **Correzione server trovata durante i test** — Uno stream batch terminava con `incomplete_generation`: il parser ammetteva
- righe 30–48: **Suite locali** — I controlli saltati restano elencati nel JSON.
- righe 49–100: **Suite live** — Le prove usano CUDA, RTX 3090, Qwen3.8-27B Q4_K_S, mmproj F16 e sidecar MTP Q4_0.
- righe 101–122: **Responsabilità e controlli ancora aperti** — Una riscrittura valida o il mancato adempimento di una richiesta delete non
- righe 123–137: **Tentativi e limiti conservati** — Il primo tentativo MTP usava `--batched-session 1`: il runtime disabilitava MTP.

## [Qwen Flash: prefill reale dall'agente e dal launcher, 8 ottobre 2026](qwen-flash-agent-launcher-20261008.md)

L'harness bench_qwen_flash_agent.py

- righe 28–52: **Riproduzione e isolamento** — L'harness bench_qwen_flash_agent.py
- righe 53–88: **Diagnosi iniziale** — Screen `agent-launcher-v71-screen-r2`: un campione per configurazione, con sampling
- righe 89–107: **Isolamento del chunk e generazioni reali** — `agent-launcher-v71-isolate` usa sampling greedy, stesso prompt e quattro turni
- righe 108–141: **Launcher originale contro bundle 2048** — `agent-launcher-v71-balanced` ripete la configurazione originale con sampling
- righe 142–150: **Stato della qualifica** — Le prove descrivono l'avvio reale e identificano la differenza rispetto al
- righe 151–212: **Correzione numerica e profilo standard v74** — Il primo gate v72 a capacità 131072 e prompt da 4758 token trova KL massimo
- righe 213–249: **Promozione finale e uso** — Verdetto **KEEP / profilo standard promosso** per il launcher Flash CUDA sul

## [Qwen Flash: nuova verifica del prefill Strata, 8 ottobre 2026](qwen-flash-prefill-strata-20261008.md)

La testa `main` verificata durante questa sessione è

- righe 21–35: **Fonte ricontrollata** — La testa `main` verificata durante questa sessione è
- righe 36–63: **Liste compatte, v64** — L'ibrido può riservare `esperti × token` interi anziché `esperti × token × slot`.
- righe 64–158: **Anticipare le copie rispetto al router** — Strata, dai chunk di 1024 righe, trasferisce tutti gli esperti non residenti
  - righe 105–143: **Confronto ripetuto degli hint v66** — Cinque coppie alternate a 4096 token effettivi, capacità 8192, chunk 4096,
  - righe 144–158: **Soglia prestazionale dell'1%** — Su richiesta dell'utente, il tuner usa `--min-gain 0.01` come default,
- righe 159–210: **Contesto effettivo 64K: pipeline completa v67** — Due confronti diagnostici separano l'anticipo dal percorso decode qualificato.
  - righe 189–210: **Contesto effettivo 128K** — Stesso confronto integrato, 131072 token effettivi, capacità 131104,
- righe 211–432: **Richiesta DSpark v2 predefinito per Flash** — Verifica del catalogo pubblico del 2026-10-08: 389 modelli trovati dalla
  - righe 256–287: **v68: NextN predefinito per Flash CUDA** — CLI e server attivano NextN embedded dopo la validazione del target Flash
  - righe 288–322: **Confronto per fase e costo del NextN v68** — Cinque coppie CLI alternate per dominio, dopo una coppia di warm-up,
  - righe 323–373: **v69: residenza degli esperti NextN entro la quota esistente** — L'ispezione di `src/core/mtp.cpp` al commit Strata
  - righe 374–403: **v70: decisione corrente per prefill e decode** — La richiesta finale è «usa NextN se funziona meglio», separando prefill
  - righe 404–432: **v71: fase esplicita, compresi i prefill brevi** — Il numero di righe non identifica da solo il workload.
- righe 433–464: **Altri punti Strata da verificare** — Strata prepara anche lo storico KV del NextN durante il prefill del target:

## [Qwen Flash — pipeline ispirata a Strata, RTX 3090 / WSL](qwen-flash-strata-20261006.md)

`codex/qwen-flash-strata-20261006`; la modifica preesistente a

- righe 104–122: **Identità e condizioni** — `codex/qwen-flash-strata-20261006`; la modifica preesistente a
- righe 123–154: **Correzioni della pipeline** — Lo scheduler esegue triplette complete: prima gli hit GPU, poi CPU e gruppi
- righe 155–178: **A/B alternati: incremento v5** — Cinque coppie A/B e B/A, processi freschi, due warm-up per processo, un
- righe 179–205: **A/B finale v10, scelta automatica** — Stesse condizioni e cinque coppie alternatamente A/B e B/A.
- righe 206–241: **KV e profilo** — Primo attraversamento reale: 16384 token, 12288 celle residenti per layer,
- righe 242–310: **Correttezza e stato di promozione** — Vision finale v12: pass tutti i 16 gate Flash/27B, target/MTP.
- righe 311–329: **HTTP agentico a 128K allocati, 4395/4401 token effettivi** — Una richiesta per condizione, quindi diagnostico e non promozione A/B.
- righe 330–353: **HTTP con MTP effettivo, cinque coppie A/B** — Fixture senza tool, reasoning `none`, 34/40 token input effettivi, 32 output,

## [Seconda ispezione delle pipeline Strata](qwen-flash-strata-audit-20261006.md)

Su richiesta dell'utente il prefill qualificato diventa il punto di partenza;

- righe 3–165: **Decode: nuova ispezione del 6 ottobre** — Su richiesta dell'utente il prefill qualificato diventa il punto di partenza;
  - righe 56–165: **Selezione QSA e pubblicazione del KV nel decode** — Riletto anche `src/kernels/cuda/qsa_select.cu` di Strata: il kernel
- righe 166–200: **Diagnosi a 64K** — La v12 conserva logits bit-identici al percorso iniziale, ma scende da
- righe 201–301: **Controlli della nuova policy** — La nuova ispezione ha inoltre rilevato un errore di integrazione: il riordino
  - righe 255–301: **Diagnosi del decode dopo 64K** — L'ispezione aggiuntiva di `ExpertCache::admit` conferma che Strata conserva

## [Riesame Strata: Flash prefill e 27B decode, 7 ottobre 2026](qwen-flash-strata-audit-20261007.md)

Aggiornamento dell'8 ottobre: la nuova verifica del prefill

- righe 9–40: **Fonte fissata** — Aggiornamento dell'8 ottobre: la nuova verifica del prefill
- righe 41–57: **Stato rispetto al piano iniziale** — La v32 Flash misura a 64K 226,08 token/s di prefill e 16,64 di decode
- righe 58–90: **Flash: ottimizzazioni estratte e omissioni concrete** — `prefill.cpp` avvia per chunk grandi uno stream in ordine fisso di tutti
- righe 91–123: **27B: cosa si può riutilizzare dal decode Strata** — La verifica Strata conserva grafi di finestre, consuma residenti e miss in
- righe 124–143: **Preparazione NextN per chunk: controllo v48** — Le primitive v47 confrontano KV e carry completi con il percorso seriale
- righe 144–177: **Incremento successivo al riesame** — `ds4_qwen4_ring.cuh` aggiunge un ring sperimentale di esperti completi:
- righe 178–233: **27B: arena di rollback dopo il prefill** — Le cinque righe di verifica conservate per i 48 layer ricorrenti occupano
- righe 234–268: **Correzione della calibrazione MTP** — Il riesame ha trovato una lacuna nel tuner locale: una sola famiglia per
- righe 269–283: **Profilo persistente sul candidato finale** — `profile-v53/report.json` registra `all_pass=true`: policy recente, LRU,
- righe 284–323: **Gate vision e correzione del replay dopo rollback** — La qualificazione finale v53 conserva un fallimento: `vision-v53/report.json`,
- righe 324–415: **Chiusura Flash breve e nuovo target 27B, 8 ottobre** — La v57 corregge due costi dell'integrazione: lo scope MTP comprende anche

## [Audit dell'inferenza Qwen: interventi e risultati](qwen-inference-audit-2026-09-09.md)

Il piano è stato esaminato fino ai punti A–E.

- righe 3–43: **Risultato e perimetro** — Il piano è stato esaminato fino ai punti A–E.
- righe 44–86: **Riproduzione e correttezza** — `tests/qwen_mtp_catchup_probe.c` controlla senza tracing:
- righe 87–156: **Misure del carry** — Screening CLI, greedy, `--nothink`, 128 token, contesto allocato 2048,
- righe 157–181: **Bug degli strumenti corretti** — input differenti, hardware/toolchain incompatibili, contesti effettivi o
- righe 182–210: **Tentativi chiusi** — Per il punto 2 sono state confrontate anche KV complete e carry su 1, 2,
- righe 211–227: **Artefatti locali** — Directory sotto `performance-results/`:
- righe 228–260: **Punto 6 — input R8 condiviso e fusione selettiva (16 settembre)** — **REJECT per Qwen3.8: guadagno end-to-end sotto il 3%.** Provati separatamente
- righe 261–313: **Punto 7 — dequantizzazione cooperativa IQ4_XS** — **KEEP e default su richiesta dell'utente**, che accetta per questo punto
- righe 314–337: **Punto 8 — attention prefill a tile senza score globali** — **REJECT numerico.** Prototipo di fattibilità F32, una query per CTA e tile
- righe 338–359: **Punto 9 — residuo e RMSNorm** — **REJECT al controllo del beneficio recuperabile**, soglia 2% invariata.
- righe 360–380: **Punto 10 — convoluzione causale parallela nel tempo** — **REJECT prima dell'implementazione**, applicando il limite superiore
- righe 381–431: **Punto A — contesto e output espliciti** — **KEEP come refactoring, nessun nuovo speedup rivendicato.** Le proiezioni
- righe 432–454: **Punto B — workspace e durata dei buffer** — **Chiuso senza nuova prenotazione.** Il profilo della baseline registra sei
- righe 455–471: **Punto C — accodamento e attesa GPU** — **REJECT; coincide con il pilota del punto 2.** Il prototipo separava enqueue
- righe 472–503: **Punto D — due stream per le proiezioni** — **REJECT nello screening.** Prima misurato il costo eliminabile delle due
- righe 504–529: **Punto E — emissione SSE e calcolo successivo** — **Chiuso al gate di profilazione, senza worker I/O aggiuntivo.** Probe del

## [Progressione di ottimizzazione CUDA — Qwen3.6 27B Q4_K_S su RTX 3090](qwen36-cuda-optimization-progression.md)

Le caselle vanno marcate solo quando sono presenti nel registro del punto:

- righe 19–37: **Stato sintetico** — Le caselle vanno marcate solo quando sono presenti nel registro del punto:
- righe 38–60: **Invarianti comuni** — F32=449, Q4_K=341, Q5_K=60 e Q6_K=1.
- righe 61–96: **Cosa significa “occupare tutta la bandwidth”** — La RTX 3090 dichiara 936 GB/s di banda GDDR6X (circa 871,7 GiB/s).
- righe 97–115: **Protocollo di rollback e diagnosi** — Ogni punto segue la stessa sequenza:
- righe 116–149: **0. Congelare artefatto, baseline e budget VRAM** — **Gap.** Le misure storiche non sono tutte artifact-identical: LM Studio ha
- righe 150–203: **1. Portare Q8_1 e MMVQ per Q4_K e Q5_K** — **Gap.** DS4 possiede Q8_K a 256 valori e un Q8_32 ingenuo, ma non il blocco
- righe 204–234: **2. Portare Q6_K × Q8_1 per l'output head** — **Gap.** L'unico tensore Q6_K è l'output head da 248.320 righe.
- righe 235–263: **3. Riutilizzare Q8_1 e fondere gate/up/SwiGLU** — **Gap.** Gate e up leggono la stessa attivazione normalizzata, ma oggi ciascun
- righe 264–299: **4. Allineare e profilare il Gated Delta Net decode fuso** — **Gap.** DS4 ha già kernel GDN dedicati e il probe numerico è verde, ma avere
- righe 300–325: **5. Preallocare tutto lo scratch del decode** — **Gap.** I candidati Q8 correnti usano `cuda_tmp_alloc_on` durante il matmul;
- righe 326–358: **6. Reintrodurre CUDA Graph sul percorso stabile** — **Gap.** Il prototipo precedente riusava il graph ma non accelerava i matvec e
- righe 359–399: **7. Autotuning controllato per saturare la bandwidth** — **Gap.** La geometria upstream è un ottimo punto iniziale, ma la RTX 3090
- righe 400–428: **8. Matrice context/VRAM e validazione finale GPU** — **Gap.** Un percorso veloce a context 128 può perdere vantaggio o superare i
- righe 429–467: **9. Valutazione e confronto finale con la CPU** — Questo è obbligatoriamente l'ultimo punto.
- righe 468–482: **Riferimenti principali** — Sezione strutturale; consultare il contenuto locale indicato.

## [Registro delle ipotesi sul drift Qwen3.6](qwen36-drift-hypotheses.md)

Il gate minimo usa lo stesso GGUF, gli stessi token di prompt e la stessa

- righe 22–44: **Contratto di successo** — Il gate minimo usa lo stesso GGUF, gli stessi token di prompt e la stessa
- righe 45–56: **Metodo** — 1.
- righe 57–82: **Stato sintetico** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 83–247: **Evidenze consolidate** — teacher-forced delle logprob oracle 0,21145.
  - righe 85–97: **2026-08-04 - corpus corto corretto** — teacher-forced delle logprob oracle 0,21145.
  - righe 98–107: **2026-08-04 - bisezione per prefisso** — 2,7811; dopo `assistant`, prefisso 19, cambia l'argmax.
  - righe 108–118: **2026-08-04 - bisezione per layer** — 1,217 al layer 63; nessun NaN/Inf.
  - righe 119–136: **2026-08-05 - testata, matrice numerica e inviluppo CPU/CUDA** — output norm i logits hanno MAE centrata 0,532, coseno 0,959 e overlap top-20
  - righe 137–154: **2026-08-05 - gate reale 32+32 contro llama CPU** — 0,22287, MAE centrata 0,17247 e MAE logprob target 0,18266.
  - righe 155–175: **2026-08-05 - esperimenti matvec quantizzato** — misurabili sono QKV 2,19e-4, attention output 1,83e-5 e FFN output 1,64e-4.
  - righe 176–183: **2026-08-05 - determinismo release** — Tre processi separati, stesso build `sm_86`, stesso GGUF, prompt e posizione
  - righe 184–205: **2026-08-05 - laboratorio numerico e audit Ollama** — CPU indipendente a input identici.
  - righe 206–219: **2026-08-05 - correzione layout tiled confermata end-to-end** — Hugging Face fissato.
  - righe 220–247: **2026-08-05 - suite corta tiled rigenerata** — passi greedy e 32 teacher-forced per caso; verifica struttura/checksum `PASS`.
- righe 248–314: **Sequenza degli esperimenti** — Stato: `COMPLETATO, H04 RESPINTA`.
  - righe 250–258: **E01 - ultimo residuo, output norm e output head** — Stato: `COMPLETATO, H04 RESPINTA`.
  - righe 259–267: **E02 - inviluppo llama.cpp CPU/CUDA** — Stato: `COMPLETATO, H05 CONFERMATA`.
  - righe 268–281: **E03 - matrice numerica DS4** — Stato: `COMPLETATO; TF32 BIT-EXACT, FAST-MATH NON RISOLUTIVO`.
  - righe 282–290: **E04 - prima primitiva divergente** — Stato: `IN CORSO; GDN DECODE RESPINTA, POLICY MATVEC QUANTIZZATO ISOLATA`.
  - righe 291–299: **E05 - gate 32 token** — Stato: `GATE CORTO CALIBRATO, ZERO FAILURE; RENDERER NATIVO NON VERIFICATO`.
  - righe 300–314: **E06 - oracle semantico e prossima localizzazione** — Stato: `DA ESEGUIRE`.
- righe 315–324: **Diario delle decisioni** — `--use_fast_math` e' una proprieta' di compilazione dei kernel custom.

## [Perché LM Studio raggiunge 37,66 token/s con Qwen3.6 27B e come replicarlo in DS4](qwen36-lmstudio-decoding-analysis-2026-08-06.md)

LM Studio ha prodotto 329 token a **37,66 token/s**, cioè circa **26,56 ms per

- righe 29–54: **1. Risultato osservato e configurazione confrontata** — LM Studio ha prodotto 329 token a **37,66 token/s**, cioè circa **26,56 ms per
- righe 55–100: **2. Il modello mentale corretto per la RTX 3090** — Per ogni nuovo token, un modello denso deve attraversare quasi tutti i pesi di
  - righe 57–77: **2.1 Perché il decoding è soprattutto limitato dalla memoria** — Per ogni nuovo token, un modello denso deve attraversare quasi tutti i pesi di
  - righe 78–100: **2.2 Warp, coalescing e occupazione** — Una RTX 3090 esegue i thread in warp da 32.
- righe 101–236: **3. Cosa fa llama.cpp nel percorso veloce** — Il sorgente della build usata da LM Studio è stato ispezionato al commit esatto
  - righe 106–162: **3.1 Pesi compressi e attivazioni Q8_1 temporanee** — I pesi rimangono Q4_K, Q5_K o Q6_K in VRAM.
  - righe 163–176: **3.2 MMVQ specializzato** — llama.cpp non usa un loop scalare generico sopra questi blocchi.
  - righe 177–187: **3.3 Full offload e memoria residente** — Con tutti i layer sulla RTX 3090, il token non attraversa PCIe fra CPU e GPU a
  - righe 188–203: **3.4 Flash Attention** — Flash Attention fonde la formazione dei punteggi, softmax e prodotto con V in
  - righe 204–220: **3.5 Gated Delta Net fusa** — Qwen3.6 alterna full attention e layer ricorrenti Gated Delta Net.
  - righe 221–236: **3.6 Riuso del grafo CUDA** — CUDA Graph registra una sequenza di kernel e la ripresenta con una sola
- righe 237–324: **4. Cosa è già implementato in DS4** — Il loader ora riconosce e valida il layout misto del Q4_K_S, inclusi i tensori
  - righe 239–254: **4.1 Supporto completo del GGUF Q4_K_S** — Il loader ora riconosce e valida il layout misto del Q4_K_S, inclusi i tensori
  - righe 255–262: **4.2 Modello interamente residente e layer-major prefill** — DS4 copia il modello sulla GPU, conserva la KV sulla GPU e usa prefill
  - righe 263–270: **4.3 Kernel float warp8 affidabili** — Q4_K, Q5_K e Q6_K hanno percorsi warp8 che riducono drasticamente il numero di
  - righe 271–311: **4.4 Q4_K/Q5_K × Q8_K diagnostico** — È stato esteso il percorso sperimentale
  - righe 312–324: **4.5 Selettori per isolare le proiezioni** — I flag diagnostici permettono di applicare Q8 soltanto a:
- righe 325–378: **5. Esperimenti scartati o non promossi** — È stato costruito un prototipo che:
  - righe 327–343: **5.1 CUDA Graph DS4** — È stato costruito un prototipo che:
  - righe 344–353: **5.2 Q8 per gruppi da 32 implementato in modo ingenuo** — DS4 possedeva già un esperimento `Q8_32`, concettualmente più vicino a Q8_1:
  - righe 354–365: **5.3 Q6_K × Q8_K artigianale** — È stato provato un kernel Q6 per accelerare l'output head da 248.320 righe.
  - righe 366–373: **5.4 KV cache a 8 bit** — Non è stata implementata né forzata.
  - righe 374–378: **5.5 MTP/speculative decoding** — Non è stato usato.
- righe 379–405: **6. Risultati prestazionali della sessione** — Benchmark comune:
- righe 406–480: **7. Gate numerici ed end-to-end** — La suite `qwen_numerics_probe` è PASS:
  - righe 408–422: **7.1 Probe di primitive: riusciti** — La suite `qwen_numerics_probe` è PASS:
  - righe 423–438: **7.2 Un singolo prompt: riuscito ma insufficiente** — Su `short_fact_english`, Q4/Q5×Q8_K completo contro la baseline float ha dato:
  - righe 439–459: **7.3 Suite corta completa: fallimenti** — La suite contiene otto casi e confronta 32 righe greedy più 32 righe
  - righe 460–480: **7.4 Perché il probe passa ma il modello fallisce** — Il probe risponde alla domanda:
- righe 481–562: **8. Come rifare meglio la conversione float→int8** — Il prossimo candidato dovrebbe adottare esattamente un blocco da 32:
  - righe 483–503: **8.1 Portare Q8_1, non rinominare Q8_K** — Il prossimo candidato dovrebbe adottare esattamente un blocco da 32:
  - righe 504–520: **8.2 Portare insieme il vec-dot e la geometria MMVQ** — La sequenza consigliata è:
  - righe 521–534: **8.3 Quantizzare una volta e riusare per gate/up** — Gate e up leggono la stessa attivazione normalizzata.
  - righe 535–549: **8.4 Preallocare lo scratch prima di CUDA Graph** — Il buffer Q8_1 massimo necessario per ciascuna dimensione deve appartenere al
  - righe 550–562: **8.5 Usare una frontiera di layer solo come strumento diagnostico** — Ridurre Q8 ai primi o agli ultimi layer aiuta a localizzare la sensibilità, ma
- righe 563–582: **9. Relazione con il riferimento CPU** — Il backend CPU rimane un riferimento utile per layout, formula e ordine
- righe 583–608: **10. Piano di lavoro consigliato** — Ordine suggerito, con un gate dopo ogni passo:
- righe 609–618: **Riferimenti principali** — Sezione strutturale; consultare il contenuto locale indicato.

## [Laboratorio numerico Qwen3.6](qwen36-numerics-lab.md)

oracle CPU indipendenti.

- righe 16–35: **Componenti** — oracle CPU indipendenti.
- righe 36–53: **Comandi** — Su RTX 3090 (`sm_86`):
- righe 54–81: **Audit del sorgente Ollama** — Audit eseguito il 2026-08-05 sul commit Ollama
- righe 82–202: **Risultati ottenuti** — Il probe CUDA sintetico passa:
  - righe 112–144: **Suite corta rigenerata dopo la correzione** — Il run `ds4-q4-cuda-tiled-short-001` usa llama.cpp non patchato
  - righe 145–159: **Regressioni e long context** — Le regressioni CUDA granulari condivise sono verdi, inclusi
  - righe 160–167: **System prompt** — Il system prompt non era stato rimosso: la CLI usa per default
  - righe 168–202: **Prefill layer-major e tentativi di decode veloce** — Il collo di bottiglia originale era strutturale: `ds4_session_sync` inoltrava

## [Esperimenti prestazionali Qwen3.6 27B CUDA](qwen36-performance-experiments-2026-08-05.md)

Velocizzare il percorso CUDA di Qwen3.6 27B Q4_K_M senza degradare la

- righe 13–24: **Obiettivo** — Velocizzare il percorso CUDA di Qwen3.6 27B Q4_K_M senza degradare la
- righe 25–38: **Ambiente di prova** — `65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921`;
- righe 39–52: **Baseline iniziale** — Prima delle modifiche, il prefill inoltrava un token alla volta attraverso i
- righe 53–70: **Modifica strutturale: prefill layer-major** — È stato introdotto un grafo Qwen multi-riga che:
- righe 71–116: **Esperimenti sul prefill** — Le misure principali usano 2.048 token, chunk 2.048 e RTX 3090.
  - righe 87–103: **Profilazione del prefill** — Su un chunk da 128 token, prima dell'attention GEMM, la profilazione aggregata
  - righe 104–116: **Tentativi di prefill scartati** — 136,39 token/s.
- righe 117–152: **Esperimenti sul decode** — Le misure corte usano context 128 e 16 token generati, salvo indicazione
  - righe 136–152: **Causa del rifiuto del decode Q8** — Nel caso `short_fact_english`, il candidato Q8 coincide con l'oracolo per i
- righe 153–245: **Test di correttezza eseguiti** — Comando:
  - righe 155–177: **Probe numerici CUDA** — Comando:
  - righe 178–187: **Pattern dei layer Qwen** — Comando:
  - righe 188–202: **Test del comparatore** — Comando:
  - righe 203–245: **Suite corta full-vocabulary finale** — Run DS4 finale:
- righe 246–264: **Configurazione finale consegnata** — Il percorso predefinito conserva soltanto le modifiche considerate affidabili:
- righe 265–414: **Confronto locale con LM Studio** — Su richiesta e' stata analizzata l'installazione locale di LM Studio senza
  - righe 296–316: **Baseline di release dopo il confronto** — Con la stessa build CUDA e lo stesso Q4_K_M DS4:
  - righe 317–344: **Crossover Q4_K per il physical batch da 128** — Il primo tentativo di imitare il physical batch 512 di LM Studio aveva lasciato
  - righe 345–397: **Verifiche reali 4K e 16K dopo il crossover** — Sul caso `long_canary_ctx4096` sono stati eseguiti i chunk 128, 512, 2048 e
  - righe 398–414: **Regressioni finali** — Il `make test` senza selezione e' stato interrotto dopo oltre 16 minuti mentre
- righe 415–424: **Conclusione** — La configurazione di release misurata con la stessa build raggiunge 549,50
- righe 425–472: **Aggiornamento 2026-08-06: Ottimizzazione Register-State Caching per GDN CUDA** — In data 2026-08-06 e' stata implementata l'ottimizzazione del caching dello stato nei registri GPU nei kernel CUDA `qwen35_gdn_rows_kernel` e `qwen35_conv_rows_kernel` in `ds4_cuda.cu`.
  - righe 429–432: **Modifiche Apportate** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 433–438: **Risultati delle Misure** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 439–455: **Verifica Q4_K_S e residency del 2026-08-06** — Il GGUF Unsloth Q4_K_S verificato ha SHA-256
  - righe 456–472: **Decode long-context: warp attention e tentativo grouped GQA del 2026-08-10** — Sul Q4_K_S, con 10.666 token di prompt e 200 token greedy a context 32.768,

## [Ledger prestazionale Qwen3.6 CUDA](qwen36-performance-ledger.md)

Aggiornamento 2026-09-09: audit operativo Qwen3.8/Qwen3.6.

- righe 12–44: **Contratto** — Aggiornamento 2026-09-09: audit operativo Qwen3.8/Qwen3.6.
- righe 45–134: **Stato corrente** — lookahead constraint engine-only, callback constraint dopo submission,
- righe 135–150: **Soluzioni mantenute** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 151–182: **Soluzioni scartate o non promosse** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 183–445: **Parity Q8_1/MMVQ sul riferimento storico, 2026-08-12** — Il checkout LM Studio esatto `1a064ab0921238c1daa397d6f4a900ef33884de2`
  - righe 185–202: **FATTO DIMOSTRATO — dispatch reale** — Il checkout LM Studio esatto `1a064ab0921238c1daa397d6f4a900ef33884de2`
  - righe 203–218: **FATTO DIMOSTRATO — A/B/C al greedy step 3** — Configurazione congelata: GGUF Q4_K_S SHA-256 `ff857ba9…232aca`, RTX 3090
  - righe 219–247: **FATTO DIMOSTRATO — primitive e quantizer parity** — Un entry point diagnostico temporaneo ha alimentato i kernel DS4 e llama.cpp
  - righe 248–260: **FATTO DIMOSTRATO — frozen output head e scala FP16** — Con lo stesso hidden pre-output-norm, il solo output head F32 riproduce
  - righe 261–269: **INFERENZA E DECISIONE** — La precisione Q8_1 può cambiare una decisione a margine basso anche quando DS4
  - righe 270–302: **FATTO DIMOSTRATO — sensibilità Q4/Q5/Q6** — Due selettori temporanei, poi rimossi, hanno separato Q4-only e Q5-only.
  - righe 303–322: **FATTO DIMOSTRATO — precision staircase** — La ricostruzione offline dello stesso `output_norm` F32 e il risultato
  - righe 323–340: **FATTO DIMOSTRATO — Q8_1 LSQ offline e proiezione** — Sul vero vettore critico, un passo LSQ per blocco da 32 riduce activation MAE
  - righe 341–445: **FATTO DIMOSTRATO — residuo diffuso e Q8_1-R8** — Il simulatore offline ha decodificato direttamente le righe 310 e 728 del vero
- righe 446–470: **Analisi del residuo a 8K, 2026-08-11** — Il conto dei soli byte di peso per operazione è: ricorrente 3,426 GB, full
- righe 471–491: **Esperimenti long-context 2026-08-11** — Fonti recuperate: guida locale `problem-to-source.md`, FlashDecoding++,
- righe 492–543: **Trasferimento vLLM/Marlin e visibilità R8, 2026-08-12** — È stato ispezionato vLLM al commit fissato
- righe 544–623: **Stabilizzazione R8 long-context e riuso GQA, 2026-08-12** — La riproduzione a 10.666 token ha separato due fenomeni che prima venivano
- righe 624–653: **Curva completa 2K–30K e soglia split-K, 2026-08-13** — La nuova suite non rapida `context-curve-full` misura 15 frontiere a passo 2K,
- righe 654–699: **MTP long-context e crossover split-K, 2026-08-13** — La curva server target-only 0–28K a passo 2K, capacità 28.737, due run per
- righe 700–722: **Coda ordinata** — 1.
- righe 723–740: **Template per il prossimo record** — ID / data / commit / dirty state:

## [Qwen3.8 agentico: `SEARCH` statica e MTP grammar-aware](qwen38-agent-search-static-2026-08-27.md)

Questo ciclo avvicina il decode del server agentico Qwen3.8 alla velocità del

- righe 7–28: **Obiettivo e invarianti** — Questo ciclo avvicina il decode del server agentico Qwen3.8 alla velocità del
- righe 29–51: **Baseline prima della modifica** — Hardware: RTX 3090, build CUDA `sm_86`, contesto allocato 22.593, prompt
- righe 52–74: **Risultato appaiato** — Il masking target-only scende del 99,21%.
- righe 75–127: **MTP grammar-aware: implementazione e vantaggio** — Il server passa al verifier un callback per ogni distribuzione target della
- righe 128–161: **Gate di correttezza** — La storia completa è stata ripetuta con `compare_new_vs_oracle`: 433 confronti
- righe 162–229: **Matrice temperatura, thinking e MTP** — Il gate `tests/test_server_sampling_matrix.py` esplicita il prodotto cartesiano
- righe 230–278: **Fonti primarie e scelte implementative** — I valori campionati non sono inventati dal benchmark.
- righe 279–289: **Verifica corrente** — Le misure sopra descrivono fixture storici ritirati e non certificano ExMekh Agent.

## [Flash Next Q2: residency CUDA sulla RTX 3090 — 3 ottobre 2026](qwen38-flash-3090-residency-20261003.md)

**Esperimenti diagnostici, non qualificati per il percorso predefinito.**

- righe 3–9: **Stato** — **Esperimenti diagnostici, non qualificati per il percorso predefinito.**
- righe 10–29: **Diagnosi e provenienza** — Macchina locale: RTX 3090 sm_86, i9-14900K, 64 GB fisici, WSL con 48 GB e
- righe 30–40: **Variabili isolate** — 1.
- righe 41–65: **Prime misure direction** — Contesto allocato 64, frontiere 16/32, teacher-forced 4 token per frontiera,
- righe 66–77: **Gate di promozione** — Richiesti: warm-up e almeno cinque ripetizioni con CV <= 5%, logit del
- righe 78–113: **Verifiche successive** — Baseline pulita a frontiere 16/32, 4 token teacher-forced, un primo sweep
- righe 114–125: **Nota sulla scheda madre** — L'inventario Windows identifica MSI MAG Z790 TOMAHAWK MAX WIFI (MS-7E25),
- righe 126–357: **Gerarchia adattiva e checkpoint: estensione richiesta dall’utente** — La cache statica precedente è superata.
  - righe 140–175: **Policy e trasferimenti** — per tensor/layer.
  - righe 176–190: **Persistenza della graduatoria** — Il payload Flash può aggiungere una sezione versionata contenente storia
  - righe 191–215: **Correttezza e integrazione verificate** — conservazione dei conteggi e sparizione completa della storia oltre 128 token.
  - righe 216–238: **Selezione dell’algoritmo** — Con cache abbastanza grandi da contenere il working set breve, LRU e finestre
  - righe 239–259: **Scelta misurata e varianti scartate** — Cinque sweep misurati dopo il warm-up, cache 1 GiB VRAM / 2 GiB RAM, prompt
  - righe 260–277: **Parità dei checkpoint con Qwen 27B** — Due guard preesistenti escludevano Flash dalla cattura del prefisso di sistema
  - righe 278–333: **Misure finali della policy selezionata** — Medesimo Q2, testo letterario, ctx allocato 256, frontiere 64/128, 16 token
  - righe 334–357: **Esito della verifica richiesta** — Con il root SSD corretto, i test HTTP invariati sono:
- righe 358–502: **Audit upstream e ripresa del 4 ottobre 2026** — Il fetch di `antirez/ds4/main` restituisce ancora
  - righe 400–502: **Budget degli esperti rispetto al contesto configurato** — Il GGUF locale dichiara `qwen4exp.context_length=262144`.

## [Flash Next: pipeline CUDA e protezione Qwen27 — 4 ottobre 2026](qwen38-flash-pipeline-20261004.md)

Implementazione del piano di azione sul working tree `codex/qwen-flash-3090`, HEAD `bd89b13f0c839d3d9e17c7afab1f8a4dcf1b98e6`.

- righe 3–8: **Perimetro** — Implementazione del piano di azione sul working tree `codex/qwen-flash-3090`, HEAD `bd89b13f0c839d3d9e17c7afab1f8a4dcf1b98e6`.
- righe 9–20: **Modifiche** — I kernel aritmetici, il router, top-k/tie-break, BF16 PLE, KV e il formato dei checkpoint non sono cambiati.
- righe 21–23: **Avvio** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 24–26: **Default Qwen27, comando revisionabile senza build/inferenza:** — ./start-ds4-server.sh --print-command
- righe 27–30: **Launcher dedicati dell’agente, uno per modello:** — ./agent/start-ds4-server.sh # Qwen27B, MTP, ctx 28000
- righe 31–105: **Il contesto e il chunk restano configurabili; MTP è attivo:** — DS4_MODEL_PROFILE=flash DS4_CTX=16384 DS4_PREFILL_CHUNK=256 \
- righe 42–49: **Evidenze e protocollo** — Artefatti locali: `performance-results/flash-pipeline-20261004/`.
- righe 50–66: **Risultati qualificati localmente** — Il record JSON conserva campioni, statistiche complete, comandi, hash di sorgenti/binari e risultati dei gate.
- righe 67–79: **Correttezza e checkpoint** — I checkpoint del 27B restano utilizzabili con il 27B; i loro scenari/contratti vengono esercitati sul Flash, non i file KV di un modello diverso.
- righe 80–105: **Riproduzione e lavoro rimanente** — make -j2 ds4 ds4-server ds4-bench CUDA_ARCH=sm_86

## [Flash Next CUDA: letture anticipate degli esperti](qwen38-flash-prefill-async-20261004.md)

I tempi del primo workload derivano dal CSV arrotondato a due decimali,

- righe 9–31: **Risultato** — I tempi del primo workload derivano dal CSV arrotondato a due decimali,
- righe 32–75: **Modifica mantenuta** — La ricerca precedente
- righe 76–99: **Correttezza e checkpoint** — I test serializzano i processi del modello, rispettando l’instance lock.
- righe 100–140: **Misure e limiti** — Il modello Flash è il GGUF Q2 locale da 147207127040 byte; Qwen27 è
- righe 141–161: **Esperimento scartato: chunk più grandi** — A parità di prompt da 1024 token, una prova cold misura 68,69 / 68,43 /

## [Flash Next: lentezza del prefill e budget RAM, 4 ottobre 2026](qwen38-flash-prefill-latency-20261004.md)

Baseline congelata: `5d374d4978c644a3de9ce40353b3dcba9b97d235`.

- righe 9–37: **Metodo e provenienza** — Baseline congelata: `5d374d4978c644a3de9ce40353b3dcba9b97d235`.
- righe 38–67: **Perché la GPU attende** — Sul prompt HTTP profilato, un solo prefill richiede **185,241 s**.
- righe 68–98: **Modifica mantenuta** — `expert_ram_slot` ora usa il budget derivato dalla memoria disponibile,
- righe 99–148: **Benchmark e correttezza** — Flash: `speed-bench/promessi_sposi.txt`, ctx allocato 131072, chunk 256,
- righe 149–163: **Limiti e prossime verifiche** — Il verdetto **KEEP** riguarda il budget RAM e i workload Flash misurati;

## [Flash: doppio prefill del prefisso agentico — 4 ottobre 2026](qwen38-flash-prefill-replay-20261004.md)

Il server lanciato con `agent/start-ds4-server-flash.sh` riceve un prompt

- righe 3–22: **Diagnosi del caso segnalato** — Il server lanciato con `agent/start-ds4-server-flash.sh` riceve un prompt
- righe 23–32: **Correzione e compatibilità** — Cache residente e checkpoint cold Qwen usano lo stesso prefisso di sistema
- righe 33–67: **Verifiche** — cold inferiore al prefisso e cache disabilitata.
- righe 68–81: **Lentezza residua e hardware** — Nella rilevazione del processo originale: circa 22799 MiB di VRAM usata,
- righe 82–94: **Default dei launcher confermati** — Su richiesta dell'utente, Flash conserva il contesto predefinito 131072 e

## [Qualificazione numerica Qwen3.8 — 17 settembre 2026](qwen38-qualification-2026-09-17.md)

RTX 3090 24 GB, WSL Ubuntu 24.04, CUDA 12.4, sm_86, driver 610.62.

- righe 8–58: **Ambiente e metodo** — RTX 3090 24 GB, WSL Ubuntu 24.04, CUDA 12.4, sm_86, driver 610.62.
- righe 59–86: **Oracolo completo: esito FAIL, nessuna promozione** — DS4 default e llama.cpp completano tutti i nove casi, con inventari validi
- righe 87–109: **FP16 preesistente: qualifica parziale, non bit-exact** — Il run `no-k-f16` completa il medesimo corpus disabilitando soltanto
