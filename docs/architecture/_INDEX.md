# Indice: architecture

Indice generato; non modificare a mano.

## [Accettazione merge DS4 e QS — 2 ottobre 2026](ds4-merge-acceptance-20261002.md)

**PASS per Qwen3.8-27B sulla RTX 3090.** Sono concluse le prove residue:

- righe 3–20: **Esito e provenienza** — **PASS per Qwen3.8-27B sulla RTX 3090.** Sono concluse le prove residue:
- righe 21–51: **Foto e memoria tra tool** — La fixture JPEG reale `earth.jpg` proviene dal corpus upstream.
- righe 52–66: **Regressioni e prestazioni** — QS: **1.958 PASS**, copertura 100%, Ruff e Pyright puliti.
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
