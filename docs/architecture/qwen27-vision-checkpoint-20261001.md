# Qwen27 vision e checkpoint completi — 1 ottobre 2026

## Implementazione e perimetro

Sviluppo sul branch `codex/merge-ds4-upstream-2026-10-01`, dopo il merge
`0edbded0ff3476190060450149dd74a10fa12043`. L’implementazione è contenuta in
`82d4e51b47d85c1e7155e09580a0c7349f148e96`. `main` resta il riferimento
precedente. FE/QS rimane l’unico frontend agentico; nessuna delega è stata usata.

Qwen3.8-27B è nativamente multimodale: il [config ufficiale Qwen](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)
include un encoder vision a 27 layer, hidden 1152 e uscita 5120. DS4 riusa
preprocessing, decoder immagini e kernel ViT importati da antirez per Flash Next,
con il proiettore specifico del 27B. Loader, grafo testuale e MTP restano `qwen35`,
distinti da `qwen4exp`. Il mmproj Flash ha uscita 2560 e non è intercambiabile.

Il percorso testo conserva il precedente kernel RoPE. Per immagini vengono
usate posizioni T/H/W e 32 coppie di frequenze interleaved `[11,11,10]`, incluse
le KV del sidecar MTP. Indici di cache e posizioni rotary rimangono distinti.
Le posizioni dei token precedenti alla prima immagine restano quelle testuali.

## Ordine delle immagini e del thinking

Il [forward ufficiale Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
calcola gli embedding delle immagini, li inserisce nei rispettivi placeholder
nel prompt e poi esegue il modello linguistico. Più immagini nello stesso turno
non richiedono thinking intermedio. DS4 segue questo ordine: encoder sequenziali,
embedding in RAM, prefill multimodale completo, poi thinking/generazione.
Un turno separato dopo ogni immagine resta una scelta applicativa possibile.

La letteratura sulla [separazione degli stadi MLLM](https://arxiv.org/abs/2512.17574)
tratta preprocessing, encoder e modello linguistico come stadi distinti.
L’alternanza su una singola 3090 con checkpoint SSD descritta qui è una scelta
implementativa locale, verificata sperimentalmente; non viene attribuita al
paper né presentata come requisito del modello.

## Ciclo VRAM effettivo

Per ogni immagine non presente nella cache embedding, sotto il lock d’inferenza:

1. Salvare su SSD il payload completo di ogni sessione Qwen27 viva.
2. Distruggere i grafi delle sessioni: KV target/MTP, stato ricorrente e scratch
   vengono liberati. I pesi linguistici target/MTP restano residenti.
3. Caricare pigramente il mmproj ed eseguire il ViT; conservare l’output in RAM.
4. Sincronizzare CUDA e liberare soltanto la mappatura GPU dei pesi vision.
5. Ricreare i grafi con il modo MTP originario di ciascuna sessione e ripristinare
   i payload completi, senza rigenerare il contesto.

Salvare un file senza distruggere il grafo non libera VRAM. Questo percorso
esegue entrambe le operazioni. Un errore durante l’encoder tenta comunque il
ripristino; un checkpoint la cui ricostruzione fallisce resta disponibile fino
alla chiusura della sessione. L’API encoder richiede inferenza serializzata con
le altre operazioni sulle sessioni dello stesso engine, come nel server.

I checkpoint temporanei usano il meccanismo esistente in `/tmp`, sul filesystem
WSL. Richiedono spazio per la somma dei payload delle sessioni vive. Il ciclo
per immagine comporta I/O: non è un nuovo scheduler di offload e non è un modo
per superare la capacità del contesto dopo il ripristino. Gli embedding già in
cache evitano encoder e parcheggio. L’encoder non resta residente tra richieste.

## Payload, skill e subagent

Un payload multimodale contiene un envelope `VMCP`, versione 1, con:

- dimensione del payload famiglia e numero di token;
- fingerprint e intervallo token di ogni immagine;
- posizioni assolute T/H/W e delta mRoPE per Qwen27;
- payload completo preesistente: KV full-attention, Gated DeltaNet, logits,
  token, pending hidden, draft logits e KV MTP quando abilitato.

Il corpo Flash Next include già posizioni e delta, oltre a cache attention,
PLE e stato lineare; l’envelope aggiunge le identità delle immagini. Gli header
sono validati prima del ripristino. I wrapper server verificano il checksum
prima di mutare la sessione.

Il layout testuale mantiene ABI 3 e gli stessi byte. I rollback compatti sono
esclusi quando non rappresentano tutto lo stato, in particolare con immagini;
Flash Next usa un payload completo anche per le frontiere request-local.
I checkpoint `.dsk` delle skill e `.dsh` dei contesti agentici condividono questo
percorso completo. Copiare il live tool state conserva anche gli offset delle
immagini. I prefissi JSON/tool forzati vengono aggiunti alla KV viva, evitando
la ricostruzione testuale che perderebbe gli embedding. Per le continuazioni
Qwen27 con immagini e senza thinking, il binding visibile già presente collega
la history client alla KV campionata, controllando marker e fingerprint;
il template testuale e l’ordine dei token generati restano quelli precedenti.

## Artefatti riproducibili

| Artefatto | Byte | SHA-256 |
| --- | ---: | --- |
| `Qwen3.8-27B-UD-Q4_K_S.gguf` | 15358213024 | `75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017` |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | 1369590656 | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |
| `mmproj-Qwen3.8-27B-F16.gguf` | 927607488 | `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` |

Il [mmproj Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/mmproj-F16.gguf)
è fissato alla revisione `4ca720788d1e01f1bff70c033e0d0028fd02e502`, come
target e MTP. `download_model.sh qwen38-27b-vision` verifica dimensione e hash
senza cambiare il modello selezionato. I due launcher locali aggiungono il
matching `--vision` se presente; `DS4_VISION_MODEL` consente un percorso esplicito.

Hardware: RTX 3090 24 GiB, `sm_86`, CUDA 12.4, driver 610.62, i9-14900K,
64 GB fisici; WSL 48 GB e swap 12 GB. I risultati locali e i binari congelati
sono in `performance-results/qwen27-vision-20261001/` (ignorato da Git).
Il [riepilogo JSON](../performance/qwen27-vision-checkpoint-20261001.json)
conserva le misure compatte.

## Verifica

I test numerici non cambiano tolleranze né golden. Il confronto del ViT usa
Transformers 5.18.0 CPU float32 con gli stessi pesi GGUF F16 dequantizzati,
non un encoder diverso. Su due immagini 320×256, 80 token e hidden 5120:

| Immagine | Coseno minimo | Coseno medio | Esito |
| --- | ---: | ---: | --- |
| Rosso | 0.99999842925489 | 0.9999999794277137 | PASS |
| Blu | 0.9999999860360147 | 0.9999999997199651 | PASS |

Il gate ViT upstream richiede coseno ≥0.99. Il test mRoPE indipendente float64
richiede errore assoluto <2e-5 su Q/K e identità bit-exact del percorso testo
e della KV MTP. Il massimo misurato è 6.49012663e-6 su Q e 5.95160248e-6 su K.

| Verifica | Esito effettivo |
| --- | --- |
| Build CPU e CUDA `sm_86`, unit server/core | PASS |
| Probe numerico Qwen, formati quantizzati e mRoPE | PASS; tolleranze e golden invariati |
| Contratti Python | 101 PASS, 4 SKIP su 105 test |
| Suite FE, inclusi E2E disponibili | 612 PASS, 16 SKIP; coverage 100% |
| HTTP vision, target-only e MTP esterno | 9 casi PASS per variante: immagine, continuazione con KV riusata, due immagini, pixel cambiati, thinking, JSON, tool e continuazione, SSE, Anthropic |
| Checkpoint skill/subagent multimodali Qwen27 | PASS: 248.320 logits e token identici, sessione ricreata, continuazione, tre livelli, checksum corrotto rifiutato senza mutazione |
| Immagine da 1024 token con chunk prefill 256 e MTP | PASS |
| Checkpoint testuali Flash Next Q2 | PASS con target-only e MTP embedded, sessione ricreata e tre livelli |
| CLI vision, nuova immagine su contesto attivo tramite HTTP, client QS reale e launcher FE/QS con vision | NOT_VERIFIED; rimandati dall’utente a un’altra sessione |

La prova lunga ha **20.000 token testuali già popolati**, poi una immagine e
20.100 token nel checkpoint, con capacità 22.593 e MTP attivo. Ciascun file
skill/subagent misura 2.960.394.456 byte. Salvare da solo lascia invariata la
VRAM libera (3.018.530.816 byte); distruggere il grafo la porta a
7.342.129.152 byte: **4.323.598.336 byte, circa 4,03 GiB**, effettivamente
liberati. Dopo encoder, scarico dei pesi vision e restore, la VRAM libera torna
al valore iniziale e i logits sono bit-exact. La prova di nuova immagine dopo
un prefisso testuale lungo è quindi verificata a livello del motore; la nuova
immagine su history multimodale tramite HTTP resta fra le prove rimandate.

## Prestazioni del percorso testo

Baseline vision: merge `0edbded`, con binari congelati; implementazione verificata
`82d4e51`. Ogni workload usa warm-up e cinque campioni; prompt e binari sono
identificati nel JSON. Il confronto diretto dei dump conserva esattamente tutti
i 248.320 logits alle frontiere e dopo decode, oltre ai 64 token greedy, a
2K, 8K, 16K e 22K. I tempi TTFT usano richieste streamed non riusate, con 32 token
in uscita e contesti effettivi controllati.

| Contesto | Prefill prima/dopo (token/s), prima misura | Decode prima/dopo (token/s), prima misura | TTFT prima/dopo (ms) |
| ---: | ---: | ---: | ---: |
| 2048 | 662.07 / 635.25 | 29.98 / 28.71 | 2738.07 / 2726.74 |
| 8192 | 749.17 / 690.94 | 26.78 / 24.51 | 11564.05 / 11138.14 |
| 16384 | 666.30 / 665.60 | 23.04 / 22.74 | 24544.87 / 23834.68 |
| 22528 | 633.76 / 632.89 | 21.10 / 20.73 | 35695.80 / 34538.33 |

La prima misura resta **REJECT_CANDIDATE** nell’harness: decode −4,24% a 2K e
−8,48% a 8K. Non viene cancellata né vengono allargate le soglie. Per verificare
la riproducibilità sono stati ripetuti entrambi i contesti nell’ordine
baseline/candidate e poi candidate/baseline, con warm-up nel processo e cinque
campioni per eseguibile e per giro.

| Contesto | Giro | Variazione prefill | Variazione decode | CV entro 5% |
| ---: | ---: | ---: | ---: | --- |
| 2048 | 1 | -0.60% | -0.54% | sì |
| 2048 | 2 | +0.08% | +0.48% | sì |
| 8192 | 1 | -0.44% | +0.11% | sì |
| 8192 | 2 | +0.38% | -0.04% | sì |

Nelle ripetizioni il calo iniziale non è riprodotto; tutte rispettano sia il
limite richiesto del 5% sia la soglia del 2% sul decode dell’harness. TTFT è
PASS in tutti i contesti, con output identici e CV sotto 5%. Il verdetto di
questa verifica di conservazione è **PASS**, senza dichiarare uno speedup;
il verdetto della prima serie resta nel report. Le misure non identificano
la causa della variazione fra serie: non viene attribuita al kernel o
all’hardware senza ulteriori prove.


## Limiti osservati e verifiche mancanti

La prima fixture OCR con scritta `RED7` ha prodotto `R7`; l’oracle llama.cpp
alla revisione `1a064ab0921238c1daa397d6f4a900ef33884de2`, con gli stessi GGUF,
ha prodotto anch’esso `R7`. Il test OCR resta **FAIL**: non viene sostituito da
un golden corretto artificialmente. Altri primi casi OCR sono falliti e non
hanno tutti un confronto cross-engine. Riconoscere colori non qualifica OCR
universale o qualità su immagini complesse.

Un errore reale distinto riguardava il prefisso JSON forzato: la sync testuale
perdeva lo stato immagine. È stato corretto; il caso blu ora produce JSON blu
sia con sia senza MTP, e il test del checkpoint include quel suffisso.

Thinking è verificato dal campo `reasoning_content` non vuoto; il contatore
`usage.completion_tokens_details.reasoning_tokens` osservato resta 0. Questa
limitazione della telemetria non certifica una corretta contabilizzazione del
thinking.

Metal/ROCm/CPU vision Qwen27, video, inferenza distribuita, immagini complesse
su un corpus esteso e checkpoint vision Flash Next pratici sono
**NOT_VERIFIED**. Il nuovo percorso Qwen27 vision è abilitato solo su CUDA.
Flash Next mantiene il proprio encoder e il proprio grafo; i checkpoint
skill/subagent testuali sono stati provati realmente con Q2 e con MTP embedded.
L’envelope rende rappresentabile anche il suo stato immagine, ma non sostituisce
un test pratico vision. QS ha già mapper `ImagesUserMessage` per Chat Completions e Responses.
Gli allegati FE attuali vengono referenziati come file; questa modifica non
aggiunge una UI di invio diretto dei pixel. I test vision live eseguiti qualificano
HTTP sui casi elencati; la suite FE verifica i contratti agentici esistenti.
CLI vision, nuova immagine su contesto attivo tramite HTTP e client QS reale
sono rimandati dall’utente alla prossima sessione e restano NOT_VERIFIED.
I relativi test opt-in sono presenti per riprenderli senza doverli riscrivere.

## Riproduzione

```sh
./download_model.sh qwen38-27b-vision
make cuda CUDA_ARCH=sm_86
make tests/test_qwen35_mrope tests/test_server_agentic_payload CUDA_ARCH=sm_86
./tests/test_qwen35_mrope
DS4_TEST_CTX=22593 DS4_TEST_PREFIX_TOKENS=20000 DS4_VISION_TIMING=1 \
  ./tests/test_server_agentic_payload gguf/Qwen3.8-27B-UD-Q4_K_S.gguf --mtp \
  --vision gguf/mmproj-Qwen3.8-27B-F16.gguf --image image.png
./start-ds4-server.sh
# In un secondo terminale, con server fermo rispetto ad altri client:
python3 tests/test_qwen35_vision_api.py --url http://127.0.0.1:8080
# Ripetere con ./agent/start-ds4-server.sh per MTP.
DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=16 DS4_TEST_CTX=256 \
  ./tests/test_server_agentic_payload gguf/Qwen3.8-Flash-Next-Q2.gguf --embedded-mtp
```