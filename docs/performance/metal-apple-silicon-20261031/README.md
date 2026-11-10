# T04–T05 Metal: implementazione e qualifiche limitate

31 ottobre 2026. MacBook Pro M5 Pro, 24 GiB. Modifiche lasciate senza commit.

Il porting dei percorsi condivisi è disponibile. **Il gate completo T04/T05
resta IN_PROGRESS**: Quick Chat e i flussi con i GGUF linguistici completi
richiedono T06-A per Flash e T06-B/T03 per Qwen27. I test host, i provider mock,
la fixture Metal e gli encoder reali hanno ambiti distinti.

Esito delle verifiche eseguite: **21 gruppi C/Metal finali PASS**, build CPU
isolata PASS, **1210 test frontend PASS / 46 SKIP**, copertura statement e
branch **100%**. Entrambi gli encoder reali superano le due soglie numeriche.
Gli hash dei sorgenti e dei log finali sono stati ricontrollati.

## Modifiche

- `ds4_gpu_tensor_copy` usa il batch attivo oppure completa una copia autonoma.
  Il catch-up MTP non dipende più da un dettaglio del backend CUDA.
- Il workspace MTP possiede il proprio rilascio: errore parziale e retry non
  lasciano allocazioni né azzerano lo stato del target. Lo stesso stimatore
  conta i buffer fisici per ammissione, numero di sessioni e report memoria.
- Vision Qwen27 ammette Metal e conserva il controllo del proiettore 5120;
  quello Flash locale è 2560. Dopo il parcheggio delle sessioni si verifica
  il picco simultaneo di scratch device e posizioni/output host. Il fallimento
  di ammissione ripristina le conversazioni già attive.
- I due ingressi Qwen condividono pianificazione delle dimensioni e bicubic
  uint8. La modalità raw controlla soltanto la normalizzazione. Prima della
  correzione il percorso normalizzato sceglieva un altro resizer e un diverso
  arrotondamento dei mezzi interi.
- I checkpoint multimodali Qwen usano VMCP versione 2: gli stati condizionati
  da immagini con il vecchio preprocessing vengono rifiutati prima di mutare
  la sessione. I checkpoint testuali conservano il formato precedente.
- I test live agentici e callback MTP accettano `DS4_TEST_BACKEND`,
  `DS4_TEST_CTX` e `DS4_TEST_PREFILL`; su Apple il default è Metal 8192/64.
  Gli input invalidi vengono rifiutati. Il test checkpoint ora compila anche
  con Metal. I test live non vengono eseguiti dalla qualifica sintetica.

Non sono stati modificati kernel aritmetici Metal o implementazioni delle
quantizzazioni. I riferimenti S1–S4 delle fondamenta rimangono quelli già
registrati; qui cambiano contratto di copia, scheduling, accounting e
preprocessing host, riusando le implementazioni esistenti.

## Verifiche e confini

| Prova | Ambito |
| --- | --- |
| [Regressione finale](checkpoint-compatibility/verification.json) | Build Metal, test CPU; risorse; quantizzazioni/GDN/attenzione; grafo ibrido; lifecycle; protocolli; sampling; nuovi casi T04/T05 |
| [Frontend e regressione completa](verification/verification.json) | Suite frontend con provider mock, compresi Playwright, compaction, history e contratti client; nessuna inferenza LLM Metal |
| [Qwen27 vision](qwen27-vision-final.json) | Encoder GGUF F16 reale, output 121 × 5120 |
| [Flash vision](flash-vision-final.json) | Encoder GGUF Q8_0 reale, output 121 × 2560 |

La fixture Qwen27 contiene quattro layer ibridi, embedding/FFN 256 e
vocabolario 32, mantenendo le dimensioni reali delle teste GDN/attenzione.
Usa il grafo di produzione su Metal, contesto 16, prefill 5 e cap device
128 MiB. Copre copie autonome/in batch, allocazione MTP negata e retry,
`padre → skill → ritorno`, cancellazione, metadati RNG, SSD e restore in una
nuova sessione; accettazione/rifiuto, callback invalida e rollback RNG;
confronto dello stato KV/ricorrente/MTP con l'esecuzione senza speculazione;
contesto quasi pieno, testo/immagine/testo, fingerprint cambiato, mRoPE,
park/resume e rifiuto del picco encoder senza perdita delle due sessioni.
Il vecchio checkpoint multimodale viene respinto conservando lo stato vivo.
Il restore e la continuazione testuale funzionano anche con encoder vision
disabilitato; il formato dipende dalla famiglia del modello. La build CPU
isolata è conservata anche nel [report dopo revisione](post-review/verification.json).

Il confronto numerico del grafo usa `2e-5 + 5e-5 * abs(reference)` e controlla
valori finiti. I payload dei rollback e dei restore dello stesso percorso
sono confrontati byte per byte. Flash ha una prova MTP separata di staging e
combine contro FP64, con una mappa da 4 MiB; il test lifecycle Flash con MTP
usa stato sintetico e non esegue il drafter completo.

Gli encoder sono eseguiti uno alla volta, senza caricare i pesi LLM. Il
riferimento indipendente è Transformers su CPU con gli stessi pesi GGUF.
Si confrontano prima ingressi identici esportati dal preprocessore nativo,
poi il percorso completo contro il processore PIL di Transformers. La soglia
minima per token resta **coseno ≥ 0,99**, fissata prima dei tentativi.
I report includono massimi/medie degli errori, layout, hash di sorgenti,
binario, modello e immagine, oltre alle versioni delle librerie.

Sul JPEG `earth.jpg`, 121 token effettivi e cap 128:

| Encoder | Coseno minimo, patch identiche | Coseno minimo, immagine completa |
| --- | ---: | ---: |
| Flash 2560 | 0,9999999997 | 0,9995476 |
| Qwen27 5120 | 0,9999999997 | 0,9999743 |

[Prima](preprocessing.json) e [dopo](preprocessing-after.json) documentano il
confronto delle patch: sul PNG `maple.png` rimane solo roundoff FP32; sul JPEG
lo scarto medio normalizzato scende da circa 0,0013723 a 0,00002762. I report
iniziali e gli esperimenti non promossi restano conservati separatamente;
non sostituiscono i risultati finali. La prima regressione frontend
([bounded](bounded/verification.json)) ha rilevato una lettura immediata dopo
il clic del menu mobile: il test ora attende la visibilità del pannello.

## Riproduzione

Dalla radice del repository, con le dipendenze già installate:

```sh
python3 tools/qualify_metal_foundations.py \
  --output build/metal-qualification/t04-t05-new \
  --features --cpu-build --frontend

build/metal-venv/bin/python -m tools.qualify_qwen_metal_vision \
  --mmproj gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image tests/vision-fixtures/glm53/earth.jpg --tokens 128 \
  --projection 2560 --report build/metal-qualification/flash-vision-new.json

build/metal-venv/bin/python -m tools.qualify_qwen_metal_vision \
  --mmproj gguf/mmproj-Qwen3.8-27B-F16.gguf \
  --image tests/vision-fixtures/glm53/earth.jpg --tokens 128 \
  --projection 5120 --report build/metal-qualification/qwen27-vision-new.json
```

La qualifica generale richiede una directory nuova e registra comandi, tempi,
exit code, memoria e hash dei log. Le compilazioni CPU temporanee vengono
rimosse dopo la verifica. I due comandi vision vanno eseguiti in sequenza.

Restano **NOT_VERIFIED** inferenza dei due LLM completi a 8192/64, MTP con il
sidecar 27B reale corrispondente, Flash MTP completo, integrazione multimodale
LLM/server/client reale, restart HTTP con i GGUF completi, CUDA/ROCm e altri
Mac. Nessuno speedup MTP è dedotto dalle prove di correttezza. La disponibilità
locale del sidecar e i gate streamed T06 restano prerequisiti delle prove
integrate, senza fallback a inferenza LLM su CPU.
