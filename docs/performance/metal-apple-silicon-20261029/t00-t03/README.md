# T00–T03: fondamenta Metal e grafo Qwen27

29 ottobre 2026. Implementazione delle fondamenta e gate numerico sintetico
**PASS**; inferenza reale Qwen27 **NOT_VERIFIED**, dipendente da T06-B secondo
la [roadmap](../../../roadmaps/metal-apple-silicon-support-20261028.md).
Il riferimento di partenza è `8e8ac45a7b463f7d4d72d859234f794b815019ef`;
[verification.json](verification.json) identifica sorgenti modificati, binari,
ambiente, comandi, tempi e hash dei log. Le modifiche locali preesistenti
non sono state annullate.
Il successivo [riesame delle responsabilità](../../../architecture/metal-foundations-clean-code-review-20261029.md)
e il [nuovo registro](../foundation-cleanup/README.md) identificano separatamente
i sorgenti dopo il refactoring; gli hash di questa prima verifica rimangono storici.

## Inventario e portabilità

Mac17,9/M5 Pro, RAM fisica **25.769.803.776 byte (24 GiB)**, macOS 26.6.2,
clang 21.0.0, SDK 26.5; spazio SSD iniziale circa **299,7 GiB**. GB e GiB
restano distinti. Il device riporta working set raccomandato 19.069.665.280
byte e `maxBufferLength` 14.302.248.960 byte. La raccomandazione non è RAM libera.

Il GGUF Qwen27 locale coincide con l'artefatto fissato:
15.358.213.024 byte, SHA-256
`75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017`.
[qwen27-tensors.json](qwen27-tensors.json) collega tutti gli **866 tensori**
del metadata snapshot verificato a tipo, shape, byte, offset, operazione e
uso target/NextN. Comprende i dodici tipi del target. Q4_0 è verificato come
formato aggiuntivo; il nome del sidecar non ne descrive da solo i formati
effettivi. Il sidecar reale non è stato caricato.

L'ispezione del GGUF non esegue forward. Flash e i modelli suite mantengono
identità e limiti delle evidenze precedenti; questa consegna non verifica
l'integrità del Flash locale, che appartiene a T06-A.

- `make -j2 all` e link suite Metal: PASS. Tutti i sette simboli pubblici
  `ds4_gpu_qwen35_*` risultano definiti, senza stub di successo.
- Build CPU in una copia nuova dei sorgenti: PASS, quattro eseguibili;
  `otool -L` di `ds4` mostra soltanto libSystem. Nessuna inferenza CPU.
- Quattordici test launcher: PASS; dry-run di launcher comune e dei due
  wrapper agent conservati nei log. Metal usa 512/64 e target-only; suite e
  vision richiedono selezione esplicita. I default CUDA restano coperti.
- Una vision esplicita inesistente resta visibile nel dry-run e produce
  errore prima della build. Il runtime rifiuta vision Qwen27 non CUDA prima
  del riscaldamento e dell'ammissione del target.

## Memoria e numerica

Il registro Metal applica il budget anche all'inizializzazione ordinaria,
usa `hasUnifiedMemory` e riconcilia i campioni prima di nuove sessioni Qwen
senza sovrascrivere cap o prenotazioni esplicite. Il budget incrementale
conserva il minore fra disponibilità host e working set raccomandato meno
allocazioni Metal, con riserva di **2 GiB** applicata una volta.
Le pagine inattive recuperabili sono limitate dal numero di pagine file-backed:
le pagine anonime inattive non sono trattate tutte come cache liberabile.
La telemetria pressione è best-effort: `0` significa sconosciuta, `1` normale,
un valore superiore azzera l'ammissione aggiuntiva. Buffer oltre la capacità
del device vengono rifiutati prima dell'allocazione. La riserva resta una
policy iniziale; RSS, storage registrato e memoria del driver sono misure distinte.

| Gate | Risultato |
| --- | --- |
| Policy CPU, registry, runtime, executor | PASS: budget fisico condiviso, staging distinto, alias, fence, guasti e recovery |
| Quantizzazioni | PASS: dodici formati quantizzati, embedding singolo/gather, decode e righe 1/3/5, colonne 256/512; decoder scalare indipendente |
| GDN e attention | PASS: oracle GDN FP32 e Q/K/RoPE/attention FP64; soglie precedenti conservate, controlli espliciti dei valori non finiti |
| Grafo ibrido | PASS: quattro layer, tre GDN e uno full-attention, dimensioni head reali; embedding/FFN 256 e vocabolario 32, pesi Q8_0 sintetici |
| Chunk e continuazione | Undici posizioni teacher-forced sintetiche; chunk 1–5 e decode finale; logits completi max abs `2,86847353e-7`, stato/KV max abs `2,98023224e-6` |
| Riproducibilità e scratch | Replay decode dopo reset e destinazione SwiGLU separata bit-exact; input invalido senza mutazioni, allocazione parziale ripulita, ledger tornato alla baseline |
| Percorso generale | Stesso replay PASS con `DS4_METAL_DISABLE_METAL4=1`; non è una prova su un altro chip |
| Lifecycle Qwen27/Flash | PASS con stato sintetico, cap device 1 GiB, errori restore e park/release/resume; nessun forward reale |
| Suite numerica condivisa | 37 test, tre skip CUDA, nessun fallimento; nessun modello image/embedding caricato |

Il grafo ridotto ha un cap di **128 MiB** sullo storage device registrato.
Il confronto fra chunk usa soglie congelate `2e-5 + 5e-5*abs(ref)` per valore;
replay dello stesso percorso e prova dell'alias richiedono uguaglianza esatta.
Non è un oracle del modello 27B completo. Pressione normale e swap a zero
prima/dopo i casi finali; sono campioni ai confini delle prove.

## Riesame Strata S1–S4

Il confronto resta fissato a
[Strata 46c152a](https://github.com/Niko1221/Strata/tree/46c152a1310f77cd2f0ce2c7cd213962df14fe83/metal),
licenza MIT già conservata in `third_party/strata/`. File consultati, URL e
hash del testo ottenuto dal connettore sono in
[strata-review.json](strata-review.json). Non viene importato nuovo codice kernel.
La baseline del fork `837894c2` non definiva i binding Qwen35 in Metal;
il grafo Flash preesistente rimane un riferimento con semantica distinta.

| Famiglia | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| Quant/GEMV | Riesame `gemv.metal`: riuso pesi fra token, codebook e layout | Decoder scalare e tail 1/3/5; layout target inventariato | Confronto statico, nessuna nuova ottimizzazione numerica; il tempo del probe include apertura/compilazione | Dispatch e scratch esercitati nel grafo ridotto Q8; gli altri dtype hanno parity isolata |
| GDN | `gdn.metal` e `gdn_parity.cpp`: 512 thread e limiti pipeline; conservare stato head-major, quattro slot conv e SiLU del fork | Oracle FP32, finitezza e soglie precedenti | Nessuna modifica delle formule o promessa di speedup; capability di SIMD/threadgroup controllate nel binding | Prefill/decode, reset, KV e stato completo confrontati attraverso i chunk |
| Memoria | `arena_budget.h`: cache file distinta da anonimo inattivo, cap e working set separati | Aritmetica pura e registro sotto budget sintetico | Nessun tuning della riserva di 2 GiB; mantenuta la policy del piano | Allocazioni parziali, alias e lifecycle su buffer Metal veri; telemetria separata |

Non è stato eseguito Strata alle stesse shape/dtype. Il riesame statico e i
tempi dei test del fork non costituiscono un benchmark comparativo.

## Matrice delle funzioni fondamentali

La colonna reale conserva i limiti delle prove precedenti: una build o una
fixture non promuove una capability pubblica. CPU ha solo il gate build;
Qwen27/Flash non supportano inferenza CPU. CUDA/ROCm non sono verificati qui.

| ID/modello Metal | Copertura disponibile | Gate reale ancora aperto |
| --- | --- | --- |
| F01 Qwen27 testo | Build, quant, GDN, attention e replay ibrido PASS | NOT_VERIFIED: T03/T06-B, logits del 27B e contesto effettivo 8k |
| F01 Qwen27 MTP/vision | Simboli, mRoPE ridotto e lifecycle target-only | NOT_VERIFIED: T05 |
| F01 Flash | Build, primitive e streaming sintetico precedenti; lifecycle PASS | NOT_VERIFIED: T06-A e inferenza streamed |
| F02 Qwen Image generazione | Pipeline reale ridotta nelle evidenze precedenti; primitive condivise PASS | Oracle latenti/RGBA indipendente e qualità completa |
| F03 Qwen Image editing/RGBA | Editing e reference RGBA precedenti | Oracle indipendente, preservation e alpha |
| F04 EmbeddingGemma testo | Output reale e ranking precedenti; primitive PASS | Gate interno BF16 precedente FAIL, non allargato |
| F05 EmbeddingGemma media | Encoder reali e oracle precedenti | Gate interno BF16 precedente FAIL, qualifica completa aperta |
| F06 alternanza | Checkpoint e park sintetici per entrambi i LLM | Alternanza con forward reali T07 |
| F07 tool/skill/checkpoint | Test host esistenti e lifecycle sintetico | Sequenza reale padre/skill/ritorno T04 per famiglia |
| F08 editing interattivo | Fixture protocollo esistenti | Client reale, replay e ripresa LLM T04/T07 |
| F09 ricerca/ranking | Ranking Gemma ridotto nelle evidenze precedenti | Flusso ricerca ExMekh completo |
| F10 API/media/discovery | Contratti host e factory esistenti | Client/server reali con output verificati |
| F11 frontend/persistenza | Preparazione Quick Chat precedente | Chat/media reali e riapertura della conversazione |
| F12 guasti/risorse | Registry/executor, OOM e checkpoint corrotti/troncati/mancanti PASS | Cancellazione/retry e recovery dei flussi reali |

## Riproduzione

```sh
python3 tools/qualify_metal_foundations.py \
  --output build/metal-qualification/nuova-verifica \
  --cpu-build --model gguf/Qwen3.8-27B-UD-Q4_K_S.gguf
```

La directory deve essere nuova. `--model` esegue soltanto hash e ispezione;
omettendolo, i gate sintetici non richiedono GGUF. La build CPU avviene in
una copia separata; i processi GPU vengono eseguiti in sequenza con timeout.
La selezione breve è anche disponibile come `make test-metal-foundations`.
Non eseguire benchmark o grandi inferenze come parte di questo target.

T03 resta **IN_PROGRESS** per il gate del modello reale: la roadmap richiede
prima T06-A Flash, poi T06-B Qwen27 streamed 8192/64. I test di questa consegna
non sostituiscono quell'ordine, MTP/vision o le qualifiche applicative T04–T08.
