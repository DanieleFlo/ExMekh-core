# Draft dello snapshot old_text da 32/64 token — 12 ottobre 2026

Il default CUDA Qwen27 verifica continuazioni dello snapshot quando la selezione
letterale OLD è globalmente univoca. Il target può scegliere la chiusura o una
tokenizzazione/serializzazione diversa: il server mantiene la maschera del
verificatore. Nessun testo proposto viene pubblicato o applicato prima dell'accettazione.
Il wire XML v2 resta invariato, senza code e senza marcatori @.

## Responsabilità e percorso di esecuzione

| Componente | Responsabilità |
| --- | --- |
| Documento edit | Matching nello snapshot e proposta XML da un prefisso univoco; UTF-8, entità, CDATA ed EOF |
| Comportamento/protocollo server | Escape del delimitatore nativo, autorizzazione dei percorsi, budget e checkpoint solo dei token accettati |
| Scheduler | Tokenizzazione delle proposte, scelta 32/64, verifica del target e fallback per codifiche diverse |
| Motore | Logits per riga, KV, stato ricorrente, rollback/replay, RNG e catch-up MTP |
| Client | Evidenza inspect/read, filesystem e workroot, encoding/hash, range, alias, sovrapposizioni e applicazione atomica |

La policy parte da 32 e passa a 64 dopo un batch pieno. Un mismatch ritorna a 32;
un'accettazione bassa rinvia il tentativo di 32 token, e zero draft accettati
disabilitano le proposte per quella selezione. Una nuova fase/elemento ripristina
la policy. Le code sotto 32 token e i limiti di contesto/budget usano il decode
ordinario/MTP. Il motore supporta anche batch troncati: i test coprono 1/2/3 righe.

Il motore usa un checkpoint compatto dei soli stati ricorrenti, non 64 copie.
Se il target chiude prima, ripristina il checkpoint e ricalcola il prefisso accettato;
le righe future della KV attention vengono sovrascritte per posizione. Il primo
token diverso viene restituito senza valutarlo; non viene campionato un bonus
attraverso la frontiera OLD/NEW. L'MTP mantiene il proprio percorso e riceve
gli hidden target accettati tramite catch-up.

Le proiezioni del nuovo verificatore usano GEMM già presente, FP32 pedantic,
senza F16/TF32 o nuova quantizzazione del target. La testa del vocabolario viene
divisa in blocchi di 32.768 righe per limitare il temporaneo a circa 640 MiB.
Il dispatch ordinario, il prefill e l'MTP mantengono la selezione precedente.
Le fonti sono la [guida MTP](../research/guides/qwen36-mtp-llamacpp-ds4-control-flow.md),
la [scheda llama.cpp](../research/platforms/llama-cpp.md) e la
[analisi RTX 3090](../research/forks/syv-ai-qwen38-27b-rtx3090.md).
Si riusano GEMV/GEMM e il controllo causale; non si importano kernel o compromessi di quantizzazione.

## Misure di decoding

RTX 3090, driver 610.88, Qwen3.8-27B UD-Q4_K_S. Un warm-up e cinque campioni,
senza profiler. I riepiloghi completi min/p10/mediana/media/p90/max/std/CV sono
nel [JSON](old-text-snapshot-chunks-20261012.json). Tutte le misure definitive
hanno CV entro il 5%. Il picco GPU osservato durante campioni live è
23746 MiB su 24.576; la misura riguarda l'intera scheda.

Benchmark teacher-forced del verificatore; include checkpoint, logits/readback
e catch-up, esclude il restore del checkpoint di partenza. Non è generazione libera:

| Contesto / supporto | Seriale token/s | Batch 32 token/s | Batch 64 token/s |
| --- | ---: | ---: | ---: |
| 2.053 token, MTP | 31.15 | 71.44 | 109.49 |
| 7.764 token, MTP | 28.53 | 45.06 | 56.73 |
| 2.053 token, senza MTP | 31.24 | 70.70 | 109.43 |

Client reale, selezione lunga, con identici XML di edit e identici byte finali:

| Metrica (mediana di 5) | Baseline | Default finale |
| --- | ---: | ---: |
| Decoding del tool, token/s | 36.91 | 46.04 |
| Decoding del tool, secondi | 7.749 | 6.211 |
| Flusso client completo, secondi | 77.283 | 74.256 |
| CV decoding | 0.58% | 2.08% |

Il guadagno di decoding reale è **1.248× (+24.8%)**.
Ogni campione genera 286 token nuovi; i 42 token della pausa sono esclusi dalla
velocità, pur restando nei 328 token dell'usage cumulativo. Il flusso completo
include altre chiamate, prefill, checkpoint e ritorno all'agente principale;
il guadagno del verificatore non va presentato come accelerazione di tutto il client.
Per dati con serializzazione/tokenizzazione diversa, il fallback può restare
necessario: il caso OLD HTML/Unicode in CDATA passa il controllo dei byte e torna
all'ordinario dopo la prima proposta senza accettazioni.

## Correttezza e limiti

- Native server e protocollo XML: PASS, incluso oracle completo contro cache incrementale.
- ASan/UBSan: PASS sui nuovi prefissi, frontiere UTF-8 e proprietà dei buffer.
- Client: 471 test unitari pertinenti PASS; selezioni corte, lunghe e CDATA con byte esatti.
- Motore live con/senza MTP, contesti 2.053 e 7.764: PASS. Dodici casi includono
  mismatch dopo 1/2/31/32/63 token, callback negativo/RNG, EOF, budget e rewind/sync.
  I test confrontano l'intero vocabolario anche alla frontiera finale, top-1/top-20,
  cosine e valori finiti; KV attention, stato ricorrente e MTP hanno L2 relativo <0,001.
  Un callback declinato ripristina lo stato autorevole e RNG esattamente per il solo seed.
  Le differenze numeriche del GEMM rispetto al matvec con residuo R8 sono registrate,
  non descritte come equivalenza bit-exact del batch completo.
- HTTP live batch 32/64: PASS per snapshot EOF, chiusura anticipata e markup/Unicode
  con tag di chiusura letterali. Constrained prefill nell'usage rimane zero.
- Regresso MTP V(2) di cancellazione: PASS con payload/KV/RNG bit-exact.
- Metal, SSD e distribuito: NOT_VERIFIED su questa macchina; la capacità dei draft
  esterni è zero fuori dal percorso CUDA Qwen35 non distribuito.

Il trace del primo test CDATA ha mostrato lo snapshot copiato correttamente,
poi `<![CDATA[]]>` ripetuto a EOF fino a incomplete_generation (512 token).
Il server ora impedisce nuove aperture CDATA quando non resta un carattere
selezionabile: un blocco già aperto può chiudersi, e i segmenti vuoti rimangono
ammessi prima di EOF e in NEW. Il risultato definitivo termina in 233 token.
Un controllo distinto limita i prefissi di escape nativi OLD alle codifiche
che possono davvero ricostruire lo snapshot, evitando profondità amp; incompatibili.

**Resta un FAIL di intento del modello nella prova HTML di NEW:** richiede 14
paragrafi (294 byte), ma produce un paragrafo e altra spaziatura (49 byte).
La baseline e il default definitivo producono lo stesso SHA256. L'XML è valido;
il server non possiede il contenuto atteso di NEW. Il controllo del client di test
lo rileva e questo campione è escluso dalle misure di velocità. Non è una regressione
del nuovo percorso OLD e non viene trasformato in un PASS cambiando il prompt.

L'esperimento iniziale senza GEMM è conservato come REJECT (~20 contro ~30 token/s);
la misura senza MTP inizialmente instabile (CV 7,43%) è conservata e ripetuta.
Log, wire, trace, file risultanti e script sono in
`performance-results/old-text-chunks-20261012/`; le cache KV temporanee di prova
vengono eliminate dopo l'arresto dei relativi server.

## Riproduzione

```bash
make test-server-interactive ds4_server_test tests/test_qwen_snapshot_drafts
./ds4_server_test
./tests/test_qwen_snapshot_drafts gguf/Qwen3.8-27B-UD-Q4_K_S.gguf gguf/mtp-Qwen3.8-27B-Q4_0.gguf
DS4_DRAFT_TEST_CONTEXT=7764 ./tests/test_qwen_snapshot_drafts gguf/Qwen3.8-27B-UD-Q4_K_S.gguf gguf/mtp-Qwen3.8-27B-Q4_0.gguf
# Un solo server, con DS4_SERVER_PHASE_PROFILE=1 e log diretto a un file:
exmekh_lib/.venv/bin/python tests/bench_interactive_edit_live.py --base-url http://127.0.0.1:8080/v1 --case old_text_long --repeats 5 --server-log /tmp/server.log --work-dir /tmp/new-unique-workdir
```

`DS4_OLD_TEXT_DRAFT_BATCH=32|64` e `DS4_OLD_TEXT_DRAFT_DISABLE=1` sono diagnostici
per confrontare il medesimo percorso. Il default non richiede variabili.
Binario definitivo SHA256: `0be42197c1a2ec367614af6fe6eeaf6ebc96f7ef54a35c2a272526acbe6964d0`. Nessun commit è stato creato.
