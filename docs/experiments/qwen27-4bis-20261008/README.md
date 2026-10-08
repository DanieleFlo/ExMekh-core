# Qwen 27B — esperimento 4 BIS archiviato

Il lavoro è stato ritirato l'8 ottobre 2026 su richiesta dell'utente, per
concentrare lo sviluppo su Qwen Flash. I sorgenti in questa cartella non
partecipano alla build e non possono essere attivati con flag dell'applicazione.
L'inferenza ordinaria del 27B e il NextN già presente restano nel progetto.

## Idee provate

1. **KV offload:** storico FP32 completo in RAM, parte residente GPU,
   attention a tile con copie esplicite, softmax incrementale e ordine di
   riduzione originale. Sono stati provati crescita progressiva, staging del
   layer NextN e rollback ricorrente con arena GPU per la fase decode.
2. **Speculazione estesa:** NextN fino a 15 proposte e DSpark v2 per il 27B,
   con converter, primitive, stato, checkpoint e confronto col target.
   La scelta diagnostica finale NextN era cinque proposte. Cinque proposte
   non significano cinque token accettati.
3. **Pesi in streaming:** massimo 64K, KV e ricorrenza interamente GPU,
   pack residente e solo complemento pesi RAM, layer alternati, worker
   persistente, due slot GPU e staging pinned bounded. Copia e consumo erano
   protetti da eventi; una copia serviva tutte le righe della verifica MTP.
4. **Budget VRAM:** limite anche agli override, riserva per graph/snapshot,
   supporto, scratch e ring; verifica della memoria effettiva per evitare
   paging WSL. La prima richiesta da 11 GiB bypassava la riserva ed è stata
   corretta dopo la segnalazione dell'utente.

Si riprendevano da Strata residenza separata dai miss, ordine noto di consumo,
ring e dipendenze fra copie e lettori. Il 27B è denso: ogni forward richiede
tutti i layer e l'attenzione completa. Il working set sparso di Flash non
si trasferisce direttamente a questo modello. Il riferimento Strata era
il commit `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`, licenza MIT;
l'implementazione locale non copiava i suoi kernel.

## Risultati e motivi del ritiro

La macchina locale aveva RTX 3090, WSL e PCIe Gen4 x2, storicamente circa
2,74 GiB/s. Conservare la precisione KV FP32 aumentava la capacità, ma il
traffico remoto dominava il decode. Le prove preservavano il vocabolario
completo, token e stato; non bastava un argmax coincidente.

| Prova | Prefill, token/s | Decode totale, token/s | Qualificazione |
| --- | ---: | ---: | --- |
| KV offload v27, 65536+8 token, teacher-forced | 414,10 | 0,69 | Diagnostica numerica/capacità |
| KV offload v51, 64K, NextN5 | 376,63 | 2,245 | Singolo workload; acceptance diversa dalle prove seguenti |
| KV offload v53, 128K, NextN5 | 81,15 | 0,3916 | Corretto numericamente, prefill peggiorato; nessuna promozione |
| KV offload di controllo v57, 65503+32, NextN5 | 370,71 | 1,05 | Riferimento della coppia v61 |
| Pesi in streaming v61, 65503+32, NextN5 | 393,10 | 0,91 | Logits e token bit-identici al riferimento; diagnostica singola |

Non si confrontano direttamente suffissi, profondità o binari diversi.
Nella coppia v61 il prefill migliorava del 6,04%, mentre il decode peggiorava
del 13,33%. Restavano circa 1,8 GiB liberi secondo `nvidia-smi` durante il
prefill; il problema di saturazione iniziale era corretto. Il complemento
remoto era però 5,649 GiB per forward: 91 forward trasferivano 514,056 GiB.
La sovrapposizione con il calcolo non eliminava il costo del link x2.
L'utente ha quindi deciso di abbandonare il 4 BIS invece di proseguire
l'ottimizzazione. Nessuna di queste misure costituisce cinque A/B alternati.

Le primitive v62 passavano errori di budget senza distruggere il piano,
riuso degli slot, tutti-residenti, rollback NextN5 completo, cancellazione,
checkpoint e RNG. La ripetizione finale di tali gate è stata interrotta dalla
pausa dell'utente: i precedenti pass sono conservati, non sostituiti da un
risultato inventato per la ripetizione incompleta.

## Contenuto e recupero

- [source](source/): backend esclusivi, converter e strumenti/test del 27B.
- [shared-integration.patch](shared-integration.patch): integrazione rimossa
  da engine, CUDA, header, frontend, build e strumenti condivisi. È materiale
  di studio e recupero, non una patch da applicare senza rivedere il checkout.
- [documentation](documentation/): schede originali delle due architetture.
- [results](results/): 223 report, CSV e log compatti, con manifest dei binari.
- [manifest.json](manifest.json): commit di partenza, SHA dei sorgenti e report.

I dump completi di logits, payload e binari congelati rimangono nelle
directory originali `performance-results/flash-strata-20261006`, escluse da
Git. I cinque eseguibili di test esclusivi sono stati spostati sotto
`performance-results/flash-strata-20261006/retired-qwen27-binaries`.
I GGUF e le fixture non sono stati modificati o spostati.

I flag `--qwen-kv-capacity-mode`, `--qwen-kv-resident`,
`--qwen-weight-streaming`, `--qwen-weight-resident-mb` e
`--qwen-extended-draft` sono rimossi. Le vecchie variabili ambientali non
riattivano i backend archiviati. La correzione indipendente del layout
IQ4_NL (32 valori/18 byte) e i miglioramenti condivisi necessari ai
checkpoint Flash restano nel codice attivo.
