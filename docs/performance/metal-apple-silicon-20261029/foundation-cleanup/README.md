# Verifica del riesame delle fondamenta Metal

29 ottobre 2026, baseline `8e8ac45a7b463f7d4d72d859234f794b815019ef`.
Il [riesame architetturale](../../../architecture/metal-foundations-clean-code-review-20261029.md)
descrive responsabilità, contratti e correzioni. Questo registro distingue
le nuove prove dalle evidenze originali [T00–T03](../t00-t03/README.md).

[verification.json](verification.json) è prodotto dal qualificatore rifattorizzato:
contiene ambiente, hash sorgenti/binari, inventario, comandi, log, timeout e
campioni di pressione/swap. [review-verification.json](review-verification.json)
aggiunge le prove di integrazione e i sorgenti dei test frontend.

## Gate Metal e host

| Prova | Esito |
| --- | --- |
| Build Metal e suite | PASS: eseguibili ordinari, suite e probe collegati |
| CPU isolata | PASS: quattro eseguibili, nessun riuso di oggetti Metal o forward CPU |
| Fondamenta | PASS: runtime, registry, executor, policy condivisa, staging, pressione nota/sconosciuta e campione fallito |
| Owner numerico | PASS: factory accelerator e ownership dopo fence fallita |
| Quant, GDN e attention | PASS: decoder scalari e oracle FP32/FP64, soglie invariate |
| Storage FP16 degli oracle | PASS: decoder condiviso, tutti i 65536 pattern; subnormali e zero con segno corretti |
| Grafo ibrido | PASS: chunk 1–5, decode finale, replay bit-exact, alias SwiGLU e guasto dopo la creazione delle view |
| Percorso generale | PASS: stesso grafo senza Metal 4 opzionale |
| Lifecycle Qwen27/Flash | PASS: checkpoint e pesi tramite API pubbliche, restore atomico e cleanup completo |
| Suite numerica Metal | 37 test, tre skip CUDA; nessun fallimento |
| Contratti host | 22 test launcher/qualificatore PASS; compatibilità Qwen separata: 20 test, sei skip di gate live |
| Modello locale | Hash e metadata dei 866 tensori Qwen27 corrispondenti allo snapshot fissato; nessun forward reale |

Il replay mantiene gli stessi massimi osservati prima del refactoring:
logits `2,86847353e-7`, stato/KV `2,98023224e-6`. L'uguaglianza esatta ora
controlla esplicitamente i byte e la finitezza, senza cambiare le soglie del
confronto fra chunk. Cap del grafo ridotto 128 MiB, lifecycle 1 GiB, riserva
host 2 GiB; memoria del driver e storage registrato restano misure distinte.

## Frontend e portabilità dei test

Il primo run completo è conservato in [frontend-initial.log](frontend-initial.log):
1205 PASS, cinque FAIL e 46 SKIP, con coverage 100%. I fallimenti hanno esposto
dipendenze dei test dal checkout esterno ExMekh-Lib, conteggi non monotoni durante
la navigazione, accesso concorrente/asincrono alla clipboard e Control-click
interpretato come click secondario su macOS.

La fixture bootstrap usa ora una copia isolata dei sorgenti e un progetto
fittizio, la funzione di invio attende il contenuto pubblicato, la clipboard
viene verificata dopo il completamento e in esecuzione esclusiva, e il board
usa il modificatore nativo `ControlOrMeta`. Le asserzioni funzionali restano
attive; non vengono aumentati i timeout per nascondere i fallimenti.
La selezione di recupero passa tutti i **28 test**, inclusi i cinque casi
falliti. La successiva suite completa passa **1210 test**, con **46 skip** di
gate live e coverage di statement/branch **100%**; tempo 245,94 secondi.
Il log finale e le relative misure sono conservati nel registro del riesame.

Ruff passa su tutti i sorgenti e test frontend; 288 file rispettano il formato.
Le chiamate a provider reali restano disabilitate: i test browser usano i
provider simulati del repository.

## Riproduzione

```sh
python3 tools/qualify_metal_foundations.py \
  --output build/metal-qualification/nuovo-riesame --cpu-build \
  --model gguf/Qwen3.8-27B-UD-Q4_K_S.gguf
make test-suite-numeric-metal SUITE_PYTHON=build/metal-venv/bin/python
make test-suite-accelerator test-numeric-lifetime
cd agent
EXMEKH_AGENT_RUN_LIVE_TESTS=0 PYTEST_XDIST_AUTO_NUM_WORKERS=4 uv run --frozen pytest
```

Nessuna grande inferenza, download modello o conversione di pesi fa parte di
queste prove. T03 reale, streaming 8k, MTP/vision, alternanza applicativa e gli
altri acceleratori conservano i gate aperti nella roadmap. I due warning di
parametri inutilizzati nelle funzioni Flash incluse dai test erano preesistenti.
