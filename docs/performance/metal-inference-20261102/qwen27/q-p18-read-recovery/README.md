# Q-P18: ownership stabile dopo un read retained fallito

S1 prima del codice, 3 novembre2026. Il riesame del ramo errore inP17 trova
una seconda differenza: un buffer retained non ancora pubblicato viene
liberato al primo EOF e ricaricato al retry. ds4_gpu_tensor_free può rinviare
la liberazione finché esistono command buffer pendenti, anche se quel
particolare peso non è stato codificato. Ridurre live e ripristinare la quota
prima del reclaim può quindi sottostimare lo storage effettivamente posseduto
tra due errori consecutivi. I guard globali restano attivi, ma l’owner della
cache deve conservare una contabilità coerente anche su questo ramo.

Riaperti per l’intervento: Strata46c152a1310f77cd2f0ce2c7cd213962df14fe83
metal/src/engine/stream.cpp:read_part/io_main (ready dopo tutte le parti),
mtl_rt.mm e testarena (lifetime/accounting); l’abort upstream non è recovery.
ggml dde879bc4f954b809892fd0d660515f42536394a Metal device/ops e test-backend-ops:
consumatori completati prima della distruzione. HFca8c871ede62b521a75a2ddfc5e1968ac9c8e097
Qwen3_5MLP/DecoderLayer (pesi identici e nessuna mutazione dello stato nel
reader); DS4 originale890898c ds4_cuda.cu selected upload: pending distinto
ready, drain prima di rilasciare una prenotazione fallita. Locale riletto
bundle:weight_group_load/release, tensor_free/retire e test retained EOF.

Scelta: il retained owner mantiene l’allocazione anche dopo EOF, con ready
privato false; un retry legge nello stesso buffer e lo espone solo al
completamento. La quota passa da pending a live una volta sola; nessun
nuovo buffer e nessuna attesa sul percorso di successo. Close drena e libera
anche retained non ready. Alternativa fence globale a ogni EOF corretta ma
non necessaria. Flash senza retained non cambia. Non è un nuovo speedup.

Test pubblico: due EOF consecutivi durante batch attivo, storage posseduto
sempre3row nel cap3row, ripristino backing e retry esatto; poi primitive,
grafi/errori/lifecycle e full8192/MTP4. Tempi finali dopo integrazione distinti
dalla matriceP17; nessun criterio numerico modificato.

Riproduzione pre-fix su oggettoP17 congelato: FAIL su owned==3row,
UD11/cols256, durante batch attivo. Il vecchio ramo sottraeva lo storage
appena liberato logicamente; il nuovo lo conserva non-ready e riusa sul retry.
Non è un nuovo mismatch numerico. Test/Gate successivi registrati separati.

S2 PASS: ripetuti EOF in batch attivo e retry, oracle scalare/GDN e
full128/8192/MTP4.11 vettori ordinari/spec bit-exact conP17. Nessuna
allocazione ripetuta nel retry retained; close e park conservano il drain.
