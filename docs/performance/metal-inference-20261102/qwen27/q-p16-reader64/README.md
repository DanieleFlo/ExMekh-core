# Q-P16: soglia64 e costo di first touch

S1, prima del candidato, 3 novembre2026. La soglia8 di Q-P14 peggiora
8/16/24MiB con buffer riusati; Q-P03 includeva malloc/first touch e non è un
confronto equivalente. Profilo corrente:~54GB letti per16 output MTP4.

Riletti HF ca8c871ede62b521a75a2ddfc5e1968ac9c8e097 Qwen3_5MLP/DecoderLayer
(ordine/cast/residui invariati), Strata46c152a1310f77cd2f0ce2c7cd213962df14fe83
metal/src/engine/stream.cpp:read_part/io_main e stream.h (partizioni concorrenti
su slot persistenti, ready dopo ultima parte), ggml dde879bc4f954b809892fd0d660515f42536394a
src/llama-mmap.cpp:read_raw_unsafe e Metal wrapper/device (EINTR/shortread,
nessuna soglia universale). Strata riguarda esperti/slot allineati, non tutta
la copertura densa27B. Riletto upstream DS4 Q-P15: l’ordinamento CUDA prefill
non implica che più parallelismo vinca su singola lettura densa. Locale:
reader/run_batch, Q-P03 malloc+read+free, Q-P14 warm read+wait e inventario GGUF.

Scelta: micro isolato della sola soglia32→64, stessi4worker/chunk8, destinazioni
calde riusate e nuove mmap anonime per ogni lettura, separando alloc/release
fuori timer e misurando read+wait (first touch incluso). Byte identici prima
di unmap.32/48/64/128MiB, warmup poi10 coppie alternate per forma/mode.
Nessun buffer aggiuntivo di produzione né modifica dei limiti. Integrare solo
se il micro e successivo full-model lo giustificano; altrimenti conservare32.

S2 micro PASS,88 coppie con warmup: soglia64 peggiora32/48MiB di
5.7/20.9% su buffer riusati e5.0/20.4% su nuove mmap anonime.
64/128MiB conservano la stessa policy e oscillano entro~1.2%.
Byte identici e nessun errore. **REJECT prima dell’integrazione**:
first touch non cambia il segno della regressione. La soglia32 di
produzione resta invariata. Soglia8 e64 hanno entrambe fallito; nessuno
sweep arbitrario di worker. Dati grezzi/righe in continuation-20261103/p16-*.
