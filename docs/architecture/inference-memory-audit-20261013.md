# Audit RAM e VRAM durante l'inferenza — 13 ottobre 2026

Nel workload CUDA/Qwen27 verificato non emerge una crescita continua dopo il
riscaldamento iniziale. L'audit ha però trovato e corretto tre ritenzioni
evitabili. Le evidenze e gli hash sono nel
[riepilogo JSON](../performance/inference-memory-audit-20261013.json).

## Correzioni

1. Il buffer host di un checkpoint conservava sempre la capacità massima,
   anche quando un nuovo prefisso richiedeva molto meno spazio. Ora viene
   ridimensionato quando la nuova dimensione è inferiore a un quarto della
   precedente; variazioni moderate continuano a riutilizzarlo. La scrittura
   conserva il byte aggiuntivo necessario a `fmemopen` e il payload finale.
2. Dopo un fallback host, una successiva cattura riuscita in VRAM conservava
   anche la vecchia copia RAM. Il proprietario del checkpoint libera ora la
   copia host quando la copia GPU diventa quella autorevole.
3. Un fallimento nell'allocazione dei workspace OLD lasciava vive le allocazioni
   riuscite prima dell'errore. Il grafo ora libera il workspace incompleto prima
   del fallback ordinario. Un ciclo fallito o esplicitamente declinato lo libera
   dopo il rollback/replay e il catch-up MTP. I cicli riusciti continuano a
   riutilizzarlo. Il rilascio appartiene al grafo, senza accessi ai tensor dal server.

I test hanno riprodotto i difetti di ridimensionamento host e allocazione OLD
prima delle correzioni. Il fallback con tre fallimenti simulati e una verifica
declinata conserva il primo token, RNG e logits bit-exact. Il passaggio da
checkpoint host a GPU mantiene bit-exact il ripristino di sistema/skill.

## Memoria che rimane intenzionalmente disponibile

- Pesi, KV e buffer di prefill/decode appartengono a engine/sessione. Le righe KV
  non più visibili possono restare nello spazio prenotato dal contesto e vengono
  sovrascritte; azzerarle non libererebbe la capacità GPU.
- Il workspace OLD completo occupa **223.739.904 byte, cioè 213,375 MiB** per
  sessione. La quantità resta costante attraverso le finestre riuscite; la
  distruzione della sessione lo libera. Non vengono aggiunte nuove copie a ogni batch.
- I temporanei CUDA crescono fino al massimo necessario e vengono riutilizzati.
  La testa dei draft può portare il temporaneo GEMM a 640 MiB. È capacità di
  lavoro conservata fino al cleanup del backend, non una lista di output vecchi.
- Trie del vocabolario e maschere statiche sono cache del sampler. I buffer
  temporanei dei probe vengono liberati; snapshot ripetuti nella stessa call
  condividono l'allocazione originale.
- Le continuazioni interattive hanno budget e massimo 64 record. Il payload
  attivo scade dopo 300 secondi; le ricevute di replay durano 600 secondi.
  Il listener esegue la pulizia ogni secondo anche senza nuove richieste.
  Il termine della richiesta libera snapshot, parser e stream; le ricevute
  restano per l'idempotenza e poi vengono eliminate.
- La RAM anonima può restare disponibile nel runtime CUDA o nell'allocator
  dopo `free`. La prova misura separatamente RSS e RAM anonima e usa
  `malloc_trim` soltanto come diagnostica finale, senza aggiungerlo al decoding.

## Verifica e limiti delle misure

Qwen3.8-27B UD-Q4_K_S, sidecar MTP, RTX 3090, CUDA/Linux su WSL. La prova usa
contesto 4.096, prompt 2.053, prefill 2.048 e proposte da 64 token. Esegue
32 finestre, quattro creazioni/distruzioni di sessione, fallimenti simulati
di allocazione e declino del callback. Controlla la stabilità dopo warm-up,
il rilascio delle risorse e la correttezza del fallback. Il gate numerico
full-vocabulary resta distinto da questa misura di durata delle allocazioni.

`cudaMemGetInfo` misura memoria GPU globale, includendo altri client Windows/WSL.
Una prima soglia di 32 MiB ha intercettato un'oscillazione di circa 37 MiB,
con workspace e RAM anonima invariati; quella prova è conservata nel report.
La soglia GPU è ora 128 MiB, con controllo separato dei workspace posseduti.
La soglia della RAM anonima dopo warm-up è 32 MiB. Queste prove non escludono
qualsiasi leak minuscolo o che richieda sessioni più lunghe del workload verificato.

Sono stati eseguiti altri dodici casi HTTP con una traccia diagnostica di
`cudaMalloc`, `cudaMallocManaged` e `cudaFree`. Dopo ciascuno dei quattro giri,
il processo conserva esattamente **19.404.026.784 byte in 347 allocazioni**;
alla chiusura del server i contatori tornano a **zero**, senza free non abbinati.
Il temporaneo GEMM da 640 MiB spiega la crescita iniziale dei byte tracciati.
Le allocazioni private del driver non sono incluse. La VRAM globale oscilla
di 105 MiB mentre i byte tracciati restano identici: misurare l'intera scheda
non consente di attribuire queste oscillazioni ai buffer del server.
La RAM anonima passa da 668.232 a 668.488 KiB dopo warm-up, cioè +256 KiB,
compatibili con ricevute/metadata ancora conservati entro il TTL.
La traccia riguarda la memoria e non costituisce un benchmark di velocità.

Il server trovato attivo usava vision e contesto 26.000, con circa 24 GB di
VRAM globale occupata. Quel valore da solo non dimostra una perdita: include
pesi, contesto riservato, runtime e altri client GPU. L'utente lo ha fermato
prima delle prove dedicate; non sono stati avviati due modelli grandi insieme.

Metal, SSD streaming, distribuito, numerica vision e altre famiglie non sono
qualificati da queste prove. La lettura statica dei rami GLM ha inoltre trovato
workspace locali statici non agganciati al cleanup CUDA (`xq_scratch`,
`midq_scratch`, `mid_local`, `out_local`, `map_scratch`, `down_terms_scratch`,
`gu_scratch` in `ds4_cuda.cu`). Sono bounded e riutilizzati durante l'inferenza,
ma la loro durata oltre la chiusura del backend richiede una bonifica e test
dedicati; non fanno parte del percorso Qwen27 verificato qui.

I log, le prove rosse e i dati grezzi sono conservati localmente in
`performance-results/inference-memory-audit-20261013/`.
