# Q-P15: aggiornamenti SSD del DS4 originale

Audit richiesto dall’utente, 3 novembre 2026. Copia di ricerca separata e
ignorata, upstream fissato a `890898cfb437ed256be58ebcf7c58a6dfa8f2f53`
(2 novembre). Nessun merge, nessun commit nel fork. Provenienza in sources.json.
Le modifiche upstream includono contributi oltre a quelli di Sanfilippo.

| Intervento verificato | Applicabilità al fork |
| --- | --- |
| `22e417286c27050b07cf715798f64b651567d031`, ds4.c: static weights residenti entro budget | Già presente per GLM/DeepSeek. Qwen usa un owner diverso: Q-P06 applica lo stesso principio con storage esplicito bounded, senza mlock del GGUF intero. Image ha già keep per gruppi rivisitati e una riserva pari al gruppo più grande. |
| `ee7e8405232c8971000cc7ba2b945b00bf6d20d3`, ds4_cuda.cu: letture selected ordinate/coalescenti nel solo prefill | Idea trasferibile al reader Flash Metal, preservando remap e ready solo dopo drain. Qwen27 è denso e non ha selected experts. Image usa bundle mmap + stage per tensore, non questo reader: servirebbe un distinto adapter con budget host/device. Non copiare CUDA stream/event o ordinare arbitrariamente il decode. |
| `347c2e1fbee313824b997a3220d37c1ba8e8f57d` + `64baaa0678d70b8baf7143d38a461b4a034b4d89` + `c97964b8aa77034a8e17afb10ca7a686c40ab511`, ds4.c e test ngram | Prefetch PLE del prossimo chunk con storia privata, validazione prima di mutare stato, join prima dei callback e retry sincrono. Il fork ha un diverso prefetch CUDA; Metal resta sincrono. L’upstream abilita solo chunk >=256, mentre la qualifica corrente è64: non promette un guadagno qui. Qwen27 e Image non hanno PLE. |
| `5382eac5e07c8ea6ee122f134b0311200892da27`, ds4.c/test_deepseek41_prefill.c | Sweep di append medio2048..8192 conservando partizioni aritmetiche. Nel fork il caso medio è limitato a CUDA; il nuovo enable Metal non è importato. Riguarda V4.1, non è un kernel Qwen. Il principio di ammortizzare i pesi su più righe è pertinente a verifier MTP5 e ai blocchi DiT ripetuti. Cambiare chunk GDN/attention richiede gate distinti. |
| `890898cfb437ed256be58ebcf7c58a6dfa8f2f53`, ds4.c | Selezione read-only/untracked dopo validazione della famiglia V4.1. Nessuna prova di utilità universale per i buffer riscritti dello streaming. Non disabilitare hazard tracking per storage riutilizzato senza una pipeline esplicita. |

Confronto locale: ds4_qwen4_metal_stream.inc possiede il reader e pubblica
ready soltanto dopo lettura completa; Q-P13 rimanda il riciclo alla pressione
del cap, preservando fence prima di sovrascrivere. ds4_bundle.c:weight_group_load
stages tensor-by-tensor da mmap e pubblica resident dopo availability/quiesce;
ds4_weight_cache.c:limit/trim protegge gruppi popolari e largest-group headroom.
Queste API diverse impediscono un unico trapianto di cache per tutti i modelli.

Prossimi esperimenti motivati, in sequenza: consolidare prima Qwen27; sul
reader comune confrontare soglia32/64 e buffer nuovi/riusati (Q-P14 ha respinto8).
Per Flash, ordinamento fisico dei soli miss prefill e coalescenza dove anche
la destinazione è adiacente, senza cambiare router. Per Image, prima profilare
page fault/staging e acquire per blocco durante DiT: nessun claim di speedup
prima di un confronto del pipeline e degli oracle Diffusers. Nessuna modifica
Image in questa campagna; il suo gate resta separato, non ereditato dai LLM.

Qwen Image: riletti i tre sorgenti Diffusers al pin
`86ef420b17363c28e36ebad6b58555e9aa8feda2`, SHA256 verificati con il manifest
T06/T07: pipeline_qwenimage21.py (loop timestep e cond/uncond),
transformer_qwenimage21.py (ordine transformer_blocks),
autoencoder_kl_qwenimage21.py (_decode/tiled_decode). I blocchi DiT sono
rivisitati a ogni timestep: residenza parziale con headroom è pertinente;
la lettura anticipata può sovrapporre staging del blocco successivo senza
cambiare BF16, maschere causali, CFG o tile VAE. Il percorso Image non usa
`ds4_file_reader.h`, quindi le modifiche Qwen/Flash non ne accelerano già
l’I/O. Servono misure phase-specific di encoder/DiT/VAE e full pipeline:
non esiste un guadagno misurato Image da attribuire a questo audit.

Ulteriore capacità già esposta per Image: ds4_bundle.c:hint_worker/start/close
può anticipare page-cache advice bounded/cancellabile senza copiare interi
pesi in RAM anonima. Per una prova DiT conviene riusare questa API o estenderne
la selezione ai soli gruppi prossimi, invece di aggiungere un secondo reader
che conosca gli interni Numeric. La cache Image protegge già i gruppi più
rivisitati: non reimplementare LRU/pinning dal percorso MoE upstream.
