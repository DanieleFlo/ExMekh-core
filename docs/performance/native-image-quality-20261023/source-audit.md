# Indagine sulle fonti e sul dettaglio residuo — 23 ottobre 2026

La perdita di dettaglio segnalata dall'utente resta **non risolta**. La correzione
del timestep ripara una divergenza concreta dal checkpoint; la parità numerica
non certifica la qualità percettiva. Le prove successive distinguono la ricetta
ufficiale, gli artefatti del VAE e le ricette sperimentali della community.
Il confronto successivo con Unsloth, i casi panda/ape e la precisione del
denoiser sono documentati in [unsloth-audit.md](unsloth-audit.md).

## Fonti primarie verificate

| Fonte | Riscontro e conseguenza per DS4 |
| --- | --- |
| [model_index.json Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/model_index.json) | Gli otto `sample_sigmas` coincidono con il modello locale. Lo zero finale è aggiunto dallo scheduler. |
| [scheduler_config.json Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/scheduler/scheduler_config.json) | `shift=1`, `use_dynamic_shifting=false`, `shift_terminal=null`: non applicare lo shift del modello base o di un altro distillato. Il JSON locale coincide. |
| [vae/config.json Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/vae/config.json) | RGBA, 64 canali latenti, fattore spaziale 16; tutte le 128 costanti di media/deviazione standard coincidono con l'header C. Anche il JSON locale coincide. |
| [Pipeline Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py), [VAE](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py), [DiT](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py) | I tre file correnti coincidono, normalizzati i newline, con la revisione locale `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`. Il cast BF16 prima della divisione per 1000 resta necessario. |
| [Documentazione Diffusers Qwen-Image 2.1](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21) e [PR 14950](https://github.com/huggingface/diffusers/pull/14950) | La configurazione del checkpoint determina il numero di passi. Alzare soltanto `num_inference_steps` non cambia gli otto passi Turbo. La revisione locale include questa correzione upstream. |
| [Model card Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/README.md) | Ricetta raccomandata: otto passi, CFG 1 e preset da circa 4 MP. I preset includono 2400×1792, 2528×1696, 2752×1536 e orientamenti inversi: 2048 per lato è un limite DS4, non del checkpoint. |
| [Discussione VAE #12](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/12) e [documentazione Comfy](https://github.com/Comfy-Org/docs/blob/main/tutorials/image/qwen/qwen-image-2-1.mdx#vae-decoding-artifacts) | Il decoder ufficiale può introdurre griglie fini anche in un semplice encode/decode, senza denoising. Questo fenomeno non equivale a tutta la perdita di dettaglio percepita. |
| [Texture-Fix VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) e [chiarimenti dell'autore](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1/discussions/1) | Fine-tuning non ufficiale del decoder: riduce la griglia a 1/2/4 pixel; i blocchi inferiori sono congelati. L'autore dichiara un compromesso fra texture e accuratezza della ricostruzione. Non garantisce la rimozione dei pattern a 8 pixel. |
| [Turbo #4](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/discussions/4), [Comfy #22](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/discussions/22), [workflow Comfy](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_turbo_t2i.json) | I risultati torbidi con scheduler generici sono riprodotti da utenti. Il workflow corrente usa Euler, CFG 1 e gli stessi sigma ufficiali; contiene anche un preset 1024×1024. Due griglie alternative nei commenti sono ipotesi da provare, non nuove raccomandazioni Qwen. |
| [Turbo #5, implementazione Swift/Metal](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/discussions/5) | Un confronto con stessi pesi e scheduler diversi segnala un aspetto più inciso con la griglia Comfy, anche eccessivo. È una testimonianza utile per un A/B controllato, non prova di un bug DS4. |
| [stable-diffusion.cpp #2041](https://github.com/leejet/stable-diffusion.cpp/issues/2041) | Un'altra implementazione segnala pattern anche nei latenti e a 2K: motivo per verificare tutti i passi ad alta risoluzione, non per attribuire automaticamente tutto al VAE. |
| [Viggle Turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/README.md) | Distillato e scheduler diversi: trasferire la sua ricetta a Qwen Turbo non costituisce una correzione equivalente. |

Le testimonianze della community sono osservazioni dei rispettivi autori,
non diagnosi già confermate su DS4. I suggerimenti di 35–40 passi per il
modello base non sono direttamente applicabili alla griglia fissa Turbo.

## Prove controllate

Gli originali conservano prompt, seed e rumore C esplicito. La terza prova usa
lo stesso prompt/seed Alpi a 2048×1536, generando il rumore alla nuova forma.
Non è un upscale del PNG e non è un confronto dello stesso contenuto pixel.

Il vero `QwenImage21Pipeline.__call__` è eseguito per intero con i componenti
ufficiali locali BF16, text encoder eager, cache KV e attention MATH limitata
solo lungo le query, con tutto l'asse delle chiavi. Il modello usa CPU offload;
non vengono confrontati due soli script che ricostruiscono la stessa ricetta.

| Caso | Conditioning esatto | Latenti esatti | RMSE RGBA nativo/pipeline completa |
| --- | --- | --- | ---: |
| Alpi 1024×768 | sì | 8/8 | 0,001387 |
| Cyberpunk 1024×768 | sì | 8/8 | 0,001119 |
| Alpi 2048×1536 | sì | 8/8 | 0,001335 |

I 24 latenti esatti escludono ulteriore drift del denoising nativo in queste
prove. Non provano che i dettagli generati dal checkpoint siano soddisfacenti.
La pipeline completa postprocessa l'output BF16; i PNG non sono bit-exact.
Il confronto VAE separato resta nel gate numerico già documentato.

Due controlli sul decoder ufficiale usano gli stessi latenti finali: prima la
sola denormalizzazione FP32, poi l'intero decoder FP32. Rispetto al decode BF16,
l'indicatore di sbilanciamento delle quattro fasi pari/dispari cambia meno
dello 0,4% per entrambi i casi. Il passaggio a FP32 non rimuove la griglia e
l'ispezione non mostra un recupero evidente dei dettagli richiesti.
L'indicatore rileva un pattern periodico, non misura la qualità assoluta.
Questa prima prova FP32 converte i valori BF16 del VAE Turbo e non ricrea
bits assenti dai pesi. Il controllo successivo usa anche i pesi originali
FP32 del VAE base: tutti i 238 tensori coincidono dopo cast BF16 con il VAE
Turbo; il decode originale FP32 non elimina il pattern. I valori e la
provenienza sono in `unsloth-audit.json`. Nessuna inferenza del DiT base
sostituisce il checkpoint Turbo in queste prove.

Il Texture-Fix VAE è provato offline sugli stessi latenti, senza sostituire il
VAE di produzione. Il medesimo indicatore scende da 0,005398 a 0,001857 per
Alpi e da 0,003956 a 0,001842 per Cyberpunk (circa 66% e 53%). L'utente vede
ancora la perdita di dettaglio nei risultati: il miglioramento della griglia
non chiude la segnalazione. Il candidato è congelato alla revisione
`9a10df811f183323e4e94d255c18add1924ce340`; hash e provenienza sono nel JSON.

## Griglie alternative: esito diagnostico

Sono eseguiti sei denoising completi, due prompt per tre griglie: ufficiale,
`community-roscosmos` da Comfy #22 e `community-low-sigma` da Turbo #4.
Il controllo ufficiale riproduce esattamente entrambi i latenti nativi finali.
Le altre due mantengono gli stessi pesi BF16, conditioning, rumore, CFG 1,
Euler e otto valutazioni del modello. Gli elenchi della community includono
spesso lo zero finale nel conteggio dei passi: qui lo zero è terminale e non
costituisce una valutazione aggiuntiva.

La variante con ultimo sigma 0,25 rende più incisi bordi e texture rispetto
all'ultimo sigma ufficiale 0,414568, ma conserva pattern periodici. Non c'è
una reference fotografica che dimostri che l'incisione recuperi dettagli veri.
Decodificando anche questi latenti con Texture-Fix, lo sbilanciamento delle
fasi scende da 0,005054 a 0,001297 (Alpi) e da 0,003739 a 0,001341 (Cyberpunk).
Questa combinazione resta sperimentale e non chiude il problema percettivo.

Confronti apribili alla risoluzione originale:
[Alpi, sigma alternativo](../../../performance-results/image-quality-20261023/source-audit/schedule-diagnostic/alps-community-low-sigma/result.png),
[Cyberpunk, sigma alternativo](../../../performance-results/image-quality-20261023/source-audit/schedule-diagnostic/cyberpunk-community-low-sigma/result.png),
[Alpi, sigma alternativo + Texture-Fix](../../../performance-results/image-quality-20261023/source-audit/alps-low-sigma-texture-fix/result.png),
[Cyberpunk, sigma alternativo + Texture-Fix](../../../performance-results/image-quality-20261023/source-audit/cyberpunk-low-sigma-texture-fix/result.png).

## Artefatti e limiti

`source-audit.json` conserva provenienza, confronti completi e hash degli script.
I dati grandi e gli script diagnostici sono in
`performance-results/image-quality-20261023/source-audit/`:
`whole_pipeline.py`, `decoder_precision.py`, `schedule_diagnostic.py` e
`compare_followup.py`, `compare_vae.py` e `combined_diagnostic.py`. Nessuna prova modifica checkpoint, griglia o precisione
della ricetta di produzione; nessun filtro di sharpening viene aggiunto.

Le immagini si possono aprire senza conversioni:
[ufficiale 2K](../../../performance-results/image-quality-20261023/source-audit/whole-pipeline/alps-2048x1536/result.png),
[nativo 2K](../../../performance-results/image-quality-20261023/source-audit/native-2048x1536/case-0/result.png),
[decoder FP32](../../../performance-results/image-quality-20261023/source-audit/decoder-precision/alps-f32/result.png),
[Texture-Fix Alpi](../../../performance-results/image-quality-20261023/source-audit/alps-texture-fix/result.png),
[Texture-Fix Cyberpunk](../../../performance-results/image-quality-20261023/source-audit/cyberpunk-texture-fix/result.png).
