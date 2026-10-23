# Unsloth, precisione e nuovi casi — 23 ottobre 2026

Il confronto non individua un passaggio di denoising mancante nella ricetta
Turbo DS4. Il problema percettivo delle prime immagini 1024×768 resta aperto;
non è attribuito a un limite del checkpoint. L'utente giudica corretti il panda
2048×2048 e l'ape 1024×1024 generati con il riferimento Diffusers. Il secondo
caso impedisce di considerare la sola risoluzione una spiegazione completa.

## Confronto delle pipeline

Audit della [model card GGUF Unsloth](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF)
a revisione `5d1f9f759ea877ae556cda88dbec958b62ad3809` e dei sorgenti
[Unsloth](https://github.com/unslothai/unsloth/tree/9f59a73ff9084f848fe600d87c4c21e17e666f97/studio/backend/core/inference)
a `9f59a73f`. Sono conservati i manifest di 343 file d'inferenza e dei file
selezionati, senza trasferire questo codice nel runtime di produzione.

| Passaggio | Riscontro |
| --- | --- |
| Scheduler | Stessi otto sigma ufficiali, Euler, CFG 1, zero terminale; Unsloth salta lo shift Comfy generico quando trova `sample_sigmas`. |
| Diffusers | Pipeline, transformer e VAE del pin Unsloth `d42ebf9c` coincidono con i tre file locali `aea7b70b`. |
| Precisione | Su NVIDIA Ampere o successiva Unsloth seleziona BF16; RTX 3090 rientra in questo caso. Il nome F16 del GGUF descrive lo storage e non certifica un'inferenza uniforme FP16 o FP32. |
| VAE | Tutti i 238 tensori del VAE base originale FP32 coincidono con il VAE Turbo dopo cast BF16. Il decoder con pesi originali FP32 è provato separatamente: il pattern pari/dispari non scompare. |
| Text encoder | Stessa architettura Qwen3-VL e conditioning prima della RMSNorm finale; Unsloth può quantizzare il text encoder. DS4 usa BF16. |
| Rumore | DS4 usa SplitMix64/Box-Muller, Diffusers un `torch.Generator` sul device selezionato. Lo stesso seed non garantisce lo stesso rumore fra runtime. I confronti numerici passano esplicitamente lo stesso rumore. |
| Geometria | Il runtime Unsloth ammette lati fino a 2752 e 4.300.800 pixel; DS4 conserva 2048 per lato e 4.194.304 pixel totali nel batch. |

La prova diretta delle ottimizzazioni importa i moduli originali Unsloth in uno
spazio diagnostico isolato: forward veloce installato e 120 moduli VAE fusi.
Su ape 1024×1024, stessi pesi BF16, prompt, seed e rumore, **tutti gli otto
latenti sono bit-exact** rispetto a Diffusers senza queste ottimizzazioni.
Il VAE fuso produce RMSE RGBA 0,000984 rispetto allo stock. Questo confronto
isola i processi, non qualifica la quantizzazione GGUF o l'intera app Studio.

## Nuove inferenze native

Sono eseguite realmente due nuove inferenze C/CUDA, con il rumore iniziale
catturato nei riferimenti. Il tempo monotono della chiamata completa, inclusa
l'apertura e verifica del bundle, è 289,6 s per panda 2048×2048 e 123,2 s per
ape 1024×1024. Il successivo comando che visualizza il PNG non rigenera l'immagine.

| Caso | RMSE RGBA nativo / SDPA | Errore relativo latente finale |
| --- | ---: | ---: |
| Panda 2048×2048 | 0,017550 | 0,070233 |
| Ape 1024×1024 | 0,011231 | 0,023689 |

Questi riferimenti usano SDPA predefinita; DS4 usa eager/MATH. I nuovi output
non sono bit-exact e non sostituiscono i 24 latenti esatti del confronto con
MATH riportato in `source-audit.md`. Il riscontro positivo dell'utente riguarda
i riferimenti mostrati prima delle due nuove inferenze native.

[Panda nativo](../../../performance-results/image-quality-20261023/source-audit/native-showcase/case-0/result.png),
[ape nativa](../../../performance-results/image-quality-20261023/source-audit/native-honeybee/case-0/result.png).

## Default del prodotto

La wheel installata nell'agente espone ora 2048×2048 per client, helper e tool.
Le dimensioni minori esplicite restano rispettate; per più immagini 2048×2048
si usano chiamate separate entro il limite del batch. È una scelta di default
per il dettaglio, non una diagnosi conclusiva del difetto alle altre risoluzioni.

Le metriche, revisioni, hash e limiti del confronto sono in `unsloth-audit.json`.

## Confronto del denoiser BF16 / FP16 / FP32

Sono completate sei inferenze, entrambe le immagini originali in ciascuna
precisione. Restano identici i pesi sorgente Turbo con valori BF16, il
conditioning BF16 salvato, il rumore iniziale arrotondato BF16, gli otto sigma,
i timestep normalizzati con le stesse frontiere BF16 e il decoder BF16.
Cambia la precisione del calcolo del denoiser; nessun risultato usa il DiT base.
Le operazioni mantengono FP32 dove previsto dall'implementazione Diffusers.

| Caso | Calcolo DiT | Errore relativo latente / BF16 | RMSE RGBA / BF16 | Denoising, s |
| --- | --- | ---: | ---: | ---: |
| alps | bf16 | 0.000000 | 0.000000 | 14.2 |
| cyberpunk | bf16 | 0.000000 | 0.000000 | 13.7 |
| alps | fp16 | 0.012404 | 0.008009 | 13.7 |
| cyberpunk | fp16 | 0.078106 | 0.047679 | 14.0 |
| alps | fp32 | 0.011629 | 0.007550 | 182.5 |
| cyberpunk | fp32 | 0.079304 | 0.048339 | 182.5 |

Tutti i 48 passi producono valori finiti; i 16 passi BF16 di controllo sono
bit-exact con DS4. FP32 supera la VRAM disponibile e usa offload di un blocco
alla volta: questi tempi includono i trasferimenti, non rappresentano la
prestazione di una GPU che contiene il modello intero. Il decode è successivo
ed è escluso dai tempi della tabella.

L'ispezione dei sei PNG non evidenzia un recupero generale netto del dettaglio;
la trama periodica resta presente. L'utente conferma sul confronto appena mostrato:
«no non cambia niente». Le differenze numeriche non misurano da sole
un aumento o una riduzione della qualità. Il risultato indebolisce l'ipotesi
che la sola aritmetica BF16 del denoiser spieghi la segnalazione; non certifica
una pipeline completa FP32. Il text encoder rimane BF16 e non vengono ricreati
bits assenti dai pesi sorgente. Anche i timestep sono mantenuti uguali per
isolare la variabile aritmetica: non è il confronto fra tre ricette complete
con timestep arrotondati ciascuno nel proprio formato.

Il controllo iniziale della nuova prova diagnostica ha intercettato un cast
indebito del buffer FP32 delle frequenze sinusoidali del timestep. È stato
corretto caricando i pesi separatamente per dtype. Il controllo BF16 finale
richiede e ottiene 16/16 latenti esatti; il problema del primo harness non era
presente nel percorso nativo di produzione. Il log fallito è conservato con
provenienza distinta e non costituisce evidenza di un nuovo bug DS4.

[Alpi BF16](../../../performance-results/image-quality-20261023/source-audit/dit-precision/alps-bf16/result.png),
[Alpi FP16](../../../performance-results/image-quality-20261023/source-audit/dit-precision/alps-fp16/result.png),
[Alpi FP32](../../../performance-results/image-quality-20261023/source-audit/dit-precision/alps-fp32/result.png),
[città BF16](../../../performance-results/image-quality-20261023/source-audit/dit-precision/cyberpunk-bf16/result.png),
[città FP16](../../../performance-results/image-quality-20261023/source-audit/dit-precision/cyberpunk-fp16/result.png),
[città FP32](../../../performance-results/image-quality-20261023/source-audit/dit-precision/cyberpunk-fp32/result.png).
