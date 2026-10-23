# Correzione qualità immagini e geometria del tool — 23 ottobre 2026

Il bug è riprodotto sulle prime due immagini della chat «Generazione di immagini»:
Alpi e città cyberpunk, 1024×768. Prompt, seed e rumore iniziale sono identici;
i PNG della baseline coincidono pixel per pixel in RGB con gli originali.
La conservazione dei PNG e le anteprime a scala 1 non introducono ridimensionamenti.

## Stato della qualità visiva

L'utente vede ancora una perdita di dettaglio ad alta frequenza nelle immagini
con timestep corretto e nel confronto con VAE Texture-Fix. La parità numerica
riportata sotto qualifica l'implementazione rispetto al riferimento; **non
chiude il problema di qualità percettiva**. L'indagine aggiuntiva e le fonti
Hugging Face sono in [source-audit.md](source-audit.md). Il confronto successivo
con Unsloth e i nuovi casi panda/ape sono in [unsloth-audit.md](unsloth-audit.md).
L'utente giudica corretti i due nuovi riferimenti Diffusers (panda 2048×2048
e ape 1024×1024); questo riscontro non risolve da solo le prime due immagini.

## Causa e correzione

La pipeline ufficiale congelata `QwenImage21Pipeline` in Diffusers `aea7b70b`
calcola `t.expand(batch).to(latents.dtype) / 1000`. Il forward nativo arrotondava
soltanto sigma, dopo la normalizzazione. Gli stessi oracle locali ripetevano
questo errore e potevano quindi certificare una traiettoria diversa da quella
della pipeline. Il sigma 0,978453 produceva un timestep normalizzato 0,9765625,
mentre la pipeline richiede 0,98046875. Cambiano tre degli otto passi.

`qi_image_timestep` conserva entrambe le frontiere BF16, senza cambiare i sigma
dello scheduler o l'aggiornamento Euler. Entrambi gli oracle passano ora il
timestep con lo stesso ordine del vero `__call__`; il test C indipendente
congela tutti gli otto valori attesi. Le vecchie qualifiche restano evidenze
storiche: la loro parità con l'oracle precedente non prova questa conformità.

| Caso | Errore relativo latente finale prima | Dopo | RMSE RGBA prima | Dopo | SSIM RGB dopo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alpi | 0,026032 | 0 (bit-exact) | 0,011567 | 0,000926 | 0,999825 |
| Cyberpunk | 0,094595 | 0 (bit-exact) | 0,052823 | 0,000977 | 0,999693 |

Tutti i 16 latenti sono bit-exact con l'oracle corretto. Le soglie T00 non
sono state modificate. Nella banda FFT radiale da 0,25 cicli/pixel, l'errore
relativo scende da 20,05% a 2,30% sulle Alpi e da 65,35% a 2,98% sulla città.
L'energia ad alta frequenza corretta è rispettivamente 99,989% e 100,006%
di quella dell'oracle. Sono confronti con il riferimento, non una metrica
assoluta di bellezza o nitidezza. La causa completa della perdita percepita
nelle prime immagini non è ancora dimostrata. Otto passi sono la ricetta Turbo
raccomandata dal checkpoint, non una compressione del PNG.

## Dimensioni

`ImageDimension` enumera tutti i multipli di 32 tra 32 e 2048. Larghezza e
altezza sono indipendenti, quindi restano due parametri Literal. Client,
helper e tool registrato condividono il tipo; `n` enumera 1..8. Il limite
aggiuntivo del batch è `n * width * height <= 4194304`. La validazione locale
precede anche gli upload delle reference.

Client, helper e tool ExMekh usano ora 2048×2048 quando le dimensioni vengono
omesse. La descrizione indica anche 2048×1536, 1536×2048 e 2048×1152;
le richieste esplicite di formati minori restano rispettate. Il limite di batch
richiede chiamate separate per più immagini 2048×2048. Il nuovo default riduce
l'uso involontario di bassa risoluzione, ma non dimostra che questa sia l'unica
causa del difetto segnalato. Il default del protocollo HTTP non cambia.

Il limite 2048 per lato e 4194304 pixel è operativo del servizio DS4, non un
massimo del modello Qwen. La model card ufficiale pubblica anche 2400×1792,
2528×1696 e 2752×1536 (e le corrispondenti orientazioni verticali), oggi fuori
dal contratto del servizio. 2560×1440 supera il limite DS4 per lato;
1920×1080 ha altezza non multipla di 32.
2048×1152 è il formato 2K 16:9 ammesso. I test verificano tutte le 4096 coppie
sia sul trasporto Python sia sul parser HTTP C e controllano lo schema del
tool realmente caricato. Tutte e cinque le generazioni GPU reali passano:
32×32, 2048×32, 32×2048, 2048×1152 e 2048×2048. I PNG hanno le dimensioni
esatte richieste e i buffer RGBA sono finiti e nel range 0..1. Anche il piano
RTX 3090 ammette tutti i casi. L'indagine successiva in `source-audit.md` estende il confronto con la pipeline
completa a 2048×1536: conditioning e tutti gli otto latenti sono bit-exact
anche in questo caso, con RMSE RGBA 0,001335.

## Verifica e riproducibilità

Dopo il default 2K la suite libreria passa 3642 test (76 skip); Ruff passa. La suite agente parallela passa
1123 test (45 skip) e registra 10 fallimenti transitori di steering/UI:
tutti e 10 passano nella riesecuzione seriale. Dopo la sincronizzazione della
wheel passano anche i tre test dedicati ad allegati, anteprime e persistenza,
compreso Playwright, rieseguiti anche dopo il default 2K (3 pass, senza
il gate di copertura globale nella selezione mirata). Contratti e guardia
testi rieseguiti: 11 pass e 20 subtest. La suite agente completa sopra è
storica, non una nuova esecuzione integrale dopo il cambio del default. Build server, timestep, input/output/PNG/alpha, parser HTTP,
contratti, guardia testi e Ruff passano.

`verification.json` conserva metriche complete, hash sorgenti/binari/wheel,
revisioni e limiti. `exmekh-lib.patch` conserva la modifica della dipendenza
sorgente esterna; la wheel aggiornata è già installata nell'agente. I grandi
artefatti riproducibili restano in `performance-results/image-quality-20261023/`
(cases/fixture/rumore, originali, baseline, native, oracle e log).
Il [riesame clean code](clean-code-review.md) aggiunge controlli coerenti di
richieste e risposte, responsabilità esplicite e prove complete della libreria
e del frontend. I risultati di questa revisione sono separati in `verification.json`.
Questo è un gate sui due casi segnalati, non l'intero corpus T00 né una nuova
qualifica di Metal, ROCm, distribuito o SSD LLM. Non cambiano kernel o percorsi LLM.


Immagini corrette, con gli stessi input degli originali:
[Alpi](../../../performance-results/image-quality-20261023/native/case-0/result.png)
e [cyberpunk](../../../performance-results/image-quality-20261023/native/case-1/result.png).
