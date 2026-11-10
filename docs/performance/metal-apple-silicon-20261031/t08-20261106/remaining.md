# T08 — lavoro residuo, aggiornamento 9 novembre 2026

Stato della qualifica completa: `IN_PROGRESS`. La campagna iniziale era
senza commit; il successivo riesame autorizzato viene integrato per scopo,
con esiti in `review-verification.json`. La lista
separa limiti misurati, gate non ancora eseguiti e hardware non disponibile.
La ripresa successiva usa riserva host0su richiesta dell'utente; le prove
con2GiBrestano storiche e non vengono sommate ai50cicli nuovi.
Non attribuisce tutti i punti aperti alla RAM. I precedenti PASS locali
F01–F07/F09–F12 e L4 non costituiscono chiusura degli ambiti estesi.

| Priorità | Gate | Prova necessaria | Stato / limite attuale |
| --- | --- | --- | --- |
| 1 | F08 Flash | Workflow reale completo con receipt, effetti esatti, history, ritorno al caller e nessuna ripetizione indebita | `NOT_VERIFIED`: tre aperture della skill dopo la creazione; non è un rifiuto RAM. Restore248320logits bit-exact, scelta diversa riprodotta con diversa segmentazione prefill/decode. Prima di un intervento servono replay delle singole operazioni e nuovo audit specifico. |
| 1 | F08 Qwen27 | Stesso workflow su un server distinto, poi le ulteriori coperture XMLv2 previste dalla matrice | `NOT_VERIFIED`: nuova prova con cache automatica/riserva0crea il file, restituisce il risultato e riapre la skill tre volte. Interrotto con cleanup ordinato, crescita swap0. Serve isolamento specifico della famiglia prima di attribuire la causa; la diagnosi numerica Flash non prova la causa Qwen. |
| Chiuso locale | Stress Flash | Almeno50alternanze dopo warmup, immagine256, embedding multimodali, cache/restore e risorse bounded | `PASS`50/50+warmup con riserva0, server exit0 e nessun cleanup forzato. Crescita swap globale241.12MiB, monitor256MiB rispettato; nessuna dichiarazione di assenza assoluta di swap. La vecchia prova27cicli resta incompleta e separata. |
| Chiuso locale | Stress Qwen27 | Stessi50cicli, output deterministici e codice della conversazione conservato | `PASS`9novembre:50/50+warmup nello stesso processo,6956.335s,crescita swap0,pressione max2,server exit0/nessun cleanup forzato. Stato idle esatto e nessuna prenotazione viva; riserva0,contesto8192,prefill64,Image256 e criteri originali. [Audit del fix e checkpoint](../../metal-inference-20261102/qwen27/swap-transition-20261106/README.md). I precedenti SKIP e21cicli fermati restano separati. |
| Chiuso locale | Image512 indipendente | Encoder/DiT indipendenti, VAE con device dichiarato, tutti gli otto latenti e RGBA entro le soglie congelate | `NUMERIC_PASS`alpha512, encoder CPU/DiT MPS/VAE MPS offloaded. Conditioning bit-exact, max latent RMSE0.010359, SSIM RGB0.997166, crescita swap0. Non qualifica automaticamente il corpus esteso o l'aderenza visiva di tutte le recipe. |
| 2 | Image corpus esteso |24recipe ×2seed T00, forme originali1024/1536, reference/alpha/preservation e revisione qualitativa | Ammissione56/56forme PASS con riserva0, ma non è inferenza. Noise congelati assenti anche negli altri checkout; rigenerazione con lo script originale non coincide con nessuno dei sei hash. Recuperare gli asset prima della qualifica numerica T00. |
| 2 | Gemma corpus/ranking estesi | Intero corpus104item, extra audio/media, dimensioni128/256/512/768, input/frontiere, ranking e soglie congelate | I480confini BF16 esatti del subset locale restano distinti. Manca anche il `vectors.npz` della baseline T00 FP32-reduction nel checkout originale; recuperare l'asset prima del confronto completo con i golden. |
| 2 | Primitive numeriche | Risolvere e riverificare i due assert sin/cos riprodotti anche sul controllo precedente | Debito preesistente; nessuna soglia allargata o test eliminato in questa campagna. |
| 3 | Flash residente | Inferenza reale e suite funzionale distinta dallo streaming | Non qualificato su questo Mac; verificare ammissione e RAM su hardware adeguato. |
| 3 | Secondo Mac e matrice Apple | Almeno una seconda famiglia; distinguere M1/M2/M3/M4/M5 e capability effettive | Disponibile solo M5 Pro24GiB. Non estendere empiricamente i PASS ad altri chip. |
| 3 | CUDA/ROCm e distribuito | Gate dei backend condivisi sulle rispettive macchine | Hardware non disponibile in questa sessione; accesso CUDA e test distribuiti richiedono le autorizzazioni previste da AGENT.md. |
| 4 | Consegna finale | Riconciliare README, launcher, discovery, guide, PROJECT_INDEX e matrice con tutti gli esiti reali | Da chiudere dopo i gate obbligatori. Conservare `ready:false` e nessuna qualifica completa finché restano aperti. |

Per ogni rifiuto RAM registrare comando, forma, stato restituito, picco e
minimo host misurato. Non ridimensionare il corpus né ridurre passi, contesto,
precisione o criteri per trasformare il rifiuto in PASS. I timeout e i
problemi di workflow rimangono cause distinte dai limiti di memoria.

Le prove lunghe della suite ora salvano ogni ciclo convalidato su disco.
Il controller `qualify_metal_server.py --resume` conserva i tentativi e
riparte dal ciclo incompleto, verificando binario, modelli, configurazione,
override e artefatti. Provato con arresto reale dopo1ciclo e ripresa dei2
rimanenti, senza rigenerare il PNG completato. I cicli funzionali cumulati
fra processi sono distinti dal lifetime continuo; il PASS Qwen50 sopra è
stato ottenuto senza interruzioni. F08 e gli altri ambiti estesi restano aperti.
