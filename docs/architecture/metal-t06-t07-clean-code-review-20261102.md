# Riesame Clean Code T06–T07, 2 novembre 2026

Rilettura delle modifiche successive a `9805acb`, inclusi runtime C/Metal,
shader numerici, lifecycle, SDK distribuito, frontend, tool e qualificatori.
Le prove live rimanenti sono sospese per istruzione dell'utente: questo
riesame usa build, test host, UI con provider mock e fixture sintetiche.
La serie di commit distingue runtime nativo, strumenti, frontend/SDK ed evidenze.

## Responsabilità e correzioni

| Confine | Risultato del riesame |
| --- | --- |
| Runtime Metal e numerica | `DS4MetalNumeric` possiede device, pipeline, buffer e completamento; i callback C usano i suoi metodi. I modelli interrogano l'API numerica e il costo del workspace senza conoscere gli oggetti MPS |
| Streaming Qwen | L'owner conserva binding, ID compattati, pin e accounting. Eviction e cambio layer drenano i consumer. Il chiamante riceve un binding coerente e non ricostruisce il layout interno |
| Engine/sessione e checkpoint | Identità del modello, layout e ripristino restano nel lifecycle; il registro delle risorse possiede budget e lifetime, senza conoscere token o HTTP |
| Qualifica server/client | Nuovo `QualificationClient` immutabile: possiede comando, cwd, hash sorgente e risultati JUnit. Il runner conserva avvio, monitor memoria e cleanup. Rimossa la deduzione dei test saltati dal testo libero del terminale |
| Provenienza client | Gli hash dei test del frontend erano cercati dalla root del repository anziché dalla cwd del client. Ora il percorso effettivo viene risolto dal proprietario dell'invocazione |
| JUnit | Sono supportate entrambe le opzioni pubbliche pytest, con valore separato o `=`, percorso relativo o assoluto e separatore `--`. Un percorso personalizzato viene letto correttamente; report assente/vuoto non produce PASS. XML parziale, tempi invalidi e skip mantengono i rispettivi verdetti |
| Readiness | Prima di avviare il modello viene verificata la disponibilità della porta. Un listener preesistente non può essere usato come prova di avvio del server posseduto; il test verifica che non venga lanciato alcun processo |
| Oracle indipendente | `SafetensorsWeights` usa `safe_open.keys()` e `get_slice(...).get_shape()` per i metadata. Rimosso il parsing manuale dell'header binario; le mappe di storage sono private. Chiavi duplicate fra shard vengono rifiutate anziché sovrascritte implicitamente |
| Contratto della Mapping | Restano lazy loading, ordine dei pesi, dtype e buffer del modello. Test reali su un modello minuscolo verificano valori, prefissi, buffer, nessuna cache dei pesi e ritorno dei parametri su meta |
| Shader inutilizzato | Rimosso il vecchio entrypoint Metal `nm_softmax` e la pipeline compilata senza alcun dispatch. Il callback C resta e seleziona esclusivamente `nm_ordered_softmax`, come nei gate precedenti |
| Ambiente Make | I target bundle e contratti suite ora rispettano `SUITE_PYTHON`, già usato dagli altri target numerici. Non installano dipendenze nel Python di sistema |
| Pacchetto SDK | Verificati proprietà della compaction, confronto del request completo, metodi pubblici e assegnazione dell'handler da parte di `Client` dopo il request hook. La patch riproducibile coincide esattamente con il diff del checkout sorgente locale |
| Distribuzione e file locali | Wheel e patch SDK, test, configurazioni e prove compatte sono versionati. Checkout SDK d'audit, coverage database e screenshot temporanei restano output locali ignorati; non viene aggiunto un gitlink al repository SDK |

L'adattatore greedy usa `ProviderRequest.with_extra`, così non deve conoscere
l'espansione degli allegati o ricreare handler e opzioni. Il coding-agent
dichiara la propria skill file-processing nel resource HDS. Il dominio SDK
continua a possedere sostituzione della history, scope e timeout validato.
Queste correzioni della fase precedente sono incluse nella serie di commit.

La rimozione dello shader morto non cambia operazioni aritmetiche raggiungibili,
recipe o oracle. Nessuna ottimizzazione TPS è stata introdotta. Gli audit
upstream fissati e i precedenti gate full-model rimangono evidenze storiche
separate, senza attribuire loro nuove esecuzioni durante questo riesame.

## Verifica

Gli esiti, i comandi, gli hash dei sorgenti e i log sono conservati in
[`docs/performance/metal-clean-code-review-20261102/`](../performance/metal-clean-code-review-20261102/).
Il riepilogo verificabile è `verification.json` in quella directory.
La suite frontend completa conserva copertura statement/branch al 100%; i suoi
skip live non qualificano i consumer nativi rimasti aperti.

I primi tentativi dei target bundle hanno individuato l'interprete sbagliato
e il bind loopback negato dal sandbox. Sono conservati con il loro fallimento;
il target corretto passa usando l'ambiente predisposto e accesso al loopback.
Anche la verifica offline del lock usa la cache uv locale autorizzata.
Non sono stati avviati modelli completi per risolvere questi errori di ambiente.

## Limiti architetturali residui

Non si può certificare che l'intero repository soddisfi ogni principio Clean
Code: `ds4.c` e `ds4_server.c` conservano ampie responsabilità legacy e il runtime
Metal mantiene stato globale serializzato. La loro separazione richiede una
migrazione progressiva, con oracle per famiglia e backend; non è risolta da uno
spostamento meccanico di funzioni. Il riesame ha corretto i confini modificati e
ha mantenuto le responsabilità numeriche nei rispettivi owner.

F08 resta `NOT_VERIFIED` e sospeso; T07 resta aperto. Qualità estesa, stress,
altre famiglie Apple e nuova verifica CUDA/ROCm non sono promossi dai test host.
La macchina CUDA non è disponibile, come già stabilito; non sono state eseguite
prove distribuite. Una nuova differenza numerica o di prestazioni richiederà
una nuova indagine upstream secondo `AGENTS.md`, prima di modificare un kernel.
