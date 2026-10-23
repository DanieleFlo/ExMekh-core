# Suite nativa nell'agente: launcher e accettazione live

La chat di generazione immagini restituiva `404 unknown endpoint`: il launcher
dell'agente avviava il binario LLM `ds4-server`. Ora avvia `ds4-server-suite`
con i bundle configurati, sulla stessa porta 8080. Contesto 26000, MTP e
proiettore visivo Qwen27 restano attivi. Il launcher della root e quello Flash
conservano i propri default; `DS4_SUITE_CONFIG` seleziona esplicitamente la suite.

Accettazione finale: **PASS**, 11 test live della libreria e 1 test browser,
eseguiti consecutivamente sulla stessa istanza. Il
[manifest di verifica](native-suite-agent-live-20261022.json) conserva hash,
risultati e corrispondenza fra wheel e libreria installata.

## Correzioni emerse dai test

Il caricamento di un payload multimodale creava il buffer RoPE visivo con
l'owner del chiamante. Il successivo park/restore non lo contava nella capienza
della sessione e poteva fallire con `restore_failed`. Il loader ora attribuisce
anche queste allocazioni lazy alla sessione. La nuova asserzione CUDA fallisce
prima della correzione e passa dopo; tre restore ripetuti entro il budget
esatto dichiarato conservano logits e token bit-exact, con MTP e visione.

I pesi Gemma conservati fra richieste potevano inoltre occupare spazio richiesto
dal successivo prefill LLM. Il servizio li rilascia prima di pubblicare il
risultato embedding. La conservazione fra richieste resta un'ottimizzazione
aperta, subordinata a un bound completo dei transitori LLM; le cache host dei
media e i planner standalone mantengono i propri contratti.

## Scenari verificati

| Percorso reale | Asserzioni |
| --- | --- |
| Tool immagini registrato | Testo, soli riferimenti ordinati, testo+riferimento, output multipli, PNG RGBA/alpha, seed, ricetta, idempotenza e download identico |
| Lettura visiva del risultato | Un LLM con history nuova riceve soltanto il PNG generato e riconosce `red square`, senza il prompt di generazione |
| Paesaggio richiesto | Generazione 1024×768 delle Alpi in inverno, originale e anteprima con dimensioni corrette |
| Client embedding tipizzato | Dimensioni 128/256/512/768, tutti i task pubblici, titolo document, vettori normalizzati e finiti |
| Quattro modalità | Testo, immagine, audio, video; batch riordinato, interleaving ordinato e fusione distinta dalla media dei vettori |
| Video con audio | Include/exclude della traccia controllati da contatori e vettori, fixture H264/AAC versionata con checksum |
| Consumer del ranking | Similarità semantica, cache esplicita e ranking visivo su embedding reali della stessa URL |
| Recupero e conversazione | Input invalidi/overflow/media mancante, servizio nuovamente usabile, tool ranking reale, ritorno cached e nuovo prefill con due immagini dopo gli embedding |
| Browser dell'agente | DeepAgent → tool → GPU → allegati vision, due miniature e dialog, responsività, reload e secondo turno senza rigenerazione |

Le fixture di ricerca sono locali. Modelli, encoder, trasporto HTTP, tool,
consumer del ranking e frontend non sono sostituiti da mock. Il test della
continuazione misura i cached token al ritorno del tool; una nuova richiesta
visiva può ricostruire correttamente il prefisso e non è obbligata a un cache hit.

## Riproduzione

Preparare i bundle e le dipendenze secondo la
[guida suite](../architecture/native-suite-release.md), impostando i percorsi
in `config/native-suite.env` dall'esempio versionato. Avviare un solo modello:

```sh
./agent/start-ds4-server.sh
# In un altro terminale, dopo il log listening:
DS4_SUITE_HTTP_URL=http://127.0.0.1:8080/v1 \
DS4_SUITE_E2E_ARTIFACTS=/absolute/path/to/new-evidence \
make test-suite-agent-live SUITE_PYTHON=agent/.venv/bin/python
```

Il target esegue 11 test della libreria e il test browser live; senza URL
fallisce esplicitamente. Non serve PyAV nei test: solo il generatore offline
della fixture lo usa. L'agente usa la propria wheel installata.

Log, richieste, contatori, PNG e schermate sono conservati sotto
`performance-results/native-suite-agent-repair-20261022/`. Le prove negative
sono separate dall'accettazione finale. Build, regressioni CPU/HTTP, otto
test launcher e regressioni CUDA lifecycle/executor accompagnano la prova live.

Ambito: Ubuntu 24.04 WSL, RTX 3090, suite BF16 nativa, Qwen3.8-27B Q4_K_S,
MTP e proiettore vision F16, contesto 26000, worker seriale. Questi test non
chiudono i gate di qualità completa T11 né gli studi di ottimizzazione rimasti
aperti nel [piano T19–T22](native-suite-t19-t22-20261022/README.md).
