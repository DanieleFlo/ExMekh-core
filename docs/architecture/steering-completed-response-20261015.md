# Steering dopo una risposta completata — 15 ottobre 2026

La chat `Cancellami tutto quello che ve...` con ID
`26a4665e-e816-4396-8c84-7250ff75fa09` mostra un caso diverso dalla
[cancellazione di una pausa edit](steering-cache-fix-20261014.md): il server
completa una normale chiamata, ma lo steering arriva prima del dispatch del
client. Il client scartava la risposta completata e conservava soltanto
`interactive.cancelled/code=steered`. La KV del server includeva quindi una
chiamata assente dalla richiesta successiva. I call ID del precedente turno
eseguito non corrispondevano più alla frontiera live.

Il secondo steering è stato accettato il 14 ottobre alle 18:40:11, ora di Roma.
Il server ha ripristinato i 5.119 token di sistema e ricalcolato i 13.374 token
della conversazione in 24,691 secondi. Non ha ricalcolato il system prompt.
L'esaurimento successivo del contesto rimane distinto da questa regressione.

## Corrispondenza fra chiamate ed esecuzioni

I call ID uniscono le chiamate generate nel log, le chiamate e risposte nella
raw history delle skill e i `tool_events` della chat. Il log allegato contiene
42 chiamate generate. Tre non hanno né chiamata conservata in history, né tool
output, né evento di esecuzione: sono le risposte completate e poi scartate.

| Steering | Chiamata generata e scartata | Ripresa del prefill | Token ricalcolati |
| --- | --- | --- | ---: |
| 1, accettato 18:38:10 | `call_3b024531c96e68125be6e0b3e00f97b3`, bash, generata 18:38:07 | 18:38:11 | 9.151 |
| 2, accettato 18:40:11 | `call_47b6b59991e46a60407cd491e68c3093`, bash, generata 18:40:06 | 18:40:11 | 13.374 |
| 3, accettato 18:51:06 | `call_75724f12898242dd311cd58c524b3faf`, bash, generata 18:51:04 | 18:51:06 | 16.273 |

Tutti gli orari della tabella si riferiscono al 14 ottobre 2026, Europe/Rome.
Prima del secondo steering è stata eseguita
`call_9abbcbf9fe01a4b47d5609fbc549137c`; dopo il ricalcolo è stata eseguita
`call_080e26aa428f77375e8a43cce9817e74`. La chiamata intermedia `call_47b...`
compare soltanto nel log di generazione: non risulta eseguita. Il log termina
durante il prefill della terza ripresa; le chiamate successive sono disponibili
soltanto nella chat salvata.

La correlazione completa, con posizioni delle history e timestamp degli eventi,
è conservata localmente in
`performance-results/steering-completed-response-20261015/original-call-execution-trace.json`.
Le copie dei dati originali e lo script di correlazione sono nella stessa cartella.

## Correzione

`InteractiveResponseOpenAI` distingue il risultato completato dal server e
superato dallo steering da una generazione realmente interrotta. Nel primo caso
restituisce il contenuto completato come `ProviderResponse.superseded_response`,
mentre il risultato destinato al dispatcher continua a non avere chiamate.

`Client` conserva quella risposta nella history e aggiunge per ciascun call ID
un tool output XML con `status=cancelled`, `executed=false`, `code=steered`.
Poi il runtime aggiunge l'evidenza dell'interruzione e lo steering. La history
spiega così perché una chiamata generata non è stata eseguita, e il server può
riusare la frontiera corrispondente. Anche una risposta di solo testo viene
conservata per mantenere il prefisso visibile. Skill ed exit tool superati dallo
steering non vengono aperti o eseguiti.

Il grafo dell'agente riconosce la ricevuta `cancelled/ executed=false` quando
non esiste un evento di esecuzione. Mostra chiamata e risposta di annullamento
senza aggiungere una falsa esecuzione completata. Le ricevute rimangono nella
raw history, associate ai call ID originali.

Le pause edit cancellate e i rifiuti parziali conservano i loro percorsi
precedenti: non sono risposte completate e non ricevono questa evidenza.
Il server, i kernel e il formato dei checkpoint non cambiano.

## Verifica

Le regressioni con SDK e runtime reali, trasporto simulato e file temporanei
falliscono prima della correzione e passano dopo. Coprono JSON/SSE, testo senza
tool e batch di cancellazione più `exit_agent`; i file restano invariati e
l'exit scartato non chiude la skill.

Il test opt-in `exmekh_lib/tests/live/test_steering_history_live.py` usa il client
e il server reali con tool sintetici mai eseguiti. Per ogni trasporto invia due
steering consecutivi e verifica direttamente i contatori wire della cache.
La cancellazione è impostata prima del controllo del dispatch, dopo che il server
ha prodotto la risposta completa; il test non dipende da una gara temporale.

| Prova live | Token riusati | Token nuovi di prefill |
| --- | ---: | ---: |
| Vecchio comportamento, JSON, primo steering | 292 | 1.952 |
| Correzione, JSON, primo steering | 2.222 | 159 |
| Correzione, JSON, secondo steering | 2.483 | 159 |
| Correzione, SSE, primo steering | 2.222 | 159 |
| Correzione, SSE, secondo steering | 2.483 | 159 |

Il controllo negativo usa lo stesso server e disabilita soltanto la registrazione
della risposta superata: emula il comportamento precedente del client, non è
un confronto con un binario storico. I tool sono sintetici e non scrivono file.
Configurazione: Qwen3.8-27B UD-Q4_K_S, CUDA RTX 3090, MTP, contesto 26.000.
La qualifica live non è estesa ad altri modelli o backend.

I [contatori e gli hash](../performance/steering-completed-response-20261015.json)
collegano sorgenti, wheel e prove. La wheel dell'agente è ricostruita e installata.

Passano 2.920 test della libreria, con 62 skip delle prove opt-in, i due test
live del client e tutti i 10 controlli della suite server della cache. Ruff e
Pyright passano sui moduli modificati. La copertura di linee e rami dei quattro
moduli della correzione client è 100%. Il gate globale della libreria resta a
99,99% per un ramo non coperto in `edit_filesystem.py:147`, appartenente alle
modifiche concorrenti del tool di eliminazione; quel file non è modificato da
questa correzione.

Nell'ultima esecuzione completa dell'agente passano 990 test, con 43 skip e
copertura di linee e rami al 100%. Un test browser sui collegamenti del board
va in timeout nella suite parallela e passa isolato. Un timeout di un precedente
run sulla visibilità dell'output skill passa sia isolato sia nel run successivo.
La suite browser completa non è quindi dichiarata stabilmente verde. Passano
anche tutti i 100 test mirati di history, grafo e steering, incluso il test
browser del recupero dello steering. La chat originale conserva lo stesso hash;
il server temporaneo delle prove è stato arrestato.

Comando per ripetere le due prove live da `exmekh_lib/`, su un server inattivo:

```bash
DS4_STEERING_CACHE_BASE_URL=http://127.0.0.1:8080 \
DS4_COMPLETED_STEERING_REPORT=/tmp/completed-steering \
.venv/bin/python -m pytest tests/live/test_steering_history_live.py --run-live -n 0 -s
```
