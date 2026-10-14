# Correzione del riuso KV dopo steering — 14 ottobre 2026

La correzione interviene sul riconoscimento della continuazione Responses e
sull'ownership dei checkpoint. I kernel, il formato dei payload e il client
non cambiano. La [riproduzione precedente](steering-cache-reproduction-20261014.md)
conserva le misure del difetto; i
[risultati della correzione](../performance/steering-cache-fix-20261014.json)
riportano configurazioni, contatori e hash.

## Cause e comportamento corretto

`responses_prepare_live_continuation` cercava soltanto risultati di tool in
coda. Un messaggio user di steering o di interruzione dopo quei risultati
nascondeva il confine della continuazione. Il matching del prompt completo
falliva anche perché il blocco sintetico `agent_runtime` viene aggiunto al
messaggio più recente: il replay non è un prefisso byte per byte del precedente
prompt. Il riconoscitore ora cerca il confine prima dei nuovi messaggi user,
ma include tutti quei messaggi nel suffisso da valutare. Rimangono il binding
ai call ID della frontiera live, la verifica delle immagini e il controllo
successivo nel worker. Una risposta assistant intermedia impedisce di
riutilizzare un vecchio batch di chiamate.

Una richiesta ausiliaria senza `agentic` rimuoveva il proprietario dello slot
prima di salvare il contesto. Ora il salvataggio della frontiera vincolata
precede sia una richiesta stateless sia il passaggio a un altro contesto.

La cancellazione di una pausa edit eliminava il checkpoint sospeso, lasciando
nella sessione la generazione incompleta e nella struttura live il binding
del turno precedente. Sono ora distinti il checkpoint del resume e quello
del rollback. Prima di estendere una frontiera Responses riusabile il runtime
conserva una copia compatta dello stato e del binding. Soltanto quando si
verifica una pausa la converte in un payload portabile. Il resume ripristina
la generazione sospesa; cancel ed expiry pubblicano invece il checkpoint
dell'ultimo turno completato. La thread di controllo trasferisce ownership;
soltanto il worker ripristina la sessione. I due checkpoint sono conteggiati
nel budget esistente e liberati dai rispettivi proprietari.

## Contratto delle prove

Il test di cancellazione confronta la ripresa con i token già riusati prima
della pausa: è la frontiera dell'ultimo turno completato. Non pretende di
conservare il prompt della richiesta cancellata né la chiamata incompleta.
Il risultato del tool sostituito, le evidenze di interruzione e lo steering
sono il nuovo suffisso e devono essere elaborati. Questo distingue un piccolo
prefill necessario dal ricalcolo della storia completa.

I tool nelle fixture sono sintetici e non scrivono file. Le risposte devono
avere stato e call ID validi; il test di resume completa anche una selezione
vincolata allo snapshot. Le regressioni native verificano il confine del
suffisso per quattro sintassi, l'ownership del rollback e il budget aggregato.

Le prove live usano Qwen3.8-27B UD-Q4_K_S, CUDA/RTX 3090, MTP, contesto 26.000
ed encoder vision caricato. Le altre combinazioni di modello/backend non sono
qualificate live da questo intervento. Il gate di cancellazione riguarda una
pausa `file_snapshot`, non un errore GPU o l'interruzione di un prefill.

## Risultati verificati

Tutte le 13 prove live della cache passano, senza errori HTTP nelle fixture.
Passano anche i 6 test live del protocollo edit e i 3 controlli locali del
parser di riferimento. Le matrici comprendono controlli senza steering,
richieste ausiliarie con e senza contesto, SSE, thinking e resume.

| Prova | Token riusati | Token nuovi di prefill |
| --- | ---: | ---: |
| matrix: steering-after-cancelled-edit-pause | 2627 | 329 |
| matrix: steering-after-completed-tool | 2627 | 132 |
| matrix: auxiliary-without-agent-context | 2627 | 116 |
| thinking-sse: steering-after-resumed-edit | 2930 | 129 |
| long-sse: steering-after-cancelled-edit-pause | 12959 | 326 |
| ordinary-sse: steering-after-completed-tool | 2472 | 130 |

Passano i test ds4_server_test, test-server-interactive e il test di lifetime host.
AddressSanitizer/UndefinedBehaviorSanitizer sul parser interattivo.
Il server CUDA è ricompilato; il server temporaneo delle prove è stato terminato.

## Ripetibilità

```bash
DS4_STEERING_CACHE_BASE_URL=http://127.0.0.1:8080 \
DS4_STEERING_CACHE_REPORT=/tmp/steering-cache.json \
python3 -m unittest tests.test_steering_cache_live -v
```

`DS4_STEERING_CACHE_STREAM=1` abilita SSE,
`DS4_STEERING_CACHE_REASONING=low` abilita thinking,
`DS4_STEERING_CACHE_ARCHIVE_LINES=640` costruisce la conversazione lunga e
`DS4_STEERING_CACHE_INTERACTIVE=0` isola i tool ordinari.
Senza BASE_URL le prove live sono saltate e i controlli unitari sono eseguiti.
Le richieste complete e i log della correzione sono in
`performance-results/steering-cache-fix-20261014/`.


## Verifica successiva: cancel seguito da richiesta ausiliaria

La [revisione con test live](clean-code-live-review-20261014.md) ha aggiunto un
caso prima non coperto: una richiesta ausiliaria dopo cancel poteva sostituire
il rollback pubblicato con la sessione interrotta ancora residente. Il checkpoint
è ora autorevole fino al ripristino e la pubblicazione è idempotente. Il report
precedente conserva le misure della prima qualifica; il report successivo include
la riproduzione indipendente e la nuova matrice sul binario corretto.
