# Istruzioni agentiche e diagnosi dei test live — 14 ottobre 2026

Il server possiede il protocollo di inferenza, i vincoli sui token, il
trasporto e i checkpoint. La libreria e l'agente possiedono le istruzioni
operative, le descrizioni dei tool e il ritorno dalle skill.

## Rimozioni dal server

Eliminati il blocco `agent_runtime` aggiunto a user/tool message, le istruzioni
di ritorno dalle skill e il caso speciale che sollecitava una chiamata a un
tool senza argomenti. Eliminata anche la riscrittura degli schemi che
aggiungeva una seconda descrizione dell'edit già documentato dal client.
I relativi campi, funzioni e costanti inutilizzati sono rimossi.

Il recupero di una chiamata malformata conserva l'errore e i delimitatori
del protocollo, senza aggiungere istruzioni di correzione né ripetere il
system prompt. I template delle famiglie di modelli, gli schemi ricevuti,
il masking e la generazione degli ID edit rimangono parte del protocollo.

## Criteri delle prove

Un test sul software confronta i byte scritti con gli argomenti realmente
emessi dal modello. Un contenuto diverso richiesto nella domanda è una
valutazione distinta dell'intento. La verifica degli effetti precede
l'eventuale classificazione `MODEL_LIMITATION`.

| Osservazione | Esito del test sul software |
| --- | --- |
| Scrittura diversa da una chiamata riuscita | Fallimento |
| Modifica dopo una chiamata rifiutata | Fallimento |
| Creazione/cancellazione valida che fallisce | Fallimento |
| Parsing, trasporto, schema, ID o lifecycle incoerenti | Fallimento |
| Cache non conservata dopo steering, anche se il modello sceglie un altro tool | Fallimento |
| Setup o payload richiesto non prodotto dal modello | `MODEL_LIMITATION`, percorso non verificato |
| Calcolo o parola finale sbagliati, con streaming corretto | Osservazione separata; il test di trasporto conserva le proprie asserzioni |
| Modello riapre un lavoro già completato | Effetti verificati, limite del modello registrato, checkpoint della ripetizione rilasciato in teardown |

Non si convertono eccezioni generiche o timeout di rete in errori del
modello. Le richieste dei test specificano esplicitamente quando un lavoro
è terminato, senza introdurre nuove istruzioni nel prodotto.

I controlli deterministici del server includono parsing letterale, ID,
rifiuti, messaggi di recupero e una regressione di cache accompagnata da
una scelta errata del modello. Dimostrano che la classificazione non
nasconde questi difetti. Le verifiche degli effetti sui file appartengono
alle fixture del client, il cui lavoro resta sospeso. Le fixture Python e C
non diventano messaggi del prodotto.

## Esecuzione e risultati

Passano build CUDA, test nativi del server e del protocollo interattivo,
lifetime della frontiera host, controllo dei testi e verifiche degli indici.
I test C confrontano l'intero suffisso di recupero per DeepSeek, Qwen,
Qwen4 e GLM: contiene soltanto errore, dettaglio e delimitatori del protocollo.

Passano i 7 scenari HTTP della cache e il caso SSE con reasoning low e
384 righe di storia. Quest'ultimo conserva 7749 token e ne rivaluta 236 dopo
cancellazione della pausa edit e richiesta ausiliaria senza contesto agentico.
Passano 4 dei 6 test live HTTP dell'edit; gli altri 2 sono inconcludenti
per il modello: sostituzione diversa nel caso con ID omesso e mancata
emissione della seconda risorsa nel caso di rifiuto. Non sono conteggiati
come successi. Il parser di riferimento ha 3 test locali separati.

Il [riepilogo verificabile](../performance/server-agent-guidance-removal-20261014.json)
conserva misure e hash del server. I test live di libreria e frontend sono
sospesi su richiesta dell'utente. Le precedenti prove della libreria hanno
rilevato anche import incoerenti durante modifiche parallele ai sorgenti;
non costituiscono una qualifica della versione in lavorazione nell'altra chat.

Le tracce delle esecuzioni sono conservate in
`performance-results/agent-guidance-removal-20261014/`, con log del server,
richieste della libreria e risultati JUnit. La qualifica riguarda il server
locale Qwen3.8-27B, CUDA/RTX 3090, MTP e contesto 26000; gli altri backend e
modelli richiedono prove dedicate.
