# Revisione di responsabilità e coerenza della suite nativa

20 ottobre 2026. Revisione del lavoro T00–T02 prima del commit. Il perimetro
comprende runtime, bridge LLM, diagnostic vision, build, tool numerici e test;
sono stati esaminati anche i confini CLI/server e le classi di configurazione,
sessione e persistenza dell'agente. Non è una certificazione dell'intera repository.

## Correzioni applicate

| Problema | Correzione | Responsabile |
| --- | --- | --- |
| Il diagnostic vision controllava un flag e poi inizializzava autonomamente CUDA: un altro thread poteva aprire un engine fra le due operazioni. Non chiudeva mapping/backend e perdeva il mapping anche al fallimento della decodifica | Prenotazione atomica dello stesso runtime degli engine, cleanup comune su successo/errore e mapping host mantenuto fino al drain GPU; controllato anche l'errore di `fclose` | Il runtime possiede backend e lock; il diagnostic possiede immagine, patch, output e mapping locale |
| L'engine esponeva un handle modello borrowed che dava accesso al destructor generico, aggirando il controllo sulle sessioni vive | Rimosso `ds4_engine_runtime_model`; rimane il solo runtime borrowed necessario alla registrazione dei componenti | Apertura/chiusura LLM esclusivamente tramite l'engine |
| Attivazione di contesto duplicata fra esecuzione e distruzione | Unico helper `runtime_activate_model`, con fence e invalidazione dopo callback fallita | Runtime, senza conoscenza di shape o tensor |
| La funzione leave si chiamava model_leave ma gestiva anche scope di costruzione senza modello | Nome unico `ds4_runtime_scope_leave` e contratto sul thread chiamante | Scope runtime |
| Getter privati decidevano ripetutamente quale shape leggere usando la presenza del descrittore di catalogo | Unico helper `ds4_engine_shape`, basato sulla presenza dello snapshot del runtime | Engine; il catalogo descrive l'identità, non la validità dello snapshot |
| `tp_bind` e `tp_vocab_split`, API con semantica booleana intera, restituivano -1 quando lo scope era rifiutato | Ritorno 0, coerente con gli altri fallimenti e con i chiamanti | Bridge API legacy, senza cambiare l'ABI |
| Un processo di probe terminato con 0 poteva essere promosso senza file, con logits vuoti o con payload mancante; il confronto Flash poteva limitarsi agli hash di JSON non validi numericamente | Validatore condiviso di metadata, conteggi, shape, payload, coppie frontier e numeri finiti; errore di validazione registrato distinto dal codice del processo | `tools/suite_artifacts.py`; i runner decidono soltanto comandi, sequenza e confronto |
| Alcuni test CPU elencavano manualmente i support object, omettendo il nuovo runtime; altri object non dipendevano dal suo header | Gruppo comune `CPU_SUPPORT_OBJS` e dipendenze esplicite dei test hook | Makefile |
| Il runner C conservava engine fra casi che possono scegliere profili o aprire modelli autonomamente | Chiusura delle fixture dopo ogni entry | Runner dei test |
| I nuovi tool avevano import disordinati e molte istruzioni compresse sulla stessa riga | Formattazione uniforme e lint su tutti i nuovi moduli Python | Singoli moduli; nessuna nuova dipendenza del server |
| La conversione automatica LF/CRLF di Git poteva cambiare i checksum del corpus e dei manifest su Windows | Regole LF esplicite per i JSON congelati e gli schemi della suite | `.gitattributes` |

## Principi verificati e limiti

- **Responsabilità singola:** il modulo runtime coordina lifetime e contesti;
  l'engine conosce shape e modello; il backend possiede stream e allocazioni;
  il validatore dei probe conosce il formato degli artefatti. Non sono state
  introdotte classi per sostituire funzioni prive di stato.
- **Incapsulamento:** CLI/server continuano a usare handle engine/sessione
  opachi. I componenti ricevono il runtime, non l'handle interno del LLM.
  Nell'agente le chiamate a metodi privati trovate fuori da `self`/`cls`
  appartengono a copie della stessa `ChatSession`, entro la sua implementazione;
  non sono violazioni del confine fra classi. Dati pubblici di DTO e fixture
  non sono equiparati a servizi da rendere artificialmente opachi.
- **Dipendenze e interfacce:** i callback del runtime non dipendono da HTTP,
  GGUF o CUDA. Il bridge traduce il modello legacy. I test CUDA usano
  deliberatamente una verifica interna dell'identità di stream/cuBLAS;
  quell'accesso non viene introdotto nei client o nel server.
- **Coerenza degli errori:** un errore del runtime non diventa un booleano vero;
  processo riuscito e risultato numerico valido sono condizioni distinte.
  Non si ricrea una baseline per nascondere una regressione.
- **Leggibilità e duplicazione:** nomi relativi al proprietario, helper comuni
  per attivazione/shape/hash, formatter e lint. Le tolleranze e i dati congelati
  non vengono cambiati dalla formattazione.
- **Testabilità:** test senza GPU per dati mancanti, shape incoerenti,
  NaN/Infinity, JSON fuori standard e processo riuscito senza output; fault
  test del runtime e test CUDA di ritorni booleani, errore diagnostic e rebind.

Non tutti i principi sono già soddisfatti globalmente. `ds4.c` e
`ds4_server.c` conservano più responsabilità e ampio stato legacy; il bridge
continua a installare una vista globale della shape sotto lock. I backend
condividono inoltre header con dettagli interni del multi GPU. Questi confini
richiedono una migrazione per responsabilità con oracle e qualifiche per
backend, non uno spostamento meccanico di migliaia di righe. T02 resta aperto
per Metal, ROCm e TP/distribuito; questa revisione non cambia quella qualifica.

## Evidenze

Comandi, hash e log sono nel [manifest di verifica](../performance/native-suite-t00-20261020/verification.json)
e nel [report T00–T02](../performance/native-suite-t00-20261020/README.md).
I log del riesame restano in `performance-results/native-suite-review-20261020`;
i tentativi di build falliti sono conservati insieme alla build corretta.
Le modifiche preesistenti a wheel, lock e ordine dell'indice esperimenti sono
escluse dal commit della suite.
