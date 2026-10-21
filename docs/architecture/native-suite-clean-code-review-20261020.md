# Revisione di responsabilità e coerenza della suite nativa

20 ottobre 2026. Revisione del lavoro T00–T04 prima del commit. Il perimetro
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

## Riesame di T03–T04

Il secondo passaggio esamina registro, accounting CUDA, checkpoint, adapter
sessione, integrazione vision e build. La lettura si estende a getter e
invalidazione legacy, CLI/server e ai confini fra `ChatSession`, manager e
persistenza JSON dell'agente. Nessuna modifica all'agente è necessaria: i
metodi privati osservati sulle copie di `ChatSession` sono chiamati dalla
classe stessa; il manager usa l'interfaccia della sessione e dello store.

| Problema | Correzione e responsabilità |
| --- | --- |
| Query memoria con un elenco manuale di decine di tensor e lettura del campo backend `owner`, destinato a divergere dalle nuove allocazioni | La sessione chiede al registro i byte device del proprio owner. L'adapter interpreta soltanto le categorie ricorrenti/MTP tramite l'API tensoriale |
| `trim` poteva togliere un'entry di accounting senza destructor e dichiarare liberati byte ancora allocati | Il registro considera solo entry ricostruibili con destructor, senza alias o eventi pendenti; un errore mantiene entry e riferimento |
| I due free CUDA duplicavano il cleanup e invalidavano diversamente la route cache; gli errori di `cudaFree` erano ignorati | Unico helper del backend per storage, cache e wrapper. Un rilascio fallito conserva handle e accounting; il destructor propaga il fallimento |
| Invalidare una sessione parcheggiata lasciava un checkpoint che poteva ripristinare la conversazione invalidata | La sessione elimina il proprio checkpoint interno e rimane parcheggiata con stato vuoto; il resume non resuscita la history |
| `set_logits` sembrava riuscire su uno stato parcheggiato, ma il successivo restore ne annullava l'effetto | Rifiuto prima della modifica, coerente con l'autorità del checkpoint su disco |
| Controllo di residenza ripetuto in 14 entry point con possibilità di divergenza | Unico controllo nel dominio sessione/engine; mantiene i codici legacy e precede l'avanzamento della speculazione |
| Handle null e directory vuote producevano esiti incoerenti; due getter dereferenziavano sessioni null | `INVALID` per argomenti lifecycle invalidi; getter null a zero come `prefill_cap`, preservando il codice di errore `BUSY` |
| Risigillare un checkpoint su uno stream più lungo conservava un trailer che lo rendeva invalido | Il serializer tronca il file alla dimensione dell'envelope prima di `fsync`; nessuna modifica del payload ABI |
| Hash già scritti nell'envelope erano conservati anche nell'handle senza più essere letti | Rimossi dall'handle; la validazione legge il file autorevole |
| Dipendenze incomplete degli header/frammenti potevano lasciare test compilati contro codice vecchio | Dipendenze comuni per i test che includono l'engine e per il probe CUDA; regole LF per frammenti C e nuovi manifest |

I test aggiunti verificano effetti osservabili: rilascio reale e fallito,
contenuto di view dopo free del proprietario, equivalenza dei due cleanup,
checkpoint con trailer, mancata resurrezione dopo invalidazione e assenza di
allocazioni/prenotazioni residue. I gate Qwen27 confrontano tutti i logits e
i token della continuazione nelle tre modalità di sospensione.

Responsabilità, incapsulamento, nomi, duplicazione, errori e testabilità sono
migliorati nei confini esaminati. Restano il monolite engine/server, i globali
legacy confinati dal runtime, l'inventario host incompleto e gli adapter non
qualificati. La revisione non chiude T03/T04 né certifica Metal, ROCm o TP.

La [verifica del riesame](../performance/native-suite-t03-t04-20261020/review-verification.json)
registra sorgenti/binari finali, comandi, log, confronto con T00 e gate SSD CUDA.
Gli artefatti completi rimangono in
`performance-results/native-suite-review-t03-t04-20261020`.

## Riesame di T05–T06

Il terzo passaggio comprende planner, executor, registro, loader, tool offline e
build. Sono stati riletti anche lifecycle, runtime, confini CLI/server e i
metodi di branching/revisione di `ChatSession`. I tre accessi a metodi privati
su `branch`/`revised` avvengono dentro la classe che li definisce, su proprie
copie: non richiedono di esporre dettagli interni al manager.

| Problema | Correzione e responsabilità |
| --- | --- |
| Il planner ricostruiva capacità host/device, margini e disponibilità globale leggendo più getter e duplicando la politica del registro | `ds4_resources_can_reserve` verifica ammissione e reclaim proiettato sotto un unico lock. La prenotazione reale usa lo stesso calcolo. Il planner descrive soltanto domanda e vittime |
| Un disconnect durante caricamento, save, restore o attesa del recupero poteva arrivare dopo l'ultimo controllo e consentire compute o commit | Controlli alle frontiere di fase e immediatamente prima della pubblicazione; restore e drain proseguono comunque. Nessun replay |
| La cancellazione di una reservation controllava solo il TLS del thread chiamante: poteva liberare un lease attivo su un altro thread o nascosto da uno scope nidificato | Il registro conta gli scope attivi del lease. La cancellazione restituisce `BUSY` finché uno scope è vivo; le allocazioni scoped ereditano anche il suo evento, come quelle committate esplicitamente |
| Rilascio dei gruppi duplicato fra percorso normale e recupero; una funzione orchestrava anche tutti i dettagli delle fasi | Helper comuni per save, rilascio delle vittime, visite e rilascio inverso dei gruppi. Il journal mantiene i bit prima dei callback potenzialmente parziali |
| Un gruppo poteva continuare a dichiararsi residente dopo un drain/free fallito | Il proprietario del gruppo invalida la disponibilità prima di tentare il rilascio, conserva lo storage per il recupero e impedisce reload/uso prematuro |
| Il reader dell'indice classificava anche gli errori di allocazione come dati invalidi | Il reader conserva il primo stato tipizzato; `NOMEM` e indice invalido restano distinti nel cleanup comune |
| Discovery rileggeva gli stessi pesi condivisi per ogni componente; verifica degli artefatti ripetuta fra più funzioni | Un solo controllo di identità condiviso, hash una volta per file nella discovery; nessun cambiamento ai byte degli artefatti |
| Regex di layer invalide, selector su asset/revisioni incompatibili o indici troppo grandi fallivano tardi con eccezioni di packing o nel loader | Selector validati prima della lettura dei pesi e limite della directory prima della pubblicazione; buffer incrementale e limite di formato a 64 MiB |
| Le dipendenze transitive degli header pubblici erano elencate diversamente fra client, test, engine e probe | Gruppo comune `DS4_API_HEADERS`; nessuna modifica ai flag numerici o ai kernel |

Non sono state aggiunte classi per funzioni senza stato. Il registro possiede
budget e lifetime, l'executor possiede l'operazione, il loader possiede directory
e gruppi, l'engine conosce i payload delle proprie sessioni. CLI/server restano
client delle API opache e l'agente usa i metodi delle proprie sessioni/store.

La revisione migliora responsabilità singola, incapsulamento, assenza di
duplicazione, nomi, errori e testabilità nei confini esaminati. Non tutti i
principi sono già soddisfatti nell'intera repository: rimangono i monoliti
engine/server, lo stato globale legacy, l'accounting host incompleto e gli
adapter non qualificati. Estrarli senza oracle/backend disponibili sposterebbe
il rischio, senza dimostrare un miglioramento. Metal, ROCm e distribuito non
ricevono una qualifica da questi test locali.

Le evidenze finali del riesame sono nel
[registro T05–T06](../performance/native-suite-t05-t06-20261020/review-verification.json).
I log restano in `performance-results/native-suite-review-t05-t06-20261020`.

## Riesame di T07–T08, 21 ottobre 2026

Riletti i nuovi moduli numerici, encoder/tokenizer, preprocessing, bridge dei
pesi, test, build e tool di verifica. Il controllo dei confini include anche
runtime/executor/registro, API engine/server e, nell'agente, `ChatSession`,
`SessionManager`, `JsonRecordStore`, `RecordedHistory` e supporto dei test UI.
La repository è C: le responsabilità sono assegnate a moduli e handle opachi,
senza introdurre classi C++ o interfacce prive di una necessità concreta.

| Problema osservato | Correzione e proprietario |
| --- | --- |
| Input e conditioning esponevano puntatori mutabili, buffer posseduti e record privati di accounting | Proprietari opachi con snapshot pubblici borrowed e di sola lettura. I client chiamano gli accessor; non liberano né ricostruiscono lo stato interno |
| Il forward possedeva anche compositing, tokenizzazione, padding, posizioni e costruzione dell'input | `ds4_qwen_image_input` possiede preparazione, validazione e lifetime. Passaggi separati per reference, posizioni e pixel. L'encoder consuma la view pubblica e orchestra vision/testo |
| Il core numerico conosceva bundle e layout del buffer dei gruppi | `ds4_numeric_weights` è il bridge; usa metodi pubblici per scrittura in byte e reinterpretazione. `ds4_numeric.h` dipende soltanto da risorse e tipi numerici |
| Softmax CUDA copiava tutta la capacità della sorgente; reorder/nearest rifiutavano output più grandi, diversamente dalle altre primitive | Il front-end delimita il prefisso del risultato prima del dispatch; tutti i backend preservano la coda inutilizzata |
| I controlli di alias consideravano solo l'uguaglianza dei puntatori; broadcast dall'output e cast impliciti potevano cambiare il risultato | Controllo dei range nel core; in-place esatto solo dove supportato, overlap parziale rifiutato, cambio dtype esplicito con CAST |
| Una cancellazione richiesta dalla callback finale poteva pubblicare un risultato | Controllo dopo la callback e prima della pubblicazione. Il test cancella proprio allo stage conditioning e verifica output nullo e risorse invariate |
| Il buffer liberava il backend prima di sapere se l'evento consentiva il rilascio del record; un'allocazione fallita perdeva metadata in-flight | Lo storage ha un distruttore invocato dal registro dopo l'evento. Wrapper e record incompleti sono trattenuti dal contesto fino al recupero, senza replay |
| `numeric_close(NULL)` differiva dagli altri close; il reader del tokenizer calcolava l'hash anche dopo una lettura incompleta e dipendeva da calloc(0) | Close nullo uniforme e testato; verifica di lettura/chiusura prima dell'hash; nessuna allocazione per una lista vuota di token speciali |
| Smart resize comune era concentrato in espressioni poco leggibili | Nomi e rami espliciti, con formule e ordine numerico invariati; percorso legacy verificato dalle regressioni |
| Il costruttore di gruppi selezionati conservava un output precedente quando la selezione era vuota | Output azzerato anche sugli argomenti invalidi, come negli altri costruttori; regressione riprodotta e corretta |

I test nuovi sono prima falliti contro i comportamenti precedenti: capacità,
aliasing, dtype e rilascio di storage con evento pendente. Dopo le correzioni
verificano gli effetti osservabili e il saldo finale del registro. I gate
encoder confrontano tutti i confini con l'oracle T00; le tolleranze non sono
state cambiate. Sanitizer, contratti e test dell'agente affiancano le verifiche
del modello reale. Comandi, hash, esiti e limiti sono nel
[registro del riesame](../performance/native-suite-t07-t08-20261020/review-verification.json).

Gli accessi privati `branch._discard_branch_state_from` e
`revised._revision_boundary` avvengono dentro `ChatSession`, su proprie copie:
sono responsabilità correttamente collocate. Il manager usa i metodi pubblici
di sessioni e store. Non sono stati aggiunti getter/setter che consentano al
chiamante di aggirare gli invarianti del proprietario.

La revisione migliora responsabilità singola, incapsulamento, dipendenze,
coerenza, nomi e testabilità nei confini esaminati. Non certifica che tutti i
principi clean code siano soddisfatti nell'intera repository: restano i grandi
moduli engine/server, lo stato globale dei backend legacy e la qualifica dei
backend non disponibili. Anche il BPE image usa ancora una ricerca quadratica
nel caso limite di un singolo segmento molto lungo: il limite in byte non è
un limite di latenza. Un algoritmo differente richiederà il proprio corpus di
equivalenza e benchmark, senza cambiare la semantica per guadagnare velocità.
