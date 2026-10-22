# Revisione clean code di T13/T14

22 ottobre 2026. Riesame di decoder, processor, encoder vision, orchestratore
visual, backbone Gemma, primitive numeriche e strumenti di qualifica.
Controllati anche i confini adiacenti di runtime, registro risorse, gruppi
pesi, decoder immagini condiviso e link delle quattro applicazioni LLM.
La revisione riguarda questi percorsi; non certifica ogni modulo della repository.

## Correzioni

| Problema | Correzione e responsabilità |
| --- | --- |
| Raw video e container dimensionavano separatamente gli indici: il primo rifiutava upsampling senza cap, arrotondamenti validi e cap superiori al numero effettivo di frame | Un piano di selezione nel modulo media valida il numero effettivo prima dell'allocazione; entrambi i percorsi usano gli stessi indici |
| Un frame rate positivo estremamente piccolo poteva produrre `0 * infinity`, poi convertire NaN a intero | Il sampler definisce esplicitamente l'indice iniziale zero; valida NaN/Inf, rate negativi e numero di campioni prima dei cast |
| Il processor controllava il contesto prima di sommare i soft token audio del video | Controllo del totale visual/audio prima delle allocazioni dei tensori; 31 frame quadrati a budget 280 e 30 secondi audio sono ora rifiutati |
| L'orchestratore conosceva gli ID numerici dei placeholder e ne duplicava il conteggio | Il backbone espone una validazione privata di numero e tipo; lo stesso metodo è usato prima del forward e prima della copia delle feature |
| Ogni prompt prenotava 256 KiB aggiuntivi indipendentemente dai media | Lo stesso builder misura e scrive l'espansione di marker/timestamp; si alloca il numero effettivo di byte incluso NUL |
| Il forward testuale duplicava allocazione, registrazione e rilascio dei buffer host | Uso di `ds4_host_buffer`, già proprietario dello storage host sincrono nella suite |
| Il parser EXIF estratto aveva indentazione incoerente; i contratti del processor/cache non esplicitavano la serializzazione | Formattazione locale e commenti di lifetime/serializzazione nelle API proprietarie |

## Confini conservati

Il decoder possiede byte originali, frame e waveform; il processor possiede
patch, feature, maschere e cache; la torre vision possiede pesi e grafo;
l'orchestratore possiede ordine dei media e durata della richiesta. Il backbone
possiede tokenizzazione, riconoscimento dei token e scatter prima della PLE.
I chiamanti usano metodi e viste pubbliche immutabili, senza accedere alle
strutture opache dei componenti.

Il refactoring non cambia equazioni, dtype, pesi, ordine delle operazioni,
geometria GEMM o kernel qualificati. Nessun nuovo flag numerico. Il limite
completo della sequenza resta verificato dal tokenizer del backbone dopo
l'espansione del prompt e l'applicazione del task.

## Verifica e limiti

Il [report della revisione](../performance/native-suite-t13-t14-20261021/review-verification.json)
registra hash di sorgenti, binari, log e confronti. Le regressioni aggiunte
riproducono i difetti di admission nella versione precedente; coprono anche
frame rate non finiti, negativi, subnormali e overflow. La matrice media passa
anche con UBSan e controllo dei cast floating point.

Il replay visual confronta tutti i 3573 file con la qualifica precedente:
input, hidden e vettori restano identici byte per byte. Sono rieseguiti tutti
i quattro output Matryoshka, sei punti di cancellazione, budget negato e
rilascio/reload dei pesi. La qualifica testuale completa riesegue i 54 casi e
l'oracle congelato. Sono verificati anche i test numerici CPU/CUDA, i test C
di lifetime e le build LLM senza dipendenze codec nel link.

GCC `-fanalyzer` sui sei moduli rileva due possibili argomenti NULL: uno nel
forward vision, uno nel forward testuale. I percorsi segnalati combinano un
conteggio zero dell'allocatore (che restituisce storage NULL per zero byte)
con frame/token presenti nel successivo ciclo o copia. I media sono opachi e
immutabili, il piano vision richiede frame, e il tokenizer produce BOS/EOS:
quei conteggi zero non raggiungono i forward validi. Le diagnostiche sono
conservate nel log, senza soppressioni né dichiarazione di zero warning.

Restano gli sviluppi già documentati: bound di workspace conservativi,
qualifica encoder per budget/aspect estremi, riduzione del rumore nei tempi
video e copertura di codec/backend aggiuntivi. Il monolite LLM e i backend
Metal/ROCm richiedono revisioni mirate sul rispettivo hardware; questa
revisione non li qualifica. Non vengono introdotti refactoring trasversali
del grafo numerico senza misure o hardware per verificarli.
