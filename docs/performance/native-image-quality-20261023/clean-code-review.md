# Riesame del codice e delle responsabilità — 23 ottobre 2026

La revisione rilegge il diff C/Python e i percorsi collegati: client/tool,
provider e wrapper, ranking, codec/collector degli allegati, sessioni e storage,
parser HTTP, timestep, oracle e packaging. La scansione AST dei sorgenti Python
è un controllo aggiuntivo dei confini, non una certificazione di ogni file
C/CUDA della repository. Gli accessi privati trovati appartengono a copie della
stessa classe, a chiamate della classe base o ad API pubbliche come `_replace`.
Non richiedono di esporre nuovi metodi o riscrivere quelle classi.

| Problema concreto | Proprietario e correzione |
| --- | --- |
| La configurazione conteneva validazione delle richieste | `NativeSuiteSettings` conserva dati e testi; `NativeSuiteClient.validate_image_options()` possiede il contratto nativo e viene usato anche prima degli upload. |
| Formato, background o seed non valido potevano essere rifiutati dopo gli upload | La stessa validazione pubblica controlla queste opzioni; il seed lascia spazio per tutte le immagini del batch, come nel server C. |
| Un riferimento successivo mancante o directory poteva essere scoperto dopo un upload | L'orchestratore del workspace verifica tutti i file prima delle chiamate native. |
| Una risposta vuota o con dimensioni diverse risultava un successo | `ImageBatch.validate_geometry()` verifica numero e dimensioni rispetto alla richiesta. |
| La persistenza conosceva il campo base64 del protocollo | `GeneratedImage.png_bytes()` possiede decodifica e verifica di formato, dimensioni e integrità PNG; il salvataggio usa il metodo pubblico. |
| Un secondo PNG invalido lasciava il primo file salvato; con anteprime disabilitate un payload invalido veniva conservato | L'helper verifica tutti i payload prima di creare la directory, indipendentemente dalle anteprime. Non è una garanzia di atomicità contro errori del filesystem durante le scritture. |
| L'entrypoint inoltrava dodici argomenti per posizione | Inoltra le opzioni per nome, conservando signature e Literal esposti al modello. |

Le funzioni nuove rimangono brevi e i testi prodotto appartengono alla loro
configurazione. Gli errori del protocollo restano espliciti. Il context manager
mantiene l'ownership del trasporto; il tool non accede al client OpenAI interno.
Le prove usano trasporto iniettato e API pubbliche, non campi privati.

Il riesame C conserva il calcolo del timestep e le sue frontiere BF16 esplicite:
la semantica resta nell'adapter. Il test usa i valori indipendenti catturati dalla
pipeline ufficiale; le tolleranze non cambiano. Gli import differiti degli oracle
sono necessari a selezionare i sorgenti congelati prima di caricare le dipendenze.

La differenza fra il default HTTP v1 1024×1024 e il default ExMekh 2048×2048
rimane esplicita e documentata: il contratto HTTP congelato e la scelta di default
del prodotto hanno scope diversi. Le dimensioni richieste vengono rispettate;
il limite del batch produce un errore, senza riduzione silenziosa.

La libreria è nel commit `6065032`. La wheel viene ricostruita da quei sorgenti,
reinstallata e confrontata integralmente con il pacchetto e il lock. La patch
versionata conserva il delta dal precedente commit `bd908cf`, anche dopo il
commit della dipendenza. Risultati e hash finali sono in `verification.json`.

Questo riesame migliora contratti e leggibilità; il problema percettivo delle
prime immagini resta aperto. Non vengono qualificati nuovi kernel, altri
backend o inferenza distribuita.

Verifica finale: 3668 test libreria e 1133 test frontend
passano; skip live rispettivamente 76 e 45.
Copertura frontend 100%, inclusi i browser offline. Passano anche 33
test di contratto con 39 sottocasi, Ruff, Pyright della libreria, build CUDA
e i gate C elencati nel manifest. Nessuna nuova generazione GPU viene
attribuita a questa revisione: il replay usa i PNG nativi già prodotti.
