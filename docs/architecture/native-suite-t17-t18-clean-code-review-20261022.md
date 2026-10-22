# Revisione di T17/T18

22 ottobre 2026. Riesame del servizio seriale CUDA e dell'integrazione ExMekh.
Evidenze e confini nel [report](../performance/native-suite-t17-t18-20261022/README.md).

La [revisione successiva](../performance/native-suite-clean-review-20261022/README.md)
corregge ownership HTTP, confini del codec allegati, timestamp e close parziali;
elimina inoltre la capability embedding autonoma dopo il chiarimento di scope.
Le evidenze originali restano conservate come prima qualifica.

## Responsabilità

| Proprietario | Confine |
| --- | --- |
| `ds4_server.c` | Trasporto esistente, coda e worker, avvio/chiusura del servizio esplicito |
| `ds4_server_suite.inc` | Stato del catalogo, storage, validazione condivisa, binding alle transazioni e recovery |
| `ds4_server_suite_http.inc` | Normalizzazione multipart, upload/download e dispatch nel worker comune |
| `ds4_server_suite_operations.inc` | Contratti image/embedding, recipe effettiva e commit delle ricevute dopo restore |
| Executor e adapter | Ammissione, vittime, checkpoint, copie, modelli e recupero; il server non legge tensori |
| `ds4_suite_text.h` / config ExMekh | Errori, diagnostica e descrizioni del rispettivo dominio |
| Client tipizzato | Serializzazione, upload, identità implementate e trasporto del provider esistente |
| Tool media | Percorsi workspace, output locali e allegati; il renderer comune presenta i valori come XML |
| Converter/collector/UI | Persistenza degli allegati e miniature tramite il componente immagini esistente |

## Correzioni verificate

- Il wrapper Gemma non apre una seconda transazione quando l'executor possiede
  già quella esterna. La query legge il mutex ricorsivo del runtime; il nesting
  pubblico delle transazioni continua a restituire BUSY. Il planning annidato
  usa la capacità della reservation corrente senza sottrarla una seconda volta.
- Le operazioni in recovery sono heap-owned e conservano vittime/callback;
  il puntatore al job HTTP è azzerato prima che il client termini. `close`
  conferma il ripristino anche quando `recover` riporta il precedente errore
  dell'operazione. In mancanza di ripristino non si chiudono i checkpoint come
  se le sessioni fossero pronte.
- Nessun risultato native viene inviato prima del restore. Ricevute scritte
  dopo chiusura dell'operazione; un errore disco lascia una tombstone bounded
  che impedisce di rieseguire una generazione già completata.
- Seed uint64 preservati dal parser attraverso la serializzazione, senza
  passare dal double. Canonicalizzazione degli oggetti con ordine degli array
  invariato; per i riferimenti di editing si confrontano i byte effettivi,
  non gli ID temporanei o il boundary multipart.
  Il parsing delle dimensioni controlla l'overflow prima della conversione
  a uint32; valori enormi non diventano dimensioni valide per wraparound.
- Multipart su file/mmap e download a blocchi; niente video integrale sullo
  heap HTTP. Limiti espliciti per connessioni, body, quota, file e ricevute.
  Header Content-Length duplicati e Transfer-Encoding vengono rifiutati.
- La pipeline espone la propria validazione pura. Riferimenti che richiedono
  resize implicito sono rifiutati con 400 prima delle vittime/compute, anziché
  diventare un errore 500 tardivo. L'editing image-only è ammesso con riferimenti
  validi e prompt vuoto; non si inventa testo di conditioning.
- Modelli tipizzati con Literal, singolo modello scelto dall'operazione e
  width/height interi nel client/tool. Il catalogo chat filtra le capability,
  senza offrire un modello embedding come destinazione delle conversazioni.
- Le entrypoint dei tool sono funzioni possedute dai moduli registrati,
  come richiede il loader. L'HDS main dichiara i due sub-tool opzionali;
  esclusioni, override e schema restano verificati dall'intera suite offline.
- Il benchmark prompt storico congela esplicitamente il proprio insieme di
  capability, escludendo i tool aggiunti dopo la baseline. Casi, snapshot,
  punteggi e contratti del confronto precedente restano quelli congelati.
- Converter e collector conservano gli allegati come proprietà dei tool e
  della risposta assistant; la UI non crea un messaggio user artificiale né
  avvia inferenza per visualizzare la miniatura.

## Verifica e limiti

La revisione comprende build, protocollo host, runtime/risorse/executor,
regressioni server e processor, libreria offline completa, Pyright/Ruff,
test frontend e browser. Le prove live usano SDK OpenAI e Client/LocalProvider
reali sullo stesso server CUDA: quattro modalità, risultati multipli,
editing con più riferimenti, cancellazione e ripresa dal contesto precedente.
La wheel è ricostruita dai sorgenti e reinstallata con lock aggiornato;
la dipendenza esterna è registrata mediante patch e hash.

Le prove host di fault/recovery appartengono all'executor; in questo task non
viene simulata una perdita hardware del device nel server HTTP. La cancellazione
per disconnessione è provata live e il deadline è provato sul protocollo host.
Il successo del trasporto non chiude i gate qualità image T11 né quelli di
rilascio integrato T19–T22. Non si estende la qualifica a Metal/ROCm/TP/SSD.
