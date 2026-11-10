# Riesame di convoluzione, memoria e prefetch Image

Il riesame del 5 novembre comprende tutte le modifiche ancora da integrare
su `4fef55f` nel worktree1340. Le qualifiche I-P02/I-P04/I-P05 restano
associate ai rispettivi binari congelati nei report prestazionali.

Il backend Metal possiede la selezione delle pipeline e la preparazione
dell'ABI della convoluzione. Numeric condivide con CPU e Metal la geometria
per asse e valida le viste prima del dispatch. Le tabelle CPU dichiarano i
callback per nome, rendendo visibili le operazioni supportate.

Il planner Image conserva la politica del modello e usa le API pubbliche
di executor e risorse. Il costo delle varianti separa trasferimenti ripetuti,
cache host e costo base. L'espressione conserva gli stessi termini uint64
della formula precedente. Il fallback mantiene la possibilità di riprovare
dopo un errore temporaneo di capacità o costruzione dell'executor.

Il DiT conosce l'ordine dei blocchi e sceglie il prossimo gruppo. La cache
espone l'hint e conserva LRU, visite e contatori. Il bundle possiede file,
selezione dei tensori e limiti dell'advice. Il massimo16MiB è definito una
volta nell'API del bundle. Il chunk1MiB rimane una costante privata condivisa
dal worker cancellabile e dal consiglio per gruppo.

Il test della cache confronta il budget pubblico prima e dopo l'hint.
Il fixture privato del bundle intercetta soltanto le syscall per controllare
gli intervalli. La verifica evita il confronto byte per byte del layout
privato della cache, compresi i byte di padding della struttura C.
L'intercettazione variadica legge il tipo di puntatore effettivamente passato
alla syscall e applica successivamente la qualificazione const.

Le modifiche di questo riesame sono equivalenti nelle operazioni e nei limiti.
I sorgenti dei kernel sono identici allo snapshot qualificato. L'hint parte
nello stesso punto e con lo stesso limite.
Gli audit upstream sono fissati in
`docs/performance/metal-inference-20261102/qwen-image/ordered-conv-20261104/sources.json`
e `group-prefetch-20261105/sources.json`. I manifest sono stati verificati
prima del riesame. Diffusers `aea7b70be0f594bcdc15a6f2677ee2ca85e49e8e`
fissa il VAE e l'ordine dei blocchi DiT. Transformers
`d4fcbccd0966ace9505b9cbd3041d28ad822d8a7` fissa l'encoder. Strata
`46c152a1310f77cd2f0ce2c7cd213962df14fe83` e ggml
`dde879bc4f954b809892fd0d660515f42536394a` conservano i confronti specifici
di capacità, lifetime e suggerimenti mmap descritti nei report.

I file locali `build/qwen27-vision-server/` contengono log, PID e probe
diagnostici di una sessione precedente. Rimangono sul filesystem fra gli
artefatti di sviluppo ignorati.

Esiti, hash e limiti delle verifiche successive al riesame vengono registrati
nel manifest `group-prefetch-20261105/review-verification.json`.

Build prodotto e CPU47 con13skip PASS. Intervalli/budget dell'hint, ASan/UBSan,
planner, memoria, executor, risorse, lifetime, Image input/output/graph,
fondazioni Metal residenti e streamed, Flash SSD, bundle, HTTP locale e
contratti/rilascio PASS. Numeric Metal47 con3skip conserva soltanto le due
asserzioni sin/cos riprodotte sul controllo. Gemma480/480frontiere bit-exact
e vettori128PASS. I gate indipendenti Image testo/RGB/RGBA passano con35
artefatti identici e crescita swap0. Riserva2GiB conservata.

La qualifica nativa512e le cinque coppie prestazionali I-P05 mantengono gli
hash precedenti. La nuova qualifica riguarda i binari riesaminati e conserva
il limite aperto dell'oracle indipendente512.
