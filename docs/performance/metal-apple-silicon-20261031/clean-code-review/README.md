# Revisione Clean Code T04–T05

31 ottobre 2026. La revisione comprende runtime, gestione dello stato,
strumenti di qualifica e codice collegato ai test. Gli artefatti delle fasi
precedenti restano conservati con i risultati e gli hash originari, compreso
lo stato del working tree al momento delle misure.

## Correzioni e responsabilità

- Il backend Metal espone `ds4_gpu_qwen4_vision_scratch_bytes`: la stima dei
  suoi buffer non appartiene alla struttura dei pesi. Il runtime usa questa
  API e non conosce il conteggio delle allocazioni dell'encoder.
- Gli export MTP completi, skill e device scrivono zero per la proposta
  inattiva, senza leggere scratch non valido o modificare la sessione.
  I test usano le API di snapshot senza azzerare manualmente i buffer.
  Una prova avvelena lo scratch con NaN e un token non valido e verifica
  export identici, stato vivo preservato e restore corretto.
- `VisionReference` possiede selezione della famiglia, caricamento differito
  e inferenza CPU. Usa `set_attn_implementation`, il metodo pubblico di
  Transformers. Il chiamante usa `encode` senza accedere alla configurazione
  interna del modello. L'encoder CPU viene caricato dopo la prova Metal.
- Parsing degli artefatti, metriche e conversione dei pesi sono condivisi in
  `tools/qwen_vision_reference.py`; il qualificatore non importa dalla
  cartella dei test. Layout incoerenti, file troncati, valori non finiti e
  array vuoti vengono rifiutati prima del confronto numerico.
- I report persistono anche dopo errori o interruzioni. I timeout restano
  `NOT_VERIFIED`; una sola comparazione fallita impedisce `PASS`. Il profilo
  del report è impostato dal suo costruttore anche se la build non parte.
- La configurazione dei test live viene applicata soltanto dopo la convalida
  completa. Gli errori mantengono le opzioni precedenti. La CLI vision
  rifiuta budget invalidi prima di aprire file o inizializzare il backend.
- Playwright attende la visibilità del menu mobile dopo il clic.

I kernel aritmetici Metal e le soglie numeriche non sono stati modificati
nella revisione. Non viene dedotto alcuno speedup dalle prove di correttezza.

## Verifica finale

[Report completo](native/verification.json): **25 gruppi PASS**, con build
CPU isolata, build Metal, risorse, primitive numeriche, lifecycle, server,
sampling, snapshot MTP/vision e contratti del qualificatore. Il gruppo dei
contratti vision esegue **13 unit test** senza GPU o modello preaddestrato.

La suite frontend comprende unit test e Playwright con provider mock:
**1210 PASS, 46 SKIP**, copertura statement e branch **100%**.
[Log frontend](native/frontend-mock.log).

Gli encoder reali sono stati provati in sequenza, una sola immagine da
121 token, cap 128, con gli stessi pesi GGUF del riferimento indipendente
Transformers. La soglia minima per token resta **0,99**.

| Encoder | Patch identiche: coseno minimo | Percorso immagine: coseno minimo |
| --- | ---: | ---: |
| [Flash 2560](flash-vision.json) | 0,9999999997 | 0,9995476 |
| [Qwen27 5120](qwen27-vision.json) | 0,9999999997 | 0,9999743 |

I report includono sorgenti, binari, immagini, versioni e identità GGUF;
i log e i sorgenti sono stati ricontrollati contro gli hash registrati.
Le prove sono state eseguite prima dei commit, sui medesimi file identificati
per SHA-256. I commit di codice sono:

- `b6f2e42` — runtime e test nativi;
- `b8d2574` — adapter vision, artefatti e report di qualifica;
- `b9ea4cb` — sincronizzazione del test frontend.

## Riproduzione e limiti

Dalla radice del repository, con le dipendenze già presenti:

```sh
python3 tools/qualify_metal_foundations.py \
  --output build/metal-qualification/review-new \
  --features --cpu-build --frontend
```

Le prove vision usano `python -m tools.qualify_qwen_metal_vision`, come nella
[procedura iniziale](../README.md#riproduzione), e vanno eseguite in sequenza.
Le qualifiche LLM complete, MTP con sidecar reale, integrazione server/client
con i GGUF completi, CUDA/ROCm e altri Mac restano `NOT_VERIFIED` e dipendono
dai gate T06 della roadmap. Le prove mock e sintetiche non chiudono tali gate.
