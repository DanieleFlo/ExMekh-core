# Indice: agentic

Indice generato; non modificare a mano.

## [Task: supporto Skill gerarchiche nel motore di inferenza senza full prefill](agent-api.md)

Una skill viene chiamata dal modello esattamente come un normale tool.

- righe 21–58: **1. Concetto generale di Skill** — Una skill viene chiamata dal modello esattamente come un normale tool.
- righe 59–96: **2. Registry dei tool statico** — Tutti i tool e tutte le skill devono essere presenti nel registry iniziale inviato al modello.
- righe 97–125: **3. Il runtime decide quali tool e skill sono utilizzabili** — Il motore di inferenza non deve implementare la semantica della gerarchia.
- righe 126–179: **4. Estensione API** — Mantenere il normale payload OpenAI-compatible e aggiungere le informazioni specifiche del sistema in un namespace:
- righe 180–215: **5. `allowed_tools` e `allowed_skills`** — `allowed_tools` contiene i tool ordinari utilizzabili nello scope corrente.
- righe 216–241: **6. Tool masking** — Durante una call il modello deve poter generare solamente uno dei nomi presenti in:
- righe 242–267: **7. Il path non serve all'inference engine** — Il modello può ricevere informazioni sul path corrente tramite le istruzioni della skill o altri messaggi costruiti dal runtime.
- righe 268–305: **8. Apertura di una skill** — Una skill viene aperta attraverso una normale tool call.
- righe 306–344: **9. Il `call_id` è un riferimento logico** — Non utilizzare il `call_id` stesso come rappresentazione fisica del checkpoint.
- righe 345–376: **10. Posizione esatta del checkpoint** — Il checkpoint associato alla skill deve rappresentare:
- righe 377–414: **11. Entrata nella skill** — Dopo aver salvato su SSD lo stato ricorrente e registrato la frontiera parent, il runtime può aggiungere i nuovi token necessari alla skill:
- righe 415–449: **12. Non serve un'operazione `enter`** — Non è necessario introdurre:
- righe 450–470: **13. Esecuzione interna della skill** — Durante l'esecuzione interna la sequence può continuare normalmente:
- righe 471–493: **14. Chiusura della skill** — La skill termina tramite il proprio tool di uscita e produce un risultato.
- righe 494–549: **15. API di return** — In fase di risalita il runtime invia:
- righe 550–582: **16. Semantica di `return`** — Quando riceve:
- righe 583–608: **17. History prima e dopo il return** — Prima:
- righe 609–656: **18. Prefill alla risalita** — Il risultato della skill deve essere processato nuovamente nel contesto parent.
- righe 657–703: **19. Implementazione della frontiera e checkpoint SSD** — La sessione deve rappresentare logicamente una struttura tipo:
- righe 704–752: **20. Skill annidate** — Il sistema deve supportare skill annidate arbitrariamente.
- righe 753–780: **21. Skill ricorsive** — Non assumere che ogni `skill_id` compaia una sola volta.
- righe 781–833: **22. API finale da supportare** — "agentic": {
- righe 783–802: **Caso normale** — "agentic": {
- righe 803–833: **Return da skill** — "agentic": {
- righe 834–865: **23. Compatibilità** — Il campo:
- righe 866–888: **24. Stato interno da verificare durante restore** — Il restore non riguarda necessariamente solo i tensori KV.
- righe 889–926: **25. Test fondamentale: apertura** — Creare un parent prefix molto lungo.
- righe 927–965: **26. Test fondamentale: return** — Esempio:
- righe 966–1031: **27. Test API sintetici, classificazione e masking** — Implementare una suite Python sottile che invii richieste HTTP fittizie alle API esistenti e concateni normali response/tool output/skill return.
- righe 1032–1056: **28. Metriche utili** — Aggiungere o verificare metriche equivalenti a:
- righe 1057–1084: **29. Logging utile** — Apertura:
- righe 1085–1147: **30. Requisito finale** — Il comportamento che vogliamo ottenere è:
- righe 1148–1164: **31. Implementazione e gate di validazione** — Il server usa il payload KV completo della sessione, comprendente full-attention,

## [Validazione checkpoint agentici Qwen3.6 — 2026-08-07](agentic-checkpoint-validation-2026-08-07.md)

`PASS` / `PROMOTE` per i task 10 e 11 sul percorso CUDA singola GPU.

- righe 3–19: **Esito** — `PASS` / `PROMOTE` per i task 10 e 11 sul percorso CUDA singola GPU.
- righe 20–45: **Comandi riproducibili** — make tests/test_agentic_checkpoint ds4-server ds4_test
- righe 46–68: **Gate lungo Q4_K_S target-only** — Il caso usa una frontiera parent da 10.000 token, seguita dalla SkillCall.
- righe 69–88: **Gate MTP** — Il gate MTP usa una frontiera corta mirata, perché il sidecar aggiorna la propria
- righe 89–123: **Robustezza e lifetime** — Il gate model-backed verifica che i seguenti errori non modifichino token o

## [ExMekh Agent](frontend.md)

Da root, avviare il server in un terminale:

- righe 8–31: **Avvio in WSL / Linux** — Da root, avviare il server in un terminale:
- righe 32–48: **Configurazione e stato** — Copiare `agent/.env.example` in `agent/.env` per personalizzare il provider.
- righe 49–76: **Modifica dei messaggi** — Il pulsante con la matita compare accanto a quello di copia quando si passa
- righe 77–96: **Confini dell'integrazione** — ExMekh-Lib possiede i messaggi e il risultato dell'esecuzione dei tool.
- righe 97–142: **Dipendenza ExMekh-Lib** — `agent/pyproject.toml` installa ExMekh-Lib dalla wheel versionata in `agent/wheels/`.
- righe 143–170: **Verifica** — Da `agent/`:
