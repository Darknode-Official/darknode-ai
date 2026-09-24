# darknode-web integration

Use the from-scratch **Darknode AI** model inside the darknode-web console
without replacing the existing multi-provider assistant (`webai.js`).

## What this is
`darknode-native.js` is a drop-in ES module that talks to the Darknode AI
serving layer (`darknode_ai/serve`). It exposes the native model as a library
call and as a small floating panel. It changes nothing on the site until you
import it.

## 1. Run the model server
```bash
pip install -e ".[serve]"
DARKNODE_CKPT=runs/darknode-small/best.pt \
DARKNODE_TOKENIZER=runs/tokenizer.json \
darknode-ai-serve            # serves http://127.0.0.1:8799  (panel at /)
```
For remote use, host it behind HTTPS and set the URL in the browser:
```js
localStorage.setItem('darknode_ai_url', 'https://darknode-ai.your-host');
```

## 2a. Quickest: floating panel on any page
Copy `darknode-native.js` into `public/js/` and add to a page (or the console):
```js
import { darknodeNative } from '/js/darknode-native.js';
darknodeNative.mountPanel();     // bottom-right panel, no other changes
```

## 2b. Wire as a provider in webai.js
`webai.js` already routes to providers. Add the native model as one more option
by calling the library where you produce a completion:
```js
import { darknodeNative } from '/js/darknode-native.js';
// when the selected provider is 'darknode-native':
const reply = await darknodeNative.generate(userPrompt, { temperature: 0.7 });
```
Add a matching entry to the model/provider selector so users can pick
"Darknode AI (native model)".

## Notes
- The model never executes actions; `/api/generate` only returns text. Treat
  output as a draft to verify against evidence.
- CORS is open on the server for local development; lock `allow_origins` down to
  your console origin before exposing it.
- The native model is small and trained from scratch — see the top-level
  `MODEL_CARD.md` for honest scope. It complements, not replaces, the hosted
  providers in `webai.js`.
```
