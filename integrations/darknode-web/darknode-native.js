// Darknode AI — native model provider for darknode-web.
//
// Drop-in, non-breaking: it does NOT replace the existing multi-provider
// assistant (webai.js). It adds the from-scratch Darknode AI model as an extra
// provider that talks to the Darknode AI serving layer (darknode_ai/serve).
//
// Two ways to use it:
//   1. As a library:  import { darknodeNative } from './darknode-native.js'
//                      const text = await darknodeNative.generate(prompt, opts)
//   2. As a widget:   darknodeNative.mountPanel(document.body)  // floating panel
//
// Configure the server URL (default http://127.0.0.1:8799) via:
//   localStorage.setItem('darknode_ai_url', 'https://your-darknode-ai-host')

const DEFAULT_URL = "http://127.0.0.1:8799";

function baseUrl() {
  try { return (localStorage.getItem("darknode_ai_url") || "").trim() || DEFAULT_URL; }
  catch (_) { return DEFAULT_URL; }
}

export const darknodeNative = {
  id: "darknode-native",
  label: "Darknode AI (native model)",

  async health() {
    const r = await fetch(baseUrl() + "/api/health");
    return r.json();
  },

  // Prompts are sent to a model that never executes actions. Output is a draft.
  async generate(prompt, opts = {}) {
    const r = await fetch(baseUrl() + "/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_new_tokens: opts.maxTokens ?? 200,
        temperature: opts.temperature ?? 0.8,
        top_k: opts.topK ?? 40,
        top_p: opts.topP ?? 0.95,
      }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || ("Darknode AI server error " + r.status));
    }
    return (await r.json()).completion;
  },

  // Minimal floating panel so the native model is usable from any page without
  // touching webai.js. Call once, e.g. darknodeNative.mountPanel(document.body).
  mountPanel(root = document.body) {
    if (document.getElementById("dn-native-panel")) return;
    const el = document.createElement("div");
    el.id = "dn-native-panel";
    el.style.cssText =
      "position:fixed;right:18px;bottom:18px;width:360px;max-height:70vh;z-index:99999;" +
      "background:#121820;color:#e6edf3;border:1px solid #1e2a36;border-radius:12px;" +
      "font:13px ui-monospace,monospace;display:flex;flex-direction:column;overflow:hidden";
    el.innerHTML =
      '<div style="padding:10px 12px;border-bottom:1px solid #1e2a36;font-weight:700">' +
      'DARKNODE <span style="color:#39d0d8">AI</span> · native model' +
      '<span id="dn-x" style="float:right;cursor:pointer;color:#8aa0b2">close</span></div>' +
      '<div id="dn-out" style="flex:1;overflow:auto;padding:10px;white-space:pre-wrap;color:#8aa0b2">' +
      'Ask the from-scratch Darknode AI model. Draft only; no action executed.</div>' +
      '<div style="padding:8px;border-top:1px solid #1e2a36;display:flex;gap:6px">' +
      '<input id="dn-in" placeholder="Triage a login..." style="flex:1;background:#0e141b;color:#e6edf3;' +
      'border:1px solid #1e2a36;border-radius:8px;padding:8px"/>' +
      '<button id="dn-go" style="background:#39d0d8;color:#04222b;border:0;border-radius:8px;padding:8px 12px;font-weight:700;cursor:pointer">Ask</button></div>';
    root.appendChild(el);
    const out = el.querySelector("#dn-out");
    const input = el.querySelector("#dn-in");
    el.querySelector("#dn-x").onclick = () => el.remove();
    const ask = async () => {
      const p = input.value.trim(); if (!p) return;
      out.textContent = "analyzing...";
      try { out.textContent = await this.generate("<|user|> " + p + "\n<|assistant|>\n"); }
      catch (e) { out.textContent = "Server unavailable: " + e.message; }
    };
    el.querySelector("#dn-go").onclick = ask;
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
  },
};

export default darknodeNative;
