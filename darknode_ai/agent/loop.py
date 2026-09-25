"""The agentic loop -- ReAct-style JSON protocol over any chat provider.

Each step the model returns ONE JSON object: either a tool call or a final
answer. The loop parses it, runs the tool (via the registry + approval gate),
feeds the observation back, and repeats until a final answer or step budget.

The protocol is plain JSON (not vendor function-calling) so it works on any
base model served through Ollama/vLLM, from 13B to 500B. A bigger base simply
makes each decision better; the loop is identical.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

AGENT_PROTOCOL = """\
You are operating as an autonomous agent. Work in steps. At EACH step, reply
with EXACTLY ONE JSON object and nothing else, in one of these two shapes:

  {"thought": "...", "tool": "<tool_name>", "args": { ... }}
  {"thought": "...", "final": "<your complete answer>"}

Rules:
- Use tools to gather facts and take actions; do not invent tool results.
- Prefer rag_search for security/knowledge facts before answering from memory.
- Tools marked [needs approval] may be denied; if denied, adapt.
- When you have enough to answer the task, return "final".
- Keep going until the task is done. One JSON object per reply, no prose around it.

Available tools:
%s
"""

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_action(text: str) -> dict:
    """Extract the JSON action from a model reply. Tolerant of code fences/prose."""
    # strip code fences
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    candidates = []
    m = _JSON_RE.search(t)
    if m:
        candidates.append(m.group(0))
    candidates.append(t)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and ("tool" in obj or "final" in obj):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    # no parseable action -> treat the whole reply as a final answer
    return {"final": text.strip()}


@dataclass
class Step:
    thought: str
    tool: str | None
    args: dict
    observation: str | None
    approved: bool | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    stopped: str = "final"  # "final" | "budget"


@dataclass
class AgentLoop:
    provider: object                      # .chat(user_content, history=, **kw) -> str
    registry: object                      # ToolRegistry
    approve: Callable[[str, dict], bool] = lambda name, args: False  # deny by default
    max_steps: int = 8
    system: str | None = None             # optional system override for the provider

    def _ask(self, content: str, history: list) -> str:
        kw = {}
        if self.system is not None:
            kw["system"] = self.system
        # OllamaProvider.chat ignores unknown kw via **overrides; system handled by caller
        try:
            return self.provider.chat(content, history=history)
        except TypeError:
            return self.provider.chat(content)

    def run(self, task: str) -> AgentResult:
        protocol = AGENT_PROTOCOL % self.registry.catalog()
        history: list = []
        content = f"{protocol}\n\nTASK: {task}\n\nBegin. Reply with one JSON object."
        result = AgentResult(answer="", stopped="budget")
        for _ in range(self.max_steps):
            reply = self._ask(content, history)
            action = parse_action(reply)
            if "final" in action and "tool" not in action:
                result.answer = str(action.get("final", "")).strip()
                result.stopped = "final"
                result.steps.append(Step(action.get("thought", ""), None, {}, None))
                return result
            name = action.get("tool", "")
            args = action.get("args", {}) or {}
            tool = self.registry.get(name)
            approved = None
            if tool is not None and tool.mutating:
                approved = bool(self.approve(name, args))
                if not approved:
                    observation = (f"DENIED: '{name}' needs approval and was not "
                                   "granted. Choose a read-only approach or stop.")
                else:
                    observation = self.registry.call(name, args)
            else:
                observation = self.registry.call(name, args)
            result.tools_used.append(name)
            result.steps.append(Step(action.get("thought", ""), name, args,
                                     observation, approved))
            history.append({"role": "user", "content": content})
            history.append({"role": "assistant", "content": reply})
            content = f"OBSERVATION from {name}:\n{observation}\n\nNext JSON action."
        # budget exhausted
        result.answer = result.steps[-1].observation if result.steps else ""
        return result
