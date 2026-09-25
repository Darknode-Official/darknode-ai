"""DarknodeAgent -- ties the foundation model, tools, RAG, and persona into one
agentic runner. This is the entrypoint the serving layer / Nexus / CLI use to
run a multi-step task.

    from darknode_ai.foundation.provider import OllamaProvider
    from darknode_ai.retrieval.store import KnowledgeStore
    from darknode_ai.agent import DarknodeAgent

    store = KnowledgeStore.load("runs/knowledge.json")
    agent = DarknodeAgent(OllamaProvider(), store=store, autonomous=False)
    out = agent.run("Audit ./config for exposed secrets and summarize findings.")
    print(out.answer)
"""
from __future__ import annotations

from typing import Callable

from darknode_ai.agent.tools import default_tools, ToolRegistry
from darknode_ai.agent.loop import AgentLoop, AgentResult
from darknode_ai.foundation.persona import DARKNODE_SYSTEM


class DarknodeAgent:
    """A Darknode agent over a chat provider.

    approve: callback (tool_name, args) -> bool for mutating tools. If omitted,
    `autonomous=True` approves everything (use only in a sandbox / authorized
    engagement); otherwise mutating tools are denied (read-only, safe default).
    """

    def __init__(self, provider, store=None, tools: ToolRegistry | None = None,
                 approve: Callable[[str, dict], bool] | None = None,
                 autonomous: bool = False, max_steps: int = 8):
        self.provider = provider
        self.registry = tools or default_tools(store=store)
        if approve is not None:
            gate = approve
        elif autonomous:
            gate = lambda name, args: True
        else:
            gate = lambda name, args: False
        self.loop = AgentLoop(provider=provider, registry=self.registry,
                              approve=gate, max_steps=max_steps,
                              system=DARKNODE_SYSTEM)

    def run(self, task: str) -> AgentResult:
        return self.loop.run(task)
