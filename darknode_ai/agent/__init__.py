"""Darknode AI — agent harness.

Parameter count makes each step smarter; this harness is what makes Darknode an
*agent* at all. It is a provider-agnostic tool-calling loop (ReAct-style JSON
protocol) that lets the Darknode foundation model plan, call tools, observe
results, and iterate to complete multi-step tasks -- the same shape as
Claude-Code-like coding/security agents.

It composes with everything else in the repo: the foundation model (provider),
the RAG knowledge store (a read-only tool), and the Darknode persona. Powerful
tools (shell, file writes, network) are gated behind an approval callback that
denies by default -- authorized, human-in-the-loop by construction.
"""
from darknode_ai.agent.tools import Tool, ToolRegistry, default_tools
from darknode_ai.agent.loop import AgentLoop, AGENT_PROTOCOL
from darknode_ai.agent.agent import DarknodeAgent

__all__ = ["Tool", "ToolRegistry", "default_tools", "AgentLoop",
           "AGENT_PROTOCOL", "DarknodeAgent"]
