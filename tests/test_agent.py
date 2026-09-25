"""Agent harness tests. No torch/network -- a scripted fake provider."""
import json

import pytest

from darknode_ai.agent.tools import Tool, ToolRegistry, default_tools
from darknode_ai.agent.loop import AgentLoop, parse_action
from darknode_ai.agent.agent import DarknodeAgent
from darknode_ai.retrieval.store import KnowledgeStore


class ScriptedProvider:
    """Returns queued replies in order, ignoring the prompt."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, content, history=None, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else json.dumps({"final": "done"})


def test_parse_action_variants():
    assert parse_action('{"tool":"x","args":{}}')["tool"] == "x"
    assert parse_action('```json\n{"final":"hi"}\n```')["final"] == "hi"
    assert parse_action('here you go {"final":"ok"} thanks')["final"] == "ok"
    # unparseable -> whole reply becomes the final answer
    assert parse_action("just prose")["final"] == "just prose"


def test_loop_runs_read_tool_then_finals():
    reg = default_tools()
    prov = ScriptedProvider([
        json.dumps({"thought": "list first", "tool": "list_dir", "args": {"path": "."}}),
        json.dumps({"thought": "done", "final": "there are files"}),
    ])
    loop = AgentLoop(provider=prov, registry=reg, max_steps=5)
    res = loop.run("what is here")
    assert res.stopped == "final"
    assert res.answer == "there are files"
    assert "list_dir" in res.tools_used


def test_mutating_tool_denied_by_default(tmp_path):
    reg = default_tools()
    target = tmp_path / "x.txt"
    prov = ScriptedProvider([
        json.dumps({"tool": "write_file", "args": {"path": str(target), "content": "hi"}}),
        json.dumps({"final": "could not write"}),
    ])
    loop = AgentLoop(provider=prov, registry=reg, max_steps=5)  # deny-by-default
    res = loop.run("write a file")
    assert not target.exists()  # write was blocked
    assert any(s.tool == "write_file" and s.approved is False for s in res.steps)


def test_mutating_tool_runs_when_approved(tmp_path):
    reg = default_tools()
    target = tmp_path / "y.txt"
    prov = ScriptedProvider([
        json.dumps({"tool": "write_file", "args": {"path": str(target), "content": "hello"}}),
        json.dumps({"final": "written"}),
    ])
    loop = AgentLoop(provider=prov, registry=reg, approve=lambda n, a: True, max_steps=5)
    res = loop.run("write a file")
    assert target.read_text() == "hello"
    assert any(s.tool == "write_file" and s.approved is True for s in res.steps)


def test_rag_tool_grounds_agent():
    store = KnowledgeStore()
    store.add("SMB signing blocks NTLM relay attacks.", "smb.txt", "authored")
    store.build()
    reg = default_tools(store=store)
    assert "rag_search" in reg.names()
    prov = ScriptedProvider([
        json.dumps({"tool": "rag_search", "args": {"query": "ntlm relay", "k": 1}}),
        json.dumps({"final": "Use SMB signing."}),
    ])
    loop = AgentLoop(provider=prov, registry=reg, max_steps=5)
    res = loop.run("how to stop ntlm relay")
    assert res.steps[0].tool == "rag_search"
    assert "SMB signing" in res.steps[0].observation


def test_unknown_tool_does_not_crash():
    reg = default_tools()
    prov = ScriptedProvider([
        json.dumps({"tool": "nonexistent", "args": {}}),
        json.dumps({"final": "recovered"}),
    ])
    res = AgentLoop(provider=prov, registry=reg, max_steps=5).run("t")
    assert res.answer == "recovered"
    assert "ERROR: unknown tool" in res.steps[0].observation


def test_budget_exhaustion_stops():
    reg = default_tools()
    # always returns a tool call -> never finals
    prov = ScriptedProvider([json.dumps({"tool": "list_dir", "args": {}})] * 10)
    res = AgentLoop(provider=prov, registry=reg, max_steps=3).run("loop forever")
    assert res.stopped == "budget"
    assert len(res.tools_used) == 3


def test_darknode_agent_autonomous_flag(tmp_path):
    store = KnowledgeStore()
    store.add("x", "a.txt", "authored")
    store.build()
    target = tmp_path / "z.txt"
    prov = ScriptedProvider([
        json.dumps({"tool": "write_file", "args": {"path": str(target), "content": "ok"}}),
        json.dumps({"final": "done"}),
    ])
    agent = DarknodeAgent(prov, store=store, autonomous=True, max_steps=4)
    agent.run("write z")
    assert target.read_text() == "ok"
