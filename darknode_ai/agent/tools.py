"""Agent tools -- the capabilities Darknode can invoke during a task.

Each Tool declares whether it is `mutating` (writes files, runs commands, or
reaches the network). Read-only tools run freely; mutating tools are routed
through the agent's approval callback, which denies by default. This keeps the
agent authorized and human-in-the-loop even when driven by a large, capable
base model.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., str]
    params: dict = field(default_factory=dict)   # name -> short description
    mutating: bool = False                        # needs approval before running

    def spec(self) -> str:
        args = ", ".join(f"{k} ({v})" for k, v in self.params.items()) or "none"
        tag = " [needs approval]" if self.mutating else ""
        return f"- {self.name}{tag}: {self.description} | args: {args}"


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def catalog(self) -> str:
        return "\n".join(t.spec() for t in self._tools.values())

    def call(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self.names())}"
        try:
            return str(tool.fn(**(args or {})))
        except TypeError as e:
            return f"ERROR: bad arguments for '{name}': {e}"
        except Exception as e:  # tools must never crash the loop
            return f"ERROR: tool '{name}' failed: {e}"


# -- built-in tools ----------------------------------------------------------

def _read_file(path: str, max_bytes: int = 20000) -> str:
    p = Path(path)
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    data = p.read_text(encoding="utf-8", errors="replace")
    return data[:max_bytes] + ("\n...[truncated]" if len(data) > max_bytes else "")


def _list_dir(path: str = ".") -> str:
    p = Path(path)
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir()))


def _write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def _run_shell(command: str, timeout: int = 60) -> str:
    r = subprocess.run(command, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    return f"exit={r.returncode}\n{out[:20000]}"


def rag_tool(store) -> Tool:
    """A read-only retrieval tool backed by a KnowledgeStore."""
    def _search(query: str, k: int = 4) -> str:
        hits = store.search(query, k=int(k))
        if not hits:
            return "no matching source (treat as UNKNOWN)"
        return "\n".join(f"RETRIEVED [{h.doc.source}]: {h.doc.text[:300]}" for h in hits)
    return Tool("rag_search", "search Darknode's knowledge base for grounded facts",
                _search, {"query": "search text", "k": "results (default 4)"},
                mutating=False)


def default_tools(store=None) -> ToolRegistry:
    """The standard toolset. `store` enables rag_search when provided."""
    reg = ToolRegistry([
        Tool("read_file", "read a text file", _read_file, {"path": "file path"}),
        Tool("list_dir", "list a directory", _list_dir, {"path": "dir (default .)"}),
        Tool("write_file", "create/overwrite a text file", _write_file,
             {"path": "file path", "content": "text"}, mutating=True),
        Tool("run_shell", "run a shell command", _run_shell,
             {"command": "shell command", "timeout": "seconds (default 60)"},
             mutating=True),
    ])
    if store is not None:
        reg.add(rag_tool(store))
    return reg
