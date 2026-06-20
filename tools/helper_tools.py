"""
tools/helper_tools.py
Accuracy-boosting helper tools: calculate, run_python, web_search.
"""
import ast
import asyncio
import operator as op
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse

import httpx
from state import RunState


def make_helper_tools(state: RunState):
    """Returns [web_search, calculate, run_python] tool functions."""

    # ── Web Search ─────────────────────────────────────────────────────────────
    async def web_search(query: str) -> str:
        """Search the internet for current facts, documentation, and answers.

        Args:
            query: The search query string.
        """
        if not query or not query.strip():
            return "Validation Error: 'query' cannot be empty."

        print(f"  [web_search] Query: '{query}'")
        state.estimate_and_record_tokens(query, is_input=True, phase="research")

        results = []

        # 1. DuckDuckGo Instant Answer API
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_redirect=1"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers={"User-Agent": "AgentArenaCompetitor/2.0"})
                if resp.status_code == 200:
                    data = resp.json()
                    abstract = data.get("AbstractText", "")
                    if abstract:
                        results.append(
                            f"[DDG Instant Answer]\n{abstract}\nSource: {data.get('AbstractURL', '')}\n"
                        )
                    # Also collect RelatedTopics summaries
                    for topic in data.get("RelatedTopics", [])[:3]:
                        text = topic.get("Text", "")
                        if text:
                            results.append(f"- {text}")
        except Exception as e:
            print(f"  [web_search] DDG error: {e}")

        # 2. Wikipedia Search API
        try:
            url = (
                f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                f"&srsearch={urllib.parse.quote(query)}&utf8=&format=json&srlimit=4"
            )
            headers = {"User-Agent": "AgentArenaCompetitor/2.0 (admin@agentarena.com)"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data    = resp.json()
                    hits    = data.get("query", {}).get("search", [])
                    if hits:
                        results.append("\n[Wikipedia Search Results]")
                        for r in hits:
                            title   = r.get("title", "")
                            snippet = re.sub(r'<[^>]+>', '', r.get("snippet", ""))
                            results.append(f"• {title}: {snippet}")
        except Exception as e:
            print(f"  [web_search] Wikipedia error: {e}")

        output = "\n".join(results) if results else "No search results retrieved."
        state.estimate_and_record_tokens(output, is_input=False, phase="research")
        return output

    # ── Safe Math Evaluator ───────────────────────────────────────────────────
    _ops = {
        ast.Add:  op.add,  ast.Sub:  op.sub,  ast.Mult: op.mul,
        ast.Div:  op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
        ast.FloorDiv: op.floordiv,
        ast.USub: op.neg,  ast.UAdd: lambda x: x,
    }

    def _eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):   # Python < 3.8
            return node.n
        if isinstance(node, ast.BinOp):
            return _ops[type(node.op)](_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            return _ops[type(node.op)](_eval_node(node.operand))
        raise TypeError(f"Unsupported node: {type(node).__name__}")

    async def calculate(expression: str) -> str:
        """Safely evaluate a numeric math expression using AST (no eval/exec).
        For complex algorithms, use run_python() instead.

        Args:
            expression: e.g. "2 + 3 * (4 - 1)" or "2 ** 10 / 3"
        """
        if not expression or not expression.strip():
            return "Validation Error: 'expression' cannot be empty."

        print(f"  [calculate] {expression}")
        state.estimate_and_record_tokens(expression, is_input=True, phase="solve")

        try:
            cleaned = re.sub(r'[^0-9+\-*/().\s**%//]', '', expression)
            tree    = ast.parse(cleaned, mode='eval')
            result  = str(_eval_node(tree.body))
        except Exception as e:
            result = (
                f"Error: {e}. "
                "For complex math, write a Python script and use run_python() instead."
            )

        state.estimate_and_record_tokens(result, is_input=False, phase="solve")
        return result

    # ── Python Sandbox Runner ─────────────────────────────────────────────────
    async def run_python(code: str) -> str:
        """Execute Python code in a subprocess sandbox and return stdout + stderr.
        MANDATORY for all CODING tasks. Never submit code that hasn't been run here.

        Args:
            code: Valid Python source code to execute.
        """
        if not code or not code.strip():
            return "Validation Error: 'code' cannot be empty."

        print(f"  [run_python] Executing {len(code)} bytes...")
        state.estimate_and_record_tokens(code, is_input=True, phase="solve")

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False,
                                         mode="w", encoding="utf-8") as f:
            f.write(code)
            tmp = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20.0)
                parts = []
                if stdout:
                    parts.append(f"STDOUT:\n{stdout.decode('utf-8', errors='replace')}")
                if stderr:
                    parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
                if not parts:
                    parts.append("Code executed successfully with no output.")
                res = "\n".join(parts)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                res = "Error: Execution timed out after 20 seconds."
        except Exception as e:
            res = f"Error: Failed to launch subprocess: {e}"
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

        state.estimate_and_record_tokens(res, is_input=False, phase="solve")
        return res

    return [web_search, calculate, run_python]
