"""
agents/researcher.py
Phase 3 — Research agent.
Calls web_search up to 2 times when RESEARCH_REQUIRED = YES.
No LLM tokens spent — uses the tools directly.
"""
from state import RunState
from tools.helper_tools import make_helper_tools


async def research_task(task_description: str, plan_steps: list[str],
                         state: RunState) -> str:
    """
    Phase 3: Execute up to 2 web searches and return a bullet-fact string.
    If research yields nothing useful, returns "RESEARCH_FACTS: none".
    """
    print(f"\n  [Researcher] Running up to 2 searches...")
    tools = make_helper_tools(state)
    web_search = tools[0]   # index 0

    all_facts: list[str] = []

    # Build targeted queries from the task + plan
    queries: list[str] = []
    queries.append(task_description[:200])   # query 1: task itself
    if plan_steps:
        # query 2: focus on the first concrete sub-goal
        queries.append(plan_steps[0].lstrip("0123456789. "))

    for i, q in enumerate(queries[:2]):
        result = await web_search(q)
        if result and "No search results" not in result and "Validation Error" not in result:
            # Extract bullet lines from result
            for line in result.split("\n"):
                stripped = line.strip()
                if stripped and stripped not in ("", "[DDG Instant Answer]",
                                                  "[Wikipedia Search Results]"):
                    all_facts.append(f"- {stripped}")

    if not all_facts:
        return "RESEARCH_FACTS: none"

    facts_str = "RESEARCH_FACTS:\n" + "\n".join(all_facts[:20])  # cap at 20 facts
    print(f"  [Researcher] Extracted {len(all_facts)} facts.")
    return facts_str
