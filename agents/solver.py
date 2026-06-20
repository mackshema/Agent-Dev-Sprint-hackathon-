"""
agents/solver.py
Phase 4 -- Solver agent.
Uses the best available model from config.SOLVE_FALLBACK_CHAIN.
Falls back down the chain automatically on quota/rate-limit errors.
Has access to calculate() and run_python() tools for accurate execution.
"""
import os
import re
import google.genai as genai
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

import config
from state import RunState
from tools.helper_tools import make_helper_tools


def _load_prompt(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", f"{name}_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_SOLVER_SYSTEM = _load_prompt("solver")


def _is_quota_error(e: Exception) -> bool:
    err = repr(e)
    return any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "quota", "limit: 0"))


async def _run_solver_with_model(
    model: str,
    message: str,
    session_id: str,
    state: RunState,
) -> str:
    """Run the ADK solver with a specific model. Returns reply text or raises."""
    helper_tools = make_helper_tools(state)

    solver_agent = LlmAgent(
        name=f"{config.AGENT_NAME}_solver",
        model=model,
        instruction=_SOLVER_SYSTEM,
        tools=helper_tools,
    )

    sessions = InMemorySessionService()
    await sessions.create_session(
        session_id=session_id,
        app_name=f"{config.AGENT_NAME}_solver",
        user_id="solver",
    )
    runner = Runner(
        agent=solver_agent,
        session_service=sessions,
        app_name=f"{config.AGENT_NAME}_solver",
    )

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)]
    )

    final_reply = ""
    async for event in runner.run_async(
        session_id=session_id, new_message=content, user_id="solver"
    ):
        if getattr(event, "turn_complete", False):
            if event.content and event.content.parts:
                final_reply = event.content.parts[0].text or ""
            break
        if hasattr(event, "content") and event.content and event.content.parts:
            text = event.content.parts[0].text or ""
            if text:
                final_reply = text

    return final_reply


async def solve_task(
    task_description: str,
    task_type: str,
    plan: list[str],
    research_facts: str,
    memory_context: str,
    state: RunState,
    session_id: str,
) -> str:
    """
    Phase 4: Solve the task using the best available model.
    Automatically falls back down SOLVE_FALLBACK_CHAIN on quota errors.
    Returns the SOLUTION string (clean, ready for submission).
    """
    plan_str = "\n".join(plan)
    memory_section = (
        f"\n\nPAST SIMILAR TASKS (for reference only):\n{memory_context}"
        if memory_context else ""
    )
    message = (
        f"TASK:\n{task_description}\n\n"
        f"TASK_TYPE: {task_type}\n\n"
        f"PLAN:\n{plan_str}\n\n"
        f"{research_facts}"
        f"{memory_section}\n\n"
        f"Execute the plan completely. Output your answer under SOLUTION:."
    )

    # Deduplicate fallback chain while preserving order
    seen = set()
    chain = []
    for m in config.SOLVE_FALLBACK_CHAIN:
        if m not in seen:
            seen.add(m)
            chain.append(m)

    final_reply = ""
    used_model = chain[0]

    for model in chain:
        print(f"\n  [Solver] Solving with {model}...")
        state.estimate_and_record_tokens(message, is_input=True,
                                         model=model, phase="solve")
        try:
            final_reply = await _run_solver_with_model(
                model, message, f"{session_id}_{model.replace('-', '_')}", state
            )
            used_model = model
            break   # success -- stop trying fallbacks
        except Exception as e:
            if _is_quota_error(e):
                print(f"  [Solver] {model} quota exhausted -- trying next model...")
                continue
            else:
                print(f"  [Solver] {model} error (non-quota): {e}")
                final_reply = ""
                used_model = model
                break

    state.estimate_and_record_tokens(final_reply, is_input=False,
                                     model=used_model, phase="solve")

    # Extract the SOLUTION section
    sol_match = re.search(r"SOLUTION:\s*\n?(.*)", final_reply, re.DOTALL | re.IGNORECASE)
    solution = sol_match.group(1).strip() if sol_match else final_reply.strip()

    print(f"  [Solver] Solution generated ({len(solution)} chars) using {used_model}.")
    return solution
