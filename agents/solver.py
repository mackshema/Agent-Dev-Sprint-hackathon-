"""
agents/solver.py
Phase 4 — Solver agent.
Uses gemini-2.5-pro (most capable) to execute the plan and produce a solution.
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
    Phase 4: Use gemini-2.5-pro to solve the task with tool access.
    Returns the SOLUTION string (clean, ready for submission).
    """
    print(f"\n  [Solver] Solving with {config.MODEL_SOLVE}...")

    helper_tools = make_helper_tools(state)

    # Build the solver agent with tool access
    solver_agent = LlmAgent(
        name=f"{config.AGENT_NAME}_solver",
        model=config.MODEL_SOLVE,
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

    # Compose the full input message for the solver
    plan_str = "\n".join(plan)
    memory_section = f"\n\nPAST SIMILAR TASKS (for reference only):\n{memory_context}" if memory_context else ""
    message = (
        f"TASK:\n{task_description}\n\n"
        f"TASK_TYPE: {task_type}\n\n"
        f"PLAN:\n{plan_str}\n\n"
        f"{research_facts}"
        f"{memory_section}\n\n"
        f"Execute the plan completely. Output your answer under SOLUTION:."
    )

    state.estimate_and_record_tokens(message, is_input=True,
                                     model=config.MODEL_SOLVE, phase="solve")

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)]
    )

    final_reply = ""
    try:
        async for event in runner.run_async(
            session_id=session_id, new_message=content, user_id="solver"
        ):
            if getattr(event, "turn_complete", False):
                if event.content and event.content.parts:
                    final_reply = event.content.parts[0].text or ""
                break
            # Capture any intermediate text in case turn_complete isn't fired
            if hasattr(event, "content") and event.content and event.content.parts:
                text = event.content.parts[0].text or ""
                if text:
                    final_reply = text
    except Exception as e:
        print(f"  [Solver] Runner error: {e}")
        final_reply = ""

    state.estimate_and_record_tokens(final_reply, is_input=False,
                                     model=config.MODEL_SOLVE, phase="solve")

    # Extract the SOLUTION section
    sol_match = re.search(r"SOLUTION:\s*\n?(.*)", final_reply, re.DOTALL | re.IGNORECASE)
    if sol_match:
        solution = sol_match.group(1).strip()
    else:
        solution = final_reply.strip()

    print(f"  [Solver] Solution generated ({len(solution)} chars).")
    return solution
