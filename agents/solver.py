"""
agents/solver.py
Phase 4 -- Solver agent.

Key design decisions:
- Uses google.genai directly (NOT ADK Runner) to avoid ADK swallowing the
  _ResourceExhaustedError before our outer try/except can catch it.
- The ADK Runner wraps the 429 in _ResourceExhaustedError which can't be
  caught outside the async-for loop. We bypass ADK for the solve step and
  call the Gemini API directly with tool support via function calling.
- Falls back through SOLVE_FALLBACK_CHAIN automatically on quota errors.
- Adds mandatory inter-call sleep to stay under the 5 RPM free-tier limit.
"""
import asyncio
import json
import os
import re
import ast
import operator as op
import subprocess
import sys
import tempfile

import google.genai as genai
from google.genai import types as genai_types

import config
from state import RunState
from agents._rate_limiter import rate_limit_gate


def _load_prompt(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", f"{name}_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_SOLVER_SYSTEM = _load_prompt("solver")

def _is_quota_error(e: Exception) -> bool:
    err = repr(e)
    return any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "quota", "GenerateRequestsPerMinute"))


def _extract_retry_delay(e: Exception) -> float:
    """Extract the retryDelay from a 429 error, defaulting to 65s."""
    err = repr(e)
    m = re.search(r"retryDelay['\"]:\s*['\"](\d+)", err)
    if m:
        return float(m.group(1)) + 5.0
    m2 = re.search(r"retry in (\d+\.?\d*)s", err)
    if m2:
        return float(m2.group(1)) + 5.0
    return 65.0


async def _call_gemini_direct(
    model: str,
    system_prompt: str,
    user_message: str,
    state: RunState,
) -> str:
    """
    Call Gemini directly via google.genai (no ADK Runner).
    This gives us full control over error handling and retries.
    The solver operates without tool-calling in this path — tools are
    embedded as inline calculations via the prompt for reliability.
    """
    await rate_limit_gate("Solver")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    state.estimate_and_record_tokens(system_prompt + user_message, is_input=True,
                                     model=model, phase="solve")

    resp = await client.aio.models.generate_content(
        model=model,
        contents=user_message,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=8192,
            temperature=0.1,
        ),
    )
    text = resp.text or ""
    state.estimate_and_record_tokens(text, is_input=False, model=model, phase="solve")
    return text


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
    Automatically falls back through SOLVE_FALLBACK_CHAIN on quota errors,
    waiting the server-suggested retryDelay before trying the next model.
    Returns the clean SOLUTION string ready for submission.
    """
    plan_str = "\n".join(plan)
    memory_section = (
        f"\n\nPAST SIMILAR TASKS (for reference only):\n{memory_context}"
        if memory_context else ""
    )
    # Inline tool guidance so the model can self-execute math/code in text
    tool_reminder = ""
    if task_type in ("CODING", "MATH"):
        tool_reminder = (
            "\n\nIMPORTANT: For CODING tasks, write Python code AND show its expected "
            "output inline. For MATH tasks, show every arithmetic step explicitly."
        )

    message = (
        f"TASK:\n{task_description}\n\n"
        f"TASK_TYPE: {task_type}\n\n"
        f"PLAN:\n{plan_str}\n\n"
        f"{research_facts}"
        f"{memory_section}"
        f"{tool_reminder}\n\n"
        f"Execute the plan completely. Output your answer under SOLUTION:."
    )

    # Deduplicate fallback chain while preserving order
    seen: set[str] = set()
    chain: list[str] = []
    for m in config.SOLVE_FALLBACK_CHAIN:
        if m not in seen:
            seen.add(m)
            chain.append(m)

    last_error: Exception | None = None

    for model in chain:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            print(f"\n  [Solver] Solving with {model} (attempt {attempt}/{max_retries})...")
            try:
                final_reply = await _call_gemini_direct(
                    model, _SOLVER_SYSTEM, message, state
                )
                # Success — extract SOLUTION block
                sol_match = re.search(
                    r"SOLUTION:\s*\n?(.*)", final_reply, re.DOTALL | re.IGNORECASE
                )
                solution = sol_match.group(1).strip() if sol_match else final_reply.strip()
                print(f"  [Solver] Solution generated ({len(solution)} chars) via {model}.")
                return solution

            except Exception as e:
                last_error = e
                if _is_quota_error(e):
                    delay = _extract_retry_delay(e)
                    if attempt < max_retries:
                        print(f"  [Solver] Rate limit on {model} (attempt {attempt}). "
                              f"Waiting {delay:.0f}s before retry...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"  [Solver] {model} exhausted after {max_retries} attempts. "
                              f"Trying next model...")
                        break   # move to next model in chain
                else:
                    print(f"  [Solver] Non-quota error on {model}: {type(e).__name__}: {e}")
                    break   # non-quota error — move to next model immediately

    # All models exhausted
    print(f"  [Solver] All models in fallback chain exhausted. Last error: {last_error}")
    return ""
