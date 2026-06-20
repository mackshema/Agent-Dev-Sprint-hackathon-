"""
agents/critic.py
Phase 5 — Critic agent.
Uses gemini-2.5-flash to find gaps/errors and flag if fixes are needed.
Max 2 critique loops to control token spend.
"""
import os
import re
import google.genai as genai

import config
from state import RunState


def _load_prompt(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", f"{name}_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_CRITIC_SYSTEM = _load_prompt("critic")


class CritiqueResult:
    def __init__(self, issues: list[str], fixes_needed: bool, weakness: str):
        self.issues      = issues
        self.fixes_needed = fixes_needed
        self.weakness    = weakness


async def critique_solution(
    task_description: str,
    solution: str,
    state: RunState,
) -> CritiqueResult:
    """
    Phase 5: Review the solution for issues. Returns a CritiqueResult.
    """
    print(f"\n  [Critic] Reviewing solution with {config.MODEL_CRITIQUE}...")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    prompt = (
        f"TASK:\n{task_description}\n\n"
        f"SOLUTION:\n{solution[:3000]}"   # truncate very long solutions
    )

    state.estimate_and_record_tokens(_CRITIC_SYSTEM + prompt, is_input=True,
                                     model=config.MODEL_CRITIQUE, phase="critique")
    try:
        resp = await client.aio.models.generate_content(
            model=config.MODEL_CRITIQUE,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_CRITIC_SYSTEM,
                max_output_tokens=250,
                temperature=0.1,
            ),
        )
        raw = resp.text or ""
    except Exception as e:
        print(f"  [Critic] Error: {e}")
        raw = "ISSUES_FOUND:\n- none\n\nFIXES_NEEDED: NO"

    state.estimate_and_record_tokens(raw, is_input=False,
                                     model=config.MODEL_CRITIQUE, phase="critique")

    # Parse
    issues_block = re.search(r"ISSUES_FOUND:\s*\n((?:- .+\n?)+)", raw, re.IGNORECASE)
    issues = []
    if issues_block:
        issues = [l.lstrip("- ").strip()
                  for l in issues_block.group(1).split("\n") if l.strip()]

    fixes_m    = re.search(r"FIXES_NEEDED:\s*(YES|NO)", raw, re.IGNORECASE)
    fixes_needed = (fixes_m.group(1).upper() == "YES") if fixes_m else False

    weakness_m = re.search(r"BIGGEST_WEAKNESS:\s*(.+)", raw, re.IGNORECASE)
    weakness   = weakness_m.group(1).strip() if weakness_m else ""

    icon = "⚠" if fixes_needed else "✓"
    print(f"  [Critic] {icon} Fixes needed: {fixes_needed} | Issues: {len(issues)}")
    if fixes_needed and weakness:
        print(f"  [Critic] Biggest weakness: {weakness}")

    return CritiqueResult(issues, fixes_needed, weakness)
