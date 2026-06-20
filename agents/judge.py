"""
agents/judge.py
Phase 6 — Judge agent.
Uses gemini-2.5-flash to score (0–100) and decide whether to loop or submit.
"""
import os
import re
import google.genai as genai

import config
from state import RunState
from agents._rate_limiter import rate_limit_gate


def _load_prompt(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", f"{name}_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_JUDGE_SYSTEM = _load_prompt("judge")


class JudgeResult:
    def __init__(self, score: int, reason: str, weakest_axis: str):
        self.score       = score
        self.reason      = reason
        self.weakest_axis = weakest_axis
        self.should_submit = score >= 85


async def judge_solution(
    task_description: str,
    solution: str,
    state: RunState,
) -> JudgeResult:
    """
    Phase 6: Score the solution 0–100. Returns a JudgeResult.
    """
    print(f"\n  [Judge] Scoring solution with {config.MODEL_JUDGE}...")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    prompt = (
        f"TASK:\n{task_description}\n\n"
        f"SOLUTION:\n{solution[:3000]}"
    )

    state.estimate_and_record_tokens(_JUDGE_SYSTEM + prompt, is_input=True,
                                     model=config.MODEL_JUDGE, phase="judge")
    try:
        await rate_limit_gate("Judge")
        resp = await client.aio.models.generate_content(
            model=config.MODEL_JUDGE,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_JUDGE_SYSTEM,
                max_output_tokens=150,
                temperature=0.0,
            ),
        )
        raw = resp.text or ""
    except Exception as e:
        print(f"  [Judge] Error: {e}")
        raw = "CONFIDENCE_SCORE: 80\nREASON: Unable to judge, proceeding with submission.\nWEAKEST_AXIS: Completeness"

    state.estimate_and_record_tokens(raw, is_input=False,
                                     model=config.MODEL_JUDGE, phase="judge")

    # Parse
    score_m  = re.search(r"CONFIDENCE_SCORE:\s*(\d+)", raw, re.IGNORECASE)
    score    = int(score_m.group(1)) if score_m else 75

    reason_m = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE)
    reason   = reason_m.group(1).strip() if reason_m else ""

    axis_m   = re.search(r"WEAKEST_AXIS:\s*(Completeness|Accuracy|Clarity|Correctness)", raw, re.IGNORECASE)
    axis     = axis_m.group(1) if axis_m else "Completeness"

    icon = "[OK]" if score >= 85 else "[retry]"
    print(f"  [Judge] {icon} Score: {score}/100 | Reason: {reason}")

    return JudgeResult(score, reason, axis)
