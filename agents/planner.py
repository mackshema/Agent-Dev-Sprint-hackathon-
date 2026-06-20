"""
agents/planner.py
Phase 1 (Classify) + Phase 2 (Plan) agent.
Uses gemini-2.5-flash-lite for classification (cheapest) and
gemini-2.5-flash for planning.
"""
import re
import os
import google.genai as genai
import config
from state import RunState


def _load_prompt(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", f"{name}_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_PLANNER_SYSTEM = _load_prompt("planner")


class PlanResult:
    def __init__(self, task_type: str, tools_needed: list[str],
                 research_required: bool, plan: list[str], raw: str):
        self.task_type         = task_type
        self.tools_needed      = tools_needed
        self.research_required = research_required
        self.plan              = plan
        self.raw               = raw


async def plan_task(task_description: str, state: RunState) -> PlanResult:
    """
    Phase 1+2: Classify the task and decompose it into a plan.
    Uses flash-lite for classify then flash for the plan.
    """
    print(f"\n  [Planner] Classifying & planning task...")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    prompt = (
        f"TASK:\n{task_description}\n\n"
        f"Follow the instructions exactly. Output TASK_TYPE, TOOLS_NEEDED, "
        f"RESEARCH_REQUIRED, and PLAN."
    )

    state.estimate_and_record_tokens(_PLANNER_SYSTEM + prompt, is_input=True,
                                     model=config.MODEL_CLASSIFY, phase="classify")
    try:
        resp = await client.aio.models.generate_content(
            model=config.MODEL_CLASSIFY,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_PLANNER_SYSTEM,
                max_output_tokens=300,
                temperature=0.1,
            ),
        )
        raw = resp.text or ""
    except Exception as e:
        print(f"  [Planner] Model error ({config.MODEL_CLASSIFY}): {e}")
        # Fallback to flash
        try:
            resp = await client.aio.models.generate_content(
                model=config.MODEL_PLAN,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=_PLANNER_SYSTEM,
                    max_output_tokens=300,
                    temperature=0.1,
                ),
            )
            raw = resp.text or ""
        except Exception as e2:
            print(f"  [Planner] Fallback also failed: {e2}")
            raw = "TASK_TYPE: ANALYSIS\nTOOLS_NEEDED: none\nRESEARCH_REQUIRED: NO\nPLAN:\n1. Solve the task."

    state.estimate_and_record_tokens(raw, is_input=False,
                                     model=config.MODEL_CLASSIFY, phase="classify")

    # ── Parse output ──────────────────────────────────────────────────────────
    task_type_m = re.search(r"TASK_TYPE:\s*(\w[\w\-]*)", raw, re.IGNORECASE)
    task_type   = (task_type_m.group(1).upper() if task_type_m else "ANALYSIS")

    tools_m     = re.search(r"TOOLS_NEEDED:\s*\[?([^\]\n]+)\]?", raw, re.IGNORECASE)
    tools_needed = []
    if tools_m:
        tools_needed = [t.strip().lower() for t in tools_m.group(1).split(",") if t.strip()]

    research_m       = re.search(r"RESEARCH_REQUIRED:\s*(YES|NO)", raw, re.IGNORECASE)
    research_required = (research_m.group(1).upper() == "YES") if research_m else False

    plan_lines = []
    plan_block = re.search(r"PLAN:\s*\n((?:\d+\..*\n?)+)", raw, re.IGNORECASE)
    if plan_block:
        plan_lines = [l.strip() for l in plan_block.group(1).strip().split("\n") if l.strip()]
    else:
        plan_lines = ["1. Analyse and solve the task completely."]

    print(f"  [Planner] Type={task_type} | Research={research_required} | Steps={len(plan_lines)}")
    return PlanResult(task_type, tools_needed, research_required, plan_lines, raw)
