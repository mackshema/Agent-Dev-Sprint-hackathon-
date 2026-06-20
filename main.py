"""
main.py — Arena Champion Agent Orchestrator
Multi-agent 6-phase workflow:
  Phase 1+2 → Planner  (flash-lite + flash)
  Phase 3   → Researcher (no LLM, just search APIs)
  Phase 4   → Solver   (gemini-2.5-pro, with tools)
  Phase 5   → Critic   (flash, ≤2 loops)
  Phase 6   → Judge    (flash, ≤2 loops)
  Submit    → submit_task()
"""
import asyncio
import json
import re
import sys

import config
from state import RunState
from memory.task_memory import TaskMemory
from tools.arena_tools import make_arena_tools, mcp_call

from agents.planner    import plan_task
from agents.researcher import research_task
from agents.solver     import solve_task
from agents.critic     import critique_solution
from agents.judge      import judge_solution


# ── Helpers ──────────────────────────────────────────────────────────────────

async def with_retry(coro_fn, max_retries: int = 5, label: str = ""):
    """Wrap an async callable with rate-limit retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_fn()
        except Exception as e:
            err = repr(e)
            if any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "quota", "limit")):
                delay = 65.0
                m = re.search(r"retry[Dd]elay':\s*'(\d+)", err) or \
                    re.search(r"retry in (\d+\.?\d*)s", err)
                if m:
                    delay = float(m.group(1)) + 2.0
                print(f"\n  [{label}] Rate limit hit (attempt {attempt}/{max_retries}). "
                      f"Waiting {delay:.0f}s...")
                await asyncio.sleep(delay)
            else:
                raise
    raise RuntimeError(f"{label}: Failed after {max_retries} retries due to rate limits.")


# ── Task Solver Pipeline ──────────────────────────────────────────────────────

async def run_task_pipeline(
    task_data: dict,
    state: RunState,
    memory: TaskMemory,
    arena_tools,
    task_idx: int,
) -> bool:
    """
    Run the full 6-phase pipeline for a single task.
    Returns True if submission was made.
    """
    register_agent, get_tasks, submit_task, skip_task = arena_tools

    task_id    = task_data.get("id", state.task_id)
    title      = task_data.get("title", "Unknown Task")
    description = task_data.get("description", title)
    level      = task_data.get("level", state.current_level)

    print(f"\n{'='*60}")
    print(f"  TASK: {title}")
    print(f"  Level: {level} | ID: {task_id}")
    print(f"{'='*60}")

    # ── Phase 1+2: Plan ───────────────────────────────────────────────────────
    plan_result = await with_retry(
        lambda: plan_task(description, state),
        label="Planner"
    )

    # ── Phase 3: Research (conditional) ───────────────────────────────────────
    research_facts = "RESEARCH_FACTS: none"
    if plan_result.research_required:
        print("\n  [Phase 3] Research required — running search...")
        research_facts = await research_task(description, plan_result.plan, state)
    else:
        print("\n  [Phase 3] Skipped — no research required.")

    # ── Recall memory context ─────────────────────────────────────────────────
    past = memory.recall_similar(title, plan_result.task_type, top_k=2)
    memory_context = ""
    if past:
        lines = []
        for r in past:
            lines.append(
                f"- Title: {r['title']} | Score: {r['score']}\n"
                f"  Solution preview: {r['solution'][:300]}..."
            )
        memory_context = "\n".join(lines)

    # ── Phase 4+5+6: Solve → Critique → Judge loop ───────────────────────────
    session_id = f"{state.run_id}_task{task_idx}"
    solution   = ""
    max_loops  = 2

    for loop in range(1, max_loops + 1):
        print(f"\n  ── Loop {loop}/{max_loops} ──────────────────────────────")

        # Phase 4: Solve
        solution = await with_retry(
            lambda sol=solution, rc=research_facts, mc=memory_context: solve_task(
                description, plan_result.task_type, plan_result.plan,
                rc, mc, state, f"{session_id}_loop{loop}"
            ),
            label="Solver"
        )

        if not solution:
            print("  [Solver] Empty solution — skipping task.")
            await skip_task(state.agent_id, task_id)
            return False

        # Phase 5: Critique
        critique = await with_retry(
            lambda s=solution: critique_solution(description, s, state),
            label="Critic"
        )

        # Phase 6: Judge
        judgment = await with_retry(
            lambda s=solution: judge_solution(description, s, state),
            label="Judge"
        )

        print(f"\n  Confidence: {judgment.score}/100 | Fixes needed: {critique.fixes_needed}")

        if judgment.should_submit:
            print(f"  ✓ Score {judgment.score} ≥ 85 — submitting.")
            break

        if loop == max_loops:
            print(f"  Max loops reached — submitting best answer (score={judgment.score}).")
            break

        if critique.fixes_needed:
            print(f"  ↻ Fixing: {critique.weakness or 'general improvements'}...")
            # Augment memory context with critique feedback for the next loop
            memory_context += (
                f"\n\nPREVIOUS ATTEMPT (score ~{judgment.score}):\n"
                f"{solution[:500]}\n"
                f"ISSUE TO FIX: {critique.weakness}"
            )

    # ── Submit ────────────────────────────────────────────────────────────────
    print(f"\n  [Submit] Submitting solution ({len(solution)} chars)...")
    submit_result = await with_retry(
        lambda s=solution: submit_task(state.agent_id, task_id, s),
        label="Submit"
    )

    # Save to memory
    score_m = re.search(r"score[:\s]*(\d+)", submit_result or "", re.IGNORECASE)
    actual_score = int(score_m.group(1)) if score_m else judgment.score
    memory.record(
        task_id=task_id,
        title=title,
        task_type=plan_result.task_type,
        solution=solution,
        score=actual_score,
        feedback=judgment.reason,
    )
    return True


# ── Main Loop ─────────────────────────────────────────────────────────────────

async def main():
    config.validate_config(exit_on_failure=True)

    state  = RunState()
    memory = TaskMemory()

    print("\n" + "="*60)
    print(f"  Arena Champion Agent: {config.AGENT_NAME}")
    print(f"  Stack:  {config.AGENT_STACK}")
    print(f"  Models: classify={config.MODEL_CLASSIFY}")
    print(f"          plan/critique/judge={config.MODEL_PLAN}")
    print(f"          solve={config.MODEL_SOLVE}")
    mem_stats = memory.stats()
    print(f"  Memory: {mem_stats['total']} past tasks | avg score {mem_stats['avg_score']:.1f}")
    print("="*60 + "\n")

    arena_tools = make_arena_tools(state)
    register_agent, get_tasks, submit_task, skip_task = arena_tools

    # ── Register ──────────────────────────────────────────────────────────────
    print("--- [Step 1] Registering Agent ---")
    reg_result = await with_retry(
        lambda: register_agent(config.AGENT_NAME, config.AGENT_STACK),
        label="Register"
    )
    print(f"[Register] {reg_result[:200]}")

    if not state.agent_id:
        print("ERROR: Registration failed. Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # ── Task Loop ─────────────────────────────────────────────────────────────
    task_count = 0
    while task_count < config.MAX_TURNS:
        task_count += 1
        print(f"\n--- [Task {task_count}/{config.MAX_TURNS}] Fetching task ---")

        tasks_result = await with_retry(
            lambda: get_tasks(state.agent_id),
            label="GetTasks"
        )

        # Parse task JSON
        task_data = {}
        try:
            task_data = json.loads(tasks_result)
        except Exception:
            # Some responses are plain text with key=value pairs
            print(f"  [Main] Non-JSON response: {tasks_result[:300]}")
            if "no task" in tasks_result.lower() or "error" in tasks_result.lower():
                print("  No active task available. Waiting 10s...")
                await asyncio.sleep(10)
                continue

        if not task_data.get("id"):
            print("  No task ID in response. Waiting 15s...")
            await asyncio.sleep(15)
            continue

        # Run the 6-phase pipeline
        submitted = await run_task_pipeline(
            task_data, state, memory, arena_tools, task_count
        )

        if submitted:
            await asyncio.sleep(3)   # brief pause between tasks

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Arena Champion Agent — Session Complete")
    print(f"  Tasks Attempted: {state.tasks_attempted}")
    print(f"  Total Score:     {state.total_score}")
    print(f"  Final Level:     {state.current_level}")
    state.print_cost_summary()
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
