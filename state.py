import uuid
import config


class RunState:
    """Tracks agent competitor state, task metrics, and per-phase estimated credit usage."""

    def __init__(self):
        self.run_id           = str(uuid.uuid4())
        self.agent_id         = ""
        self.task_id          = ""
        self.current_level    = 1
        self.total_score      = 0
        self.tasks_attempted  = 0
        self.level_history    = []

        # Per-phase cost tracking
        self.phase_stats: dict[str, dict] = {}   # {phase_name: {input_tk, output_tk, cost}}
        self.input_tokens    = 0
        self.output_tokens   = 0
        self.estimated_cost  = 0.0

    # ── Token Recording ───────────────────────────────────────────────────────

    def record_tokens(self, input_tks: int, output_tks: int, model: str = "", phase: str = ""):
        """Records token usage + increments cumulative estimated cost."""
        self.input_tokens  += input_tks
        self.output_tokens += output_tks

        in_price, out_price = config.get_model_pricing(model)
        cost = (input_tks / 1_000_000) * in_price + (output_tks / 1_000_000) * out_price
        self.estimated_cost += cost

        # Per-phase breakdown
        if phase:
            if phase not in self.phase_stats:
                self.phase_stats[phase] = {"input_tk": 0, "output_tk": 0, "cost": 0.0}
            self.phase_stats[phase]["input_tk"]  += input_tks
            self.phase_stats[phase]["output_tk"] += output_tks
            self.phase_stats[phase]["cost"]      += cost

    def estimate_and_record_tokens(self, text: str, is_input: bool = True,
                                   model: str = "", phase: str = ""):
        """Estimates token count (1 token ≈ 4 chars) and records usage."""
        estimated = max(1, len(text) // 4)
        if is_input:
            self.record_tokens(estimated, 0, model=model, phase=phase)
        else:
            self.record_tokens(0, estimated, model=model, phase=phase)

    # ── Task Result Recording ─────────────────────────────────────────────────

    def record(self, level: int, title: str, score: int, levelled_up: bool):
        """Records a submitted task result and updates level progression."""
        self.tasks_attempted += 1
        self.total_score     += score
        if levelled_up:
            self.current_level = level + 1

        self.level_history.append({
            "level": level, "task": title, "score": score, "up": levelled_up
        })
        icon = "✓" if levelled_up else ("~" if score >= 70 else "✗")

        print("\n" + "=" * 55)
        print(f"  [RunState] Task Complete: {icon} Level {level} → Score: {score}/100")
        print(f"  Attempted: {self.tasks_attempted}  |  Total Score: {self.total_score}")
        print(f"  Current Level: {self.current_level}")
        self.print_cost_summary()
        print("=" * 55 + "\n")

    # ── Cost Reporting ────────────────────────────────────────────────────────

    def print_cost_summary(self):
        """Prints current total and per-phase cost breakdown."""
        print(f"  Est. Tokens: {self.input_tokens} in / {self.output_tokens} out")
        print(f"  Est. Cost:   ${self.estimated_cost:.6f} USD")
        if self.phase_stats:
            print("  Phase Breakdown:")
            for ph, s in self.phase_stats.items():
                print(f"    {ph:12s}: {s['input_tk']:>6} in / {s['output_tk']:>6} out  ${s['cost']:.6f}")
