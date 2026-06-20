"""
memory/task_memory.py
JSON-backed task memory store.

Stores past tasks, solutions, scores and feedback so the agent can:
  1. Avoid repeating mistakes on similar tasks.
  2. Bootstrap answers using verified past solutions.
"""
import json
import os
from datetime import datetime, timezone

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "task_memory.json")


class TaskMemory:
    """Simple JSON-backed memory of past task attempts."""

    def __init__(self, max_entries: int = 200):
        self.max_entries = max_entries
        self.records: list[dict] = []
        self._load()

    def _load(self):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.records = []

    def _save(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.records[-self.max_entries:], f, indent=2, ensure_ascii=False)

    def record(self, task_id: str, title: str, task_type: str,
               solution: str, score: int, feedback: str = ""):
        """Save a completed task attempt to memory."""
        entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "task_id":    task_id,
            "title":      title,
            "task_type":  task_type,
            "solution":   solution[:2000],   # truncate for storage
            "score":      score,
            "feedback":   feedback,
        }
        self.records.append(entry)
        self._save()

    def recall_similar(self, title: str, task_type: str, top_k: int = 3) -> list[dict]:
        """
        Very lightweight similarity: returns recent records of the same task_type,
        with title keyword overlap, sorted by score descending.
        """
        title_words = set(title.lower().split())
        scored = []
        for r in self.records:
            if r.get("task_type") != task_type:
                continue
            overlap = len(title_words & set(r.get("title", "").lower().split()))
            scored.append((overlap + r.get("score", 0) / 10, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def stats(self) -> dict:
        """Summary statistics."""
        if not self.records:
            return {"total": 0, "avg_score": 0, "best_score": 0}
        scores = [r.get("score", 0) for r in self.records]
        return {
            "total":      len(self.records),
            "avg_score":  sum(scores) / len(scores),
            "best_score": max(scores),
        }
