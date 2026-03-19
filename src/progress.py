import json
from pathlib import Path

PROGRESS_FILE = Path("progress.json")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def record_quiz_result(topic: str, correct: int, total: int):
    progress = load_progress()
    if topic not in progress:
        progress[topic] = {"attempts": 0, "total_correct": 0, "total_questions": 0}
    progress[topic]["attempts"] += 1
    progress[topic]["total_correct"] += correct
    progress[topic]["total_questions"] += total
    save_progress(progress)
    return progress


def get_weak_topics(progress: dict, threshold: float = 0.6) -> list:
    weak = []
    for topic, data in progress.items():
        if data["total_questions"] > 0:
            rate = data["total_correct"] / data["total_questions"]
            if rate < threshold:
                weak.append((topic, rate))
    return sorted(weak, key=lambda x: x[1])


def format_progress(progress: dict) -> str:
    if not progress:
        return "No quiz history yet. Try /quiz <topic> to get started."

    lines = ["Quiz Progress:", "-" * 45]
    for topic, data in progress.items():
        total_q = data["total_questions"]
        total_c = data["total_correct"]
        rate = total_c / total_q if total_q > 0 else 0
        attempts = data["attempts"]
        lines.append(f"  {topic:<25} {total_c}/{total_q} ({rate:.0%}) — {attempts} attempt(s)")

    weak = get_weak_topics(progress)
    if weak:
        lines.append("")
        lines.append("Suggested review:")
        for topic, rate in weak[:3]:
            lines.append(f"  → /quiz {topic}  ({rate:.0%} correct so far)")

    return "\n".join(lines)
