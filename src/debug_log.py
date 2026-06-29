"""
Debug logger — writes chunks and answers to logs/ folder.
Each run creates a timestamped file.
"""
import os
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

_run_file = None


def _get_file():
    """Get or create the log file for this run."""
    global _run_file
    if _run_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _run_file = LOGS_DIR / f"run_{timestamp}.md"
    return _run_file


def log_section(title: str, content: str):
    """Append a section to the run log."""
    with open(_get_file(), "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n{content}\n")


def log_chunks(query: str, iteration: int, items: list):
    """Log retrieved chunks for a query."""
    lines = [f"**Query:** {query}  \n**Iteration:** {iteration}\n"]
    for i, item in enumerate(items, 1):
        lines.append(f"### Chunk {i} — `{item.file_name}` (score={item.score:.3f})")
        lines.append(f"```\n{item.content[:500]}\n```\n")
    log_section(f"Chunks — iter {iteration}", "\n".join(lines))


def log_answer(query: str, status: str, answer: str):
    """Log the full answer."""
    log_section(f"Answer — {status}", f"**Query:** {query}\n\n{answer}")


def log_feedback(iteration: int, feedback):
    """Log evaluate feedback."""
    log_section(f"Feedback — iter {iteration}",
                f"- can_answer: {feedback.can_answer}\n"
                f"- missing_info: {feedback.missing_info}\n"
                f"- suggestion: {feedback.suggestion}")


# === ANALYST / NL2SQL ===

def log_plan(question: str, reasoning: str, steps: list):
    """Log the analyst plan."""
    lines = [f"**Question:** {question}\n", f"**Reasoning:** {reasoning}\n"]
    for i, step in enumerate(steps, 1):
        if step.action == "query":
            lines.append(f"{i}. `{step.id}` — query `{step.table}`: {step.sub_question}")
        elif step.action == "tool":
            lines.append(f"{i}. `{step.id}` — tool `{step.tool_name}` on {step.input_steps} | params={step.params}")
    log_section("Analyst Plan", "\n".join(lines))


def log_step_result(step_id: str, action: str, description: str, status: str,
                    row_count: int = 0, error: str = "", sql: str = "",
                    df_preview: str = ""):
    """Log a single execution step (query or tool)."""
    lines = [f"**Step:** `{step_id}` ({action})  \n**Status:** {status}"]
    if description:
        lines.append(f"**Description:** {description}")
    if sql:
        lines.append(f"\n```sql\n{sql}\n```")
    if error:
        lines.append(f"**Error:** {error}")
    if row_count:
        lines.append(f"**Rows:** {row_count}")
    if df_preview:
        lines.append(f"\n```\n{df_preview}\n```")
    log_section(f"Step — {step_id}", "\n".join(lines))


def log_analyst_answer(question: str, status: str, answer: str):
    """Log the analyst's final synthesized answer."""
    log_section(f"Analyst Answer — {status}",
                f"**Question:** {question}\n\n{answer}")
