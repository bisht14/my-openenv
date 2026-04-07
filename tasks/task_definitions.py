"""
Task definitions for the Code Review environment.

Each task bundles a code snippet, the ground-truth issues, and metadata
needed by the grader.  Three difficulty levels:

  easy   – obvious bug (off-by-one / null dereference)
  medium – security vulnerability (SQL injection / hardcoded secret)
  hard   – multi-issue review (bug + security + style + performance)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskSpec:
    task_id: str
    difficulty: str           # "easy" | "medium" | "hard"
    language: str
    description: str
    code_snippet: str
    ground_truth_issues: List[str]        # canonical issues the agent must find
    issue_keywords: List[List[str]]       # keyword groups — agent must match ≥1 keyword per group
    severity_answers: List[str]           # expected severity per issue
    hints: List[str] = field(default_factory=list)
    max_steps: int = 5


# ─── EASY ────────────────────────────────────────────────────────────────────
TASK_EASY = TaskSpec(
    task_id="spot-the-bug",
    difficulty="easy",
    language="Python",
    description=(
        "You are a code reviewer. The snippet below contains exactly ONE bug. "
        "Identify the bug, state which line it is on, explain why it is wrong, "
        "and suggest a fix. Rate its severity (low / medium / high / critical)."
    ),
    code_snippet="""\
def calculate_average(numbers):
    \"\"\"Return the average of a list of numbers.\"\"\"
    total = 0
    for i in range(len(numbers) + 1):   # line 4
        total += numbers[i]
    return total / len(numbers)

scores = [85, 90, 78, 92, 88]
print(calculate_average(scores))
""",
    ground_truth_issues=[
        "Off-by-one error in range(): using len(numbers)+1 causes an IndexError on the last iteration"
    ],
    issue_keywords=[
        ["off-by-one", "index", "range", "len", "indexerror", "out of range", "line 4", "+1"]
    ],
    severity_answers=["high"],
    hints=[
        "Look carefully at the loop boundary in range().",
        "What happens when i equals len(numbers)?"
    ],
    max_steps=4,
)

# ─── MEDIUM ──────────────────────────────────────────────────────────────────
TASK_MEDIUM = TaskSpec(
    task_id="security-audit",
    difficulty="medium",
    language="Python",
    description=(
        "You are a security-focused code reviewer. The snippet below contains "
        "security vulnerabilities. Identify ALL vulnerabilities, explain the "
        "attack vector for each, and suggest secure alternatives. "
        "Rate each severity (low / medium / high / critical)."
    ),
    code_snippet="""\
import sqlite3

DB_PASSWORD = "admin123"          # line 3
SECRET_KEY  = "my_secret_key_42"  # line 4

def get_user(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    # Build query by string formatting
    query = f"SELECT * FROM users WHERE username = '{username}'"  # line 10
    cursor.execute(query)
    return cursor.fetchone()

def login(username, password):
    user = get_user(username)
    if user and user[2] == password:   # line 16 – plaintext comparison
        return True
    return False
""",
    ground_truth_issues=[
        "SQL injection via f-string query construction on line 10",
        "Hardcoded credentials / secrets on lines 3-4",
        "Plaintext password comparison on line 16 (no hashing)",
    ],
    issue_keywords=[
        ["sql injection", "sql", "f-string", "string format", "parameterized", "line 10", "query"],
        ["hardcoded", "credential", "secret", "password", "plain text secret", "line 3", "line 4"],
        ["plaintext", "plain text", "hash", "bcrypt", "password comparison", "line 16"],
    ],
    severity_answers=["critical", "critical", "high"],
    hints=[
        "Think about what happens if username contains a single quote.",
        "Should credentials ever appear as string literals in source code?",
        "How should passwords be stored and compared securely?",
    ],
    max_steps=6,
)

# ─── HARD ────────────────────────────────────────────────────────────────────
TASK_HARD = TaskSpec(
    task_id="full-review",
    difficulty="hard",
    language="Python",
    description=(
        "You are a senior engineer doing a thorough code review. The snippet "
        "below has multiple issues spanning correctness, security, performance, "
        "and style. Find ALL issues, assign a severity to each (low / medium / "
        "high / critical), explain the impact, and provide concrete fixes. "
        "Order your findings from most to least severe."
    ),
    code_snippet="""\
import os, sys, json          # line 1 – wildcard-style multi-import
from pathlib import Path

API_KEY = os.environ.get("API_KEY", "fallback_dev_key")   # line 4

def load_config(path):
    data = open(path).read()                # line 7 – resource leak
    return json.loads(data)

def process_files(directory):
    results = []
    files = os.listdir(directory)
    for filename in files:
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath) as f:
                content = f.read()          # line 17 – reads entire file into memory
            processed = eval(content)       # line 18 – arbitrary code execution
            results.append(processed)
    return results

def save_results(results, output_path):
    with open(output_path, "w") as f:
        for r in results:
            f.write(str(r) + "\\n")         # line 24 – no error handling / validation
    print("Done! Saved", len(results), "records to", output_path)

if __name__ == "__main__":
    cfg = load_config("config.json")
    data = process_files(cfg["input_dir"])
    save_results(data, cfg["output_path"])
""",
    ground_truth_issues=[
        "eval() on file content on line 18 allows arbitrary code execution (critical)",
        "Fallback API key value on line 4 — insecure default credential",
        "File handle leak in load_config() on line 7 — open() not used as context manager",
        "Reading entire file into memory on line 17 — no size guard, DoS risk on large files",
        "No input validation or error handling in save_results()",
        "Multi-import on line 1 is a style violation (PEP 8)",
    ],
    issue_keywords=[
        ["eval", "code execution", "arbitrary", "injection", "line 18"],
        ["fallback", "default", "api key", "hardcoded", "insecure default", "line 4"],
        ["resource leak", "context manager", "open()", "file handle", "line 7", "with open"],
        ["memory", "entire file", "large file", "dos", "line 17", "no size"],
        ["error handling", "validation", "exception", "save_results"],
        ["pep 8", "import", "style", "line 1", "multiple import"],
    ],
    severity_answers=["critical", "high", "medium", "medium", "medium", "low"],
    hints=[
        "Start with the most dangerous function call in process_files().",
        "Check every place a file is opened — are they all properly closed?",
        "What happens if cfg['input_dir'] contains a 10 GB file?",
        "Review PEP 8 import conventions.",
    ],
    max_steps=8,
)

ALL_TASKS: dict[str, TaskSpec] = {
    "spot-the-bug": TASK_EASY,
    "security-audit": TASK_MEDIUM,
    "full-review": TASK_HARD,
}
