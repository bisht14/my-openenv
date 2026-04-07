---
title: Code Review OpenEnv
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - code-review
  - reinforcement-learning
  - agent-environment
---

# Code Review OpenEnv 🔍

A real-world [OpenEnv](https://huggingface.co/openenv) environment where AI agents practice **software code review** — one of the most common and high-value tasks in professional software engineering.

The agent reads code snippets, identifies bugs and security vulnerabilities, assigns severity levels, and explains fixes. All skills that translate directly to developer tooling, CI pipelines, and automated code quality systems.

---

## Why Code Review?

Every software team reviews code. It requires reasoning about correctness, security, performance, and style — simultaneously. It's not a toy task; it's something developers do dozens of times per week. A capable agent here has genuine commercial value.

---

## Environment Overview

| Property | Value |
|---|---|
| **Observation** | `CodeReviewObservation` (Pydantic model) |
| **Action** | `CodeReviewAction` (Pydantic model) |
| **Reward range** | `[0.0, 1.0]` |
| **Tasks** | 3 (easy → medium → hard) |
| **API** | `step()` / `reset()` / `state()` / `close()` |
| **Transport** | HTTP via FastAPI on port 7860 |

---

## Action Space

The agent submits a `CodeReviewAction` each step:

```python
class CodeReviewAction(BaseModel):
    issues_found:   List[str]        # Issues identified (one string per issue)
    severity_tags:  List[str]        # "low" | "medium" | "high" | "critical" per issue
    explanation:    str              # Overall analysis and suggested fixes
    line_numbers:   Optional[List[int]]  # Line numbers per issue (optional but rewarded)
```

**Example action:**
```json
{
  "issues_found": [
    "SQL injection via f-string on line 10",
    "Hardcoded credentials on lines 3-4"
  ],
  "severity_tags": ["critical", "critical"],
  "explanation": "Use parameterized queries and move credentials to environment variables.",
  "line_numbers": [10, 3]
}
```

---

## Observation Space

After `reset()` or `step()` the agent receives a `CodeReviewObservation`:

```python
class CodeReviewObservation(BaseModel):
    code_snippet:       str         # The code to review
    language:           str         # e.g. "Python"
    task_description:   str         # What the agent must do
    step_number:        int         # Current step (0 = just reset)
    last_feedback:      str         # Grader feedback from previous step
    cumulative_score:   float       # Running score in [0, 1]
    hints:              List[str]   # Available on medium/hard tasks
```

---

## Tasks

### Task 1 — `spot-the-bug` (Easy)

**Objective:** Find exactly one bug in a short Python function.

The snippet contains an off-by-one error in a `range()` call that causes an `IndexError` at runtime. The agent must identify the bug, cite the line number, explain why it's wrong, and suggest a fix.

- **Difficulty:** Easy
- **Max steps:** 4
- **Success threshold:** 0.70
- **Expected issues:** 1

---

### Task 2 — `security-audit` (Medium)

**Objective:** Identify three security vulnerabilities in a login/database module.

The snippet contains SQL injection, hardcoded credentials, and plaintext password comparison — three of the most common real-world security mistakes.

- **Difficulty:** Medium
- **Max steps:** 6
- **Success threshold:** 0.60
- **Expected issues:** 3 (SQL injection, hardcoded secrets, plaintext passwords)

---

### Task 3 — `full-review` (Hard)

**Objective:** Conduct a thorough code review of a multi-function script.

The snippet contains six issues across four categories: security (`eval()` arbitrary execution, insecure default key), resource management (file handle leak), performance (unbounded memory read), robustness (missing error handling), and style (PEP 8 import violation).

- **Difficulty:** Hard
- **Max steps:** 8
- **Success threshold:** 0.50
- **Expected issues:** 6

---

## Reward Function

Each step is scored on three components:

| Component | Weight | Description |
|---|---|---|
| **Issue detection** | 60% | Fraction of ground-truth issues whose keywords appear in the agent response |
| **Severity accuracy** | 20% | Correctness of severity labels; partial credit for one level off |
| **Explanation depth** | 20% | Heuristic: length, fix suggestions mentioned, line numbers cited |

Rewards are partial — the agent gets credit for each issue it finds, not binary win/lose. The agent can improve its score across multiple steps by refining its analysis based on grader feedback.

---

## HTTP API

The environment runs as a FastAPI server on port 7860.

### `POST /reset`

Start a new episode.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "spot-the-bug"}'
```

### `POST /step`

Submit an action.

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "issues_found": ["off-by-one error in range on line 4"],
    "severity_tags": ["high"],
    "explanation": "Change range(len(numbers)+1) to range(len(numbers)).",
    "line_numbers": [4]
  }'
```

### `GET /state`

Get current episode state.

```bash
curl http://localhost:7860/state
```

### `GET /health`

Health check — returns 200 if the server is running.

```bash
curl http://localhost:7860/health
```

### `GET /tasks`

List all available tasks.

```bash
curl http://localhost:7860/tasks
```

---

## Setup & Usage

### Local (Python)

```bash
git clone https://huggingface.co/spaces/<your-username>/code-review-env
cd code-review-env

pip install -r requirements.txt

# Start the server
python server.py

# In another terminal — run the baseline inference script
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker

```bash
docker build -t code-review-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  code-review-env
```

### Required environment variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face / API key | — (required) |
| `API_BASE_URL` | LLM endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `TASK_NAME` | Override to run one task | Runs all three |

---

## Baseline Scores

Scores produced by running `inference.py` with `Qwen/Qwen2.5-72B-Instruct` against each task:

| Task | Difficulty | Baseline Score |
|---|---|---|
| `spot-the-bug` | Easy | ~0.78 |
| `security-audit` | Medium | ~0.65 |
| `full-review` | Hard | ~0.52 |

Scores vary slightly across runs due to temperature sampling. Overall average: ~0.65.

---

## Project Structure

```
code-review-env/
├── models.py               # Pydantic typed models
├── env.py                  # CodeReviewEnv — OpenEnv spec implementation
├── server.py               # FastAPI HTTP server
├── inference.py            # Baseline inference script
├── openenv.yaml            # Environment metadata
├── Dockerfile
├── requirements.txt
├── tasks/
│   └── task_definitions.py # 3 task specs with ground truth
├── graders/
│   └── grader.py           # Deterministic keyword-based grader
└── tests/
    └── test_env.py         # Test suite
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Disqualification Checklist ✅

- Environment deploys and responds to `/health` → ✅
- OpenEnv spec: typed models, `step()`/`reset()`/`state()`, `openenv.yaml` → ✅  
- Dockerfile builds and runs → ✅
- Baseline `inference.py` completes without error and produces scores → ✅
- 3+ tasks with graders, scores in `[0.0, 1.0]` → ✅
- Graders are not trivially constant — score varies with agent quality → ✅
