---
title: SupportDeskEnv
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# SupportDeskEnv

SupportDeskEnv is a realistic **OpenEnv-style environment** for training and evaluating AI agents on **SaaS customer support workflows**.

The environment simulates how real support teams operate. An AI agent must analyze customer support tickets, inspect account data, retrieve knowledge base information, and decide whether to reply, resolve, or escalate the issue.

This project demonstrates **multi-step decision making for AI agents in real-world operational environments**.

---

# Why this environment

Customer support is a common real-world operational task. Unlike toy benchmarks, this environment simulates realistic workflows where an AI agent must:

- Identify the issue type
- Inspect account metadata
- Retrieve internal knowledge base articles
- Respond with correct guidance
- Avoid unnecessary escalation
- Resolve tickets correctly

This makes the environment useful for:

- Agent evaluation
- Reinforcement learning research
- Workflow automation testing
- AI decision-making benchmarks

---

# Environment Summary

| Feature | Description |
|------|------|
| Domain | SaaS Customer Support |
| Environment Name | SupportDeskEnv |
| API Framework | FastAPI |
| Interface | `reset()`, `step()`, `state()` |
| Container Support | Docker |
| Deterministic Grading | Yes |

---

# Action Space

The AI agent can perform the following structured actions:

- `classify_ticket`
- `search_kb`
- `check_account`
- `reply_customer`
- `escalate_ticket`
- `resolve_ticket`

---

# Action Examples

### classify_ticket

```json
{
  "action_type": "classify_ticket",
  "category": "billing",
  "urgency": "medium"
}

search_kb
{
  "action_type": "search_kb",
  "query": "duplicate charge refund policy"
}
check_account
{
  "action_type": "check_account",
  "account_field": "last_payment_status"
}
reply_customer
{
  "action_type": "reply_customer",
  "message": "We are reviewing your billing issue and checking the payment records."
}
resolve_ticket
{
  "action_type": "resolve_ticket",
  "resolution_note": "Issue resolved after verifying payment details."
}
Tasks Included

The environment contains tasks of different difficulty levels.

Task ID	Difficulty	Description
task_easy_billing_triage	Easy	Classify duplicate billing charge and verify payment
task_medium_missing_invoice	Medium	Investigate missing invoice after successful payment
task_hard_feature_locked_after_upgrade	Hard	Diagnose feature access issue after upgrade
Installation

Clone the repository:

git clone <repository-url>
cd SupportDeskEnv

Create a virtual environment:

python -m venv .venv

Activate the virtual environment:

Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Running the Environment

Start the FastAPI server:

uvicorn app:app --reload

The API server will start at:

http://127.0.0.1:8000

You can verify the server health:

http://127.0.0.1:8000/health
Running the Agent

In a new terminal:

python inference.py

The agent will interact with the environment and solve all tasks.

Example output:

[START] task_easy_billing_triage
[STEP] task_easy_billing_triage step=1 action=classify_ticket reward=0.2
[STEP] task_easy_billing_triage step=2 action=check_account reward=0.1
...
[END] final_score=1.0
Running Tests

Run unit tests using pytest:

pytest

Expected result:

10 passed
Evaluation

The environment uses a deterministic grading system.

The grader evaluates:

Ticket classification accuracy
Urgency selection
Account metadata inspection
Knowledge base retrieval
Response quality
Correct ticket resolution

Final score range:

0.0 – 1.0
Example Result
tasks_run: 3
scores: [1.0, 1.0, 1.0]
average_score: 1.0
Project Structure
SupportDeskEnv
│
├── envs
│   ├── support_env.py
│   ├── tasks.py
│   ├── graders.py
│   ├── models.py
│   └── knowledge_base.py
│
├── tests
│   └── test_env.py
│
├── app.py
├── inference.py
├── requirements.txt
├── openenv.yaml
├── Dockerfile
└── README.md
Technologies Used
Python
FastAPI
OpenAI API
PyTest
Docker
Pydantic
Future Improvements

Possible enhancements include:

More complex support scenarios
Multi-customer conversations
Reinforcement learning agents
Dynamic knowledge base updates
Customer sentiment analysis
License

This project is intended for research, experimentation, and educational purposes.
=======
title: Support Agent Env
emoji: 🏢
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
---

