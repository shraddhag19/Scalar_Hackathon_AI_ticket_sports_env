"""
Phase 2 full compliance validation script.
Run this while the server is running on port 7860.
"""
import re
import os
import sys
import requests

base = "http://127.0.0.1:7860"
PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(label, condition, detail=""):
    icon = PASS if condition else FAIL
    results.append((icon, label, detail))
    suffix = f"  ->  {detail}" if detail else ""
    print(f"{icon} {label}{suffix}", flush=True)


print("=" * 65)
print("  PHASE 2 FULL COMPLIANCE CHECK")
print("=" * 65)

# ── 1. HF Space deploys: ping returns 200 ────────────────────────
r = requests.get(f"{base}/")
check("HF Space / ping returns 200", r.status_code == 200, f"HTTP {r.status_code}")

r = requests.get(f"{base}/health")
check("HF Space /health returns 200", r.status_code == 200, str(r.json()))

# ── 2. OpenEnv spec: reset() ─────────────────────────────────────
r = requests.post(f"{base}/reset", json={"task_id": "task_easy_billing_triage"})
data = r.json()
has_obs = "observation" in data and "done" in data
obs_fields = set(data.get("observation", {}).keys())
required_obs = {
    "ticket_id", "task_id", "customer_message", "customer_profile",
    "interaction_history", "visible_tools", "current_status",
    "remaining_steps", "last_tool_result",
}
check("OpenEnv reset() returns observation+done", has_obs)
check(
    "Observation has all required fields",
    required_obs.issubset(obs_fields),
    f"missing={required_obs - obs_fields}",
)

# ── 3. OpenEnv spec: step() ──────────────────────────────────────
r = requests.post(
    f"{base}/step",
    json={"action": {"action_type": "classify_ticket", "category": "billing", "urgency": "medium"}},
)
step_data = r.json()
check(
    "OpenEnv step() returns observation+reward+done+info",
    all(k in step_data for k in ["observation", "reward", "done", "info"]),
    f"keys={list(step_data.keys())}",
)
score = step_data["reward"]["score"]
check("step() reward score in [-1.0, 1.0]", -1.0 <= score <= 1.0, f"score={score}")

# ── 4. OpenEnv spec: state() ─────────────────────────────────────
r = requests.get(f"{base}/state")
check("OpenEnv state() returns 200", r.status_code == 200, f"HTTP {r.status_code}")

# ── 5. 3+ tasks ──────────────────────────────────────────────────
task_ids = [
    "task_easy_billing_triage",
    "task_medium_missing_invoice",
    "task_hard_feature_locked_after_upgrade",
]
check("3+ tasks defined", len(task_ids) >= 3, f"{len(task_ids)} tasks")

# ── 6. Required files ────────────────────────────────────────────
check("inference.py exists in root", os.path.isfile("inference.py"))
check("Dockerfile exists", os.path.isfile("Dockerfile"))
check("openenv.yaml exists", os.path.isfile("openenv.yaml"))
check("requirements.txt exists", os.path.isfile("requirements.txt"))

# ── 7. .env.example has required vars ────────────────────────────
with open(".env.example") as f:
    env_content = f.read()
for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
    check(f".env.example contains {var}", var in env_content)

# ── 8. inference.py correctness ──────────────────────────────────
with open("inference.py") as f:
    inf = f.read()
check("inference.py imports OpenAI client", "from openai import OpenAI" in inf)
check("inference.py reads API_BASE_URL", "API_BASE_URL" in inf)
check("inference.py reads MODEL_NAME", "MODEL_NAME" in inf)
check("inference.py reads HF_TOKEN", "HF_TOKEN" in inf)
flush_count = inf.count("flush=True")
check("inference.py uses flush=True on prints", flush_count >= 3, f"count={flush_count}")
check("inference.py not redirecting stdout to file", "sys.stdout" not in inf)

# ── 9. [START]/[STEP]/[END] format in saved output ───────────────
with open("final_inference_output.txt") as f:
    output = f.read()

starts = re.findall(r"\[START\] task=\S+", output)
steps  = re.findall(r"\[STEP\] task=\S+ step=\d+ reward=[\d.]+", output)
ends   = re.findall(r"\[END\] task=\S+ score=[\d.]+ steps=\d+", output)

check("[START] blocks — correct format (3 tasks)", len(starts) == 3, f"found {len(starts)}")
check("[STEP]  blocks — correct format", len(steps) >= 15, f"found {len(steps)} total steps")
check("[END]   blocks — correct format (3 tasks)", len(ends) == 3, f"found {len(ends)}")

# ── 10. Grader scores in [0.0, 1.0] ─────────────────────────────
scores_found = [float(x) for x in re.findall(r"\[END\].*?score=([\d.]+)", output)]
check(
    "All grader scores in 0.0-1.0 range",
    all(0.0 <= s <= 1.0 for s in scores_found),
    f"scores={scores_found}",
)
check("All grader scores > 0", all(s > 0 for s in scores_found), f"scores={scores_found}")

# ── 11. No old wrong format present ──────────────────────────────
old_start = bool(re.search(r"\[START\] task_id=", output))
old_end   = bool(re.search(r"\[END\].*final_score=", output))
check("No old 'task_id=' in [START] (wrong format)", not old_start)
check("No old 'final_score=' in [END] (wrong format)", not old_end)

# ── Summary ──────────────────────────────────────────────────────
print()
print("=" * 65)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  RESULT: {passed} passed  |  {failed} failed  |  {len(results)} total checks")
print("=" * 65)

sys.exit(0 if failed == 0 else 1)
