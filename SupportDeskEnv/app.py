from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from envs.support_env import SupportDeskEnv
from envs.models import Action  # IMPORTANT FIX


app = FastAPI(
    title="SupportDeskEnv API",
    description="OpenEnv-style SaaS customer support simulation environment.",
    version="1.0.0",
)

env = SupportDeskEnv()


# -------------------------------
# Request Models
# -------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


# -------------------------------
# Root & Health
# -------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "SupportDeskEnv",
        "status": "ok",
        "message": "Environment server is running."
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


# -------------------------------
# RESET API (GET + POST FIX)
# -------------------------------

# ✅ Browser-friendly
@app.get("/reset")
def reset_get(task_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        observation = env.reset(task_id=task_id)
        return {
            "observation": observation.model_dump(),
            "done": False,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ✅ OpenEnv / POST requests
@app.post("/reset")
def reset_post(request: ResetRequest) -> Dict[str, Any]:
    try:
        observation = env.reset(task_id=request.task_id)
        return {
            "observation": observation.model_dump(),
            "done": False,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------
# STEP API (FIXED)
# -------------------------------

@app.post("/step")
def step_environment(request: StepRequest) -> Dict[str, Any]:
    try:
        # 🔥 FIX: Convert dict → Action object
        action_obj = Action(**request.action)

        observation, reward, done, info = env.step(action_obj)

        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------
# STATE API
# -------------------------------

@app.get("/state")
def get_state() -> Dict[str, Any]:
    try:
        state = env.state()
        return {"state": state.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
