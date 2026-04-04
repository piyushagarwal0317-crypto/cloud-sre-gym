import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

# Adjust these imports based on your exact OpenEnv generated package names
from src.models import SREAction
from src.env import CloudSREEnv 

IMAGE_NAME = os.getenv("IMAGE_NAME") 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Default to your easy task if not specified by the evaluator harness
TASK_NAME = os.getenv("SRE_GYM_TASK", "easy-traffic")
BENCHMARK = os.getenv("SRE_GYM_BENCHMARK", "cloud-sre-gym")

# In our env, 1440 mins is a full day, but for the hackathon baseline we might limit steps 
# depending on your specific episode length configuration. Let's assume 24 steps (hourly actions).
MAX_STEPS = int(os.getenv("MAX_STEPS", "24")) 
TEMPERATURE = 0.2 # Lower temperature for more deterministic JSON outputs
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.70  # e.g., Need 70% weighted score from the grader

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous Site Reliability Engineer (SRE) managing cloud infrastructure.
    Your goal is to manage instances to keep latency under 200ms while minimizing server costs.
    
    You must respond with a valid JSON object representing your next action.
    Allowed action_types: "scale_up", "scale_down", "change_instance_type", "do_nothing".
    
    Example response:
    {
      "action_type": "scale_up",
      "amount": 1,
      "instance_type": "aws_us_east"
    }
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, tool_outputs: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Recent Tool Outputs: {json.dumps(tool_outputs)}
        
        Previous actions:
        {history_block}
        
        Analyze the metrics and provide your next JSON action.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, tool_outputs: dict, history: List[str]) -> tuple[SREAction, str, Optional[str]]:
    user_prompt = build_user_prompt(step, tool_outputs, history)
    raw_text = ""
    error_msg = None
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={ "type": "json_object" } # Force JSON output if the model supports it
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        
        # Parse text to Pydantic model
        action_data = json.loads(raw_text)
        action = SREAction(**action_data)
        
    except json.JSONDecodeError as e:
        error_msg = f"JSON Parse Error: {str(e)}"
        action = SREAction(action_type="do_nothing", amount=0)
        raw_text = raw_text or "invalid_json"
    except Exception as exc:
        error_msg = f"API/Validation Error: {str(exc)}"
        action = SREAction(action_type="do_nothing", amount=0)
        raw_text = "error_fallback"
        
    return action, raw_text, error_msg

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize environment (Assuming OpenEnv standard async wrapper)
    if IMAGE_NAME:
        env = await CloudSREEnv.from_docker_image(IMAGE_NAME)
    else:
        env = CloudSREEnv(difficulty=TASK_NAME.split('-')[0]) # Fallback for local testing without docker

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Standard OpenEnv reset
        result = await env.reset()
        tool_outputs = result.observation.tool_outputs

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_obj, raw_action_str, error = get_model_action(client, step, tool_outputs, history)

            # In a real tool-use scenario, the agent would request these. 
            # For the baseline, we auto-request the vital signs so the agent can see them next turn.
            requested_tools = ["get_metrics", "get_alerts", "get_cost"]
            
            # Step the environment
            result = await env.step(action_obj, requested_tools=requested_tools)
            
            reward = result.reward.reward if hasattr(result.reward, 'reward') else (result.reward or 0.0)
            done = result.done
            
            rewards.append(reward)
            steps_taken = step
            tool_outputs = result.observation.tool_outputs

            # Log step exactly as required
            # Ensure raw_action_str doesn't contain newlines that would break the stdout format
            flat_action_str = raw_action_str.replace('\n', '').replace('\r', '')
            log_step(step=step, action=flat_action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {flat_action_str} -> reward {reward:+.2f}")

            if done:
                break

        # Calculate final score based on your Grader (Assuming env.state() returns business metrics)
        final_state = await env.state()
        
        # Example grading logic inline, or call your CloudGrader here
        max_budget = 200.0
        cost_score = max(0.0, 1.0 - (final_state.get("cost", 0) / max_budget))
        sla_score = max(0.0, 1.0 - (final_state.get("sla_violations", 0) / 50))
        
        score = (cost_score * 0.3) + (sla_score * 0.7)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            # We don't print this to standard out if it breaks the format, use stderr or a debug format
            pass 
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())