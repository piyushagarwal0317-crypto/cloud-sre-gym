import os
import json
from openai import OpenAI
from src.env import CloudSREEnv
from src.models import SREAction

def run_openai_baseline(task_id="easy-traffic"):
    # 1. Check for API Key (Strict hackathon requirement)
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required.")
        
    client = OpenAI()
    
    # 2. Initialize the OpenEnv environment based on the task
    difficulty_map = {"easy-traffic": "easy", "medium-spikes": "medium", "hard-incidents": "hard"}
    env = CloudSREEnv(difficulty=difficulty_map[task_id])
    
    obs = env.reset()
    done = False
    
    system_prompt = """You are an autonomous Site Reliability Engineer (SRE). 
Your goal is to manage cloud instances to keep latency under 200ms while minimizing server costs.
Available actions: 'scale_up', 'scale_down', 'change_instance_type', 'do_nothing'.
You must output a JSON object representing your action.
Example: {"action_type": "scale_up", "amount": 1, "instance_type": "standard"}"""

    print(f"--- Starting Baseline Run for Task: {task_id} ---")
    
    while not done:
        # Format the observation for the LLM
        user_prompt = f"Current Time Step: {obs.time_step}\nRecent Tool Outputs: {json.dumps(obs.tool_outputs)}\nWhat is your next action?"
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # Use 3.5 for fast, cheap baseline reproduction
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            # Parse the LLM's chosen action
            llm_output = json.loads(response.choices[0].message.content)
            action = SREAction(**llm_output)
            
            # Agent explicitly asks for tool data for the next step to simulate realism
            requested_tools = ["get_metrics", "get_cost", "get_alerts"]
            
        except Exception as e:
            print(f"LLM Error, defaulting to do_nothing. Error: {e}")
            action = SREAction(action_type="do_nothing")
            requested_tools = ["get_metrics"]
            
        # 3. Step the environment (OpenEnv API)
        obs, reward, done, state = env.step(action, requested_tools=requested_tools)
        
    print(f"Run Complete. Final State: {state}")
    return state

if __name__ == "__main__":
    for task in ["easy-traffic", "medium-spikes", "hard-incidents"]:
        run_openai_baseline(task)