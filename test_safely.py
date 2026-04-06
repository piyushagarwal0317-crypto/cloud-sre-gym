"""
Safe Environment Tester 🛡️
Use this script to test the mechanics of the CloudScaleRL environment
WITHOUT making expensive OpenAI API calls and without needing the server running.

This uses a "Dummy Agent" that just sends basic HPA and no-op actions
directly into the python core logic so you can verify the math and metrics.
"""

from cloudscalerl.server.cloudscalerl_env import CloudScaleEnvServer
from cloudscalerl.models import CloudScaleAction, HPAAction
import time

def main():
    print("🚀 Starting local CloudScaleRL environment (In-Memory/No LLM required)...")
    
    # 1. Boot up the environment core physics engine directly
    env = CloudScaleEnvServer()

    try:
        # 2. Reset the environment to start a fresh simulation
        print("\n[✓] Resetting environment on task1_hpa...")
        obs = env._reset(task_id="task1_hpa")
        
        print(f"    Initial Cost: ${obs.total_cost_usd_per_hour:.2f}/hr")
        print(f"    Initial Budget: ${obs.budget_remaining_usd:.2f}")

        # 3. Simulate 10 ticks (10 minutes of simulated time)
        print("\n[✓] Running 10 ticks with a Dummy Agent (scaling to 5 replicas immediately)...")
        for step in range(1, 11):
            
            # SAFE DUMMY LOGIC: Instead of an LLM, we just hardcode an action
            if step == 1:
                # Tick 1: Proactively scale api-gateway to 5 replicas
                action = CloudScaleAction(
                    hpa=HPAAction(service="api-gateway", target_replicas=5)
                )
            else:
                # Ticks 2-10: Do nothing, let the queueing theory physics play out
                action = CloudScaleAction(no_op=True)

            print(f"\n--- Tick {step}/10 ---")
            print(f"Action Sent: {action.model_dump(exclude_none=True)}")
            
            # In-memory fast step! Returns (obs, reward, done, info)
            obs, reward, done, info = env._step(action)
            
            svc = obs.services["api-gateway"]
            print(f"Result -> p99 Latency: {svc.p99_latency_ms:.1f}ms | CPU: {svc.cpu_utilization:.0%} | Replicas: {svc.replicas}")
            print(f"Reward -> {reward:+.3f} | Cost: ${obs.total_cost_usd_per_hour:.2f}/hr")
            
            time.sleep(0.1)

        print("\n🎉 Environment mechanics tested successfully in pure memory!")
        print("To run the real LLM against it, use `python -m cloudscalerl.client task1_hpa`.")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")

if __name__ == "__main__":
    main()

