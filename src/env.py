import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .simulation import MultiCloudSimulation

class CloudSREGymEnv(gym.Env):
    def __init__(self):
        super(CloudSREGymEnv, self).__init__()
        self.sim = MultiCloudSimulation()
        
        # Action Space: [delta_aws, delta_gcp, delta_azure]
        # Each can be 0 (down), 1 (do nothing), 2 (scale up)
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        
        # Obs Space: 3 providers * (instances, cpu, latency) + time_feature
        self.observation_space = spaces.Box(low=0, high=1000, shape=(10,), dtype=np.float32)
        
        self.prev_instances = [2, 2, 2]

    def reset(self, seed=None, options=None):
        self.sim = MultiCloudSimulation()
        self.prev_instances = [2, 2, 2]
        return self._get_obs(), {}

    def _get_obs(self):
        metrics = self.sim.step()
        obs = []
        for name in ["aws_us_east", "gcp_eu_west", "azure_ap_south"]:
            obs.extend([
                self.sim.providers[name]["instances"],
                metrics[name]["cpu"],
                metrics[name]["latency"]
            ])
        obs.append(self.sim.time % 1440 / 1440.0) # Normalized time of day
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # Apply actions: map [0, 1, 2] to [-1, 0, +1]
        deltas = action - 1
        
        current_instances = []
        for i, provider in enumerate(["aws_us_east", "gcp_eu_west", "azure_ap_south"]):
            self.sim.providers[provider]["instances"] = max(0, self.sim.providers[provider]["instances"] + deltas[i])
            current_instances.append(self.sim.providers[provider]["instances"])

        # Advance simulation
        metrics = self.sim.step()
        
        # Calculate Reward
        reward = self._calculate_reward(metrics, current_instances)
        self.prev_instances = current_instances
        
        done = self.sim.time >= 1440
        truncated = False
        
        return self._get_obs(), reward, done, truncated, {"metrics": metrics}

    def _calculate_reward(self, metrics, current_instances):
        total_cost = sum([m["cost"] for m in metrics.values()])
        max_latency = max([m["latency"] for m in metrics.values()])
        
        # 1. Cost Penalty
        cost_penalty = total_cost * 0.5
        
        # 2. SLA Penalty (Exponential if > 200ms)
        sla_penalty = 0
        if max_latency > 200:
            sla_penalty = (max_latency - 200) * 0.1
            
        # 3. Scaling Instability Penalty (Penalize flip-flopping)
        instability_penalty = sum(abs(c - p) for c, p in zip(current_instances, self.prev_instances)) * 2.0
        
        return -(cost_penalty + sla_penalty + instability_penalty)