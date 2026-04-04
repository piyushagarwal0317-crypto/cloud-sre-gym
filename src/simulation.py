import math
import random
import numpy as np

class MultiCloudSimulation:
    def __init__(self):
        self.time = 0
        self.providers = {
            "aws_us_east": {"cost_per_h": 0.10, "base_latency": 30.0, "failure_prob": 0.005, "instances": 2},
            "gcp_eu_west": {"cost_per_h": 0.12, "base_latency": 20.0, "failure_prob": 0.002, "instances": 2},
            "azure_ap_south": {"cost_per_h": 0.09, "base_latency": 60.0, "failure_prob": 0.008, "instances": 2}
        }
        self.global_traffic = 0
        self.active_incidents = []

    def _generate_global_traffic(self):
        # Time-based global traffic with noise
        diurnal = 500 + 300 * math.sin(2 * math.pi * self.time / 1440)
        noise = random.gauss(0, 50)
        return max(0, diurnal + noise)

    def step(self):
        self.time += 1
        self.global_traffic = self._generate_global_traffic()
        
        metrics = {}
        total_capacity = sum([p["instances"] * 100 for p in self.providers.values()])
        
        # Simulate network routing delays and server failures
        for name, config in self.providers.items():
            # Server failure check
            if random.random() < config["failure_prob"] and config["instances"] > 0:
                config["instances"] -= 1
                self.active_incidents.append(f"Server crash in {name}")

            # Traffic distribution (proportional to instances)
            if total_capacity > 0:
                regional_traffic = self.global_traffic * ((config["instances"] * 100) / total_capacity)
            else:
                regional_traffic = self.global_traffic

            # Capacity math
            capacity = config["instances"] * 100
            cpu = min(1.0, regional_traffic / capacity) if capacity > 0 else 1.0
            
            # Latency spikes heavily if CPU > 80% (Queuing Theory)
            latency = config["base_latency"]
            if cpu > 0.8:
                latency *= math.exp(4 * (cpu - 0.8))
                
            metrics[name] = {"cpu": cpu, "latency": latency, "cost": config["cost_per_h"] * config["instances"]}
            
        return metrics