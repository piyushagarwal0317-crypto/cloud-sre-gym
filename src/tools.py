class SRETools:
    def __init__(self, sim):
        self.sim = sim

    def get_metrics(self) -> dict:
        """Returns obfuscated/noisy system metrics."""
        return {
            "cpu_percent": round(self.sim.cpu_utilization * 100, 2),
            "memory_percent": round((self.sim.cpu_utilization * 0.8) * 100, 2),
            "latency_ms": round(self.sim.latency_ms, 2),
            "active_instances": self.sim.instances
        }

    def get_logs(self) -> list:
        """Returns recent system events."""
        logs = []
        if self.sim.active_incidents:
            logs.extend([f"ERROR: {inc}" for inc in self.sim.active_incidents])
        else:
            logs.append("INFO: System operating normally.")
        return logs

    def get_cost(self) -> dict:
        return {"current_hourly_burn": round(self.sim.instances * self.sim.instance_types[self.sim.current_type]["cost"], 2)}

    def get_alerts(self) -> list:
        if self.sim.latency_ms > 200.0:
            return ["CRITICAL: P99 Latency SLA Exceeded (>200ms)"]
        return []