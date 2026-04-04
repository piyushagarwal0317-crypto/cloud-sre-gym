class CloudGrader:
    def __init__(self, max_budget=200.0, max_tolerable_violations=50):
        self.max_budget = max_budget
        self.max_violations = max_tolerable_violations

    def grade(self, final_state: dict) -> float:
        cost = final_state["cost"]
        violations = final_state["sla_violations"]

        # Cost Score (0 to 1)
        cost_score = max(0.0, 1.0 - (cost / self.max_budget))
        
        # SLA Score (0 to 1)
        sla_score = max(0.0, 1.0 - (violations / self.max_violations))

        # Weighted final score (SLA is 2x more important than cost to most businesses)
        final_score = (cost_score * 0.3) + (sla_score * 0.7)
        return round(final_score, 3)