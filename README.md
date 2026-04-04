Here is the revised `README.md` with the Mermaid diagram removed. I replaced the graph in the **System Architecture** section with a clear, text-based workflow description so the architectural flow remains easy to understand\!

-----

# ☁️ Cloud SRE Gym: OpenEnv for Autonomous Cloud Resource Optimization

**Cloud SRE Gym** is an OpenEnv-compliant reinforcement learning and LLM-agent environment that simulates a highly realistic, multi-cloud Site Reliability Engineering (SRE) ecosystem.

Instead of treating cloud infrastructure as a simple video game with perfect information, this environment operates as a **Partially Observable Markov Decision Process (POMDP)**. The agent does not receive the full system state; instead, it must proactively interact with diagnostic tools—just like a human engineer—to deduce system health, balance SLAs, and minimize cloud spend.

-----

## 🎯 Real-World Motivation

Modern autonomous software engineering (SWE) agents excel at writing code, but struggle with live, dynamic infrastructure management.
Cloud SRE Gym bridges this gap by providing an enterprise-grade sandbox where agents must manage traffic spikes, network delays, and server crashes across AWS, GCP, and Azure simulations.

## 🏗️ System Architecture

The environment relies on a mathematical simulation engine grounded in queuing theory, interacting with the agent purely through JSON-based OpenEnv contracts. The system flow is designed as follows:

  * **The Agent Layer:** The LLM acts as the central brain, reasoning about the environment and utilizing a tool-calling engine to determine the next best action.
  * **The OpenEnv Interface:** Acts as the strict bridge, handling `step()` and `reset()` calls, passing `SREAction` schemas in and returning `SREObservation` payloads out.
  * **The Simulation Engine:** The core mathematical backend managing traffic distribution and failure events across simulated AWS (us-east), GCP (eu-west), and Azure (ap-south) regions.
  * **The SRE Tools API:** The only way the agent can "see" into the simulation engine. It must explicitly call tools like `get_metrics` and `get_logs` to receive obfuscated, realistic telemetry.

-----

## 🔬 Mathematical Simulation Engine

To avoid hardcoded heuristics, latency and system degradation are modeled using exponential growth as CPU utilization approaches 100%, mimicking real-world $M/M/c$ queuing limits.

Latency spikes are calculated dynamically per region:

$$Latency = \begin{cases} Base, & \text{if } CPU \le 0.7 \\ Base \times e^{4(CPU-0.7)}, & \text{if } CPU > 0.7 \end{cases}$$

This ensures agents cannot exploit linear scaling rules and must maintain a buffer to absorb stochastic traffic anomalies.

-----

## 🛠️ The POMDP Tool-Calling Interface

The agent is blinded to the true global state and must explicitly request tool outputs during its `step()` execution to inform its next move.

| Tool Name | Real-World Equivalent | Description |
| :--- | :--- | :--- |
| `get_metrics()` | Datadog / Prometheus | Returns obfuscated/noisy CPU %, memory, and latency data. |
| `get_logs()` | ELK Stack / Splunk | Returns recent system events and incident crash reports. |
| `get_cost()` | AWS Cost Explorer | Returns current hourly burn rate across all providers. |
| `get_alerts()` | PagerDuty | Triggers only if SLA (P99 Latency \> 200ms) is actively violated. |

### Action Space (`SREAction`)

The agent issues actions via a strictly typed Pydantic JSON schema:

```json
{
  "action_type": "scale_up",
  "amount": 2,
  "instance_type": "aws_us_east"
}
```

*Allowed actions: `scale_up`, `scale_down`, `change_instance_type`, `do_nothing`.*

-----

## 📊 Tasks & Difficulty Levels

Configured natively via `openenv.yaml`.

  * **🟢 Easy (`easy-traffic`):** Predictable diurnal sine-wave traffic. No server failures. Tests basic cost vs. capacity balancing.
  * **🟡 Medium (`medium-spikes`):** Heavy Gaussian noise and random flash-crowd events (e.g., product launches). Requires reactive scaling and metric monitoring.
  * **🔴 Hard (`hard-incidents`):** Introduces random region-specific server crashes and network partitions. The agent must use `get_logs` and `get_alerts` to route traffic and recover capacity.

-----

## 🏆 Reward Design & Grading

### Dense Reward Function (Per-Step)

Provides immediate partial signals to the agent:

  * **Cost Penalty:** $-\alpha \times \text{Hourly Spend}$
  * **SLA Penalty:** Exponential negative reward if latency $> 200ms$.
  * **Stability Penalty:** $-\lambda \times |\Delta \text{Instances}|$ (Penalizes the agent for rapid scale-up/scale-down oscillations, preventing orchestration tool burnout).

### Objective Business Grader (0.0 - 1.0)

Independent from the reward function, the evaluation grader scores the agent's absolute business performance at the end of the episode:

  * **30% Weight:** Cost Efficiency (vs. maximum budget).
  * **70% Weight:** SLA Adherence (vs. maximum tolerable latency violations).

-----

