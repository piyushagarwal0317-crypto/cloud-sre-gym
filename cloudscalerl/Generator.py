"""
Workload Trace Generator
=========================
Generates and saves JSON workload trace files used by the environment.
Run directly to regenerate traces:
    python -m server.workload_traces.generator --seed 42 --pattern all
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

TRACES_DIR = Path(__file__).parent
TICK_DURATION_S = 60


def generate_trace(seed: int, pattern: str, length: int) -> list[float]:
    rng = random.Random(seed)
    trace: list[float] = []

    for t in range(length):
        hour = (t * TICK_DURATION_S / 3600) % 24

        if pattern == "diurnal":
            # Business-hours sine wave + lunch peak
            base = 0.4 + 0.4 * math.sin(math.pi * (hour - 6) / 12)
            lunch_boost = 0.15 * math.exp(-((hour - 12) ** 2) / 2)
            evening_drop = -0.1 * math.exp(-((hour - 20) ** 2) / 4)
            val = base + lunch_boost + evening_drop + rng.gauss(0, 0.02)
            # CPU spikes at ticks 60, 240, 420 (task1 requirement)
            if t in (60, 61, 240, 241, 420, 421):
                val = 1.8

        elif pattern == "flash_sale":
            # Weekly baseline + 3× spike at tick 240
            day_of_week = (t * TICK_DURATION_S / 3600 / 24) % 7
            weekend = 0.7 if day_of_week >= 5 else 1.0  # weekend dip
            base = weekend * (0.3 + 0.2 * math.sin(math.pi * (hour - 6) / 12))
            spike = 3.0 if 240 <= t <= 270 else 1.0
            val = base * spike + rng.gauss(0, 0.02)

        elif pattern == "incident":
            # Steady traffic throughout the incident
            base = 0.5 + 0.2 * math.sin(math.pi * hour / 24)
            # Slight surge as traffic hits healthy regions post-incident
            if 30 <= t <= 45:
                base *= 1.3
            val = base + rng.gauss(0, 0.02)

        elif pattern == "adversarial":
            # Designed to fool threshold-based policies
            phase = t % 20
            if phase < 2:
                val = 1.8   # force over-provisioning trigger
            elif phase < 10:
                val = 0.2   # sudden drop → wasted resources
            elif phase < 15:
                val = 0.85  # near threshold — uncertain
            else:
                val = 0.6
            val += rng.gauss(0, 0.02)

        elif pattern == "seasonal":
            # Multi-day with weekend dips
            day = (t * TICK_DURATION_S / 3600 / 24) % 7
            weekend = 0.5 if day >= 5 else 1.0
            base = weekend * (0.4 + 0.35 * math.sin(math.pi * (hour - 6) / 12))
            val = base + rng.gauss(0, 0.02)

        else:
            val = 0.5 + rng.gauss(0, 0.05)

        trace.append(round(max(0.05, min(3.0, val)), 4))

    return trace


def save_trace(pattern: str, seed: int = 42, length: int = 720) -> None:
    trace = generate_trace(seed, pattern, length)
    out = TRACES_DIR / f"{pattern}.json"
    with open(out, "w") as f:
        json.dump(trace, f, separators=(",", ":"))
    print(f"Saved {len(trace)} ticks → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CloudScaleRL workload traces")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pattern",
        default="all",
        choices=["all", "diurnal", "flash_sale", "incident", "adversarial", "seasonal"],
    )
    parser.add_argument("--length", type=int, default=720)
    args = parser.parse_args()

    patterns = (
        ["diurnal", "flash_sale", "incident", "adversarial", "seasonal"]
        if args.pattern == "all"
        else [args.pattern]
    )
    for p in patterns:
        save_trace(p, seed=args.seed, length=args.length)


if __name__ == "__main__":
    main()