"""
signal_controller.py
---------------------
The "AI brain" of the traffic management system.

Instead of fixed-timer traffic lights (e.g. always 30s green per direction),
this controller computes green-light duration dynamically based on:

  1. Real-time vehicle density per lane (from the detector)
  2. Anti-starvation logic (a lane that hasn't had green light for too long
     gets priority, even if it currently has fewer vehicles)
  3. Emergency vehicle override (ambulance/fire truck detected -> instant green)

This is a rule-based reinforcement-style controller. It's deliberately
readable so you can swap the decision function for a trained RL agent
(e.g. DQN using SUMO/TraCI) later without touching the rest of the pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import time


LANES = ["North", "East", "South", "West"]


@dataclass
class SignalConfig:
    min_green: int = 10          # seconds - every lane gets at least this much
    max_green: int = 60          # seconds - cap so one lane can't hog the cycle
    yellow_time: int = 3         # seconds
    all_red_time: int = 2        # seconds - clearance buffer between phases
    starvation_limit: int = 90   # seconds - force priority if lane waits longer than this
    emergency_green: int = 25    # seconds - forced green when emergency vehicle present


@dataclass
class LaneState:
    name: str
    vehicle_count: int = 0
    waiting_time: float = 0.0     # seconds since this lane last had green
    has_emergency: bool = False


class TrafficSignalController:
    """
    Decides, at every scheduling step, which lane gets the green light
    and for how long -- based on current vehicle counts across all lanes.
    """

    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self.lanes: Dict[str, LaneState] = {name: LaneState(name) for name in LANES}
        self.history: List[dict] = []
        self.cycle_number = 0

    def update_counts(self, counts: Dict[str, int], emergency: Dict[str, bool] = None):
        """Feed in the latest vehicle counts (from the detector) for each lane."""
        emergency = emergency or {}
        for name, lane in self.lanes.items():
            lane.vehicle_count = counts.get(name, 0)
            lane.has_emergency = emergency.get(name, False)

    def _tick_waiting_times(self, active_lane: str, elapsed: float):
        for name, lane in self.lanes.items():
            if name == active_lane:
                lane.waiting_time = 0.0
            else:
                lane.waiting_time += elapsed

    def compute_green_time(self, lane: LaneState, total_vehicles: int) -> int:
        """
        Proportional allocation: more vehicles -> more green time,
        bounded by [min_green, max_green].
        """
        cfg = self.config
        if total_vehicles == 0:
            return cfg.min_green

        share = lane.vehicle_count / total_vehicles
        green = cfg.min_green + share * (cfg.max_green - cfg.min_green)
        return int(round(max(cfg.min_green, min(cfg.max_green, green))))

    def select_next_lane(self) -> str:
        """
        Decide which lane goes next.
        Priority order:
          1. Emergency vehicle present -> immediate priority
          2. Starvation limit exceeded -> force priority
          3. Otherwise -> highest vehicle count
        """
        # 1. Emergency override
        for name, lane in self.lanes.items():
            if lane.has_emergency:
                return name

        # 2. Starvation prevention
        starved = [l for l in self.lanes.values() if l.waiting_time >= self.config.starvation_limit]
        if starved:
            return max(starved, key=lambda l: l.waiting_time).name

        # 3. Density-based selection
        return max(self.lanes.values(), key=lambda l: l.vehicle_count).name

    def run_cycle(self) -> dict:
        """
        Executes one full decision cycle: pick a lane, compute its green time,
        update waiting times for all other lanes, log the decision.
        Returns a dict describing the decision (used by simulator/dashboard).
        """
        total_vehicles = sum(l.vehicle_count for l in self.lanes.values())
        next_lane_name = self.select_next_lane()
        lane = self.lanes[next_lane_name]

        if lane.has_emergency:
            green_time = self.config.emergency_green
            reason = "EMERGENCY_VEHICLE"
        elif lane.waiting_time >= self.config.starvation_limit:
            green_time = self.compute_green_time(lane, total_vehicles)
            reason = "STARVATION_PREVENTION"
        else:
            green_time = self.compute_green_time(lane, total_vehicles)
            reason = "DENSITY_BASED"

        decision = {
            "cycle": self.cycle_number,
            "active_lane": next_lane_name,
            "green_time_sec": green_time,
            "yellow_time_sec": self.config.yellow_time,
            "reason": reason,
            "lane_counts": {n: l.vehicle_count for n, l in self.lanes.items()},
            "lane_waiting": {n: round(l.waiting_time, 1) for n, l in self.lanes.items()},
            "timestamp": time.time(),
        }

        # Simulate time passing: this lane gets green+yellow, others keep waiting
        elapsed = green_time + self.config.yellow_time + self.config.all_red_time
        self._tick_waiting_times(next_lane_name, elapsed)

        self.history.append(decision)
        self.cycle_number += 1
        return decision


if __name__ == "__main__":
    # Quick standalone test with made-up traffic counts
    controller = TrafficSignalController()
    sample_counts = [
        {"North": 12, "East": 3, "South": 7, "West": 2},
        {"North": 2, "East": 15, "South": 4, "West": 1},
        {"North": 5, "East": 5, "South": 5, "West": 20},
    ]
    for counts in sample_counts:
        controller.update_counts(counts)
        decision = controller.run_cycle()
        print(
            f"Cycle {decision['cycle']}: GREEN -> {decision['active_lane']} "
            f"for {decision['green_time_sec']}s (reason: {decision['reason']}) "
            f"| counts={decision['lane_counts']}"
        )
