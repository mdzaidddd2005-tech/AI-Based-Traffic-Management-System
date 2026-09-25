"""
simulate.py
------------
An end-to-end, runnable demo of the whole system that doesn't require
downloading YOLO weights or having real camera footage.

It does three REAL things (not mocked):
  1. Generates a synthetic 4-way intersection with vehicles that spawn,
     queue, and move through when their light is green.
  2. Runs actual computer-vision detection (color threshold + contour
     detection, OpenCV) on each rendered frame to count vehicles per
     lane -- i.e. the "detector" doesn't get to cheat by reading ground
     truth, it has to find vehicles in the image, same as vehicle_detector.py
     would with YOLO on a real camera.
  3. Feeds those counts into the same TrafficSignalController used in
     production (signal_controller.py) to decide which light turns green
     and for how long.

Run:
    python simulate.py --frames 900 --out demo.mp4

Output: an MP4 you can watch, showing the intersection, live vehicle
counts, and the AI controller's signal decisions overlaid, plus a
cycle-by-cycle log printed to the console.
"""

import argparse
import random
from dataclasses import dataclass, field
from typing import List, Dict

import cv2
import numpy as np

from signal_controller import TrafficSignalController, SignalConfig, LANES

# ---- layout constants ----
W, H = 800, 800
CX, CY = W // 2, H // 2
ROAD_HALF_WIDTH = 70
VEHICLE_COLOR_BGR = (60, 160, 250)   # orange-ish -- what our "detector" looks for
VEHICLE_SIZE = (22, 34)              # w, h for vertical-lane vehicles

LANE_ROIS = {
    # (x1, y1, x2, y2) approach box for each direction, used for detection + spawn
    "North": (CX - ROAD_HALF_WIDTH, 0, CX, CY - ROAD_HALF_WIDTH),
    "South": (CX, CY + ROAD_HALF_WIDTH, CX + ROAD_HALF_WIDTH, H),
    "East":  (CX + ROAD_HALF_WIDTH, CY - ROAD_HALF_WIDTH, W, CY),
    "West":  (0, CY, CX - ROAD_HALF_WIDTH, CY + ROAD_HALF_WIDTH),
}

STOP_LINE = {
    "North": CY - ROAD_HALF_WIDTH,
    "South": CY + ROAD_HALF_WIDTH,
    "East": CX + ROAD_HALF_WIDTH,
    "West": CX - ROAD_HALF_WIDTH,
}


@dataclass
class Vehicle:
    lane: str
    pos: float  # distance from spawn edge, grows toward stop line / through intersection


def spawn_position(lane: str) -> float:
    return 0.0


def vehicle_pixel_rect(v: Vehicle):
    """Convert a Vehicle's 1D position along its lane into a screen rect."""
    vw, vh = VEHICLE_SIZE
    if v.lane == "North":
        x = CX - ROAD_HALF_WIDTH // 2 - vw // 2
        y = v.pos
        return (int(x), int(y), int(x + vw), int(y + vh))
    if v.lane == "South":
        x = CX + ROAD_HALF_WIDTH // 2 - vw // 2
        y = H - v.pos - vh
        return (int(x), int(y), int(x + vw), int(y + vh))
    if v.lane == "East":
        x = W - v.pos - vh
        y = CY - ROAD_HALF_WIDTH // 2 - vw // 2
        return (int(x), int(y), int(x + vh), int(y + vw))
    if v.lane == "West":
        x = v.pos
        y = CY + ROAD_HALF_WIDTH // 2 - vw // 2
        return (int(x), int(y), int(x + vh), int(y + vw))
    raise ValueError(v.lane)


def at_stop_line(v: Vehicle) -> bool:
    vw, vh = VEHICLE_SIZE
    if v.lane == "North":
        return v.pos + vh >= STOP_LINE["North"]
    if v.lane == "South":
        return H - v.pos - vh <= STOP_LINE["South"]
    if v.lane == "East":
        return W - v.pos - vh <= STOP_LINE["East"]
    if v.lane == "West":
        return v.pos + vh >= STOP_LINE["West"]


class IntersectionSim:
    def __init__(self, spawn_rate: float = 0.12):
        self.vehicles: List[Vehicle] = []
        self.spawn_rate = spawn_rate
        self.active_lane = None
        self.active_remaining = 0
        self.controller = TrafficSignalController(SignalConfig(min_green=6, max_green=25, yellow_time=2, all_red_time=1))
        self.current_decision = None
        self.frame_num = 0

    def maybe_spawn(self):
        for lane in LANES:
            if random.random() < self.spawn_rate:
                self.vehicles.append(Vehicle(lane=lane, pos=spawn_position(lane)))

    def step_vehicles(self):
        speed = 6
        for v in self.vehicles:
            is_green = (self.active_lane == v.lane and self.active_remaining > 0)
            blocked = at_stop_line(v) and not is_green
            if not blocked:
                v.pos += speed
        # remove vehicles that exited the frame
        self.vehicles = [v for v in self.vehicles if v.pos < max(W, H) + 40]

    def render_detection_frame(self) -> np.ndarray:
        """Renders the raw scene (this is what a 'camera' would see)."""
        frame = np.full((H, W, 3), (40, 40, 40), dtype=np.uint8)
        # roads
        cv2.rectangle(frame, (CX - ROAD_HALF_WIDTH, 0), (CX + ROAD_HALF_WIDTH, H), (70, 70, 70), -1)
        cv2.rectangle(frame, (0, CY - ROAD_HALF_WIDTH), (W, CY + ROAD_HALF_WIDTH), (70, 70, 70), -1)
        for v in self.vehicles:
            x1, y1, x2, y2 = vehicle_pixel_rect(v)
            cv2.rectangle(frame, (x1, y1), (x2, y2), VEHICLE_COLOR_BGR, -1)
        return frame

    def detect_counts(self, frame: np.ndarray) -> Dict[str, int]:
        """
        REAL detection step: color-threshold the frame to find vehicle pixels,
        find contours, then bucket each contour centroid into a lane ROI.
        This stands in for YOLO inference in vehicle_detector.py -- same
        interface (frame in, per-lane counts out).
        """
        lower = np.array([v - 15 for v in VEHICLE_COLOR_BGR])
        upper = np.array([v + 15 for v in VEHICLE_COLOR_BGR])
        mask = cv2.inRange(frame, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counts = {lane: 0 for lane in LANES}
        for c in contours:
            if cv2.contourArea(c) < 20:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            for lane, (x1, y1, x2, y2) in LANE_ROIS.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    counts[lane] += 1
                    break
        return counts

    def draw_overlay(self, frame: np.ndarray, counts: Dict[str, int]):
        # signal lights near each stop line
        colors = {"green": (0, 200, 0), "yellow": (0, 200, 200), "red": (0, 0, 200)}
        light_pos = {
            "North": (CX - ROAD_HALF_WIDTH - 25, CY - ROAD_HALF_WIDTH - 10),
            "South": (CX + ROAD_HALF_WIDTH + 10, CY + ROAD_HALF_WIDTH - 10),
            "East":  (CX + ROAD_HALF_WIDTH + 10, CY - ROAD_HALF_WIDTH - 25),
            "West":  (CX - ROAD_HALF_WIDTH - 25, CY + ROAD_HALF_WIDTH + 10),
        }
        for lane in LANES:
            state = "red"
            if self.active_lane == lane and self.active_remaining > 0:
                state = "green" if self.active_remaining > 2 else "yellow"
            x, y = light_pos[lane]
            cv2.circle(frame, (x, y), 10, colors[state], -1)
            cv2.putText(frame, f"{lane}:{counts.get(lane,0)}", (x - 30, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if self.current_decision:
            d = self.current_decision
            txt = f"Cycle {d['cycle']} | GREEN: {d['active_lane']} for {d['green_time_sec']}s | {d['reason']}"
            cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"frame {self.frame_num}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def step(self, writer: cv2.VideoWriter):
        self.maybe_spawn()

        frame = self.render_detection_frame()
        counts = self.detect_counts(frame)

        # every time the active phase runs out, ask the AI controller for the next one
        if self.active_lane is None or self.active_remaining <= 0:
            self.controller.update_counts(counts)
            decision = self.controller.run_cycle()
            self.current_decision = decision
            self.active_lane = decision["active_lane"]
            self.active_remaining = decision["green_time_sec"] + decision["yellow_time_sec"]
            print(
                f"[Cycle {decision['cycle']:>3}] GREEN -> {decision['active_lane']:<5} "
                f"for {decision['green_time_sec']}s  ({decision['reason']})  "
                f"counts={decision['lane_counts']}"
            )
        else:
            self.active_remaining -= 1

        self.step_vehicles()
        self.draw_overlay(frame, counts)
        writer.write(frame)
        self.frame_num += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=600, help="number of frames to simulate")
    parser.add_argument("--out", type=str, default="/mnt/user-data/outputs/traffic_demo.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    sim = IntersectionSim()
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    for _ in range(args.frames):
        sim.step(writer)

    writer.release()
    print(f"\nSaved video to {args.out}")
    print(f"Total signal decisions made: {sim.controller.cycle_number}")


if __name__ == "__main__":
    main()
