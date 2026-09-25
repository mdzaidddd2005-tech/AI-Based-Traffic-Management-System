# AI-Based Traffic Management System

A dynamic traffic-signal control system that replaces fixed-timer traffic
lights with AI decisions based on real-time vehicle density. Instead of
every lane getting a flat 30 seconds regardless of how busy it is, the
system watches each approach to an intersection and allocates green-light
time proportionally, while still guaranteeing no lane is starved.

## How it works

```
 Camera / Video ─▶ VehicleDetector ─▶ per-lane vehicle counts ─▶ TrafficSignalController ─▶ signal decision
  (or synthetic       (YOLOv8, or           {North: 5, East: 2,      (density + anti-        {lane: "North",
   simulation)      color/contour in         South: 8, West: 1}       starvation +             green_time: 34s,
                     simulate.py)                                      emergency logic)         reason: ...}
```

### 1. Detection (`vehicle_detector.py`)
Uses YOLOv8 (Ultralytics, pretrained on COCO) to detect cars, buses,
trucks, and motorcycles in a video frame, then buckets each detection
into a lane based on which region-of-interest polygon its center falls in.

### 2. AI Signal Controller (`signal_controller.py`)
The decision-making core. For each cycle it:
- Computes green time proportional to vehicle count: `min_green + share * (max_green - min_green)`, bounded to `[min_green, max_green]`.
- Prevents starvation: if a lane hasn't had a green light in `starvation_limit` seconds, it's forced to the front of the queue regardless of density.
- Supports emergency vehicle override: if an ambulance/fire truck is flagged in a lane, that lane gets immediate priority.

This part is intentionally rule-based and readable so you can swap in a
trained RL agent (e.g. DQN trained via SUMO/TraCI) later without touching
the rest of the pipeline.

### 3. Simulation (`simulate.py`)
A fully self-contained, runnable demo — no model downloads needed. It:
- Renders a synthetic 4-way intersection with vehicles that spawn, queue at red lights, and move through on green.
- Runs **real** OpenCV detection (color threshold + contour finding) on each rendered frame to count vehicles — the same interface `vehicle_detector.py` uses, just swapping YOLO for a simpler CV technique so it runs instantly with no dependencies beyond OpenCV.
- Feeds those counts into the exact same `TrafficSignalController` used in production.
- Renders the result to an MP4 with live signal lights, counts, and decision reasoning overlaid.

## Setup

```bash
pip install -r requirements.txt
```

`opencv-python` and `numpy` are enough to run the simulation. `ultralytics`
and `torch` are only needed for real-camera/video mode.

## Usage

**Run the simulation demo (works immediately, no setup):**
```bash
python main.py sim --frames 600 --out traffic_demo.mp4
```

**Run on a real video file or webcam (needs YOLOv8 weights, auto-downloaded on first run):**
```bash
python main.py live --source path/to/traffic_video.mp4
python main.py live --source 0   # webcam
```

**Test the signal logic in isolation:**
```bash
python signal_controller.py
```

## Files

| File | Purpose |
|---|---|
| `signal_controller.py` | Core AI decision logic — density-based green time, anti-starvation, emergency override |
| `vehicle_detector.py` | YOLOv8-based detector for real camera/video input |
| `simulate.py` | Self-contained synthetic simulation + real CV detection + rendering |
| `main.py` | CLI entry point for both `sim` and `live` modes |
| `requirements.txt` | Dependencies |

## Extending this project

- **Emergency vehicle detection**: `is_emergency_vehicle()` in `vehicle_detector.py` is a stub — fine-tune YOLO on an ambulance/fire-truck dataset, or add a siren-light color heuristic.
- **Lane calibration**: `default_four_way_lanes()` splits the frame into 4 quadrants — for a real camera, replace with hand-drawn polygon ROIs matching your actual camera angle.
- **Reinforcement learning**: swap `TrafficSignalController.select_next_lane()` / `compute_green_time()` for a trained RL policy (SUMO + TraCI is the standard simulator for this).
- **Multi-intersection coordination**: extend the controller to share state between adjacent intersections for green-wave synchronization.
- **Dashboard**: wrap `main.py live` in a Flask/Streamlit app to show live counts and signal state in a browser.
