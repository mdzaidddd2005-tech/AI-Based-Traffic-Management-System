"""
main.py
--------
Single entry point for the AI Traffic Management System.

Two modes:
  sim  -> runs the self-contained synthetic simulation (no extra downloads,
          works out of the box). Good for demos, testing signal logic,
          and understanding the system.
  live -> runs YOLOv8 vehicle detection on a real video file or camera
          index, feeding real counts into the same AI signal controller.
          Requires `pip install ultralytics torch` (see requirements.txt).

Examples:
    python main.py sim --frames 600 --out demo.mp4
    python main.py live --source traffic_footage.mp4
    python main.py live --source 0            # webcam
"""

import argparse


def run_sim(args):
    import simulate
    import sys
    sys.argv = ["simulate.py", "--frames", str(args.frames), "--out", args.out, "--fps", str(args.fps)]
    simulate.main()


def run_live(args):
    import cv2
    from vehicle_detector import VehicleDetector, default_four_way_lanes
    from signal_controller import TrafficSignalController, SignalConfig

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    detector = VehicleDetector(conf_threshold=args.conf)
    controller = TrafficSignalController(SignalConfig())

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read first frame.")
    h, w = frame.shape[:2]
    lanes = default_four_way_lanes(w, h)

    frame_idx = 0
    active_remaining = 0
    while ok:
        if active_remaining <= 0:
            counts, emergency = detector.count_per_lane(frame, lanes)
            controller.update_counts(counts, emergency)
            decision = controller.run_cycle()
            active_remaining = decision["green_time_sec"] + decision["yellow_time_sec"]
            print(
                f"[Cycle {decision['cycle']:>3}] GREEN -> {decision['active_lane']:<5} "
                f"for {decision['green_time_sec']}s ({decision['reason']}) counts={decision['lane_counts']}"
            )
        else:
            active_remaining -= 1

        ok, frame = cap.read()
        frame_idx += 1

    cap.release()
    print(f"Processed {frame_idx} frames, {controller.cycle_number} signal decisions.")


def main():
    parser = argparse.ArgumentParser(description="AI Traffic Management System")
    sub = parser.add_subparsers(dest="mode", required=True)

    sim_p = sub.add_parser("sim", help="run the synthetic simulation demo")
    sim_p.add_argument("--frames", type=int, default=600)
    sim_p.add_argument("--out", type=str, default="/mnt/user-data/outputs/traffic_demo.mp4")
    sim_p.add_argument("--fps", type=int, default=20)
    sim_p.set_defaults(func=run_sim)

    live_p = sub.add_parser("live", help="run YOLOv8 detection on real video/camera")
    live_p.add_argument("--source", type=str, required=True, help="video file path or camera index")
    live_p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    live_p.set_defaults(func=run_live)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
