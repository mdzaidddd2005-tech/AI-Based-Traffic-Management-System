"""
vehicle_detector.py
--------------------
Detects and counts vehicles per lane in a real video feed / camera stream
using YOLOv8 (Ultralytics), pretrained on COCO.

This module is what you'd point at a real intersection camera or a
recorded video file. It is decoupled from signal_controller.py --
it only produces per-lane vehicle counts + emergency-vehicle flags,
which the controller then consumes.

Requirements: pip install -r requirements.txt
(ultralytics will auto-download the yolov8n.pt weights on first run,
 needs internet access once.)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import cv2

# COCO class ids relevant to traffic
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
# NOTE: COCO has no native "ambulance" class. In production, either
# (a) fine-tune YOLO on an emergency-vehicle dataset, or
# (b) match a secondary classifier / color+siren-light heuristic
# on detected "truck"/"car" boxes. `is_emergency_vehicle()` below is
# a placeholder hook for that second stage.


@dataclass
class LaneRegion:
    """Defines a lane as a polygon region-of-interest within the frame."""
    name: str
    polygon: np.ndarray  # Nx2 array of (x, y) points


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35):
        from ultralytics import YOLO  # imported lazily so this file can be
        # inspected/tested without requiring torch to be installed
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Run YOLO on a single frame, return list of vehicle detections."""
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append({
                "class": VEHICLE_CLASSES[cls_id],
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": float(box.conf[0]),
            })
        return detections

    def count_per_lane(self, frame: np.ndarray, lanes: List[LaneRegion]) -> Tuple[Dict[str, int], Dict[str, bool]]:
        """
        Runs detection, then assigns each detected vehicle to a lane based
        on whether its center point falls inside that lane's polygon.
        Returns (counts_per_lane, emergency_flag_per_lane).
        """
        detections = self.detect(frame)
        counts = {lane.name: 0 for lane in lanes}
        emergency = {lane.name: False for lane in lanes}

        for det in detections:
            cx, cy = det["center"]
            for lane in lanes:
                if cv2.pointPolygonTest(lane.polygon, (cx, cy), False) >= 0:
                    counts[lane.name] += 1
                    if is_emergency_vehicle(det):
                        emergency[lane.name] = True
                    break
        return counts, emergency


def is_emergency_vehicle(detection: dict) -> bool:
    """
    Placeholder hook for emergency-vehicle recognition.
    Replace with a fine-tuned classifier / siren-light color heuristic.
    Currently always returns False.
    """
    return False


def default_four_way_lanes(frame_width: int, frame_height: int) -> List[LaneRegion]:
    """
    Convenience helper: splits the frame into 4 quadrants representing
    North/East/South/West approaches to a 4-way intersection.
    Replace with hand-calibrated polygons for your actual camera angle.
    """
    w, h = frame_width, frame_height
    return [
        LaneRegion("North", np.array([[0, 0], [w // 2, 0], [w // 2, h // 2], [0, h // 2]])),
        LaneRegion("East",  np.array([[w // 2, 0], [w, 0], [w, h // 2], [w // 2, h // 2]])),
        LaneRegion("South", np.array([[0, h // 2], [w // 2, h // 2], [w // 2, h], [0, h]])),
        LaneRegion("West",  np.array([[w // 2, h // 2], [w, h // 2], [w, h], [w // 2, h]])),
    ]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vehicle_detector.py <video_path_or_camera_index>")
        sys.exit(1)

    source = sys.argv[1]
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    detector = VehicleDetector()

    ok, frame = cap.read()
    if not ok:
        print("Could not open video source.")
        sys.exit(1)

    h, w = frame.shape[:2]
    lanes = default_four_way_lanes(w, h)

    while ok:
        counts, emergency = detector.count_per_lane(frame, lanes)
        print(counts, emergency)
        ok, frame = cap.read()

    cap.release()
