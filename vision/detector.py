"""
vision/detector.py — YOLOv8-nano person detector.

Wraps ultralytics YOLOv8 to detect people in a frame and return
annotated output alongside a person count.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class 0 is "person"


@dataclass
class Detection:
    """A single detected person bounding box (pixel coords)."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    annotated_frame: np.ndarray | None = None

    @property
    def person_count(self) -> int:
        return len(self.detections)


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.4):
        log.info("Loading YOLO model: %s", model_name)
        self._model = YOLO(model_name)
        self._confidence = confidence
        log.info("Model ready (confidence threshold %.2f)", confidence)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a frame. Returns detections and an annotated copy."""
        results = self._model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self._confidence,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append(Detection(x1, y1, x2, y2, conf))

        annotated = results.plot()  # draws boxes + labels on a copy of the frame

        return DetectionResult(detections=detections, annotated_frame=annotated)
