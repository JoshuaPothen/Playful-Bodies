"""
main.py — Entry point.

Phase 2: live camera feed with YOLOv8 person detection.
Press Q to quit.
"""

import logging

import cv2

from vision.camera import Camera
from vision.detector import PersonDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

WINDOW_TITLE = "Person Tracker — Q to quit"


def run():
    detector = PersonDetector()

    with Camera(device_index=0) as cam:
        log.info("Starting detection loop — press Q in the window to quit")
        prev_count = -1

        while True:
            frame = cam.read()
            if frame is None:
                log.error("Lost camera feed — exiting")
                break

            result = detector.detect(frame)

            # Log count only when it changes to avoid log spam.
            if result.person_count != prev_count:
                log.info("%d %s detected",
                         result.person_count,
                         "person" if result.person_count == 1 else "people")
                prev_count = result.person_count

            cv2.imshow(WINDOW_TITLE, result.annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    log.info("Done")


if __name__ == "__main__":
    run()
