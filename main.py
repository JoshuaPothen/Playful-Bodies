"""
main.py — Entry point.

Usage:
  python3 main.py              # normal mode with debug window
  python3 main.py --headless   # kiosk mode, no window

All tuning lives in config.yaml — no need to touch code.
"""

import argparse
import logging
import sys

from utils.config import cfg
from utils.logger import EventLogger
from state.machine import StateMachine
from vision.camera import Camera
from vision.detector import PersonDetector

# ── Logging ──────────────────────────────────────────────────────────────────
_level = getattr(logging, cfg["app"].get("log_level", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_level,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

WINDOW_TITLE = "Person Tracker — Q to quit"


def run(headless: bool):
    if not headless:
        import cv2  # only needed for the window

    csv_path = cfg["app"].get("events_csv", "")
    event_logger = EventLogger(csv_path)

    detector = PersonDetector(
        model_name=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
    )
    machine = StateMachine(event_logger=event_logger)
    cam = Camera(device_index=cfg["detection"]["camera_index"])

    try:
        cam.open()
        log.info("Running%s — press Ctrl-C or Q to quit",
                 " (headless)" if headless else "")

        consecutive_none = 0

        while True:
            frame = cam.read()

            if frame is None:
                consecutive_none += 1
                if consecutive_none >= 10:
                    log.error("Camera feed lost — exiting")
                    break
                continue  # camera is attempting recovery; skip frame
            consecutive_none = 0

            h, w = frame.shape[:2]
            result = detector.detect(frame)
            machine.update(result, frame_w=w, frame_h=h)

            if not headless:
                import cv2
                annotated = result.annotated_frame.copy()
                cv2.putText(annotated, machine.summary(result, w, h),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(WINDOW_TITLE, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        cam.close()
        event_logger.close()
        if not headless:
            import cv2
            cv2.destroyAllWindows()
        log.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crowd-reactive light controller")
    parser.add_argument("--headless", action="store_true",
                        help="Run without a display window (kiosk/exhibition mode)")
    args = parser.parse_args()

    # Allow config.yaml to also set headless mode.
    headless = args.headless or cfg["app"].get("headless", False)
    run(headless)
