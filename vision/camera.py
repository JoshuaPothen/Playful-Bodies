"""
vision/camera.py — OpenCV webcam feed with auto-recovery.

If the feed drops mid-session, the camera will attempt to reopen
up to RECOVERY_ATTEMPTS times before giving up.
"""

import logging
import time

import cv2

log = logging.getLogger(__name__)

RECOVERY_ATTEMPTS = 5
RECOVERY_DELAY    = 2.0  # seconds between reopen attempts


class Camera:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._cap: cv2.VideoCapture | None = None
        self._consecutive_failures = 0

    def open(self):
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.device_index}")
        self._consecutive_failures = 0
        log.info("Camera opened (index %d, %dx%d)",
                 self.device_index,
                 int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self):
        """
        Return the next frame, or None if the feed is unrecoverable.
        Automatically attempts to reopen the camera on failure.
        """
        if self._cap is None:
            return None

        ok, frame = self._cap.read()
        if ok:
            self._consecutive_failures = 0
            return frame

        self._consecutive_failures += 1
        log.warning("Frame read failed (attempt %d/%d)",
                    self._consecutive_failures, RECOVERY_ATTEMPTS)

        if self._consecutive_failures >= RECOVERY_ATTEMPTS:
            log.error("Camera feed unrecoverable after %d attempts", RECOVERY_ATTEMPTS)
            return None

        # Try to reopen.
        log.info("Attempting camera recovery in %.0fs…", RECOVERY_DELAY)
        time.sleep(RECOVERY_DELAY)
        self._cap.release()
        self._cap = cv2.VideoCapture(self.device_index)
        if self._cap.isOpened():
            log.info("Camera recovered")
            self._consecutive_failures = 0

        return None  # skip this frame; caller loops

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None
            log.info("Camera closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
