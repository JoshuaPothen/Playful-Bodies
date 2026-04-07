"""
vision/camera.py — OpenCV webcam feed.
"""

import cv2
import logging

log = logging.getLogger(__name__)


class Camera:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._cap: cv2.VideoCapture | None = None

    def open(self):
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.device_index}")
        log.info("Camera opened (index %d, %dx%d)",
                 self.device_index,
                 int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self):
        """Return the next frame as a numpy array, or None on failure."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            log.warning("Failed to read frame from camera")
            return None
        return frame

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
