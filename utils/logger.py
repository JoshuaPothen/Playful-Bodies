"""
utils/logger.py — CSV event logger for state transitions.

Writes one row per state change:
  timestamp, from_state, to_state, person_count, squat_count, hands_count

Set events_csv to "" in config.yaml to disable.
"""

import csv
import logging
import pathlib
from datetime import datetime

log = logging.getLogger(__name__)


class EventLogger:
    def __init__(self, csv_path: str):
        self._path = pathlib.Path(csv_path) if csv_path else None
        self._file = None
        self._writer = None
        if self._path:
            self._open()

    def _open(self):
        new_file = not self._path.exists()
        self._file = open(self._path, "a", newline="")
        self._writer = csv.writer(self._file)
        if new_file:
            self._writer.writerow(
                ["timestamp", "from_state", "to_state", "person_count", "squat_count", "hands_count"]
            )
            self._file.flush()
        log.info("Event log: %s", self._path.resolve())

    def log_transition(
        self,
        from_state: str,
        to_state: str,
        person_count: int,
        squat_count: int,
        hands_count: int,
    ):
        if not self._writer:
            return
        self._writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            from_state,
            to_state,
            person_count,
            squat_count,
            hands_count,
        ])
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
