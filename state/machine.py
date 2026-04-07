"""
state/machine.py — Interaction state machine.

States (highest to lowest priority):
  HANDS_UP      proportion of people with wrists raised → full white
  SQUAT_ACTIVE  proportion of people squatting          → vivid purple
  HUDDLE_RIGHT  crowd drifts right                      → warm orange
  HUDDLE_LEFT   crowd drifts left                       → cool blue
  STILLNESS     no movement for X seconds               → dim warm
  IDLE          default                                 → soft warm white

Triggers use proportions ("50% of detected people") rather than hard
counts, so the system scales from a handful of people to a full gallery.
All thresholds and colours are read from config.yaml.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto

import hue.controller as hue
from utils.config import cfg
from utils.logger import EventLogger
from vision.detector import DetectionResult
from vision.gestures import huddle_direction, is_hands_raised, is_squatting, mean_movement

log = logging.getLogger(__name__)

_s = cfg["states"]


class State(Enum):
    IDLE         = auto()
    STILLNESS    = auto()
    HUDDLE_LEFT  = auto()
    HUDDLE_RIGHT = auto()
    SQUAT_ACTIVE = auto()
    HANDS_UP     = auto()


STATE_PRIORITY = [
    State.IDLE,
    State.STILLNESS,
    State.HUDDLE_LEFT,
    State.HUDDLE_RIGHT,
    State.SQUAT_ACTIVE,
    State.HANDS_UP,
]


@dataclass
class StateConfig:
    colour: tuple[int, int, int]
    min_hold: float
    label: str

    def apply(self):
        hue.set_color(*self.colour)


def _load_state(key: str, label: str) -> StateConfig:
    s = _s[key]
    return StateConfig(colour=tuple(s["colour"]), min_hold=s["min_hold"], label=label)


STATE_CONFIGS: dict[State, StateConfig] = {
    State.IDLE:         _load_state("idle",         "IDLE"),
    State.STILLNESS:    _load_state("stillness",    "STILLNESS"),
    State.HUDDLE_LEFT:  _load_state("huddle_left",  "HUDDLE LEFT"),
    State.HUDDLE_RIGHT: _load_state("huddle_right", "HUDDLE RIGHT"),
    State.SQUAT_ACTIVE: _load_state("squat_active", "SQUAT"),
    State.HANDS_UP:     _load_state("hands_up",     "HANDS UP"),
}

SQUAT_TRIGGER_RATIO   = _s["squat_trigger_ratio"]
HANDS_UP_RATIO        = _s["hands_up_ratio"]
MIN_PEOPLE            = _s["min_people"]
STILL_MOVEMENT_THRESH = cfg["gestures"]["still_movement_thresh"]
STILL_DURATION        = cfg["gestures"]["still_duration"]


class StateMachine:
    def __init__(self, event_logger: EventLogger | None = None):
        self._state = State.IDLE
        self._entered_at = time.monotonic()
        self._prev_centres: list[tuple[float, float]] = []
        self._low_movement_since: float | None = None
        self._event_logger = event_logger
        STATE_CONFIGS[State.IDLE].apply()
        log.info("StateMachine started → IDLE")

    @property
    def state(self) -> State:
        return self._state

    def update(self, result: DetectionResult, frame_w: int, frame_h: int) -> State:
        """Evaluate all conditions and transition if warranted. Call once per frame."""
        poses = result.poses
        n = len(poses)

        squat_count = sum(1 for p in poses if is_squatting(p))
        hands_count = sum(1 for p in poses if is_hands_raised(p))
        huddle_dir  = huddle_direction(poses, frame_w)
        is_still    = self._check_stillness(poses, frame_w, frame_h)

        # Proportion-based triggers — scale to crowd size.
        squat_ratio = squat_count / n if n >= MIN_PEOPLE else 0.0
        hands_ratio = hands_count / n if n >= MIN_PEOPLE else 0.0

        active: set[State] = {State.IDLE}
        if is_still and n >= MIN_PEOPLE:
            active.add(State.STILLNESS)
        if huddle_dir == "left":
            active.add(State.HUDDLE_LEFT)
        if huddle_dir == "right":
            active.add(State.HUDDLE_RIGHT)
        if squat_ratio >= SQUAT_TRIGGER_RATIO:
            active.add(State.SQUAT_ACTIVE)
        if hands_ratio >= HANDS_UP_RATIO:
            active.add(State.HANDS_UP)

        target = max(active, key=lambda s: STATE_PRIORITY.index(s))

        if target != self._state:
            held = time.monotonic() - self._entered_at
            if held >= STATE_CONFIGS[self._state].min_hold:
                self._transition(target, n, squat_count, hands_count, huddle_dir)

        return self._state

    def summary(self, result: DetectionResult, frame_w: int, frame_h: int) -> str:
        poses = result.poses
        n = len(poses)
        squat_count = sum(1 for p in poses if is_squatting(p))
        hands_count = sum(1 for p in poses if is_hands_raised(p))
        squat_pct = int(squat_count / n * 100) if n else 0
        hands_pct = int(hands_count / n * 100) if n else 0
        return (
            f"{STATE_CONFIGS[self._state].label}  |  "
            f"People:{n}  Squat:{squat_count}({squat_pct}%)  Hands:{hands_count}({hands_pct}%)"
        )

    def _transition(self, new_state: State, n: int, squat_count: int, hands_count: int, huddle: str | None):
        old = STATE_CONFIGS[self._state]
        new = STATE_CONFIGS[new_state]
        log.info("STATE %s → %s  (people=%d squat=%d hands=%d huddle=%s)",
                 old.label, new.label, n, squat_count, hands_count, huddle or "none")
        if self._event_logger:
            self._event_logger.log_transition(old.label, new.label, n, squat_count, hands_count)
        self._state = new_state
        self._entered_at = time.monotonic()
        new.apply()

    def _check_stillness(self, poses, frame_w: int, frame_h: int) -> bool:
        movement = mean_movement(poses, self._prev_centres, frame_w, frame_h)
        self._prev_centres = (
            [((p.x1 + p.x2) / 2 / frame_w, (p.y1 + p.y2) / 2 / frame_h) for p in poses]
            if frame_w and frame_h else []
        )
        now = time.monotonic()
        if movement < STILL_MOVEMENT_THRESH and poses:
            if self._low_movement_since is None:
                self._low_movement_since = now
            return (now - self._low_movement_since) >= STILL_DURATION
        self._low_movement_since = None
        return False
