"""
vision/gestures.py — Gesture detection functions.

Thresholds are loaded from config.yaml so they can be tuned without
touching code. Each per-person function takes a Pose; crowd functions
take list[Pose] + frame dimensions.
"""

from utils.config import cfg
from vision.detector import Pose

_g = cfg["gestures"]

# ── Squat ────────────────────────────────────────────────────────────────────
SQUAT_HIP_KNEE_RATIO = _g["squat_hip_knee_ratio"]


def is_squatting(pose: Pose) -> bool:
    """True if hips have dropped to near knee height."""
    hip_ys, knee_ys = [], []
    for side in ("left", "right"):
        hip = pose.kp(f"{side}_hip")
        knee = pose.kp(f"{side}_knee")
        if hip and knee and knee.y > 0:
            hip_ys.append(hip.y)
            knee_ys.append(knee.y)
    if not hip_ys:
        return False
    avg_knee_y = sum(knee_ys) / len(knee_ys)
    if avg_knee_y <= 0:
        return False
    return (sum(hip_ys) / len(hip_ys)) / avg_knee_y >= SQUAT_HIP_KNEE_RATIO


# ── Hands raised ─────────────────────────────────────────────────────────────
HANDS_RAISED_MIN_WRISTS = _g["hands_raised_min_wrists"]


def is_hands_raised(pose: Pose) -> bool:
    """True if the person has the required number of wrists raised above shoulder."""
    raised = 0
    for side in ("left", "right"):
        wrist = pose.kp(f"{side}_wrist")
        shoulder = pose.kp(f"{side}_shoulder")
        if wrist and shoulder and wrist.y < shoulder.y:
            raised += 1
    return raised >= HANDS_RAISED_MIN_WRISTS


# ── Huddle direction ─────────────────────────────────────────────────────────
HUDDLE_LEFT_THRESHOLD  = _g["huddle_left_threshold"]
HUDDLE_RIGHT_THRESHOLD = _g["huddle_right_threshold"]
HUDDLE_MIN_PEOPLE      = _g["huddle_min_people"]


def huddle_direction(poses: list[Pose], frame_w: int) -> str | None:
    """Returns 'left', 'right', or None based on crowd centre of mass."""
    if len(poses) < HUDDLE_MIN_PEOPLE or frame_w == 0:
        return None
    centres_x = [((p.x1 + p.x2) / 2) / frame_w for p in poses]
    com_x = sum(centres_x) / len(centres_x)
    if com_x < HUDDLE_LEFT_THRESHOLD:
        return "left"
    if com_x > HUDDLE_RIGHT_THRESHOLD:
        return "right"
    return None


# ── Stillness ─────────────────────────────────────────────────────────────────
def mean_movement(
    current_poses: list[Pose],
    previous_centres: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
) -> float:
    """
    Average normalised displacement of person centres between frames.
    Matches each current centre to its nearest previous centre.
    Returns 0.0 if there are no people or no previous data.
    """
    if not current_poses or not previous_centres or frame_w == 0 or frame_h == 0:
        return 0.0

    current_centres = [
        ((p.x1 + p.x2) / 2 / frame_w, (p.y1 + p.y2) / 2 / frame_h)
        for p in current_poses
    ]

    total = 0.0
    for cx, cy in current_centres:
        nearest = min(
            ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            for px, py in previous_centres
        )
        total += nearest

    return total / len(current_centres)
