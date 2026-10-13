"""
liveness_detection.py
=====================
Decides whether a detected face is a REAL person or a presentation attack
(printed photo / phone screen) using the MiniFASNetV2 anti-spoofing model.

It does NOT detect or recognise faces. It only needs the frame and the bounding
boxes your InsightFace code already produces, so your detection/recognition code
stays untouched.

Pipeline for every face
    1. Crop the face with 2.7x context around the box and resize to 80x80
       (exactly the preprocessing the model was trained with).
    2. Run the ONNX model with onnxruntime (CPU). Output = probabilities for
       [spoof, REAL, spoof]; index 1 is "real".
    3. Smooth: each face keeps its last 5 verdicts and the MAJORITY wins
       (ties go to FAKE - the safer side). Until `min_votes` verdicts exist the
       label is "PENDING" so one noisy first frame cannot trigger an alert.

Faces are followed from frame to frame by box overlap (IoU), independent of
who the person is, so a real person standing next to a photo of themselves
has a separate vote history.

Model file: models/MiniFASNetV2.onnx next to this file (create it once with
convert_minifasnet_to_onnx.py).

Requires: numpy, opencv-python, onnxruntime  (onnxruntime comes with insightface)
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Optional, Sequence

import cv2
import numpy as np

from risk_calculation import FAKE, PENDING, REAL

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "MiniFASNetV2.onnx")


@dataclass
class LivenessResult:
    label: str            # "REAL", "FAKE" or "PENDING"
    real_prob: float      # model's probability that THIS frame is a real face (0-1), -1 if not run
    votes: int            # how many frame verdicts the label is based on
    reason: str = ""      # why it is PENDING (too small, collecting votes, error)

    @property
    def decided(self) -> bool:
        return self.label in (REAL, FAKE)


@dataclass
class _Track:
    box: np.ndarray                       # x1, y1, x2, y2
    votes: Deque[bool]                    # True = real
    missed: int = 0                       # consecutive frames in which this face was not found


# ----------------------------------------------------------------------
def _crop_with_scale(img: np.ndarray, x1, y1, x2, y2, scale: float, out_size: int) -> np.ndarray:
    """Same crop as the original Silent-Face-Anti-Spoofing project (CropImage)."""
    src_h, src_w = img.shape[:2]
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)

    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x1 + box_w / 2, y1 + box_h / 2
    left, top = cx - new_w / 2, cy - new_h / 2
    right, bottom = cx + new_w / 2, cy + new_h / 2

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1

    patch = img[int(top): int(bottom) + 1, int(left): int(right) + 1]
    return cv2.resize(patch, (out_size, out_size))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_face(a: np.ndarray, b: np.ndarray) -> float:
    """Match score between two boxes: IoU, or a fallback on centre distance for fast motion."""
    iou = _iou(a, b)
    if iou >= 0.2:
        return iou
    size = 0.5 * ((a[2] - a[0]) + (b[2] - b[0]))
    dist = np.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)
    return 0.1 if size > 0 and dist < 0.6 * size else 0.0


# ----------------------------------------------------------------------
class LivenessDetector:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        real_threshold: float = 0.5,
        history: int = 5,
        min_votes: int = 3,
        min_face_px: int = 40,
        scale: float = 2.7,
        input_size: int = 80,
        max_missed_frames: int = 5,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Anti-spoofing model not found: {model_path}\n"
                "Create it once with:  python convert_minifasnet_to_onnx.py --repo <path to "
                "Silent-Face-Anti-Spoofing>   (see the instructions that came with this module)."
            )
        import onnxruntime as ort   # imported here so a missing package gives a clear error

        self.session = ort.InferenceSession(model_path, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name

        self.real_threshold = real_threshold
        self.history = history
        self.min_votes = min(min_votes, history)
        self.min_face_px = min_face_px
        self.scale = scale
        self.input_size = input_size
        self.max_missed_frames = max_missed_frames

        self._tracks: List[_Track] = []
        self._error_logged = False

    # ------------------------------------------------------------------
    def predict_raw(self, frame: np.ndarray, bbox) -> float:
        """Single-frame probability (0-1) that the face inside `bbox` is real. No smoothing."""
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        patch = _crop_with_scale(frame, x1, y1, x2, y2, self.scale, self.input_size)
        return float(self._run([patch])[0])

    def assess(
        self, frame: np.ndarray, bboxes: Sequence, now: Optional[datetime] = None
    ) -> List[LivenessResult]:
        """
        Smoothed REAL/FAKE verdict for every face of this frame.
        `bboxes` = [x1, y1, x2, y2] per face (face.bbox from InsightFace).
        Returns one LivenessResult per box, same order. Never raises.
        """
        if len(bboxes) == 0:
            self._age_tracks(set())
            return []

        boxes = [self._clean_box(b, frame.shape) for b in bboxes]
        results: List[Optional[LivenessResult]] = [None] * len(boxes)

        # 1) which faces can be judged at all?
        usable = []
        for i, b in enumerate(boxes):
            if b is None:
                results[i] = LivenessResult(PENDING, -1.0, 0, "invalid box")
            elif min(b[2] - b[0], b[3] - b[1]) < self.min_face_px:
                results[i] = LivenessResult(PENDING, -1.0, 0, "face too small")
            else:
                usable.append(i)

        # 2) run the model on all usable faces in one batch
        probs = {}
        if usable:
            try:
                patches = [
                    _crop_with_scale(frame, *boxes[i].astype(int), self.scale, self.input_size)
                    for i in usable
                ]
                for i, p in zip(usable, self._run(patches)):
                    probs[i] = float(p)
            except Exception:                                    # noqa: BLE001
                if not self._error_logged:
                    logger.exception("Liveness model failed; faces stay PENDING")
                    self._error_logged = True
                for i in usable:
                    results[i] = LivenessResult(PENDING, -1.0, 0, "model error")
                usable = []

        # 3) attach to tracks and vote
        taken = set()
        for i in usable:
            track = self._match_track(boxes[i], taken)
            if track is None:
                track = _Track(boxes[i], deque(maxlen=self.history))
                self._tracks.append(track)
            taken.add(id(track))
            track.box = boxes[i]
            track.votes.append(probs[i] >= self.real_threshold)

            n = len(track.votes)
            if n < self.min_votes:
                results[i] = LivenessResult(PENDING, probs[i], n, "collecting frames")
            else:
                real_votes = sum(track.votes)
                label = REAL if real_votes * 2 > n else FAKE        # tie -> FAKE
                results[i] = LivenessResult(label, probs[i], n)

        self._age_tracks(taken)
        return results  # type: ignore[return-value]

    def reset(self) -> None:
        self._tracks.clear()

    def _age_tracks(self, matched: set) -> None:
        """Forget faces that have been missing for more than `max_missed_frames` frames.
        (Counted in frames, not seconds, so a slow CPU does not break the smoothing.)"""
        kept = []
        for t in self._tracks:
            if id(t) in matched:
                t.missed = 0
                kept.append(t)
            else:
                t.missed += 1
                if t.missed <= self.max_missed_frames:
                    kept.append(t)
        self._tracks = kept

    # ------------------------------------------------------------------
    def _run(self, patches: List[np.ndarray]) -> np.ndarray:
        """patches: list of 80x80 BGR uint8 images -> array of P(real)."""
        # The model expects raw 0-255 BGR values (no /255, no mean/std), channels first.
        batch = np.stack(patches).astype(np.float32).transpose(0, 3, 1, 2)
        out = self.session.run(None, {self.input_name: batch})[0]
        out = np.asarray(out, dtype=np.float64)
        sums = out.sum(axis=1, keepdims=True)
        if out.min() < 0 or not np.allclose(sums, 1.0, atol=1e-3):      # logits -> softmax
            e = np.exp(out - out.max(axis=1, keepdims=True))
            out = e / e.sum(axis=1, keepdims=True)
        return out[:, 1]

    def _match_track(self, box: np.ndarray, taken: set) -> Optional[_Track]:
        best, best_score = None, 0.0
        for t in self._tracks:
            if id(t) in taken:
                continue
            score = _same_face(box, t.box)
            if score > best_score:
                best, best_score = t, score
        return best

    @staticmethod
    def _clean_box(bbox, shape) -> Optional[np.ndarray]:
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            return None
        h, w = shape[:2]
        x1, x2 = max(0.0, x1), min(float(w - 1), x2)
        y1, y2 = max(0.0, y1), min(float(h - 1), y2)
        if not (np.isfinite([x1, y1, x2, y2]).all() and x2 - x1 > 1 and y2 - y1 > 1):
            return None
        return np.array([x1, y1, x2, y2])
