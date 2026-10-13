"""
person_tracker.py
=================
Remembers people BETWEEN frames, so the other modules can stay simple.

For every frame it answers, per detected face:
    * identity          registered name, or a temporary "UNKNOWN_<n>" ID
    * duration          seconds the person has been continuously in view
    * occurrence_count  how many separate visits so far (first visit = 1)
    * spoof_attempts    FAKE episodes by this identity in the last N seconds
    * unknown_in_frame  how many unknown faces are in this frame

Definitions (all configurable in the constructor)
-------------------------------------------------
visit            Continuous presence. A NEW visit starts when the identity was not
                 seen for more than `visit_gap` seconds (default 10).
                 A new visit increases occurrence_count.
duration         Time since the current "run" of presence started. A run continues
                 through short detection gaps (`grace_gap`, default 2 s - a missed
                 frame, a head turn). If the gap is longer than that, the run (and
                 therefore duration) restarts, but it is still the same visit.
spoof attempt    One FAKE "episode": the first FAKE verdict for an identity, then
                 nothing counts again until FAKE has been absent for `visit_gap`
                 seconds. (Otherwise one photo held up for 1 second would count
                 as dozens of "attempts", one per frame.)
unknown person   Unknown faces are compared with each other using cosine similarity
                 (>= `unknown_threshold`, default 0.5 = the same threshold as your
                 recognizer) and get IDs UNKNOWN_1, UNKNOWN_2, ... Two unknown faces
                 in the same frame never share an ID.

State lives in memory only: everything resets when the script restarts.
Time: pass `datetime.now()` (local clock). Tests can pass fake times.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from risk_calculation import FAKE, PENDING, is_unknown_identity

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """One detected face, as produced by your existing code + the liveness module."""
    name: str                      # what recognize_face() returned ("UNKNOWN" or a name)
    embedding: np.ndarray          # face.embedding from InsightFace
    liveness: str = PENDING        # "REAL" / "FAKE" / "PENDING"


@dataclass
class PersonState:
    identity: str                  # name or "UNKNOWN_<n>"
    is_known: bool
    duration: float
    occurrence_count: int
    spoof_attempts: int
    unknown_in_frame: int
    new_visit: bool                # True on the first frame of a visit


@dataclass
class _Record:
    occurrence_count: int
    run_start: datetime
    last_seen: datetime
    last_fake: Optional[datetime] = None
    fake_episodes: Deque[datetime] = field(default_factory=deque)


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 and np.isfinite(n) else np.zeros_like(v)


class PersonTracker:
    def __init__(
        self,
        visit_gap: float = 10.0,
        grace_gap: float = 2.0,
        unknown_threshold: float = 0.5,
        spoof_window: float = 120.0,
        embeddings_per_unknown: int = 10,
        max_unknown_ids: int = 200,
    ):
        self.visit_gap = timedelta(seconds=visit_gap)
        self.grace_gap = timedelta(seconds=grace_gap)
        self.spoof_window = timedelta(seconds=spoof_window)
        self.unknown_threshold = unknown_threshold
        self.embeddings_per_unknown = embeddings_per_unknown
        self.max_unknown_ids = max_unknown_ids

        self._records: Dict[str, _Record] = {}
        self._gallery: Dict[str, Deque[np.ndarray]] = {}   # UNKNOWN_n -> recent embeddings
        self._next_unknown = 1

        # Measured time between frames. On a slow CPU a frame can take longer than
        # `grace_gap`; the effective grace gap grows to ~2.5 frames so the 5 s timer
        # still works instead of restarting on every frame.
        self._last_update: Optional[datetime] = None
        self._frame_dt = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._records.clear()
        self._gallery.clear()
        self._next_unknown = 1
        self._last_update = None
        self._frame_dt = 0.0

    # ------------------------------------------------------------------
    def update(
        self, observations: Sequence[Observation], now: Optional[datetime] = None
    ) -> List[PersonState]:
        """Process all faces of ONE frame; returns one PersonState per observation, same order."""
        now = now or datetime.now()
        self._measure_frame_interval(now)
        if not observations:
            return []

        identities = self._resolve_identities(observations, now)
        unknown_in_frame = sum(is_unknown_identity(i) for i in identities)

        # Pass 1: update presence once per identity, note FAKE verdicts
        new_visit: Dict[str, bool] = {}
        for obs, ident in zip(observations, identities):
            if ident not in new_visit:
                new_visit[ident] = self._touch(ident, now)
            if obs.liveness == FAKE:
                self._note_fake(self._records[ident], now)

        # Pass 2: build the answer
        states = []
        for ident in identities:
            rec = self._records[ident]
            self._prune_fake(rec, now)
            states.append(
                PersonState(
                    identity=ident,
                    is_known=not is_unknown_identity(ident),
                    duration=(now - rec.run_start).total_seconds(),
                    occurrence_count=rec.occurrence_count,
                    spoof_attempts=len(rec.fake_episodes),
                    unknown_in_frame=unknown_in_frame,
                    new_visit=new_visit[ident],
                )
            )
        return states

    # ------------------------------------------------------------------
    def _touch(self, ident: str, now: datetime) -> bool:
        """Update presence for an identity. Returns True if this starts a new visit."""
        rec = self._records.get(ident)
        if rec is None:
            self._records[ident] = _Record(1, now, now)
            return True
        gap = now - rec.last_seen
        started_new_visit = False
        if gap > self.visit_gap:
            rec.occurrence_count += 1
            rec.run_start = now
            started_new_visit = True
        elif gap > max(self.grace_gap, timedelta(seconds=2.5 * self._frame_dt)):
            rec.run_start = now           # same visit, but the duration timer restarts
        rec.last_seen = now
        return started_new_visit

    def _measure_frame_interval(self, now: datetime) -> None:
        if self._last_update is not None:
            dt = min((now - self._last_update).total_seconds(), 2.0)   # ignore long pauses
            if dt > 0:
                self._frame_dt = dt if self._frame_dt == 0 else 0.8 * self._frame_dt + 0.2 * dt
        self._last_update = now

    def _note_fake(self, rec: _Record, now: datetime) -> None:
        if rec.last_fake is None or now - rec.last_fake > self.visit_gap:
            rec.fake_episodes.append(now)           # a new spoof attempt
        rec.last_fake = now

    def _prune_fake(self, rec: _Record, now: datetime) -> None:
        while rec.fake_episodes and now - rec.fake_episodes[0] > self.spoof_window:
            rec.fake_episodes.popleft()

    # ------------------------------------------------------------------
    def _resolve_identities(self, observations: Sequence[Observation], now: datetime) -> List[str]:
        identities: List[Optional[str]] = [None] * len(observations)
        unknown_idx = []
        for i, obs in enumerate(observations):
            if is_unknown_identity(obs.name):
                unknown_idx.append(i)
            else:
                identities[i] = str(obs.name)

        if unknown_idx:
            self._assign_unknown_ids(observations, identities, unknown_idx)
            self._evict_old_unknowns(now, keep={i for i in identities if i})

        return identities  # type: ignore[return-value]

    def _assign_unknown_ids(self, observations, identities, unknown_idx) -> None:
        embs = {i: _unit(observations[i].embedding) for i in unknown_idx}

        # Similarity of every unknown face with every stored unknown person
        pairs: List[Tuple[float, int, str]] = []
        for uid, gallery in self._gallery.items():
            g = np.stack(gallery)                       # (k, 512)
            for i in unknown_idx:
                sim = float(np.max(g @ embs[i]))
                if sim >= self.unknown_threshold:
                    pairs.append((sim, i, uid))

        # Best matches first; one ID per face and one face per ID within a frame
        pairs.sort(reverse=True)
        used_ids = set()
        for sim, i, uid in pairs:
            if identities[i] is None and uid not in used_ids:
                identities[i] = uid
                used_ids.add(uid)
                # Keep the gallery varied (new angles), but skip near-duplicates
                if sim < 0.9:
                    self._gallery[uid].append(embs[i])

        for i in unknown_idx:
            if identities[i] is None:
                uid = f"UNKNOWN_{self._next_unknown}"
                self._next_unknown += 1
                self._gallery[uid] = deque([embs[i]], maxlen=self.embeddings_per_unknown)
                identities[i] = uid

    def _evict_old_unknowns(self, now: datetime, keep: set) -> None:
        """Memory guard: forget the least recently seen unknown IDs beyond the cap."""
        if len(self._gallery) <= self.max_unknown_ids:
            return
        candidates = [
            (self._records[u].last_seen if u in self._records else now, u)
            for u in self._gallery if u not in keep
        ]
        candidates.sort()
        for _, uid in candidates[: len(self._gallery) - self.max_unknown_ids]:
            self._gallery.pop(uid, None)
            self._records.pop(uid, None)
