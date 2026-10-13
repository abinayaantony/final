"""
telegram_alert.py
=================
Sends a Telegram photo alert (cropped face + caption) for suspicious events.

When an alert is sent
    * an UNKNOWN person is detected (REAL or FAKE)
    * a FAKE (photograph) of a KNOWN person is detected
  A known person with a REAL face never triggers an alert (even when the risk
  is HIGH) - the risk level still appears on screen and in alerts that do fire.

Duplicate-alert prevention
    * Cooldown per (identity, event type), default 60 s.
    * Escalation bypass: if the risk level goes up (LOW -> HIGH) after an alert,
      a new alert is sent immediately even inside the cooldown.
    * If sending fails (no internet, Telegram error) the alert is retried after a
      short delay instead of waiting for the whole cooldown.

The webcam loop is never blocked: the face is cropped immediately, then a
background thread saves the snapshot and uploads it.

Credentials (never hard-coded) come from environment variables or a .env file:
    TELEGRAM_BOT_TOKEN=123456:ABC...
    TELEGRAM_CHAT_ID=123456789
If they are missing, alerts are still logged and snapshots are saved to alerts/,
but nothing is sent.

Test your setup:   python telegram_alert.py --test

Requires: requests, opencv-python, numpy   (python-dotenv optional, for .env files)
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time as _time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests

from risk_calculation import (
    FAKE, HIGH, LOW, REAL, RISK_ORDER, display_name, format_timestamp, is_unknown_identity,
)

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DEFAULT_SAVE_DIR = HERE / "alerts"

EVENT_UNKNOWN_PERSON = "UNKNOWN_PERSON"     # unknown + REAL
EVENT_UNKNOWN_PHOTO = "UNKNOWN_PHOTO"       # unknown + FAKE
EVENT_KNOWN_PHOTO = "KNOWN_PHOTO"           # known   + FAKE

EVENT_TITLES = {
    EVENT_UNKNOWN_PERSON: "Unknown person detected",
    EVENT_UNKNOWN_PHOTO: "Unknown person detected (photograph / spoof)",
    EVENT_KNOWN_PHOTO: "Photograph of a known person detected",
}

CAPTION_LIMIT = 1024                         # Telegram's caption limit


def classify_event(identity: str, liveness: str) -> Optional[str]:
    """Which alert event (if any) does this detection correspond to?"""
    liveness = str(liveness).upper()
    if liveness not in (REAL, FAKE):
        return None                          # not decided yet
    if is_unknown_identity(identity):
        return EVENT_UNKNOWN_PERSON if liveness == REAL else EVENT_UNKNOWN_PHOTO
    return EVENT_KNOWN_PHOTO if liveness == FAKE else None


def crop_face(frame: np.ndarray, bbox: Sequence, margin: float = 0.3) -> Optional[np.ndarray]:
    """Copy of the face region with some margin around the box (None if unusable)."""
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    h, w = frame.shape[:2]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return frame[y1:y2, x1:x2].copy()


def _load_credentials(token, chat_id):
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")           # next to this file
        load_dotenv()                        # or in the current folder
    except ImportError:
        pass                                 # fine: plain environment variables still work
    return (token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip(), \
           str(chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()


@dataclass
class _Sent:
    when: float                              # time.monotonic() of the (attempted) alert
    risk: str


@dataclass
class _Job:
    key: Tuple[str, str]
    jpg: bytes
    caption: str
    snapshot_name: str


class TelegramAlerter:
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        cooldown_seconds: float = 60.0,
        retry_delay_seconds: float = 15.0,
        request_timeout: float = 15.0,
        save_snapshots: bool = True,
        save_dir: Path = DEFAULT_SAVE_DIR,
        queue_size: int = 20,
    ):
        self.token, self.chat_id = _load_credentials(token, chat_id)
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.warning(
                "Telegram is NOT configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env). "
                "Alerts will only be logged and saved locally."
            )

        self.cooldown = cooldown_seconds
        self.retry_delay = retry_delay_seconds
        self.timeout = request_timeout
        self.save_snapshots = save_snapshots
        self.save_dir = Path(save_dir)

        self._last: Dict[Tuple[str, str], _Sent] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[_Job]]" = queue.Queue(maxsize=queue_size)
        self._worker = threading.Thread(target=self._run, name="telegram-alerts", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def maybe_alert(
        self,
        frame: np.ndarray,
        bbox: Sequence,
        identity: str,
        liveness: str,
        timestamp: datetime,
        risk_level: str = LOW,
        triggered_rules: Sequence[str] = (),
    ) -> bool:
        """
        Call once per detected face per frame. Does nothing (cheaply) unless the
        event deserves an alert AND the cooldown allows it. Returns True if an
        alert was queued.
        """
        event = classify_event(identity, liveness)
        if event is None:
            return False

        key = (str(identity), event)
        now = _time.monotonic()
        with self._lock:
            last = self._last.get(key)
            if last is not None and now - last.when < self.cooldown:
                escalated = RISK_ORDER.get(risk_level, 0) > RISK_ORDER.get(last.risk, 0)
                if not escalated:
                    return False
            previous = last
            self._last[key] = _Sent(now, risk_level)        # reserve: blocks duplicates at once

        face = crop_face(frame, bbox)
        if face is None:
            self._restore(key, previous)
            return False
        ok, buf = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            self._restore(key, previous)
            return False

        job = _Job(
            key=key,
            jpg=buf.tobytes(),
            caption=self._caption(event, identity, liveness, timestamp, risk_level, triggered_rules),
            snapshot_name=(
                f"{timestamp:%Y%m%d_%H%M%S}_{timestamp.microsecond // 1000:03d}_"
                f"{re.sub(r'[^A-Za-z0-9_.-]', '_', str(identity))}_{event}.jpg"
            ),
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            logger.warning("Alert queue is full; alert for %s dropped", identity)
            self._restore(key, previous)
            return False
        return True

    def close(self, timeout: float = 5.0) -> None:
        """Let queued alerts finish (up to `timeout` seconds), then stop the worker."""
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._worker.join(timeout)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _restore(self, key, previous: Optional[_Sent]) -> None:
        with self._lock:
            if previous is None:
                self._last.pop(key, None)
            else:
                self._last[key] = previous

    @staticmethod
    def _caption(event, identity, liveness, timestamp, risk_level, rules) -> str:
        lines = [
            f"\U0001F6A8 {EVENT_TITLES[event]}",
            f"Name: {display_name(identity)}",
        ]
        if is_unknown_identity(identity) and identity != "UNKNOWN":
            lines.append(f"Track ID: {identity}")
        lines += [
            f"Liveness: {str(liveness).upper()}",
            f"Risk: {risk_level}",
        ]
        if rules:
            lines.append("Triggered: " + ", ".join(rules))
        lines.append(f"Time: {format_timestamp(timestamp)}")
        return "\n".join(lines)[:CAPTION_LIMIT]

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._handle(job)
            except Exception:                                    # noqa: BLE001
                logger.exception("Unexpected error while handling an alert")

    def _handle(self, job: _Job) -> None:
        logger.info("ALERT | %s", job.caption.replace("\n", " | "))
        if self.save_snapshots:
            try:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                (self.save_dir / job.snapshot_name).write_bytes(job.jpg)
            except OSError as exc:
                logger.warning("Could not save snapshot: %s", exc)

        if not self.enabled:
            return
        ok, retry_after = self._send_photo(job.jpg, job.caption)
        if not ok:
            # Allow another attempt soon rather than after the whole cooldown
            wait = max(self.retry_delay, retry_after or 0)
            with self._lock:
                sent = self._last.get(job.key)
                if sent is not None:
                    sent.when = _time.monotonic() - self.cooldown + wait

    def _send_photo(self, jpg: bytes, caption: str) -> Tuple[bool, Optional[float]]:
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        try:
            resp = requests.post(
                url,
                data={"chat_id": self.chat_id, "caption": caption},
                files={"photo": ("face.jpg", jpg, "image/jpeg")},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            # The exception text contains the URL (= the token) - never log it raw
            logger.warning("Telegram request failed: %s", str(exc).replace(self.token, "***"))
            return False, None

        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code == 200 and data.get("ok"):
            logger.info("Telegram alert sent")
            return True, None

        retry_after = (data.get("parameters") or {}).get("retry_after")
        logger.warning(
            "Telegram rejected the alert (HTTP %s): %s",
            resp.status_code, data.get("description", resp.text[:200]),
        )
        return False, float(retry_after) if retry_after else None


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--test" not in sys.argv:
        print("Usage: python telegram_alert.py --test")
        sys.exit(0)

    alerter = TelegramAlerter(save_snapshots=False)
    if not alerter.enabled:
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file first.")
    img = np.full((240, 240, 3), 70, np.uint8)
    cv2.putText(img, "TEST", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    caption = alerter._caption(EVENT_UNKNOWN_PERSON, "UNKNOWN_1", REAL, datetime.now(), HIGH, ["TEST"])
    ok, _ = alerter._send_photo(cv2.imencode(".jpg", img)[1].tobytes(), caption)
    print("Test alert sent - check Telegram." if ok else "Failed - see the message above.")
    alerter.close()
