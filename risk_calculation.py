"""
risk_calculation.py
===================
Decides whether a detection is LOW or HIGH risk and says WHICH rules fired.

This module is "pure": it keeps no state and never touches the camera.
Whatever tracks people across frames (see person_tracker.py) works out the
duration / occurrence count / spoof attempts and passes them in.

Shared formats (used by liveness_detection.py, person_tracker.py and
telegram_alert.py as well):
    identity   : str       registered name, or "UNKNOWN_<n>" for an unregistered person
    liveness   : str       "REAL" or "FAKE"   (the tracker may also pass "PENDING")
    timestamp  : datetime  local system time, naive (datetime.now())
    risk level : str       "HIGH" or "LOW"

Rules
-----
Mandatory (from your requirements)
    KNOWN_PERSON_PHOTO       known person + FAKE liveness
    ODD_HOURS                detection at/after 22:00 and before 06:00 (configurable)
    LONG_PRESENCE            in front of the camera for more than 5 seconds
    REPEAT_APPEARANCE        occurrence count greater than 1
Approved extras
    REPEATED_SPOOF_ATTEMPTS  3 or more spoof attempts by one identity within 2 minutes
    MULTIPLE_UNKNOWN_PERSONS 2 or more unknown faces in the same frame (applies to the
                             unknown faces)

Adding a rule later = write one function and decorate it with @register_rule.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Shared constants / helpers (imported by the other modules)
# --------------------------------------------------------------------------
HIGH = "HIGH"
LOW = "LOW"
RISK_ORDER = {LOW: 0, HIGH: 1}          # used to detect "escalation"

REAL = "REAL"
FAKE = "FAKE"
PENDING = "PENDING"                      # liveness not decided yet (not enough frames)

UNKNOWN = "UNKNOWN"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

_UNKNOWN_RE = re.compile(r"UNKNOWN(_\d+)?")


def is_unknown_identity(identity: str) -> bool:
    """True for "UNKNOWN" and temporary IDs such as "UNKNOWN_3"."""
    return bool(_UNKNOWN_RE.fullmatch(str(identity)))


def display_name(identity: str) -> str:
    """What to show on screen / in alerts: registered name, or plain UNKNOWN."""
    return UNKNOWN if is_unknown_identity(identity) else str(identity)


def format_timestamp(ts: datetime) -> str:
    return ts.strftime(TIMESTAMP_FORMAT)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    # Odd-hour window: start <= time < end, wrapping past midnight.
    # "after 10:00 PM" -> 22:00 included; the window ends at 06:00 (not included).
    odd_hour_start: time = time(22, 0)
    odd_hour_end: time = time(6, 0)

    max_duration_seconds: float = 5.0     # HIGH when duration is MORE than this
    max_occurrences: int = 1              # HIGH when occurrence_count is MORE than this
    spoof_attempt_limit: int = 3          # HIGH when spoof attempts >= this ...
    spoof_window_seconds: float = 120.0   # ... within this window (tracker enforces it)
    multiple_unknown_limit: int = 2       # HIGH when unknown faces in frame >= this

    def __post_init__(self):
        if self.odd_hour_start == self.odd_hour_end:
            raise ValueError("odd_hour_start and odd_hour_end must differ")


DEFAULT_CONFIG = RiskConfig()


# --------------------------------------------------------------------------
# Input / output types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskInput:
    identity: str
    liveness: str
    timestamp: datetime
    duration: float                 # seconds the person has been continuously visible
    occurrence_count: int           # how many separate visits so far (first visit = 1)
    spoof_attempts: int = 0         # FAKE episodes by this identity inside the window
    unknown_in_frame: int = 0       # unknown faces currently in the same frame


@dataclass
class RiskResult:
    level: str                                   # "HIGH" or "LOW"
    triggered_rules: List[str] = field(default_factory=list)   # rule names
    reasons: List[str] = field(default_factory=list)           # readable descriptions

    @property
    def is_high(self) -> bool:
        return self.level == HIGH

    def summary(self) -> str:
        if not self.triggered_rules:
            return self.level
        return f"{self.level} ({', '.join(self.triggered_rules)})"

    def __str__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------------
# Rule registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    check: Callable[[RiskInput, RiskConfig], bool]


RULES: List[Rule] = []


def register_rule(name: str, description: str):
    """Decorator: add a rule. The function returns True when the rule fires."""
    def decorator(func: Callable[[RiskInput, RiskConfig], bool]):
        RULES.append(Rule(name, description, func))
        return func
    return decorator


def is_odd_hour(ts: datetime, start: time, end: time) -> bool:
    t = ts.time()
    if start <= end:                       # window inside one day, e.g. 01:00-05:00
        return start <= t < end
    return t >= start or t < end           # window crossing midnight, e.g. 22:00-06:00


@register_rule("KNOWN_PERSON_PHOTO", "Photograph of a known person")
def _known_person_photo(r: RiskInput, cfg: RiskConfig) -> bool:
    return r.liveness == FAKE and not is_unknown_identity(r.identity)


@register_rule("ODD_HOURS", "Detection during odd hours")
def _odd_hours(r: RiskInput, cfg: RiskConfig) -> bool:
    return is_odd_hour(r.timestamp, cfg.odd_hour_start, cfg.odd_hour_end)


@register_rule("LONG_PRESENCE", "Person stayed in front of the camera too long")
def _long_presence(r: RiskInput, cfg: RiskConfig) -> bool:
    return r.duration > cfg.max_duration_seconds


@register_rule("REPEAT_APPEARANCE", "Same person appeared more than once")
def _repeat_appearance(r: RiskInput, cfg: RiskConfig) -> bool:
    return r.occurrence_count > cfg.max_occurrences


@register_rule("REPEATED_SPOOF_ATTEMPTS", "Repeated spoof attempts")
def _repeated_spoof(r: RiskInput, cfg: RiskConfig) -> bool:
    return r.spoof_attempts >= cfg.spoof_attempt_limit


@register_rule("MULTIPLE_UNKNOWN_PERSONS", "Multiple unknown persons in the same frame")
def _multiple_unknown(r: RiskInput, cfg: RiskConfig) -> bool:
    return is_unknown_identity(r.identity) and r.unknown_in_frame >= cfg.multiple_unknown_limit


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def calculate_risk(
    identity: str,
    liveness: str,
    timestamp: Optional[datetime],
    duration: float,
    occurrence_count: int,
    *,
    spoof_attempts: int = 0,
    unknown_in_frame: int = 0,
    config: RiskConfig = DEFAULT_CONFIG,
) -> RiskResult:
    """
    Evaluate every rule and return HIGH if at least one fires, otherwise LOW.

    Bad inputs never raise: a missing timestamp becomes "now", negative or
    non-numeric numbers become 0, liveness is upper-cased. A rule that crashes
    is logged and skipped so one faulty rule cannot take the camera loop down.
    """
    if not isinstance(timestamp, datetime):
        timestamp = datetime.now()
    try:
        duration = max(0.0, float(duration))
    except (TypeError, ValueError):
        duration = 0.0
    try:
        occurrence_count = max(0, int(occurrence_count))
    except (TypeError, ValueError):
        occurrence_count = 0

    data = RiskInput(
        identity=str(identity),
        liveness=str(liveness).upper(),
        timestamp=timestamp,
        duration=duration,
        occurrence_count=occurrence_count,
        spoof_attempts=max(0, int(spoof_attempts or 0)),
        unknown_in_frame=max(0, int(unknown_in_frame or 0)),
    )

    names, reasons = [], []
    for rule in RULES:
        try:
            if rule.check(data, config):
                names.append(rule.name)
                reasons.append(rule.description)
        except Exception:                                   # noqa: BLE001
            logger.exception("Risk rule %s failed and was skipped", rule.name)

    return RiskResult(HIGH if names else LOW, names, reasons)


if __name__ == "__main__":
    # Quick self-demo:  python risk_calculation.py
    now = datetime(2026, 10, 13, 14, 0, 0)
    print(calculate_risk("Abi", "REAL", now, 1.0, 1))
    print(calculate_risk("Abi", "FAKE", now, 1.0, 1))
    print(calculate_risk("UNKNOWN_1", "REAL", now.replace(hour=23), 6.0, 2))
