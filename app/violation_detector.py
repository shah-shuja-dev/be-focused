"""
violation_detector.py -- Stateful violation detection with buffer logic.

A violation fires when the user has been continuously not_focused
for >= BUFFER_SECS seconds. Each violation is appended to violations.jsonl.
"""

import time, json, os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(r"config\.env")

BUFFER_SECS = float(os.getenv("BUFFER_SECS", "30"))
LOG_PATH    = os.path.join("logs", "violations.jsonl")

os.makedirs("logs", exist_ok=True)


class ViolationDetector:
    """
    Feed it (label, confidence, timestamp) from InferenceSession.
    Fires on_violation callback the moment the buffer threshold is crossed.

    Usage:
        def handle(v): print(v)
        detector = ViolationDetector(on_violation=handle)
        for label, conf, ts in session:
            detector.update(label, conf, ts)
    """

    def __init__(self, on_violation=None):
        self.on_violation       = on_violation
        self._unfocused_since   = None
        self._current_violation = None
        self.total_violations   = 0
        self.session_violations = []

    @property
    def is_unfocused(self) -> bool:
        return self._unfocused_since is not None

    @property
    def unfocused_duration(self) -> float:
        if self._unfocused_since is None:
            return 0.0
        return time.time() - self._unfocused_since

    def update(self, label: str, confidence: float, timestamp: float):
        if label == "focused":
            # Close any open violation
            if self._current_violation:
                self._current_violation["end_ts"]  = timestamp
                self._current_violation["duration"] = (
                    timestamp - self._current_violation["start_ts"]
                )
                self._current_violation = None
            self._unfocused_since = None

        else:
            if self._unfocused_since is None:
                self._unfocused_since = timestamp

            streak = timestamp - self._unfocused_since

            # Cross the buffer -> fire violation (only once per streak)
            if streak >= BUFFER_SECS and self._current_violation is None:
                v = {
                    "id":         self.total_violations + 1,
                    "start_ts":   self._unfocused_since,
                    "end_ts":     None,
                    "duration":   None,
                    "confidence": round(confidence, 4),
                    "datetime":   datetime.fromtimestamp(
                                      self._unfocused_since
                                  ).isoformat(),
                }
                self._current_violation  = v
                self.total_violations   += 1
                self.session_violations.append(v)
                self._log(v)
                if self.on_violation:
                    self.on_violation(v)

    def _log(self, v: dict):
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(v) + "\n")

    def get_all_violations(self) -> list:
        if not os.path.exists(LOG_PATH):
            return []
        out = []
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out

    def get_today_violations(self) -> list:
        today = datetime.now().date().isoformat()
        return [v for v in self.get_all_violations()
                if v.get("datetime", "").startswith(today)]