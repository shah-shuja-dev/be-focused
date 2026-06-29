"""
scheduler.py -- Work-hours aware check window scheduler.

Outside work hours: pure sleep, zero CPU, camera closed.
Inside work hours: fires check_window_fn() every CHECK_INTERVAL seconds.
Each call to check_window_fn() should open camera, run inference for
CHECK_WINDOW_DURATION seconds, then return.
"""

import time, os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(r"config\.env")

WORK_DAYS             = [d.strip().lower()
                          for d in os.getenv("WORK_DAYS", "mon,tue,wed,thu,fri").split(",")]
WORK_START            = os.getenv("WORK_START", "09:00")
WORK_END              = os.getenv("WORK_END",   "18:00")
CHECK_INTERVAL        = float(os.getenv("CHECK_INTERVAL",        "3600"))
CHECK_WINDOW_DURATION = float(os.getenv("CHECK_WINDOW_DURATION", "60"))

_DAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _parse_hm(t: str):
    h, m = map(int, t.split(":"))
    return h, m


def _is_work_time() -> bool:
    now     = datetime.now()
    work_wd = [_DAY_MAP[d] for d in WORK_DAYS if d in _DAY_MAP]
    if now.weekday() not in work_wd:
        return False
    sh, sm = _parse_hm(WORK_START)
    eh, em = _parse_hm(WORK_END)
    start  = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end    = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= now <= end


def _secs_until_work_start() -> float:
    now     = datetime.now()
    work_wd = [_DAY_MAP[d] for d in WORK_DAYS if d in _DAY_MAP]
    sh, sm  = _parse_hm(WORK_START)

    # Maybe today, start still in the future
    if now.weekday() in work_wd:
        today_start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        if now < today_start:
            return (today_start - now).total_seconds()

    # Search forward up to 7 days
    for delta in range(1, 8):
        candidate = now + timedelta(days=delta)
        if candidate.weekday() in work_wd:
            next_start = candidate.replace(hour=sh, minute=sm, second=0, microsecond=0)
            return (next_start - now).total_seconds()

    return 86400


class Scheduler:
    """
    Usage:
        def my_check():
            with InferenceSession(model, duration=CHECK_WINDOW_DURATION) as s:
                for label, conf, ts in s:
                    detector.update(label, conf, ts)

        sched = Scheduler(my_check)
        sched.run()   # blocking — run in a thread
    """

    def __init__(self, check_window_fn):
        self.check_window_fn = check_window_fn
        self._running        = True

    def stop(self):
        self._running = False

    @property
    def status(self) -> dict:
        return {
            "is_work_time":    _is_work_time(),
            "current_time":    datetime.now().isoformat(),
            "work_days":       WORK_DAYS,
            "work_start":      WORK_START,
            "work_end":        WORK_END,
            "check_interval":  CHECK_INTERVAL,
            "window_duration": CHECK_WINDOW_DURATION,
        }

    def run(self):
        print(f"Scheduler started | work={WORK_DAYS} {WORK_START}-{WORK_END} "
              f"| interval={CHECK_INTERVAL}s | window={CHECK_WINDOW_DURATION}s")

        while self._running:
            if not _is_work_time():
                wait = _secs_until_work_start()
                wake = datetime.now() + timedelta(seconds=wait)
                print(f"[{datetime.now().strftime('%H:%M')}] Sleeping until "
                      f"{wake.strftime('%a %H:%M')} ({wait/3600:.1f}h)")
                slept = 0
                while slept < wait and self._running:
                    time.sleep(min(60, wait - slept))
                    slept += 60
                continue

            print(f"[{datetime.now().strftime('%H:%M')}] "
                  f"Check window starting ({CHECK_WINDOW_DURATION}s)...")
            try:
                self.check_window_fn()
            except Exception as e:
                print(f"Check window error: {e}")

            if self._running and _is_work_time():
                print(f"[{datetime.now().strftime('%H:%M')}] "
                      f"Next check in {CHECK_INTERVAL/60:.0f} min.")
                slept = 0
                while slept < CHECK_INTERVAL and self._running:
                    time.sleep(min(30, CHECK_INTERVAL - slept))
                    slept += 30