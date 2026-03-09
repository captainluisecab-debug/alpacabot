"""
watchdog.py — Process watchdog for Alpaca bot.
Usage: python watchdog.py
"""
from __future__ import annotations
import logging, os, subprocess, sys, time

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][WATCHDOG] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("watchdog")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable
CHECK_SEC = 30
RESTART_DELAY = 5

PROCESS = {"name": "alpaca_engine", "script": "alpaca_engine.py", "proc": None, "restarts": 0}


def start_process(entry: dict) -> None:
    script = os.path.join(BASE_DIR, entry["script"])
    if not os.path.exists(script):
        log.error("Script not found: %s", script)
        return
    log.info("Starting %s...", entry["name"])
    entry["proc"] = subprocess.Popen([PYTHON, script], cwd=BASE_DIR)
    log.info("%s started (pid=%d)", entry["name"], entry["proc"].pid)


def main() -> None:
    log.info("=" * 50)
    log.info("ALPACA BOT WATCHDOG")
    log.info("Check interval: %ds | Restart delay: %ds", CHECK_SEC, RESTART_DELAY)
    log.info("=" * 50)
    start_process(PROCESS)
    while True:
        time.sleep(CHECK_SEC)
        proc = PROCESS["proc"]
        if proc is None or proc.poll() is not None:
            exit_code = proc.poll() if proc else "never_started"
            PROCESS["restarts"] += 1
            log.warning("%s exited (code=%s, restarts=%d) — restarting in %ds",
                        PROCESS["name"], exit_code, PROCESS["restarts"], RESTART_DELAY)
            time.sleep(RESTART_DELAY)
            start_process(PROCESS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Watchdog stopped.")
        if PROCESS["proc"] and PROCESS["proc"].poll() is None:
            PROCESS["proc"].terminate()
            log.info("Terminated %s", PROCESS["name"])
