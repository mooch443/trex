"""Report pip's Windows post-link progress while retaining its transaction log."""

from contextlib import nullcontext
import os
import re
import subprocess
import sys
from threading import Event, Thread
from time import monotonic


def console_output():
    try:
        return open("CONOUT$", "w", encoding="utf-8", errors="replace")
    except OSError:
        return nullcontext(sys.stderr)


def install(arguments):
    log_path = os.environ.get("OUT_STREAM")
    log_output = open(log_path, "a", encoding="utf-8") if log_path else nullcontext(sys.stdout)
    terminal_output = console_output() if log_path else nullcontext(sys.stdout)
    with log_output as log, terminal_output as terminal:
        label = "Installing Python ML packages"
        started = last_update = monotonic()
        finished = Event()

        def report(message):
            nonlocal last_update
            print(f"[post-link] {message}", file=terminal, flush=True)
            last_update = monotonic()

        def heartbeat():
            while not finished.wait(10):
                now = monotonic()
                if now - last_update >= 10:
                    report(f"{label} ({int(now - started)}s)")

        report(f"{label}...")
        worker = Thread(target=heartbeat)
        worker.start()
        try:
            command = [sys.executable, "-X", "utf8", "-m", "pip", "install", *arguments]
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                encoding="utf-8", errors="replace",
            ) as pip:
                download = ""
                last_percent = 0
                for line in pip.stdout:
                    detail = line.strip()
                    progress = re.fullmatch(r"Progress (\d+) of (\d+)", detail)
                    if progress:
                        current, total = map(int, progress.groups())
                        if download and total:
                            percent = min(100, current * 100 // total)
                            if percent - last_percent >= 10 or (percent == 100 and last_percent != 100):
                                message = f"{download} ({percent}%)"
                                report(message)
                                if log_path:
                                    log.write(f"[post-link] {message}\n")
                                    log.flush()
                                last_percent = percent
                        continue

                    log.write(line)
                    log.flush()
                    header = re.match(r"(Downloading|Resuming download) (\S+)", detail)
                    if header:
                        download = header.group()
                        last_percent = 0
                        if log_path:
                            report(detail)
                    elif detail.startswith("Collecting "):
                        download = ""
                    elif detail.startswith(("Using cached ", "Installing collected packages:")):
                        download = ""
                        if log_path:
                            report(detail)
                status = pip.wait()
        finally:
            finished.set()
            worker.join()

        report(f"{label} {'finished' if status == 0 else 'failed'}.")
        return status


if __name__ == "__main__":
    sys.exit(install(sys.argv[1:]))
