"""Uklidnit Image Capture: launchery pozastavit, ptpcamerad jednou zabít a nové instance jen pozastavovat."""
import os, signal, subprocess, threading, time

stopped: set[int] = set()


def pids(name: str) -> list[int]:
    r = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def stop(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGSTOP)
        stopped.add(pid)
    except ProcessLookupError:
        pass


def watchdog() -> None:
    while True:
        for name in ("ptpcamerad", "icdd", "mscamerad-xpc"):
            for pid in pids(name):
                if pid not in stopped:
                    stop(pid)
        time.sleep(0.05)


def quiet_and_free() -> None:
    for name in ("icdd", "mscamerad-xpc"):
        for pid in pids(name):
            stop(pid)
    for pid in pids("ptpcamerad"):
        os.kill(pid, signal.SIGKILL)
    threading.Thread(target=watchdog, daemon=True).start()


def resume() -> None:
    for pid in stopped:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
