"""Flushed terminal progress without an additional dependency."""
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    def __init__(self, label, total):
        self.label = label
        self.total = total
        self.completed = 0
        self.started = self.last_detail = time.monotonic()
        print(f"[{label}] 0/{total} | starting", flush=True)

    def detail(self, message, force=False):
        now = time.monotonic()
        if force or now - self.last_detail >= 10:
            print(f"[{self.label}] {self.completed}/{self.total} | {message} "
                  f"| elapsed {duration(now-self.started)}", flush=True)
            self.last_detail = now

    def advance(self, loss):
        self.completed += 1
        now = time.monotonic()
        elapsed = now - self.started
        remaining = elapsed / self.completed * (self.total-self.completed)
        print(f"[{self.label}] {self.completed}/{self.total} "
              f"({100*self.completed/self.total:.1f}%) | loss={loss:.6g} "
              f"| elapsed {duration(elapsed)} | ETA {duration(remaining)}", flush=True)
        self.last_detail = now
