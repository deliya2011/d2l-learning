import time


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = round(time.time() - self.t0, 3)
        return False


with Timer() as t:
    time.sleep(0.05)

print(t.elapsed)
