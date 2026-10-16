class Counter:
    def __init__(self, start=0):
        self.start = start
        self.step = 0

    def __call__(self, step=1):
        self.start += step
        return self.start + self.step - 1

    def __len__(
        self,
    ):
        return self.step

    def __repr__(self):
        return f"Counter(value={self.start}, calls={self.step})"
