class Counter:
    def __init__(self, value):
        self.value = value
        self.step = 0

    def __call__(self, x=None):
        self.step += 1
        return self.value + self.step - 1

    def __len__(self, a=None):
        self.step += 1
        return self.step - 1

    def __repr__(self):
        return f"Counter(value={self.value}, calls={self.step})"
