class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __eq__(self, a):
        if not isinstance(a, Vec2):
            return NotImplemented
        return (self.x, self.y) == (a.x, a.y)

    def __repr__(self):
        return f"Vec2({self.x},{self.y})"
