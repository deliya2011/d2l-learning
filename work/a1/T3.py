def batch_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield list(range(start, min(start + batch_size, n)))
