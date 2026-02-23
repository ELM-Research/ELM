import time

def timeit(fn, desc="", dev=False):
    if not dev:
        return fn()
    start = time.time()
    out = fn()
    print(f"{desc} Timing: {time.time() - start:.4f}s")
    return out