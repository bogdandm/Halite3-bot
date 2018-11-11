import logging
import os
import sys
from functools import wraps

import numpy as np


class disable_print:
    def __init__(self):
        self.oldstdout = None
        self.oldstderr = None
        self.devnull = open(os.devnull, 'w')

    def __enter__(self):
        self.oldstdout, sys.stdout = sys.stdout, self.devnull
        self.oldstderr, sys.stderr = sys.stderr, self.devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.oldstdout
        sys.stderr = self.oldstderr


def mul_tuple(t: tuple, k: float, integer=False) -> tuple:
    return tuple(int(k * x) if integer else k * x for x in t)


def memoized(func):
    memory = {}

    @wraps(func)
    def memo(*args):
        logging.debug((args, hash(args)))
        if args not in memory:
            memory[args] = func(*args)
        return memory[args]

    return memo


def memoized_method(func):
    memory = {}

    @wraps(func)
    def memo(*args):
        self, args = args[0], args[1:]
        if args not in memory:
            memory[args] = func(self, *args)
        return memory[args]

    return memo


def extend_grid(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    extended_grid = np.zeros((w * 2, h * 2), dtype=grid.dtype)
    w2 = w // 2
    h2 = h // 2
    extended_grid[0:h2, 0:w2] = grid[h2:h, w2:w]
    extended_grid[0:h2, w2:w + w2] = grid[h2:h, 0:w]
    extended_grid[0:h2, w + w2:w + w] = grid[h2:h, 0:w2]

    extended_grid[h2:h + h2, 0:w2] = grid[0:h, w2:w]
    extended_grid[h2:h + h2, w2:w + w2] = grid[0:h, 0:w]
    extended_grid[h2:h + h2, w + w2:w + w] = grid[0:h, 0:w2]

    extended_grid[h + h2:h + h, 0:w2] = grid[0:h2, w2:w]
    extended_grid[h + h2:h + h, w2:w + w2] = grid[0:h2, 0:w]
    extended_grid[h + h2:h + h, w + w2:w + w] = grid[0:h2, 0:w2]
    return extended_grid
