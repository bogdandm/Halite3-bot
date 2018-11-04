import os
import sys
from functools import wraps


class disable_print:
    def __init__(self):
        self.oldstdout = None
        self.devnull = open(os.devnull, 'w')

    def __enter__(self):
        self.oldstdout, sys.stdout = sys.stdout, self.devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.oldstdout


def mul_tuple(t: tuple, k: float, integer=False) -> tuple:
    return tuple(int(k * x) if integer else k * x for x in t)


def memoized(func):
    memory = {}

    @wraps(func)
    def memo(*args):
        if args not in memory:
            memory[args] = func(*args)
        return memory[args]

    return memo
