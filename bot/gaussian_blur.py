import math
from copy import deepcopy
from functools import wraps
from typing import List


def memoized(func):
    memory = {}

    @wraps(func)
    def memo(*args):
        if args not in memory:
            memory[args] = func(*args)
        return memory[args]

    return memo


@memoized
def g(sigma, distance):
    return math.e ** (- distance ** 2 / (2 * sigma ** 2)) / math.sqrt(2 * math.pi * sigma ** 2)


def blur(data: List[List[float]], sigma: float, cut: int):
    new_data = deepcopy(data)
    h, w = len(new_data), len(new_data[0])

    for y in range(h):
        for x in range(w):
            a = 0
            for d in range(-cut, cut + 1):
                a += new_data[y][(x + d) % w] * g(sigma, d)
            new_data[y][x] = a

    for y in range(h):
        for x in range(w):
            a = 0
            for d in range(-cut, cut + 1):
                a += new_data[(y + d) % h][x] * g(sigma, d)
            new_data[y][x] = a

    return new_data
