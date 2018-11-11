import sys

CELL_SIZE = 10
FPS = 5
DEFAULT_SPEEDUP = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
PLAYERS = [
    (0, 170, 0),
    (170, 0, 0),
    (0, 0, 170),
    (0, 170, 170),
]

V2 = "--v2" in sys.argv
LOCAL = "--local" in sys.argv
