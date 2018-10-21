#!/usr/bin/env python3
# Python 3.7

import logging
import operator
import os
import sys
from typing import Tuple

from hlt.entity import Ship
from utils import disable_print

with disable_print():
    try:
        import pygame
        from pygame import display
        from pygame.sysfont import SysFont

        os.environ['SDL_VIDEO_WINDOW_POS'] = "10,40"
    except ImportError:
        pass

import hlt
from hlt import constants
from hlt.positionals import Position

CELL_SIZE = 20
FPS = 5
DEFAULT_SPEEDUP = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
PLAYERS = [
    (0, 170, 0),
    (170, 0, 0)
]

V2 = "--v2" in sys.argv
LOCAL = "--local" in sys.argv


def cord2screen(*args):
    result = []
    for value in args:
        if hasattr(value, '__iter__'):
            result.append(tuple(x * CELL_SIZE for x in value))
        else:
            result.append(value * CELL_SIZE)
    return result


def mul_tuple(t: tuple, k: float, integer=False) -> tuple:
    return tuple(int(k * x) if integer else k * x for x in t)


def pos2rect(p: Position) -> Tuple[int, int, int, int]:
    return mul_tuple((p.x, p.y, 1, 1), CELL_SIZE, integer=True)


def get_surrounding_offsets(p: Position, radius: int, center=False):
    for x in range(p.x - radius, p.x + radius):
        for y in range(p.y - radius, p.y + radius):
            if x == p.x and y == p.y and not center:
                continue
            yield x, y


class Plotter:
    def __init__(self, bot: 'Bot'):
        self.bot = bot
        self.bot.add_callback(self.run)
        pygame.init()
        self.win_size = (bot.game.map.width * CELL_SIZE, bot.game.map.height * CELL_SIZE)
        self.update_display_size()
        pygame.mouse.set_cursor(*pygame.cursors.tri_right)
        self.font = SysFont("vcrosdmono", 16)
        self.clock = pygame.time.Clock()
        self.speed_up = DEFAULT_SPEEDUP
        self.pause = False
        self.selected = None

    def run(self):
        self.clock.tick(int(FPS * self.speed_up) if not self.pause else 60)

        keys_mod = pygame.key.get_mods()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit("QUIT")
            elif e.type == pygame.KEYDOWN:
                self.key_handler(e, keys_mod)
            elif e.type == pygame.KEYUP:
                self.key_handler(e, keys_mod, True)
            if e.type == pygame.MOUSEBUTTONUP:
                self.mouse_handler(e)

        self.screen.fill(BLACK)
        if not self.selected:
            self.draw_halite()
        else:
            self.draw_selected()
        self.draw_bases()
        self.draw_ships()

        label = self.font.render(self.info, 0, (255, 0, 0))
        self.screen.blit(label, (0, 0))
        mouse = pygame.mouse.get_pos()
        pos = mul_tuple(mouse, 1 / CELL_SIZE, integer=True)
        halite_hover = self.font.render(str(self.bot.game.map[pos].halite_amount), 0, GREY)
        self.screen.blit(halite_hover, mouse)

        display.update()
        return not self.pause

    def draw_halite(self):
        for cell in self.bot.game.map:
            pygame.draw.rect(
                self.screen,
                mul_tuple((255, 255, 255), min(cell.halite_amount / constants.MAX_HALITE, 1), integer=True),
                pos2rect(cell.position)
            )

    def draw_selected(self):
        if isinstance(self.selected, Ship):
            debug = self.bot.debug_maps.get(self.selected.id, None)
            if not debug:
                self.draw_halite()
                return
            max_value = max(debug.values())
            for position, halite in debug.items():
                pygame.draw.rect(
                    self.screen,
                    mul_tuple((255, 255, 255), halite / max_value, integer=True),
                    pos2rect(position)
                )

    def draw_ships(self):
        for n, player in self.bot.game.players.items():
            color = PLAYERS[n]
            for ship in player.get_ships():
                # noinspection PyTypeChecker
                p = ship.position * CELL_SIZE + (CELL_SIZE // 2, CELL_SIZE // 2)
                pygame.draw.circle(
                    self.screen,
                    color,
                    (p.x, p.y),
                    round(CELL_SIZE / 2.5)
                )
                self.screen.blit(self.font.render(str(ship.id), 0, GREY), (p.x, p.y))
                if ship.halite_amount:
                    pygame.draw.circle(
                        self.screen,
                        WHITE,
                        (p.x, p.y),
                        round(CELL_SIZE / 2.5 * (ship.halite_amount / constants.MAX_HALITE))
                    )
                if ship.target:
                    pygame.draw.lines(
                        self.screen,
                        mul_tuple(color, .7, integer=True),
                        False,
                        (
                            tuple(ship.position * CELL_SIZE + CELL_SIZE // 2),
                            tuple(ship.target * CELL_SIZE + CELL_SIZE // 2)
                        ),
                        1
                    )

    def draw_bases(self):
        for n, player in self.bot.game.players.items():
            color = PLAYERS[n]
            for base in (player.shipyard,):
                pygame.draw.rect(
                    self.screen,
                    color,
                    pos2rect(base.position),
                    1
                )

    def update_display_size(self):
        self.screen = display.set_mode(self.win_size, pygame.DOUBLEBUF | pygame.RESIZABLE)
        display.flip()

    def key_handler(self, e, m: int, up=False):
        if not up:
            logging.debug(f"Press {e.unicode}")
            if m & pygame.KMOD_ALT and e.key == pygame.K_F4:
                raise SystemExit("QUIT")
            elif e.unicode == ',':  # '<'
                self.speed_up /= 2
            elif e.unicode == '.':  # '>'
                self.speed_up *= 2
            elif e.key == pygame.K_SPACE:
                self.pause = not self.pause

    def mouse_handler(self, e):
        map_pos = tuple(map(int, mul_tuple(e.pos, 1 / CELL_SIZE)))
        cell = self.bot.game.map[map_pos]
        logging.info(f"Cell: pos={cell.position} halite={cell.halite_amount}")
        if cell.ship:
            ship = cell.ship
            logging.info(f"Ship: id={ship.id} halite={ship.halite_amount}")
            if ship.owner == self.bot.game.me.id:
                self.selected = ship
        else:
            self.selected = None

    @property
    def info(self):
        return f"{self.bot.game.turn_number}/{constants.MAX_TURNS} X{self.speed_up:.0f} |" \
               f" G: {len(self.bot.game.players[0].get_ships()):>2d} {self.bot.game.players[0].halite_amount:>5d} | " \
               f" R: {len(self.bot.game.players[1].get_ships()):>2d} {self.bot.game.players[1].halite_amount:>5d}" \
               + (f"\n selected: #{self.selected.id} {self.selected.halite_amount / constants.MAX_HALITE:.0%}"
                  if self.selected else "")


class Bot:
    def __init__(
            self,
            ship_fill_k=.75,
            distance_penalty_k=1.3
    ):
        self.ship_fill_k = ship_fill_k
        self.distance_penalty_k = distance_penalty_k

        self.game = hlt.Game()
        self.callbacks = []
        self.debug_maps = {}

    def run(self):
        self.game.ready("BogdanDm")
        logging.info("Player ID: {}.".format(self.game.my_id))
        while True:
            self.game.update_frame()
            commands = self.loop()
            self.game.end_turn(commands)
            for fn in self.callbacks:
                while not fn():
                    pass

    def add_callback(self, fn):
        self.callbacks.append(fn)

    def loop(self):
        me = self.game.me
        gmap = self.game.map
        home = self.game.me.shipyard.position

        for ship in me.get_ships():
            if ship.halite_amount >= constants.MAX_HALITE * self.ship_fill_k:
                logging.info(f"Ship#{ship.id} moving home")
                ship.target = home
                yield ship.move(self.game.map.naive_navigate(ship, home))

            else:
                if gmap[ship].halite_amount / constants.MOVE_COST_RATIO > ship.halite_amount:
                    ship.target = None
                    yield ship.stay_still()
                    continue

                position = None
                radius = 8 if LOCAL else None
                surrounding = {}
                if radius:
                    field = get_surrounding_offsets(ship.position, radius, center=True)
                else:
                    field = map(operator.attrgetter('position'), gmap)
                for coord in field:
                    if (gmap[coord].ship is None or gmap[coord].ship is ship):
                        surrounding[Position(*coord)] = gmap[coord].halite_amount
                for coord in surrounding:
                    # surrounding[coord] /= 1 + 1 / constants.MOVE_COST_RATIO
                    surrounding[coord] /= (gmap.calculate_distance(ship.position, coord) + 1) ** self.distance_penalty_k

                if surrounding:
                    max_halite: Tuple[Position, int] = max(surrounding.items(), key=operator.itemgetter(1))
                    if max_halite[1] > 0.1:
                        position = max_halite[0]

                self.debug_maps[ship.id] = surrounding

                if position:
                    if position == ship.position:
                        logging.info(f"Ship#{ship.id} collecting halite")
                        ship.target = None
                        yield ship.stay_still()
                    else:
                        logging.info(f"Ship#{ship.id} {ship.position} moving towards {position}")
                        ship.target = position
                        yield ship.move(self.game.map.naive_navigate(ship, position))
                else:
                    logging.info(f"Ship#{ship.id} does not found good halite deposit")
                    ship.target = None
                    yield ship.stay_still()

            # else:
            #     logging.info(f"Ship#{ship.id} collecting halite")
            #     yield ship.stay_still()

        if self.game.turn_number <= constants.MAX_TURNS - 100 and me.halite_amount >= constants.SHIP_COST:
            if not gmap[me.shipyard].is_occupied:
                for pos in gmap[me.shipyard].position.get_surrounding_cardinals():
                    if gmap[pos].is_occupied:
                        break
                else:
                    yield me.shipyard.spawn()


bot = Bot()
if "--plot" in sys.argv:
    plotter = Plotter(bot)
logging.info("Initialization finished")
bot.run()
