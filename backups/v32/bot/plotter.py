import logging
import os
from typing import Tuple

import numpy as np

from bot.bot import Bot
from hlt import Position, constants
from hlt.entity import Ship
from .const import BLACK, CELL_SIZE, DEFAULT_SPEEDUP, FPS, GREY, PLAYERS, WHITE
from .utils import disable_print, mul_tuple

with disable_print():
    try:
        import pygame
        from pygame import display
        from pygame.sysfont import SysFont

        os.environ['SDL_VIDEO_WINDOW_POS'] = "10,40"
    except ImportError:
        pygame = None
        display = None
        SysFont = None


def cord2screen(*args):
    result = []
    for value in args:
        if hasattr(value, '__iter__'):
            result.append(tuple(x * CELL_SIZE for x in value))
        else:
            result.append(value * CELL_SIZE)
    return result


def pos2rect(p: Position) -> Tuple[int, int, int, int]:
    return mul_tuple((p.x, p.y, 1, 1), CELL_SIZE, integer=True)


class Plotter:
    VIEWS = [
        "__default__",
        "gbh_norm",
        "dropoff",
        "dropoff_2",
        "ships_mask",
        "halite_ex_gb_filtered",
        "own_ships_mask"
    ]

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
        self.current_view = self.VIEWS[0]

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
        if self.current_view == "__default__":
            if not self.selected:
                self.draw_halite()
            else:
                self.draw_selected()
        else:
            self.drop_custom_map(self.current_view)
        self.draw_bases()
        self.draw_ships()

        height = 0
        for info in self.info:
            label = self.font.render(info, 0, WHITE)
            self.screen.blit(label, (0, height))
            height += label.get_height() + 2
        mouse = pygame.mouse.get_pos()
        pos = mul_tuple(mouse, 1 / CELL_SIZE, integer=True)
        halite_hover = self.font.render(str(self.bot.game.map[pos].halite_amount), 0, GREY)
        self.screen.blit(halite_hover, mouse)

        display.update()
        return not self.pause

    def draw_halite(self):
        # halite = [[cell.halite_amount for cell in row] for row in self.bot.game.map.cells]
        # halite = blur(halite, 1, 5)
        # positions = [[cell.position for cell in row] for row in self.bot.game.map.cells]
        # field = [[MapCell(positions[i][j], halite[i][j]) for j in range(len(halite[0]))] for i in range(len(halite))]
        # for cell in chain(*field):
        for cell in self.bot.game.map:
            halite = cell.halite_amount / constants.MAX_HALITE
            color = mul_tuple((255, 255, 255), min(halite, 1), integer=True)
            if "gbh_norm" in self.bot.debug_maps:
                r, g, b = color
                k = 1 - self.bot.debug_maps["gbh_norm"][cell.position.y][cell.position.x]
                g *= k
                b *= k
                color = r, g, b
            pygame.draw.rect(
                self.screen,
                color,
                pos2rect(cell.position)
            )

    def draw_selected(self):
        if isinstance(self.selected, Ship):
            debug_map = self.bot.debug_maps.get(self.selected.id, None)
            if debug_map is None:
                self.draw_halite()
                return
            max_value = debug_map.max()
            it = np.nditer(debug_map, ['multi_index'])
            while not it.finished:
                halite = it[0]
                position = Position(*reversed(it.multi_index))
                pygame.draw.rect(
                    self.screen,
                    mul_tuple((255, 255, 255), halite / max_value, integer=True),
                    pos2rect(position)
                )
                it.iternext()

    def draw_ships(self):
        for n, player in self.bot.game.players.items():
            color = PLAYERS[n]
            for ship in player.get_ships():
                # noinspection PyTypeChecker
                is_selected = self.selected is None or self.selected.id == ship.id
                p = ship.position * CELL_SIZE + (CELL_SIZE // 2, CELL_SIZE // 2)
                pygame.draw.circle(
                    self.screen,
                    mul_tuple(color, 1.2 if is_selected else .1, integer=True),
                    (p.x, p.y),
                    round(CELL_SIZE / 2.5)
                )
                self.screen.blit(self.font.render(str(ship.id), 0, GREY), (p.x, p.y))
                if ship.halite_amount:
                    pygame.draw.circle(
                        self.screen,
                        mul_tuple(WHITE, 1 if is_selected else .1, integer=True),
                        (p.x, p.y),
                        round(CELL_SIZE / 2.5 * (ship.halite_amount / constants.MAX_HALITE))
                    )
                if is_selected and self.bot.ship_manager.targets.get(ship, None):
                    target = self.bot.ship_manager.targets[ship]
                    if isinstance(target, Position):
                        points = (ship.position, target)
                    else:
                        points = [ship.position, *target]

                    pygame.draw.lines(
                        self.screen,
                        color,
                        False,
                        [
                            tuple(point * CELL_SIZE + CELL_SIZE // 2) for point in points
                        ],
                        1
                    )

    def draw_bases(self):
        for n, player in self.bot.game.players.items():
            color = PLAYERS[n]
            for base in (player.shipyard, *player.get_dropoffs()):
                pygame.draw.rect(
                    self.screen,
                    color,
                    pos2rect(base.position),
                    1
                )

    def drop_custom_map(self, view):
        debug_map = self.bot.debug_maps.get(view, None)
        if debug_map is None:
            self.draw_halite()
            return
        max_value = debug_map.max()
        min_value = debug_map.min()
        it = np.nditer(debug_map, ['multi_index'])
        while not it.finished:
            halite = it[0]
            position = Position(*reversed(it.multi_index))
            k = (max_value - min_value) or .1
            pygame.draw.rect(
                self.screen,
                mul_tuple((255, 255, 255), (halite - min_value) / k, integer=True),
                pos2rect(position)
            )
            it.iternext()

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
            elif '0' <= e.unicode <= '9':
                code = int(e.unicode)
                if code == 0:
                    code = 9
                else:
                    code -= 1
                if code < len(self.VIEWS):
                    self.current_view = self.VIEWS[code]

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
        gmap = self.bot.game.map
        if self.selected:
            load = (1 - self.selected.halite_amount / constants.MAX_HALITE) * 100
        else:
            load = 0

        total = gmap.total_halite
        info = [
            f"{self.bot.game.turn_number:>3d}/{constants.MAX_TURNS} X{self.speed_up:.0f} "
            f"{total:>6.0f} {total / gmap.initial_halite:>3.0%} ({self.bot.ship_fill_k:.1f})"
        ]

        players = [
            f"{color}: {len(p.get_ships()):>2d} {p.halite_amount:>5d}"
            for color, p in zip("GRBC", self.bot.game.players.values())
        ]
        info.append(f"{players[0]} | {players[1]}")
        if len(players) > 2:
            info.append(f"{players[2]} | {players[3]}")

        if self.selected:
            info.append(f":> selected: #{self.selected.id} {load:>2.0f}%")

        return info
