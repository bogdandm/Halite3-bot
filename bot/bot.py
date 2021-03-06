import json
import logging
import operator
import time
from itertools import combinations
from pathlib import Path
from random import choice, shuffle
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import skimage.measure as msr
import yaml
from scipy.ndimage import gaussian_filter
from skimage import draw

from hlt import Direction, Game, Position, constants
from hlt.entity import Ship
from hlt.game_map import Player
from .const import FLOG, LOCAL, V2

np.seterrcall(lambda type, flag: logging.error("Floating point error ({}), with flag {}".format(type, flag)))
np.seterr(all='call')

PATH: Path = Path(__file__).parent.parent.absolute().resolve()


def ship_collecting_halite_coefficient(ship, gmap):
    """
    Return k [0.1, 1]
    :param ship:
    :param gmap:
    :return:
    """
    ship_cargo_free = constants.MAX_HALITE - ship.halite_amount
    cell = gmap[ship.position].halite_amount or 10 ** -10
    return max((cell - ship_cargo_free) / cell, .1)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def roll2d(a: np.ndarray, x, y) -> np.ndarray:
    a = np.roll(a, y, axis=0)
    return np.roll(a, x, axis=1)


class ShipManager:
    def __init__(self, game: Game, turn_waiting_until_destroy: int = 20):
        self.game = game
        self.turn_waiting_until_destroy = turn_waiting_until_destroy

        self.ships: List[Ship] = []
        self.distances: Dict[Ship, int] = {}
        self.targets: Dict[Ship, Optional[Union[Position, Tuple[int, int], List[Position]]]] = {}
        self._commands: Dict[Ship, Optional[str]] = {}
        self.blockers: Dict[Ship, Dict[Ship, int]] = {}
        self.f_log = []

    def update(self):
        self.ships = self.game.me.get_ships()

        gmap = self.game.map
        me = self.game.me
        bases = {me.shipyard.position, *(base.position for base in me.get_dropoffs())}

        self.distances = {
            ship: min(gmap.distance(ship.position, base) for base in bases)
            for ship in self.ships
        }
        self.ships = [ship for ship, distance in sorted(self.distances.items(), key=operator.itemgetter(1))]
        self.targets = {ship: None for ship in self.ships}
        if FLOG and self.game.turn_number == constants.MAX_TURNS - 1:
            with open(f"f-log-{time.time()}{'-V2' if V2 else ''}.json", "w") as f:
                json.dump(self.f_log, f, default=lambda x: int(x) if isinstance(x, np.int32) else str(x))

    def add_target(self, ship: Ship, target: Optional[Union[Position, Tuple[int, int], List[Position]]]):
        self.targets[ship] = target
        if FLOG and target:
            p = Position(*(target[-1] if isinstance(target, list) else target))
            self.f_log.append({
                "t": self.game.turn_number,
                "x": p.x,
                "y": p.y,
                "color": "#FF0000"
            })

    def resolve_moves(self, moves: List[Tuple[Ship, Iterable[Tuple[int, int]]]], ignore_enemy_ships=False):
        for i in range(4):
            commands, moves = self._resolve_moves(moves, ignore_enemy_ships=ignore_enemy_ships)
            yield from commands
            if not moves:
                break

        for ship, d in moves:
            logging.debug(f"Unsafe move: {ship} -> {d}")
            yield ship.stay_still()

    def _resolve_moves(self, moves: List[Tuple[Ship, Iterable[Tuple[int, int]]]], ignore_enemy_ships):
        gmap = self.game.map
        me = self.game.me

        tmp_moves = moves[:]
        unsafe_moves = []
        swapped_ships: Set[Ship] = set()
        commands = []

        # Resolve all regular movements until all remaining moves are not safe
        changing = True
        while changing and tmp_moves:
            changing = False
            while tmp_moves:
                ship, directions = tmp_moves.pop()
                for d in directions:
                    cell = gmap[ship.position + d]
                    if (
                            cell.ship is None
                            or ignore_enemy_ships and cell.ship.owner != me.id
                            or cell.is_occupied_base(me)
                    ):
                        logging.info(f"Ship#{ship.id} moves {Direction.Names[d]}")
                        commands.append(gmap.update_ship_position(ship, d))
                        changing = True
                        break
                else:
                    unsafe_moves.append((ship, directions))

            tmp_moves, unsafe_moves = unsafe_moves, []

        # Perform "ship swap" movements
        for a, b in combinations(tmp_moves, 2):
            resolved = False
            ship1, ld1 = a
            ship2, ld2 = b
            if ship1 in swapped_ships or ship2 in swapped_ships:
                continue
            for d1 in ld1:
                if resolved:
                    break
                for d2 in ld2:
                    if (
                            gmap.normalize(ship1.position + d1) == ship2.position
                            and ship1.position == gmap.normalize(ship2.position + d2)
                    ):
                        resolved = True
                        swapped_ships.add(ship1)
                        swapped_ships.add(ship2)
                        commands.extend(gmap.swap_ships(ship1, ship2))

        tmp_moves = [(ship, d) for ship, d in tmp_moves if ship not in swapped_ships]

        stacked_ships = {ship for ship, _ in tmp_moves}
        resolved_ships = {ship for ship in self.ships} - stacked_ships
        for ship in resolved_ships:
            if ship in self.blockers:
                del self.blockers[ship]

        remove_moves = set()
        for ship, d in tmp_moves:
            old_blockers = self.blockers.get(ship, None)
            new_blockers = {}
            for direction in d:
                other_ship = gmap[ship.position + direction].ship
                if other_ship is None:
                    commands.append(gmap.update_ship_position(ship, direction))
                    remove_moves.add(ship)
                    if ship in self.blockers:
                        del self.blockers[ship]
                    break
                enemy = other_ship.owner != me.id
                if old_blockers and other_ship in old_blockers:
                    blocked_turns = old_blockers.get(other_ship, 0)
                    blocked_turns += 1
                    if enemy and blocked_turns >= self.turn_waiting_until_destroy \
                            and ship.halite_amount <= constants.MAX_TURNS * .75:
                        commands.append(gmap.update_ship_position(ship, direction))
                        remove_moves.add(ship)
                        del self.blockers[ship]
                        break
                    new_blockers[other_ship] = blocked_turns
                else:
                    new_blockers[other_ship] = 0
            else:
                self.blockers[ship] = new_blockers

        tmp_moves = [(ship, d) for ship, d in tmp_moves if ship not in remove_moves]
        return commands, tmp_moves


class Bot:
    def __init__(self):
        self.game = Game()
        self.ship_manager = ShipManager(self.game)
        self.building_dropoff: Optional[Tuple[Position, Ship]] = None
        self.ships_to_home: Set[Ship] = set()
        self.collect_ships_stage_started = False
        self.fast_mode = False

        self.load_config()

        self.callbacks = []
        self.debug_maps = {}

        map_creator = lambda: np.empty(shape=(self.game.map.height, self.game.map.width), dtype=float)
        self.ships_mask = map_creator()
        self.own_ships_mask = map_creator()
        self.filtered_halite = map_creator()
        self.weighted_halite = map_creator()
        self.per_ship_mask = map_creator()
        self.inspiration_mask = map_creator()
        self.blurred_halite = map_creator()
        self.distances = np.zeros((self.game.map.width, self.game.map.height), dtype=float)

    # noinspection PyAttributeOutsideInit
    def load_config(self):
        h = self.game.map.height
        p = len(self.game.players)

        with (PATH / "config.yaml").open() as f:
            config = yaml.load(f.read())

        config_items = [
            item
            for item in config["params"]
            if (not item.get('map_sizes', None) or h in item['map_sizes'])
               and (not item.get('players', None) or p in item['players'])
        ]

        kwargs = {}
        for item in config_items:
            kwargs.update(item['kwargs'])

        self.distance_penalty_k = kwargs["distance_penalty_k"]
        self.ship_fill_k_base = kwargs["ship_fill_k"]
        self.ship_fill_k_reduced_after = kwargs["ship_fill_k_reduced_after"]
        self.dropoff_spawn_stop_turn = kwargs["dropoff_spawn_stop_turn"]
        self.enemy_ship_penalty = kwargs["enemy_ship_penalty"]
        self.same_target_penalty = kwargs["same_target_penalty"]
        self.turn_time_warning = kwargs["turn_time_warning"]
        self.halite_threshold = kwargs["halite_threshold"]
        self.stay_still_bonus = kwargs["stay_still_bonus"]
        self.cluster_penalty = kwargs["cluster_penalty"]
        self.blurred_halite_effect = kwargs["blurred_halite_effect"]

        self.potential_gauss_sigma = kwargs["potential_gauss_sigma"]
        self.contour_k = kwargs["contour_k"]
        self.dropoff_threshold = kwargs["dropoff_threshold"]
        self.dropoff_my_ship = kwargs["dropoff_my_ship"]
        self.dropoff_enemy_ship = kwargs["dropoff_enemy_ship"]
        self.dropoff_my_base = kwargs["dropoff_my_base"]
        self.dropoff_enemy_base = kwargs["dropoff_enemy_base"]

        self.inspiration_bonus = kwargs["inspiration_bonus"]
        self.inspiration_track_radius = kwargs["inspiration_track_radius"]
        self.inspiration_reduced = kwargs["inspiration_reduced"]

        self.ships_produce_conf = kwargs["ships_produce"]

    def run(self):
        # Create enemy ship mask (filled with discrete values: -1000, -50, 0, 1)
        # After masks overlapped it creates 3 ranges: -1000 ... -900 ... (0) 1 2
        # Zero will be replaced by 1.0
        inspiration_was_reduced = self.inspiration_reduced is None
        self.inspiration_mask.fill(0.0)
        r = 4
        self.inspiration_mask[r, r] = -1000.0
        for y in range(r * 2 + 1):
            for x in range(r * 2 + 1):
                d = abs(x - r) + abs(y - r)
                if 1 < d <= r:
                    self.inspiration_mask[y, x] = 1.0
                elif d == 1:
                    self.inspiration_mask[y, x] = -50.0
        self.inspiration_mask = roll2d(self.inspiration_mask, -r, -r)

        # Fill distances matrix
        for x in range(self.game.map.width):
            for y in range(self.game.map.height):
                self.distances[y, x] = self.game.map.distance((0, 0), (x, y))

        self.game.ready("BogdanDm" + ("_V2" if V2 else ""))
        logging.info("Player ID: {}.".format(self.game.my_id))

        while True:
            time1 = time.time()
            self.game.update_frame()
            self.ship_manager.update()

            if not inspiration_was_reduced \
                    and self.game.map.total_halite / self.game.map.initial_halite < self.inspiration_reduced:
                self.inspiration_mask.fill(0.0)
                self.inspiration_mask[r, r] = -1000.0
                for y in range(r * 2 + 1):
                    for x in range(r * 2 + 1):
                        d = abs(x - r) + abs(y - r)
                        if 2 < d <= r:
                            self.inspiration_mask[y, x] = 1.0
                        elif 0 < d <= 2:
                            self.inspiration_mask[y, x] = -1000.0
                self.inspiration_mask = roll2d(self.inspiration_mask, -r, -r)
                inspiration_was_reduced = True

            commands = self.loop() if not self.collect_ships_stage else self.collect_ships()
            self.game.end_turn(commands)

            for fn in self.callbacks:
                while not fn():
                    pass

            # Detect slow down and enable fast_mode
            turn_time = time.time() - time1
            if not LOCAL and not self.fast_mode and turn_time > self.turn_time_warning:
                self.fast_mode = True
                logging.info("Fast mod: ENABLED")
            elif self.fast_mode and turn_time < self.turn_time_warning - .5:
                self.fast_mode = False
                logging.info("Fast mod: DISABLED")

    def add_callback(self, fn):
        self.callbacks.append(fn)

    @property
    def ship_fill_k(self):
        """
        k * MAX_HALITE

        :return: (0, 1] float
        """
        gmap = self.game.map
        if gmap.total_halite / gmap.initial_halite <= self.ship_fill_k_reduced_after:
            return self.ship_fill_k_base * .65
        else:
            return self.ship_fill_k_base

    @property
    def ships_produce(self):
        halite = self.game.map.total_halite / self.game.map.initial_halite
        turn = self.game.turn_number
        for item in self.ships_produce_conf:
            if turn <= constants.MAX_TURNS * item["turn"] and halite >= item["halite"]:
                return True
        return False

    def loop(self):
        me = self.game.me
        gmap = self.game.map
        total_halite = gmap.total_halite
        n_players = len(self.game.players)
        home = self.game.me.shipyard.position
        moves: Dict[Ship, Iterable[Tuple[int, int]]] = {}
        bases = {home, *(base.position for base in me.get_dropoffs())}

        # Check whether new dropoff can be created or not
        if self.building_dropoff is None:
            if 10 < self.game.turn_number < constants.MAX_TURNS * self.dropoff_spawn_stop_turn:
                dropoff_position, dropoff_ship = self.dropoff_builder()
            else:
                dropoff_position, dropoff_ship = None, None
        else:
            dropoff_position, dropoff_ship = self.building_dropoff
            if not dropoff_ship.exists:
                dropoff_position, dropoff_ship = None, None
                self.building_dropoff = None

        if dropoff_ship is not None and dropoff_position is not None:
            self.building_dropoff = dropoff_position, dropoff_ship
            if dropoff_ship.position == dropoff_position:
                # If ship in place wait until we have enough halite to build dropoff
                if me.halite_amount + dropoff_ship.halite_amount + gmap[dropoff_position].halite_amount \
                        > constants.DROPOFF_COST + 10:
                    yield dropoff_ship.make_dropoff()
                    me.halite_amount -= constants.DROPOFF_COST - dropoff_ship.halite_amount
                    logging.info(f"Building dropoff {dropoff_position} from Ship#{dropoff_ship.id}")
                    self.building_dropoff = None
                else:
                    yield dropoff_ship.stay_still()
            else:
                # Move to new dropoff position
                if gmap[dropoff_ship].halite_amount / constants.MOVE_COST_RATIO > dropoff_ship.halite_amount:
                    yield dropoff_ship.stay_still()
                else:
                    self.ship_manager.add_target(dropoff_ship, dropoff_position)
                    logging.info(f"Ship#{dropoff_ship.id} moving to dropoff position {dropoff_position}")
                    path = gmap.a_star_path_search(dropoff_ship.position, dropoff_position)
                    moves[dropoff_ship] = (gmap.normalize_direction(path[0] - dropoff_ship.position),)

        # Ships mask
        # Contains:
        # * enemy ship penalty
        # * penalty for cells around enemy ships (4 directions)
        # * inspiration bonus based on halite estimation of player
        self.ships_mask.fill(0.)
        players_halite: Dict[Player, int] = {
            p: p.halite_estimate
            for p in self.game.players.values()
        }
        players_sorted: List[Tuple[Player, int]] = sorted(players_halite.items(), key=operator.itemgetter(1),
                                                          reverse=True)
        my_halite = players_halite[me]
        top_player_warning = players_sorted[0][0] is not me and players_sorted[0][1] - my_halite > 5000
        enable_inspiration = n_players > 2
        for i, (player, _) in enumerate(players_sorted):
            if player is me:
                continue
            for ship in player.get_ships():
                self.ships_mask += roll2d(self.inspiration_mask, *ship.position)

            # k := 1 | 2
            if enable_inspiration:
                k = min(i + 1 if i else 1, 2) if top_player_warning else 2
            else:
                k = 1
            np.clip(self.ships_mask, -np.inf, k, out=self.ships_mask)
        # Replace mask pseudo value with real coefficients
        self.ships_mask[self.ships_mask < - 300.0] = self.enemy_ship_penalty
        self.ships_mask[self.ships_mask < 0.0] = self.enemy_ship_penalty * 2
        self.ships_mask[self.ships_mask == 0.0] = 1.0
        self.ships_mask[self.ships_mask == 2.0] = self.inspiration_bonus
        np.clip(self.ships_mask, self.enemy_ship_penalty, self.inspiration_bonus, out=self.ships_mask)
        if self.callbacks:
            self.debug_maps["ships_mask"] = self.ships_mask

        self.own_ships_mask.fill(1.0)
        for ship in me.get_ships():
            self.own_ships_mask[ship.position.y, ship.position.x] = 0
        gaussian_filter(self.own_ships_mask, sigma=4, truncate=2.0, mode='wrap', output=self.own_ships_mask)
        own_ships_mask_min, own_ships_mask_max = self.own_ships_mask.min(), self.own_ships_mask.max()
        self.own_ships_mask -= own_ships_mask_min
        self.own_ships_mask /= own_ships_mask_max - own_ships_mask_min
        self.own_ships_mask *= 1 - self.cluster_penalty
        self.own_ships_mask += self.cluster_penalty
        if self.callbacks:
            self.debug_maps["own_ships_mask"] = self.own_ships_mask

        # If we have time - apply blurred mask
        if not self.fast_mode:
            gaussian_filter(gmap.halite, sigma=2, truncate=2.0, mode='wrap', output=self.blurred_halite)
            gbh_min, gbh_max = self.blurred_halite.min(), self.blurred_halite.max()
            self.blurred_halite -= gbh_min
            if gbh_max - gbh_min != 0.0:
                self.blurred_halite /= gbh_max - gbh_min
            if self.callbacks:
                self.debug_maps["gbh_norm"] = self.blurred_halite

        # Filter halite:
        # * reduce halite spikes after ships collide
        # * remove everything that is less than self.halite_threshold
        self.filtered_halite[:, :] = gmap.halite[:, :]
        self.filtered_halite[self.filtered_halite < self.halite_threshold] = 0
        # if len(players_sorted) > 2:
        #     ix = self.filtered_halite > (self.filtered_halite.mean() ** 2)
        #     self.filtered_halite[ix] = 2 * self.filtered_halite[ix] ** 0.5
        any_halite_remain = np.any(self.filtered_halite > 0)

        self.filtered_halite *= self.blurred_halite * self.blurred_halite_effect + 1 - self.blurred_halite_effect
        self.filtered_halite *= self.own_ships_mask

        if n_players == 2:
            enemy_2p = [p for p in self.game.players.values() if p is not me][0]
            ram_enabled = len(me.get_ships()) > len(enemy_2p.get_ships()) + 10 \
                          or total_halite / gmap.initial_halite <= self.ship_fill_k_reduced_after
        else:
            ram_enabled = False

        # Generate moves for ships from masks and halite field
        for ship in reversed(self.ship_manager.ships):
            if ship == dropoff_ship:
                continue

            # Ship unloading
            to_home = ship in self.ships_to_home
            if to_home and ship.position in bases:
                logging.info(f"Ship#{ship.id} unload")
                self.ships_to_home.remove(ship)
                to_home = False

            if to_home or ship.halite_amount >= constants.MAX_HALITE * self.ship_fill_k:
                logging.info(f"Ship#{ship.id} moving home")
                self.ships_to_home.add(ship)
                ship_target = min(
                    bases,
                    key=lambda base: gmap.distance(ship.position, base)
                )
                path = gmap.a_star_path_search(ship.position, ship_target, my_id=me.id)
                self.ship_manager.add_target(ship, path)
                moves[ship] = (gmap.normalize_direction(path[0] - ship.position),)

            # Ship has too low halite to move
            elif gmap[ship].halite_amount / constants.MOVE_COST_RATIO > ship.halite_amount:
                yield ship.stay_still()
                continue

            else:
                if ram_enabled:
                    # Ram enemy ship with a lot of halite
                    for direction in Direction.All:
                        other_ship = gmap[ship.position + direction].ship
                        if not other_ship or other_ship.owner == me.id:
                            continue
                        if other_ship.halite_amount / (ship.halite_amount + 1) >= 4:
                            target_found = True
                            break
                    else:
                        target_found = False
                    if target_found:
                        # noinspection PyUnboundLocalVariable
                        logging.info(f"Ship#{ship.id} ramming enemy ship #{other_ship.id}"
                                     f" ({ship.halite_amount}) vs ({other_ship.halite_amount})")
                        # noinspection PyUnboundLocalVariable
                        yield gmap.update_ship_position(ship, direction)
                        continue

                if any_halite_remain:
                    self.per_ship_mask.fill(1.0)
                    for another_ship, target in self.ship_manager.targets.items():
                        if isinstance(target, list):
                            target = target[0]
                        if another_ship is ship or target is None or target == home:
                            continue
                        self.per_ship_mask[target[1], target[0]] *= self.same_target_penalty
                    self.weighted_halite[:, :] = self.filtered_halite[:, :]

                    # Ship mask is filtered by distance
                    distances = roll2d(self.distances, *ship.position)
                    ships_mask = np.copy(self.ships_mask)
                    ix = distances > self.inspiration_track_radius
                    ships_mask[ix] = np.clip(ships_mask[ix], 0, 1)

                    # Apply masks
                    self.weighted_halite *= ships_mask
                    self.weighted_halite *= self.per_ship_mask
                    self.weighted_halite /= (distances + 1) ** self.distance_penalty_k
                    # self.weighted_halite[ship.position.y, ship.position.x] /= ship_collecting_halite_coefficient(ship, gmap)
                    self.weighted_halite[ship.position.y, ship.position.x] *= self.stay_still_bonus

                    if self.callbacks:
                        self.debug_maps[ship.id] = self.weighted_halite.copy()

                    # Found max point
                    y, x = np.unravel_index(self.weighted_halite.argmax(), self.weighted_halite.shape)
                    max_halite = self.weighted_halite[y, x]
                    if max_halite > 0:
                        target = Position(x, y)
                        if target is dropoff_position:
                            continue
                        if target == ship.position:
                            logging.info(f"Ship#{ship.id} collecting halite")
                            self.ship_manager.add_target(ship, None)
                            yield ship.stay_still()
                        else:
                            dist = gmap.distance(ship.position, target)
                            if n_players > 2 and dist == 1:
                                for d in Direction.All:
                                    enemy_pos = gmap[target + d]
                                    if enemy_pos.ship and enemy_pos.ship.owner != me.id:
                                        break
                                else:
                                    enemy_pos = None
                            else:
                                enemy_pos = None
                            if enemy_pos:
                                logging.info(f"Ship#{ship.id}({ship.position}) is afraid of {enemy_pos.ship}")
                                yield ship.stay_still()
                            else:
                                logging.info(f"Ship#{ship.id} {ship.position} moving towards {target}")
                                self.ship_manager.add_target(ship, target)
                                moves[ship] = list(self.game.map.get_unsafe_moves(ship.position, target))
                    else:
                        logging.info(f"Ship#{ship.id} does not found good halite deposit")
                        self.ship_manager.add_target(ship, None)
                        yield ship.stay_still()
                else:
                    self.ship_manager.add_target(ship, None)
                    choices = list(Direction.All) + [Direction.Still]
                    moves[ship] = choices[np.random.choice(
                        range(len(choices)),
                        p=[1 / 14, 1 / 14, 1 / 14, 1 / 14, 10 / 14]
                    )],

        moves: List[Tuple[Ship, Iterable[Tuple[int, int]]]] = list(moves.items())
        yield from self.ship_manager.resolve_moves(moves)

        # Max ships: base at 32 map size, 2*base at 64 map size
        shipyard_cell = gmap[me.shipyard]
        if (
                (
                        self.building_dropoff is None
                        and me.halite_amount >= constants.SHIP_COST
                        or
                        self.building_dropoff is not None and all(self.building_dropoff)
                        and me.halite_amount >= constants.SHIP_COST + constants.DROPOFF_COST
                )
                and (shipyard_cell.ship is None or shipyard_cell.ship.owner != me.id)
                and self.ships_produce
        ):
            yield me.shipyard.spawn()

    def dropoff_builder(self) -> Tuple[Optional[Position], Optional[Ship]]:
        GAUSS_SIGMA = 2.
        MIN_HALITE_PER_REGION = 32000

        gmap = self.game.map
        me = self.game.me
        bases = {self.game.me.shipyard.position, *(base.position for base in self.game.me.get_dropoffs())}
        halite_ex = gmap.halite_extended.copy()
        h, w, d = gmap.height, gmap.width, 2

        # Fill map border with zeros
        halite_ex[:d, 0:2 * w] = 0
        halite_ex[-d:, 0:2 * w] = 0
        halite_ex[0:2 * h, :d] = 0
        halite_ex[0:2 * h, -d:] = 0
        halite_ex_gb = gaussian_filter(halite_ex, sigma=GAUSS_SIGMA, truncate=2.0, mode='wrap')

        halite_ex_gb_filtered = halite_ex_gb.copy()
        offset = h // 2
        mask = np.zeros(halite_ex_gb_filtered.shape, dtype=int)
        r = 10
        for ship in me.get_ships():
            x, y = ship.position
            mask = roll2d(mask, r + 1 - x - offset, r + 1 - y - offset)
            mask[:2 * r + 1, :2 * r + 1] += 1
            mask = roll2d(mask, x + offset - r - 1, y + offset - r - 1)
        np.clip(mask, 0, 1, out=mask)
        halite_ex_gb_filtered *= mask
        if self.callbacks:
            self.debug_maps["halite_ex_gb_filtered"] = halite_ex_gb_filtered[offset:3 * offset, offset:3 * offset]

        # Find contours of high halite deposits
        contours = msr.find_contours(halite_ex_gb, halite_ex_gb_filtered.max() * self.contour_k, fully_connected="high")
        masks = [poly2mask(c[:, 0], c[:, 1], halite_ex.shape) for c in contours]

        # Remove small halite chunks
        good_contours = []
        for c, m in zip(contours, masks):
            if np.sum(np.multiply(gmap.halite_extended, m)) > MIN_HALITE_PER_REGION:
                good_contours.append(c)

        # Remove offset of contour points
        contours = [contour - h // 2 for contour in good_contours]

        # Set up possible dropoff points
        points = {tuple(float(round(x)) for x in point)
                  for contour in contours
                  for point in contour}
        points = [list(point) for point in points
                  if all(0 <= x < h for x in point) and Position(*map(int, point)) not in bases]
        points = np.array(points, dtype=np.int)
        if points.shape[0] == 0:
            return None, None

        # Create potential field
        potential_field = np.zeros(gmap.halite.shape, dtype=np.float)
        for player in self.game.players.values():
            me = player is self.game.me
            for ship in player.get_ships():
                if me:
                    potential_field[ship.position.y, ship.position.x] += self.dropoff_my_ship
                else:
                    potential_field[ship.position.y, ship.position.x] += self.dropoff_enemy_ship
            for base in player.get_dropoffs():
                if me:
                    potential_field[base.position.y, base.position.x] += self.dropoff_my_base
                else:
                    potential_field[base.position.y, base.position.x] += self.dropoff_enemy_base
            pos = player.shipyard.position
            potential_field[pos.y, pos.x] += self.dropoff_my_base if me else self.dropoff_enemy_base
        potential_field = gaussian_filter(potential_field, sigma=self.potential_gauss_sigma, truncate=3.0, mode='wrap')

        if self.callbacks:
            self.debug_maps['dropoff'] = potential_field

        # Overlay points and potential_field
        potential_field_filtered = np.full(potential_field.shape, 0, dtype=np.float)
        x, y = points[:, 0], points[:, 1]
        potential_field_filtered[x, y] = potential_field[x, y]

        self.debug_maps['dropoff_2'] = potential_field_filtered
        y, x = np.unravel_index(potential_field_filtered.argmax(), potential_field_filtered.shape)
        value = potential_field_filtered[y, x]
        logging.debug(f"DROPOFF -> {value:.6f}")
        if value < self.dropoff_threshold:
            return None, None

        ships = self.game.me.get_ships()
        target = Position(x, y)
        distance = lambda ship: gmap.distance(ship.position, target)
        # ship_filter= lambda ship: ship.halite_amount / constants.MAX_HALITE < self.ship_fill_k * .75
        ship = min(ships, key=distance)
        return target, ship

    @property
    def collect_ships_stage(self):
        if self.collect_ships_stage_started:
            return True
        if self.game.turn_number + 3 < constants.MAX_TURNS - self.game.map.width:
            return False

        me = self.game.me
        gmap = self.game.map
        bases = {me.shipyard.position, *(base.position for base in me.get_dropoffs())}

        distances = [
            min(gmap.distance(base, ship.position) for base in bases)
            for ship in me.get_ships()
        ]
        distances = sorted(distances)

        if not distances:
            return False
        if constants.MAX_TURNS - self.game.turn_number - 5 <= distances[-1]:
            self.collect_ships_stage_started = True
            return True

    def get_ship_parking_spot(self, ship: 'Ship', base: Position) -> Position:
        gmap = self.game.map
        s = gmap.width // 2
        x, y = gmap.normalize(ship.position - base + (s, s))
        if 0 <= x < s:
            hor = 0
        elif x == s:
            hor = 0 if y > s else 1
        else:
            hor = 1

        if 0 <= y < s:
            vert = 0
        elif y == s:
            vert = 0 if x > s else 1
        else:
            vert = 1

        return base + {
            (0, 0): (0, -1),
            (1, 0): (1, 0),
            (1, 1): (0, 1),
            (0, 1): (-1, 0),
        }[hor, vert]

    def collect_ships(self):
        gmap = self.game.map
        me = self.game.me
        enemy: Player = choice([p for p in self.game.players.values() if p is not me])
        bases = {me.shipyard.position, *(base.position for base in me.get_dropoffs())}

        collect_points = {gmap.normalize(base + d) for d in Direction.All for base in bases} | bases
        moves = []
        for i, ship in enumerate(me.get_ships()):
            if gmap[ship].halite_amount / constants.MOVE_COST_RATIO > ship.halite_amount:
                yield ship.stay_still()
                continue

            nearest_base = min(bases, key=lambda base: gmap.distance(base, ship.position))
            if ship.position not in collect_points:
                if ship.halite_amount < 20:
                    ship_moves = list(gmap.get_unsafe_moves(
                        ship.position,
                        enemy.shipyard.position + Direction.All[i % 4]
                    ))
                    shuffle(ship_moves)
                else:
                    point = self.get_ship_parking_spot(ship, nearest_base)
                    ship_moves = list(gmap.get_unsafe_moves(ship.position, point))
                moves.append((ship, ship_moves))
            else:
                direction = next(gmap.get_unsafe_moves(ship.position, nearest_base), None)
                if direction:
                    yield gmap.update_ship_position(ship, direction)
                else:
                    yield ship.stay_still()

        yield from self.ship_manager.resolve_moves(moves, ignore_enemy_ships=True)
