import logging
import operator
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple

import hlt
from hlt import Direction, Position, constants
from hlt.entity import Ship
from .const import LOCAL


def ship_collecting_halite_coefficient(ship, gmap):
    ship_cargo_free = constants.MAX_HALITE - ship.halite_amount
    cell = gmap[ship.position].halite_amount or 10 ** -10
    return max((cell - ship_cargo_free) / cell, .1)


class Bot:
    def __init__(
            self,
            ship_fill_k=.7,
            distance_penalty_k=1.3,
            ship_limit=30,
            ship_turns_stop=100,
            enemy_ship_penalty=.1,
            enemy_ship_nearby_penalty=.3,  # .9
            same_target_penalty=.7
    ):
        self.ship_fill_k = ship_fill_k
        self.distance_penalty_k = distance_penalty_k
        self.ship_limit_base = ship_limit
        self.ship_limit = self.ship_limit_base
        self.ship_turns_stop = ship_turns_stop
        self.enemy_ship_penalty = enemy_ship_penalty
        self.enemy_ship_nearby_penalty = enemy_ship_nearby_penalty
        self.same_target_penalty = same_target_penalty
        self.stay_still_bonus = None

        self.game = hlt.Game()
        self.callbacks = []
        self.debug_maps = {}
        self.ships_targets: Dict[Ship, Tuple[int, int]] = {}
        self.max_ships_reached = 0

    def run(self):
        self.game.ready("BogdanDm")
        logging.info("Player ID: {}.".format(self.game.my_id))

        self.stay_still_bonus = 1 + 1 / constants.MOVE_COST_RATIO
        self.ship_limit = round(self.ship_limit_base * (1 + (self.game.map.width - 32) / (72 - 32)))
        if len(self.game.players) == 4:
            self.ship_limit //= 1.2

        while True:
            self.game.update_frame()

            my_ships = {ship.id: ship for ship in self.game.me.get_ships()}
            # Update ships and delete destroyed ones
            self.ships_targets = {
                my_ships[ship.id]: target
                for ship, target in self.ships_targets.items()
                if ship.id in my_ships
            }

            commands = self.loop()
            self.game.end_turn(commands)

            for fn in self.callbacks:
                while not fn():
                    pass

    def add_callback(self, fn):
        self.callbacks.append(fn)

    def loop(self):
        me = self.game.me
        my_ships = me.get_ships()
        gmap = self.game.map
        home = self.game.me.shipyard.position

        # General mask
        # Contains:
        # * enemy ship penalty
        # * penalty for cells around enemy ships (4 directions)
        # * friendly dynamic penalty based on ship cargo / cell halite ratio
        mask = defaultdict(lambda: 1)
        for player in self.game.players.values():
            for ship in player.get_ships():
                if player is not me:
                    mask[ship.position] *= self.enemy_ship_penalty
                    for pos in ship.position.get_surrounding_cardinals():
                        mask[gmap.normalize_direction(pos)] *= self.enemy_ship_nearby_penalty
                else:
                    mask[ship.position] *= ship_collecting_halite_coefficient(ship, gmap)

        self.debug_maps["mask"] = mask

        # Generate moves for ships from mask and halite field
        moves: List[Tuple[Ship, Iterable[Tuple[int, int]]]] = []
        for ship in my_ships:
            # Ship unloading
            if ship.halite_amount >= constants.MAX_HALITE * self.ship_fill_k:
                logging.info(f"Ship#{ship.id} moving home")
                self.ships_targets[ship] = home
                moves.append((ship, list(self.game.map.get_unsafe_moves(ship.position, home))))

            else:
                per_ship_mask = defaultdict(lambda: 1)
                for another_ship, target in self.ships_targets.items():
                    if another_ship is ship or target is None or target == home:
                        continue
                    # Penalty for preventing long queues to happen
                    # if V2:
                    #     # Based on distance difference
                    #     ship_distance = gmap.calculate_distance(ship.position, target)
                    #     another_ship_distance = gmap.calculate_distance(another_ship.position, target)
                    #     k = ship_distance / another_ship_distance if another_ship_distance else 1
                    #     # logging.debug(
                    #     #     f"#{str(another_ship):50} -> {str(target):16} "
                    #     #     f"{k:.2f} ({ship_distance:2d} / {another_ship_distance:2d})"
                    #     # )
                    #     per_ship_mask[target] *= 1 if k <= .9 else (self.same_target_penalty ** k)
                    # else:
                    per_ship_mask[target] *= self.same_target_penalty

                # Ship has too low halite to move
                if gmap[ship].halite_amount / constants.MOVE_COST_RATIO > ship.halite_amount:
                    # self.ships_targets[ship] = None
                    yield ship.stay_still()
                    continue

                position = None
                radius = 8 if LOCAL else 16
                if radius:
                    field = ship.position.get_surrounding_cardinals(radius, center=True)
                else:
                    field = map(operator.attrgetter('position'), gmap)
                surrounding = {Position(*coord): gmap[coord].halite_amount for coord in field}

                # Apply masks
                for coord in surrounding:
                    surrounding[coord] *= mask[coord]
                    surrounding[coord] *= per_ship_mask[coord]
                    # surrounding[coord] /= 1 + 1 / constants.MOVE_COST_RATIO
                    distance = gmap.calculate_distance(ship.position, coord)
                    if distance:
                        # Distance penalty
                        # TODO: Experiment with A* algorithm
                        surrounding[coord] /= (distance + 1) ** self.distance_penalty_k
                    else:
                        # Revert "same point penalty" of current ship
                        surrounding[coord] /= ship_collecting_halite_coefficient(ship, gmap)
                        # surrounding[coord] *= self.stay_still_bonus

                self.debug_maps[ship.id] = surrounding

                # Found max point
                max_halite: Tuple[Position, int] = max(surrounding.items(), key=operator.itemgetter(1))
                if max_halite[1] > 0.1:
                    position = max_halite[0]

                if position:
                    if position == ship.position:
                        logging.info(f"Ship#{ship.id} collecting halite")
                        self.ships_targets[ship] = None
                        yield ship.stay_still()
                    else:
                        logging.info(f"Ship#{ship.id} {ship.position} moving towards {position}")
                        self.ships_targets[ship] = position
                        moves.append((ship, list(self.game.map.get_unsafe_moves(ship.position, position))))
                else:
                    logging.info(f"Ship#{ship.id} does not found good halite deposit")
                    self.ships_targets[ship] = None
                    yield ship.stay_still()

        # Resolve moves into commands in 2 iterations
        for i in range(2):
            commands, moves = self.resolve_moves(moves)
            yield from commands

        # Remaining ship has not any safe moves
        for ship, d in moves:
            logging.debug(f"Unsafe move: {ship} -> {d}")
            yield ship.stay_still()

        # Max ships: base at 32 map size, 2*base at 64 map size
        if (
                self.game.turn_number <= constants.MAX_TURNS - self.ship_turns_stop
                and me.halite_amount >= constants.SHIP_COST
                and not gmap[me.shipyard].is_occupied
                and self.max_ships_reached <= 2
        ):
            if len(my_ships) < self.ship_limit:
                yield me.shipyard.spawn()
            else:
                self.max_ships_reached += 1

    def resolve_moves(self, moves: List[Tuple[Ship, Iterable[Tuple[int, int]]]]):
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
                    if not cell.is_occupied or cell.is_occupied_base(me):
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

        return commands, tmp_moves
