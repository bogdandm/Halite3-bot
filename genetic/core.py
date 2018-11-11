import operator
import statistics
import sys
from queue import Queue
from random import randint, random, shuffle
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import sqlalchemy as sql
from tqdm import tqdm

from .bot_model import GenericBotArguments, compile_args
from .db_connection import Session
from .game_runner import play_game
from .models import BotInstance, GameResult

MAP_SIZES = {
    2: [
        32,
        40,
        46,
        54
    ],
    4: [
        32,
        50
    ]
}


def call_or_return_value(a: Union[Any, Callable], *args, **kwargs):
    return a(*args, **kwargs) if callable(a) else a


class GeneticOptimizerCore:
    def __init__(self, bot_class: GenericBotArguments,
                 bots_per_generation=24, count_2=8, count_4=4,
                 mutation_rate: Union[float, Callable[[int], float]] = .01,
                 mutation_chance: Union[float, Callable[[int], float]] = .75):
        self.bot_class = bot_class
        self.bots_per_generation = bots_per_generation
        self.count_2 = count_2
        self.count_4 = count_4
        self.mutation_rate = mutation_rate
        self.mutation_chance = mutation_chance

    def create_generation(self):
        """
        Create new generation with random values
        """
        for i in range(self.bots_per_generation):
            yield self.bot_class.generate()

    def evolve_generation(self, generation: List[BotInstance], generation_number):
        """
        Return new generation:
        * Half is best species from old generation with some mutations
        * Another half is random children from best species with mean +- mutation values
        """
        halite = [bot.halite for bot in generation]
        mean_f = statistics.mean(halite)
        sigma_f = statistics.stdev(halite, xbar=mean_f)
        F = [
            1 + abs(h - mean_f) / 2 / sigma_f
            for h in halite
        ]
        F_sum = sum(F)
        P = [f / F_sum for f in F]
        n = len(generation)
        choices = set(np.random.choice(
            list(range(len(generation))),
            randint(n // 2 - 2, n // 2 + 2),
            replace=False, p=P
        ))
        top_bots = [bot for i, bot in enumerate(generation) if i in choices]
        top_bots_P = np.fromiter((p for i, p in enumerate(P) if i in choices), dtype=float)
        top_bots_P /= top_bots_P.sum()

        for bot in top_bots:
            if random() <= call_or_return_value(self.mutation_chance, generation_number):
                yield self.bot_class.mutate(
                    bot.args_dict,
                    call_or_return_value(self.mutation_rate, generation_number)
                )
            else:
                yield bot.args_dict

        n = self.bots_per_generation - len(top_bots)
        parents_1 = np.random.choice(top_bots, n, p=top_bots_P)
        parents_2 = np.random.choice(generation, n, replace=False)
        for p1, p2 in zip(parents_1, parents_2):
            yield self.bot_class.breed(p1.args_dict, p2.args_dict)

    def run_game(self, bots: List[BotInstance]) -> dict:
        """
        Run game with bots with given params
        :param bots:
        :return:
        """
        n = len(bots)
        assert n in (2, 4), "Only 2 or 4 players supported"

        game_count_getter = operator.attrgetter("game2_count") if n == 2 else operator.attrgetter("game4_count")
        assert all(game_count_getter(bot) == game_count_getter(bots[0]) for bot in bots), \
            "All bots should have same number of games played"

        map_size = MAP_SIZES[n][game_count_getter(bots[0]) % len(MAP_SIZES[n])]
        result = play_game(
            "halite.exe" if sys.platform.startswith("win") else "./halite",
            [self.bot_class.command.format(args=compile_args(bot.args_dict))
             for bot in bots],
            map_width=map_size, map_height=map_size,
            game_output_dir="./replays/"
        )
        return result


class GeneticOptimizerSqlAdapter:
    @property
    def last_generation(self):
        return Session.query(sql.func.max(BotInstance.generation)).first()[0]

    def save_new_generation(self, generation: Iterable[dict], bot_version):
        """
        Save new generation (Generation number is N+1)
        """
        n = self.last_generation
        generation_number = (n if n is not None else -1) + 1

        bots = []
        for args in generation:
            bot = BotInstance(version=bot_version, generation=generation_number)
            bot.args_dict = args
            bots.append(bot)
        Session.add_all(bots)
        Session.commit()

    def load_generation(self, generation_number: int) -> List[BotInstance]:
        """
        Get bots from generation
        """
        bots = Session.query(BotInstance).filter_by(generation=generation_number).all()
        return bots

    def generations_stats(self):
        stats = Session.query(sql.func.sum(BotInstance.halite).label("halite")) \
            .group_by(BotInstance.generation).order_by(BotInstance.generation).all()
        return [stat.halite for stat in stats]

    def save_game_result(self, generation_number, results: dict, bots: List[BotInstance]):
        """
        Create record for game result and update bots score
        """
        bots_count = len(bots)
        assert bots_count in (2, 4)

        game = GameResult(generation=generation_number)
        for result, bot in zip(results['stats'], bots):
            if bots_count == 2:
                bot.game2_count += 1
            else:
                bot.game4_count += 1
            bot.halite += result["score"]
            setattr(game, f"bot{result['rank']}_id", bot.id)

        Session.add(game)
        Session.commit()

    def select_bots(self, count_2, count_4) -> Optional[Tuple[Queue, int]]:
        generation_number = self.last_generation

        min_games = Session.query(sql.func.min(BotInstance.game2_count)).one()[0]
        if min_games < count_2:
            bots = list(Session.query(BotInstance).filter_by(generation=generation_number, game2_count=min_games))
            n = 2
        else:
            min_games = Session.query(sql.func.min(BotInstance.game4_count)).one()[0]
            if min_games < count_4:
                bots = list(Session.query(BotInstance).filter_by(generation=generation_number, game4_count=min_games))
                n = 4
            else:
                bots = None
                n = 0
        if bots:
            shuffle(bots)
            q = Queue()
            for bot in bots:
                q.put(bot)
            return q, n
        return None


class GeneticOptimizer:
    def __init__(self, core: GeneticOptimizerCore, db: GeneticOptimizerSqlAdapter = None):
        self.core = core
        self.db = db or GeneticOptimizerSqlAdapter()
        self.g_number = None

    def initialize(self):
        self.g_number = self.db.last_generation
        if self.g_number is None:
            self.db.save_new_generation(self.core.create_generation(), self.core.bot_class.version)
            self.g_number = self.db.last_generation

    def run(self, generations=10):
        self.initialize()

        games_per_generation = int(self.core.count_2 * self.core.bots_per_generation / 2
                                   + self.core.count_4 * self.core.bots_per_generation / 4)
        games_played = Session.query(GameResult).count()

        with tqdm(
                total=games_per_generation * generations,
                ncols=70,
                initial=games_played,
                position=0
        ) as pbar:
            while self.g_number < generations:
                with tqdm(
                        total=games_per_generation, desc=f"Generation #{self.g_number:03d}",
                        ncols=68,
                        initial=games_played % games_per_generation,
                        position=1
                ) as pbar2:
                    selection = self.db.select_bots(self.core.count_2, self.core.count_4)
                    while selection:
                        bots, n = selection
                        bots = [bots.get() for _ in range(n)]
                        result = self.core.run_game(bots)
                        self.db.save_game_result(self.g_number, result, bots)
                        pbar.update()
                        pbar2.update()
                        selection = self.db.select_bots(self.core.count_2, self.core.count_4)

                games_played = Session.query(GameResult).count()
                if self.g_number + 1 < generations:
                    bots = self.db.load_generation(self.g_number)
                    self.db.save_new_generation(self.core.evolve_generation(bots, self.g_number),
                                                self.core.bot_class.version)
                    self.g_number = self.db.last_generation
                else:
                    self.g_number += 1

    def print(self):
        bots = [(bot.args_dict, bot.halite) for bot in self.db.load_generation(self.db.last_generation)]
        halite_sum = sum(bot[1] for bot in bots)
        weights = [bot[1] / halite_sum for bot in bots]
        args_names = [arg for arg, _ in self.core.bot_class]
        max_arg_len = max(map(len, args_names))
        for arg in args_names:
            values = [args[arg] for args, _ in bots]
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            weighted_mean = sum(v * k for v, k in zip(values, weights))
            formatted_arg = ("{:>%ds}" % max_arg_len).format(arg)
            print(f"{formatted_arg} | {float(mean):.3f} +-{float(std):.5f} (weighted: {weighted_mean:.3f})")

        print("-" * 50)
        stats = self.db.generations_stats()
        for i, stat in enumerate(stats):
            print(f"{i}) {stat/1e+6:.2f}kk")
