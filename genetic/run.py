import argparse

from .bot_model import BotArguments, BotArgumentsV3
from .core import GeneticOptimizer, GeneticOptimizerCore

VERSION = {
    "latest": BotArguments,
    "v3": BotArgumentsV3
}


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mutation-rate-base", default=.04, type=float)
    parser.add_argument("-n", "--bots-per-generation", default=24, type=int)
    parser.add_argument("-N", "--number-games", nargs=2, default=(8, 4), type=int)
    parser.add_argument("-v", "--version", default="latest")
    parser.add_argument("-g", "--generations", default=4, type=int)

    args = parser.parse_args(args)
    return args.mutation_rate_base, args.bots_per_generation, args.number_games, VERSION[args.version], args.generations


mutation_rate_base, bots_per_generation, (count_2, count_4), version, generations = parse_args()
go = GeneticOptimizer(GeneticOptimizerCore(
    version(),
    mutation_rate=lambda generation: mutation_rate_base / (generation + 1),
    bots_per_generation=bots_per_generation,
    count_2=count_2,
    count_4=count_4
))
go.run(generations=generations)
