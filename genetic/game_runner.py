import json
import subprocess
import sys
from typing import List, Optional

from .vars import CWD

_SPACE_DELIMITER = ' '
_BOT_ID_POSITION = 1
import logging

logging.basicConfig()


def wrap(s: str, quotes='"'):
    if not s.startswith(quotes):
        s = quotes + s
    if not s.endswith(quotes):
        s += quotes
    return s


def win_arg_wrap(arg: str):
    if sys.platform.startswith("win") and arg.startswith("--") and arg[2] != '"':
        return "--" + '"' + arg[2:] + '"'
    return arg


def _play_game(binary, bot_commands, flags):
    """
    Plays one game considering the specified bots and the game and map constraints.
    :param binary: The halite binary
    :param bot_commands: The commands to run each of the bots
    :return: The game's result string
    """
    command = [
        binary,
        "--results-as-json"
    ]
    command.extend(flags)
    for bot_command in bot_commands:
        command.append(bot_command)
    logging.debug(f'Run: \'{" ".join(command)}\'')
    logging.debug(f"CWD: '{CWD}'")
    return subprocess.check_output(" ".join(map(win_arg_wrap, command)), cwd=str(CWD), shell=True).decode()


def play_game(binary: str, bot_commands: List[str], *flags: str, map_width: int = None, map_height: int = None,
              no_timeout=True, game_output_dir: Optional[str] = None, ):
    """
    :param binary: The Halite binary.
    :param bot_commands: The commands to run each of the bots (must be either 2 or 4)
    :param flags: Additional arguments
    :param map_width: The map width, set to None for engine random choice
    :param map_height: The map height, set to None for engine random choice
    :param no_timeout: Causes game environment to ignore bot timeouts
    :param game_output_dir: Where to put replays and log files.
    :return: Game stats
    """
    flags = list(flags)

    if no_timeout:
        flags.append('--no-timeout')

    if game_output_dir is not None:
        flags.extend(("-i", wrap(game_output_dir)))
    else:
        flags.extend(("--no-logs", "--no-replay"))

    if map_width is not None:
        flags.extend(("--width", str(map_width)))
    if map_height is not None:
        flags.extend(("--height", str(map_height)))

    if not (len(bot_commands) == 4 or len(bot_commands) == 2):
        raise IndexError("The number of bots specified must be either 2 or 4.")
    bot_commands = [f'"{cmd}"' if '"' not in cmd else cmd for cmd in bot_commands]
    match_output = _play_game(binary, bot_commands, flags)
    results = json.loads(match_output)
    results['stats'] = [results['stats'][str(i)] for i in range(len(bot_commands))]
    return results
