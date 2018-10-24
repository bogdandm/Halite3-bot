#!/usr/bin/env python3
# Python 3.7

import logging
import sys

sys.path.append("pycharm-debug-py3k.egg")
import pydevd

pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)
from bot.bot import Bot

bot = Bot()
if "--plot" in sys.argv:
    from bot.plotter import Plotter

    plotter = Plotter(bot)
logging.info("Initialization finished")
bot.run()
