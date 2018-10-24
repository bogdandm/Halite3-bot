halite.exe --replay-directory replays/ -vvv --width 32 --height 32 -n 4 --"no-timeout"^
    "python MyBot.py --local --v2"^
::    "python MyBot.py --local --v2"^
    "python MyBot.py --local"^
::    "python MyBot.py --local"
:: --"no-timeout"