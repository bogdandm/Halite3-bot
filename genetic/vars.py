import logging
import os
from pathlib import Path

CWD = (Path(__file__) / ".." / "..").resolve().absolute()
DB = "genetic.db"
os.chdir(CWD)
logging.basicConfig(
    format='%(levelname)10s %(message)s',
    filename=str(CWD / 'genetic.log'), filemode='w', level=logging.DEBUG
)
