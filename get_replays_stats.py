import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tqdm
import zstandard

PATH: Path = Path(__file__).parent.absolute().resolve() / "manager"
d = zstandard.ZstdDecompressor()

players = defaultdict(lambda: {
    (2, 32): 0,
    (2, 40): 0,
    (2, 48): 0,
    (2, 56): 0,
    (2, 64): 0,
    (4, 32): 0,
    (4, 40): 0,
    (4, 48): 0,
    (4, 56): 0,
    (4, 64): 0,
    "all": 0
})
players["total"]["all"] = 10000

SCORES = {
    2: [None, 2, -1],
    4: [None, 2, 1, 0, -1],
}

files = list((PATH / "replays").glob("*.hlt"))
for file in tqdm.tqdm(files):
    with file.open("rb") as f:
        raw = f.read()
    try:
        data = json.loads(d.decompress(raw))
    except (zstandard.ZstdError, json.JSONDecodeError):
        continue

    p_n = len(data["players"])
    m_n = data["production_map"]["height"]
    if m_n not in (32, 40, 48, 56, 64):
        continue

    players["total"][(p_n, m_n)] += 1
    for player in data["players"]:
        pid = player["player_id"]
        name = player["name"]
        stat = data["game_statistics"]["player_statistics"][pid]
        rank = stat["rank"]
        players[name][(p_n, m_n)] += SCORES[p_n][rank]
        players[name]["all"] += SCORES[p_n][rank]

df = pd.DataFrame(players).T
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.loc[df["all"].sort_values(ascending=False).index])
