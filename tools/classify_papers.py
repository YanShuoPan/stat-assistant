import sys, io, csv, re
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CSVPATH = "D:/OneDrive - Morale AI/model_bridge2/papers/top20_stats_journals_2021_2025.csv"
OUTPATH = "D:/OneDrive - Morale AI/model_bridge2/papers/top20_stats_journals_2021_2025_tagged.csv"