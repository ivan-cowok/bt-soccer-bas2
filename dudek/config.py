import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Dataset paths — set these in .env
ACTION_SPOTTING_DATASET_PATH = os.getenv("ACTION_SPOTTING_DATASET_PATH", "/workspace/bas/data/broadcast_videos")
#ACTION_SPOTTING_DATASET_PATH = os.getenv("ACTION_SPOTTING_DATASET_PATH", "E:/Database/44/soccernet_action_spotting")
BAS_DATASET_PATH = os.getenv("SN_BAS_2025_DATASET_PATH", os.getenv("BAS_DATASET_PATH", "E:/Database/44/spotting-ball-2023/train"))

# Legacy aliases
VIDEOS_DATASET_PATH = os.getenv("VIDEOS_DATASET_PATH", ACTION_SPOTTING_DATASET_PATH)
SN_BAS_2025_DATASET_PATH = BAS_DATASET_PATH

# Training
EXPERIMENTS_RANDOM_SEED = int(os.getenv("EXPERIMENTS_RANDOM_SEED", 42))
TEST_SET_CHALLENGE_SEED = 116

# Device
DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "cuda")

# Temp dir — /tmp on Linux, system temp on Windows
TMP_DIR = os.getenv("TMP_DIR", tempfile.gettempdir())

_VIDEOS_WITH_FPS_PROBLEMS = [
    "italy_serie-a/2016-2017/2016-10-02 - 21-45 AS Roma 2 - 1 Inter/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-10-02 - 21-45 AS Roma 2 - 1 Inter/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-11 - 16-00 AC Milan 0 - 1 Udinese/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-11 - 16-00 AC Milan 0 - 1 Udinese/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-08-28 - 21-45 Cagliari 2 - 2 AS Roma/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-08-28 - 21-45 Cagliari 2 - 2 AS Roma/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-08-27 - 21-45 Napoli 4 - 2 AC Milan/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-08-27 - 21-45 Napoli 4 - 2 AC Milan/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-25 - 13-30 Torino 3 - 1 AS Roma/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-25 - 13-30 Torino 3 - 1 AS Roma/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-20 - 21-45 AC Milan 2 - 0 Lazio/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-20 - 21-45 AC Milan 2 - 0 Lazio/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-16 - 21-45 Sampdoria 0 - 1 AC Milan/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-16 - 21-45 Sampdoria 0 - 1 AC Milan/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-21 - 21-45 AS Roma 4 - 0 Crotone/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-21 - 21-45 AS Roma 4 - 0 Crotone/2_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-18 - 21-45 Fiorentina 1 - 0 AS Roma/1_720p.mkv",
    "italy_serie-a/2016-2017/2016-09-18 - 21-45 Fiorentina 1 - 0 AS Roma/2_720p.mkv",
]
