from pathlib import Path
from dotenv import load_dotenv
import kagglehub
import os


# Load KAGGLE_USERNAME and KAGGLE_KEY from .env
load_dotenv()

# Change this to your dataset
DATASET = "asr-2026-spoken-numbers-recognition-challenge"

# Optional: choose where you want the dataset stored
DOWNLOAD_DIR = Path("data")
os.environ["KAGGLEHUB_CACHE"] = str(DOWNLOAD_DIR)

path = kagglehub.competition_download(DATASET)

print(f"Downloaded to: {path}")
