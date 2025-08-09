import os
import time
import glob
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import warnings

from libary.get_matches import MatchScraper
from libary.structure_data import MatchProcessor
from libary.train_model import EnsemblePredictor
from libary.player_stats_db import PlayerStatsCollector

warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.context")
warnings.filterwarnings("ignore", category=UserWarning, module="subprocess")
warnings.filterwarnings("ignore", module="joblib.externals.loky.backend.context")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cs2_predictor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

MATCH_DIR = "./matches"
DATASET_CSV = "./dataset/match_training_data_full.csv"
MODEL_PATH = "./model/model.joblib"
FEATURES_PATH = "./model/feature_columns.joblib"
MATCH_FILE_PATTERN = os.path.join(MATCH_DIR, "*.json")
MAX_DAYS_BACK = 550
MATCH_FILE_DATE_FMT = "%Y-%m-%d"

def setup_directories():
    directories = [MATCH_DIR, "./dataset", "./model", "./predictions"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def update_player_stats():
    logging.info("Updating player statistics database...")
    collector = PlayerStatsCollector()

    try:
        player_ids = set()
        match_files = glob.glob(MATCH_FILE_PATTERN)

        if not match_files:
            logging.info("No match files found to process player stats")
            return

        for file in match_files:
            try:
                with open(file, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                    for game in data.get("games", []):
                        for player in game.get("player_stats", []):
                            player_id = player.get("steam_profile_player_slug")
                            if player_id:
                                player_ids.add(player_id)
            except Exception as e:
                logging.error(f"Error reading match file {file}: {e}")
                continue

        for i, player_id in enumerate(player_ids):
            logging.info(f"Updating player {i+1}/{len(player_ids)}: {player_id}")
            try:
                collector.fetch_player_history(player_id)
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error updating stats for player {player_id}: {e}")
                continue

        logging.info(f"Updated stats for {len(player_ids)} players")
    except Exception as e:
        logging.error(f"Error in update_player_stats: {e}")
        raise

def train_model_with_ensemble():
    logging.info("Training ensemble model...")
    try:
        df = pd.read_csv(DATASET_CSV)

        ensemble = EnsemblePredictor()

        X = df.drop(columns=['winner', 'match_id', 'team1', 'team2', 'label', 'team1_score', 'team2_score'])
        y = df['winner']

        ensemble.train(X, y, tune_hyperparams=True)

        ensemble.save(MODEL_PATH, FEATURES_PATH)

        logging.info("Ensemble model trained and saved")
    except Exception as e:
        logging.error(f"Error in train_model_with_ensemble: {e}")
        raise

def get_match_file_dates():
    files = glob.glob(MATCH_FILE_PATTERN)
    dates = []
    for file in files:
        fname = os.path.basename(file)
        date_str = fname.split("_")[0]
        try:
            dates.append(datetime.strptime(date_str, MATCH_FILE_DATE_FMT))
        except ValueError:
            continue
    return sorted(dates)

def delete_old_matches(max_age_days=MAX_DAYS_BACK):
    now = datetime.utcnow()
    cutoff = now - timedelta(days=max_age_days)
    deleted = 0
    for file in glob.glob(MATCH_FILE_PATTERN):
        fname = os.path.basename(file)
        date_str = fname.split("_")[0]
        try:
            file_date = datetime.strptime(date_str, MATCH_FILE_DATE_FMT)
            if file_date < cutoff:
                os.remove(file)
                deleted += 1
        except ValueError:
            continue
    logging.info(f"Deleted {deleted} old match files.")

def get_latest_match_date():
    dates = get_match_file_dates()
    return max(dates) if dates else None

def download_new_matches():
    today = datetime.utcnow()
    latest = get_latest_match_date()
    if latest is None:
        days_back = MAX_DAYS_BACK
    else:
        delta = (today - latest).days
        if delta == 0:
            logging.info("All matches up to today are already downloaded.")
            return 0
        days_back = min(delta, MAX_DAYS_BACK)
    logging.info(f"Downloading matches for the past {days_back} days...")
    try:
        scraper = MatchScraper(output_dir=MATCH_DIR, days_back=days_back)
        matches_downloaded = scraper.fetch_and_save_matches()
        return matches_downloaded
    except Exception as e:
        logging.error(f"Error downloading matches: {e}")
        return 0

def create_dataset():
    logging.info("Creating new dataset from match files...")
    try:
        processor = MatchProcessor(MATCH_DIR, exclude_recent_days=0)
        train_df = processor.process_all_matches()
        train_df.to_csv(DATASET_CSV, index=False)
        logging.info(f"Saved {len(train_df)} training matches to {DATASET_CSV}")
        test_df = processor.get_test_matches()
        test_df.to_csv("./dataset/test_matches.csv", index=False)
        logging.info(f"Saved {len(test_df)} test matches to test_matches.csv")
    except Exception as e:
        logging.error(f"Error creating dataset: {e}")
        raise

def main():
    try:
        logging.info("Starting training pipeline...")
        setup_directories()
        delete_old_matches(max_age_days=MAX_DAYS_BACK)
        new_data = download_new_matches()

        if new_data and new_data > 0:
            logging.info("New matches found, updating player statistics...")
            update_player_stats()
            create_dataset()
            train_model_with_ensemble()
        elif not os.path.exists(MODEL_PATH):
            logging.info("No model found, training initial model...")
            update_player_stats()
            create_dataset()
            train_model_with_ensemble()
        else:
            logging.info("No new matches, using existing model.")

        logging.info("Pipeline completed successfully")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
