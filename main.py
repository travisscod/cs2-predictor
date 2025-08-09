import os
import glob
import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

import catboost as cb
import lightgbm as lgb
import xgboost as xgb


class MatchScraper:
    def __init__(self, output_dir="matches", days_back=1):
        self.output_dir = output_dir
        self.days_back = days_back
        self.base_url = "https://api.bo3.gg/api/v2/matches/finished"
        self.match_detail_url = "https://api.bo3.gg/api/v1/matches/{slug}?scope=show-match&with=games,streams,teams,tournament_deep,stage"
        self.player_stats_url = "https://api.bo3.gg/api/v1/games/{map_id}/players_stats"
        
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        
        os.makedirs(self.output_dir, exist_ok=True)

    def _make_request(self, url, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                    raise
                logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                time.sleep(2 ** attempt)
        return None

    def fetch_and_parse_player_stats(self, map_id):
        url = self.player_stats_url.format(map_id=map_id)
        try:
            response = self._make_request(url)
            if not response:
                logging.error(f"Empty response for map {map_id}")
                return []

            if isinstance(response, list):
                players = response
            elif isinstance(response, dict):
                players = response.get("data", [])
                if not players:
                    players = response.get("players", [])
            else:
                logging.error(f"Unexpected response type for map {map_id}: {type(response)}")
                return []

            if not isinstance(players, list):
                logging.error(f"Players data is not a list for map {map_id}")
                return []

            player_stats_list = []
            for p in players:
                if not isinstance(p, dict):
                    logging.warning(f"Skipping invalid player data: {p}")
                    continue

                try:
                    kills = p.get("kills", 0)
                    deaths = p.get("death", 0)
                    assists = p.get("assists", 0)
                    headshots = p.get("headshots", 0)
                    first_kills = p.get("first_kills", 0)
                    first_death = p.get("first_death", 0)
                    trade_kills = p.get("trade_kills", 0)
                    trade_death = p.get("trade_death", 0)
                    
                    multikills = p.get("multikills", {})
                    if not isinstance(multikills, dict):
                        multikills = {}
                    mk2 = multikills.get("2", 0)
                    mk3 = multikills.get("3", 0)
                    mk4 = multikills.get("4", 0)
                    mk5 = multikills.get("5", 0)

                    kd_ratio = kills / deaths if deaths > 0 else 0
                    headshot_percentage = (headshots / kills) if kills > 0 else 0
                    opening_duels = first_kills + first_death
                    opening_kill_success_rate = (first_kills / opening_duels) if opening_duels > 0 else 0
                    traded_death_rate = (trade_death / deaths) if deaths > 0 else 0
                    multikill_score = mk2 * 2 + mk3 * 3 + mk4 * 4 + mk5 * 5

                    money_spent = p.get("money_spent", 0)
                    money_save = p.get("money_save", 0)
                    utility_value = p.get("utility_value", 0)
                    total_equipment_value = p.get("total_equipment_value", 0)
                    additional_value = p.get("additional_value", 0)

                    clutches = p.get("clutches", {})
                    if not isinstance(clutches, dict):
                        clutches = {}
                    clutch_successes = clutches.get("success", 0)
                    clutch_attempts = clutches.get("attempts", 1)
                    clutch_success_rate = (clutch_successes / clutch_attempts) if clutch_attempts > 0 else 0

                    entry_kills = p.get("entry_kills", 0)
                    entry_deaths = p.get("entry_deaths", 1)
                    entry_kill_ratio = (entry_kills / entry_deaths) if entry_deaths > 0 else 0

                    plants = p.get("plants", 0)
                    plant_attempts = p.get("plant_attempts", 1)
                    plant_success_rate = (plants / plant_attempts) if plant_attempts > 0 else 0

                    steam_profile = p.get("steam_profile", {})
                    if not isinstance(steam_profile, dict):
                        steam_profile = {}
                    steam_player = steam_profile.get("player", {})
                    if not isinstance(steam_player, dict):
                        steam_player = {}

                    team_clan = p.get("team_clan", {})
                    if not isinstance(team_clan, dict):
                        team_clan = {}

                    record = {
                        "id": p.get("id"),
                        "game_id": p.get("game_id"),
                        "steam_profile_id": p.get("steam_profile_id"),
                        "steam_profile_player_id": steam_profile.get("player_id"),
                        "steam_profile_nickname": steam_profile.get("nickname"),
                        "steam_profile_player_slug": steam_player.get("slug"),
                        "clan_name": p.get("clan_name"),
                        "team_id": team_clan.get("team_id"),
                        "enemy_clan_name": p.get("enemy_clan_name"),
                        
                        "kills": kills,
                        "death": deaths,
                        "assists": assists,
                        "headshots": headshots,
                        "adr": p.get("adr", 0),
                        "kast": p.get("kast", 0),
                        "player_rating": p.get("player_rating", 0),
                        
                        "first_kills": first_kills,
                        "first_death": first_death,
                        "trade_kills": trade_kills,
                        "trade_death": trade_death,
                        "multikills_2k": mk2,
                        "multikills_3k": mk3,
                        "multikills_4k": mk4,
                        "multikills_5k": mk5,
                        "multikill_score": multikill_score,
                        "opening_duels": opening_duels,
                        "opening_kill_success_rate": opening_kill_success_rate,
                        "traded_death_rate": traded_death_rate,
                        
                        "money_spent": money_spent,
                        "money_save": money_save,
                        "utility_value": utility_value,
                        "total_equipment_value": total_equipment_value,
                        "additional_value": additional_value,
                        
                        "clutch_successes": clutch_successes,
                        "clutch_attempts": clutch_attempts,
                        "clutch_success_rate": clutch_success_rate,
                        
                        "entry_kills": entry_kills,
                        "entry_deaths": entry_deaths,
                        "entry_kill_ratio": entry_kill_ratio,
                        
                        "plants": plants,
                        "plant_attempts": plant_attempts,
                        "plant_success_rate": plant_success_rate,
                        
                        "kd_ratio": kd_ratio,
                        "headshot_percentage": headshot_percentage,
                        "is_awper": p.get("is_awper", False),
                        "is_igl": p.get("is_igl", False)
                    }
                    player_stats_list.append(record)
                except Exception as e:
                    logging.error(f"Error processing player stats: {e}")
                    continue

            return player_stats_list
        except Exception as e:
            logging.error(f"Error fetching player stats: {e}")
            return []

    def fetch_and_save_matches(self):
        today = datetime.utcnow()
        dates = [today - timedelta(days=i) for i in range(self.days_back)]
        total_matches = 0

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            logging.info(f"Fetching matches for {date_str}...")

            url = (
                f"{self.base_url}?date={date_str}"
                "&utc_offset=3600"
                "&filter%5Btier%5D%5Bin%5D=s,a,b"
                "&filter%5Bdiscipline_id%5D%5Beq%5D=1"
            )

            try:
                data = self._make_request(url)
                if not data:
                    continue

                matches = data.get("data", {}).get("tiers", {}).get("high_tier", {}).get("matches", [])
                teams = data.get("included", {}).get("teams", {})
                tournaments = data.get("included", {}).get("tournaments", {})

                for match in matches:
                    try:
                        team1_id = str(match["team1_id"])
                        team2_id = str(match["team2_id"])
                        team1_name = teams.get(team1_id, {}).get("name", f"team_{team1_id}")
                        team2_name = teams.get(team2_id, {}).get("name", f"team_{team2_id}")
                        tournament_id = str(match["tournament"])
                        tournament_name = tournaments.get(tournament_id, {}).get("name", f"tournament_{tournament_id}")

                        team1_score = match.get("team1_score")
                        team2_score = match.get("team2_score")
                        if team1_score is not None and team2_score is not None:
                            if team1_score > team2_score:
                                winner_team = team1_name
                            elif team2_score > team1_score:
                                winner_team = team2_name
                            else:
                                winner_team = "draw"
                        else:
                            winner_team = None

                        slug = match.get("slug")
                        extended_games = []
                        if slug:
                            try:
                                detail_url = self.match_detail_url.format(slug=slug)
                                detail_data = self._make_request(detail_url)
                                if detail_data:
                                    extended_game_dict = {}
                                    for ext_game in detail_data.get("games", []):
                                        key = ext_game.get("number")
                                        extended_game_dict[key] = ext_game

                                    for game in match.get("games", []):
                                        number = game.get("number")
                                        ext_game = extended_game_dict.get(number, {})
                                        map_id = ext_game.get("id")
                                        merged_game = {
                                            "number": game.get("number"),
                                            "map_name": game.get("map_name"),
                                            "map_id": map_id,
                                            "rounds_count": ext_game.get("rounds_count"),
                                            "winner_team_id": ext_game.get("winner_team_clan", {}).get("team_id"),
                                            "winner_team_name": ext_game.get("winner_team_clan", {}).get("name"),
                                            "loser_team_id": ext_game.get("loser_team_clan", {}).get("team_id"),
                                            "loser_team_name": ext_game.get("loser_team_clan", {}).get("name"),
                                            "winner_score": ext_game.get("winner_clan_score"),
                                            "loser_score": ext_game.get("loser_clan_score"),
                                            "begin_at": ext_game.get("begin_at"),
                                            "status": ext_game.get("status"),
                                            "state": ext_game.get("state"),
                                        }
                                        merged_game = {k: v for k, v in merged_game.items() if v is not None}

                                        if map_id:
                                            player_stats = self.fetch_and_parse_player_stats(map_id)
                                            merged_game["player_stats"] = player_stats
                                        else:
                                            merged_game["player_stats"] = []

                                        extended_games.append(merged_game)
                            except Exception as e:
                                logging.error(f"Error fetching extended data for {slug}: {e}")
                                extended_games = [
                                    {"number": game.get("number"), "map_name": game.get("map_name")}
                                    for game in match.get("games", [])
                                ]
                        else:
                            extended_games = [
                                {"number": game.get("number"), "map_name": game.get("map_name")}
                                for game in match.get("games", [])
                            ]

                        filtered_match = {
                            "match_id": match["id"],
                            "slug": match.get("slug"),
                            "bo_type": match.get("bo_type"),
                            "team1_id": match.get("team1_id"),
                            "team2_id": match.get("team2_id"),
                            "team1_name": team1_name,
                            "team2_name": team2_name,
                            "team1_score": team1_score,
                            "team2_score": team2_score,
                            "winner_team": winner_team,
                            "games": extended_games,
                            "tournament_id": tournament_id,
                            "tournament_name": tournament_name,
                            "scrape_date": date_str
                        }

                        filename = f"{date_str}_{team1_name.lower().replace(' ', '-')}_vs_{team2_name.lower().replace(' ', '-')}.json"
                        filepath = os.path.join(self.output_dir, filename)

                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(filtered_match, f, indent=2, ensure_ascii=False)
                            
                        total_matches += 1

                    except Exception as e:
                        logging.error(f"Error processing match {match.get('id')}: {e}")
                        continue

                logging.info(f"{len(matches)} matches saved for {date_str}")

            except Exception as e:
                logging.error(f"Error processing date {date_str}: {e}")
                continue

            time.sleep(1.2)

        logging.info("Done scraping match data.")
        return total_matches

class MatchProcessor:
    def __init__(self, input_dir, exclude_recent_days=0):
        self.input_dir = input_dir
        self.exclude_recent_days = exclude_recent_days
        self.cutoff_date = datetime.utcnow() - timedelta(days=exclude_recent_days)
        self.test_matches = []


    def extract_team_features(self, players):
        def safe_mean(values):
            return mean(values) if values else 0

        ratings = [p.get("player_rating") for p in players if p.get("player_rating") is not None]
        hs_percents = [p.get("headshot_percent", 0) for p in players if p.get("headshot_percent") is not None]
        kd_ratios = [p.get("kd_ratio") for p in players if p.get("kd_ratio") is not None]
        adrs = [p.get("adr") for p in players if p.get("adr") is not None]
        
        clutch_successes = [p.get("clutch_successes", 0) for p in players]
        clutch_attempts = [p.get("clutch_attempts", 1) for p in players]
        clutch_rate = sum(clutch_successes) / sum(clutch_attempts) if sum(clutch_attempts) > 0 else 0
        
        entry_kills = [p.get("entry_kills", 0) for p in players]
        entry_deaths = [p.get("entry_deaths", 1) for p in players]
        entry_kill_ratio = sum(entry_kills) / sum(entry_deaths) if sum(entry_deaths) > 0 else 0
        
        first_kills = [p.get("first_kills", 0) for p in players]
        first_deaths = [p.get("first_deaths", 1) for p in players]
        opening_duel_win_rate = sum(first_kills) / (sum(first_kills) + sum(first_deaths)) if (sum(first_kills) + sum(first_deaths)) > 0 else 0
        
        multikill_rounds = [p.get("multi_kill_rounds", 0) for p in players]
        multikill_score = sum(multikill_rounds) / len(players) if players else 0
        
        money_spent = [p.get("money_spent", 0) for p in players]
        utility_value = [p.get("utility_value", 0) for p in players]
        avg_money_spent = safe_mean(money_spent)
        avg_utility_value = safe_mean(utility_value)
        
        plants = [p.get("plants", 0) for p in players]
        plant_attempts = [p.get("plant_attempts", 1) for p in players]
        plant_success_rate = sum(plants) / sum(plant_attempts) if sum(plant_attempts) > 0 else 0
        
        top2_rating = safe_mean(sorted(ratings, reverse=True)[:2])
        bottom2_rating = safe_mean(sorted(ratings)[:2])
        rating_spread = top2_rating - bottom2_rating if ratings else 0
        
        avg_rating = safe_mean(ratings)
        avg_hs_percent = safe_mean(hs_percents)
        avg_kd_ratio = safe_mean(kd_ratios)
        avg_adr = safe_mean(adrs)
        
        return {
            "avg_rating": avg_rating,
            "avg_hs_percent": avg_hs_percent,
            "avg_kd_ratio": avg_kd_ratio,
            "avg_adr": avg_adr,
            "clutch_rate": clutch_rate,
            "total_clutch_attempts": sum(clutch_attempts),
            "entry_kill_ratio": entry_kill_ratio,
            "total_entry_kills": sum(entry_kills),
            "opening_duel_win_rate": opening_duel_win_rate,
            "total_first_kills": sum(first_kills),
            "multikill_score": multikill_score,
            "total_multikill_rounds": sum(multikill_rounds),
            "avg_money_spent": avg_money_spent,
            "avg_utility_value": avg_utility_value,
            "plant_success_rate": plant_success_rate,
            "total_plants": sum(plants),
            "top2_rating": top2_rating,
            "bottom2_rating": bottom2_rating,
            "rating_spread": rating_spread,
            "team_size": len(players),
            "has_awper": any(p.get("is_awper", False) for p in players),
            "has_igl": any(p.get("is_igl", False) for p in players)
        }

    def process_match_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        match_id = data.get("id") or data.get("match_id")
        if not match_id:
            return None

        team1_id, team2_id = data.get("team1_id"), data.get("team2_id")
        team1_players, team2_players, map_names = [], [], []

        for game in data.get("games", []):
            map_name = game.get("map_name")
            if map_name:
                map_names.append(map_name)
            for p in game.get("player_stats", []):
                if p["team_id"] == team1_id:
                    team1_players.append(p)
                elif p["team_id"] == team2_id:
                    team2_players.append(p)

        if not team1_players or not team2_players or not map_names:
            return None

        team1_feats = self.extract_team_features(team1_players)
        team2_feats = self.extract_team_features(team2_players)

        team1_score = data.get("team1_score")
        team2_score = data.get("team2_score")
        if team1_score is None or team2_score is None:
            return None

        winner = int(team1_score > team2_score)

        row = {
            "match_id": match_id,
            "team1": data.get("team1_name"),
            "team2": data.get("team2_name"),
            "bo_type": data.get("bo_type"),
            "map_list": map_names,
            "team1_score": team1_score,
            "team2_score": team2_score,
            "label": f"{team1_score}-{team2_score}",
            "winner": winner,
            "scrape_date": data.get("scrape_date"),
            "slug": data.get("slug")
        }
        row.update({f"team1_{k}": v for k, v in team1_feats.items()})
        row.update({f"team2_{k}": v for k, v in team2_feats.items()})
        return row

    def process_all_matches(self):
        files = glob.glob(os.path.join(self.input_dir, "*.json"))
        training_rows, test_rows = [], []

        for file_path in files:
            try:
                fname = os.path.basename(file_path)
                date_str = fname.split("_")[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                row = self.process_match_file(file_path)
                if row and row.get("map_list"):
                    if file_date >= self.cutoff_date:
                        test_rows.append(row)
                    else:
                        training_rows.append(row)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        self.test_matches = test_rows
        df = pd.DataFrame(training_rows)
        if df.empty:
            return df

        df = df[df["map_list"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
        if df.empty:
            return df

        mlb = MultiLabelBinarizer()
        map_df = pd.DataFrame(mlb.fit_transform(df["map_list"]), columns=[f"map_{m}" for m in mlb.classes_])
        return pd.concat([df.drop(columns=["map_list"]), map_df], axis=1)

    def get_test_matches(self):
        df = pd.DataFrame(self.test_matches)
        if df.empty:
            return df

        mlb = MultiLabelBinarizer()
        map_df = pd.DataFrame(mlb.fit_transform(df["map_list"]), columns=[f"map_{m}" for m in mlb.classes_])
        return pd.concat([df.drop(columns=["map_list"]), map_df], axis=1)

class EnsemblePredictor:
    def __init__(self):
        self.models = {
            "xgboost": CalibratedClassifierCV(
                xgb.XGBClassifier(
                    objective='binary:logistic',
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                ),
                method='isotonic',
                cv=5
            ),
            "lightgbm": CalibratedClassifierCV(
                lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                method='isotonic',
                cv=5
            ),
            "catboost": CalibratedClassifierCV(
                cb.CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.1,
                    depth=5,
                    subsample=0.8,
                    colsample_bylevel=0.8,
                    random_seed=42,
                    verbose=False
                ),
                method='isotonic',
                cv=5
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                random_state=42
            )
        }
        
        self.ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft',
            weights=[1.2, 1.1, 1.1, 1.0]
        )
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.held_out_matches = None
        self.feature_importance = None
        
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        try:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss'
            )
            
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='neg_log_loss',
                cv=3,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            
            logging.info("Best hyperparameters:")
            for param, value in best_params.items():
                logging.info(f"  {param}: {value}")
            
            self.models["xgboost"] = CalibratedClassifierCV(
                xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss',
                    **best_params
                ),
                method='isotonic',
                cv=5
            )
            
            self.ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in self.models.items()],
                voting='soft'
            )
            
            return best_params
            
        except Exception as e:
            logging.error(f"Error tuning hyperparameters: {e}")
            raise
        
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        interaction_features = pd.DataFrame(index=X.index)
        
        for stat in ['avg_rating', 'top2_rating', 'avg_adr', 'avg_kd_ratio']:
            team1_stat = f'team1_{stat}'
            team2_stat = f'team2_{stat}'
            if team1_stat in X.columns and team2_stat in X.columns:
                interaction_features[f'{stat}_ratio'] = X[team1_stat] / (X[team2_stat] + 1e-6)
                interaction_features[f'{stat}_diff'] = X[team1_stat] - X[team2_stat]
                interaction_features[f'{stat}_product'] = X[team1_stat] * X[team2_stat]
        
        for stat in ['avg_rating', 'top2_rating']:
            team1_stat = f'team1_{stat}'
            team2_stat = f'team2_{stat}'
            if team1_stat in X.columns and team2_stat in X.columns:
                interaction_features[f'{team1_stat}_squared'] = X[team1_stat] ** 2
                interaction_features[f'{team2_stat}_squared'] = X[team2_stat] ** 2
        
        return pd.concat([X, interaction_features], axis=1)
        
    def _preprocess_features(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if 'scrape_date' in X.columns:
            X['scrape_date'] = pd.to_datetime(X['scrape_date'])
            X['year'] = X['scrape_date'].dt.year
            X['month'] = X['scrape_date'].dt.month
            X['day'] = X['scrape_date'].dt.day
            X = X.drop(columns=['scrape_date'])
        
        if 'bo_type' not in X.columns:
            X['bo_type'] = 3
        X['bo_type'] = pd.to_numeric(X['bo_type'], errors='coerce').fillna(3)
        
        X = self._create_interaction_features(X)
        
        X_numeric = X.select_dtypes(include=['number'])
        
        if is_training:
            self.feature_columns = X_numeric.columns.tolist()
        
        if hasattr(self.scaler, 'mean_'):
            missing_cols = set(self.feature_columns) - set(X_numeric.columns)
            if missing_cols:
                for col in missing_cols:
                    X_numeric[col] = 0
            
            X_numeric = X_numeric.reindex(columns=self.feature_columns, fill_value=0)
            
            if hasattr(self.scaler, 'mean_'):
                X_numeric = pd.DataFrame(
                    self.scaler.transform(X_numeric),
                    columns=X_numeric.columns,
                    index=X_numeric.index
                )
            else:
                X_numeric = pd.DataFrame(
                    self.scaler.fit_transform(X_numeric),
                    columns=X_numeric.columns,
                    index=X_numeric.index
                )
        
        return X_numeric
        
    def train(self, X: pd.DataFrame, y: pd.Series, tune_hyperparams: bool = False) -> None:
        try:
            X_processed = self._preprocess_features(X, is_training=True)
            
            if len(X_processed) > 20:
                self.held_out_matches = {
                    'X': X_processed.tail(20),
                    'y': y.tail(20)
                }
                X_processed = X_processed.iloc[:-20]
                y = y.iloc[:-20]
            
            if tune_hyperparams:
                self.tune_hyperparameters(X_processed, y)
            
            for name, model in self.models.items():
                logging.info(f"Training {name} model...")
                model.fit(X_processed, y)
                if not hasattr(model, 'classes_'):
                    raise ValueError(f"{name} model was not properly fitted")
            
            logging.info("Training ensemble...")
            self.ensemble.fit(X_processed, y)
            
            self._calculate_feature_importance(X_processed, y)
            
            self._evaluate_model(X_processed, y)
            
            if self.held_out_matches is not None:
                logging.info("\nEvaluating on held-out data:")
                self._evaluate_model(
                    self.held_out_matches['X'],
                    self.held_out_matches['y']
                )
            
            for name, model in self.models.items():
                if not hasattr(model, 'classes_'):
                    raise ValueError(f"{name} model was not properly fitted after training")
            
            logging.info("All models successfully trained and fitted")
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise
        
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            xgb_importance = self.models['xgboost'].base_estimator.feature_importances_
            
            rf_importance = self.models['random_forest'].feature_importances_
            
            avg_importance = (xgb_importance + rf_importance) / 2
            
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            
            logging.info("\nTop 10 most important features:")
            for _, row in self.feature_importance.head(10).iterrows():
                logging.info(f"  {row['feature']}: {row['importance']:.4f}")
                
        except Exception as e:
            logging.error(f"Error calculating feature importance: {e}")
        
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            cv_scores = cross_val_score(
                self.ensemble,
                X,
                y,
                cv=5,
                scoring='neg_log_loss'
            )
            logging.info(f"Cross-validation log loss scores: {cv_scores}")
            logging.info(f"Mean CV log loss: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            y_pred = self.ensemble.predict(X)
            y_prob = self.ensemble.predict_proba(X)[:, 1]
            
            accuracy = accuracy_score(y, y_pred)
            brier = brier_score_loss(y, y_prob)
            logloss = log_loss(y, y_prob)
            roc_auc = roc_auc_score(y, y_prob)
            
            logging.info("Training set metrics:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  Brier score: {brier:.4f}")
            logging.info(f"  Log loss: {logloss:.4f}")
            logging.info(f"  ROC AUC: {roc_auc:.4f}")
            
            sample = pd.DataFrame({
                "winner_true": y.values,
                "winner_pred": y_pred,
                "prob_team1_win": y_prob
            })
            logging.info(f"\nSample predictions:\n{sample.head()}")
            
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'feature_columns') or self.feature_columns is None:
            raise NotFittedError("Model has not been trained yet. Call train() first.")
            
        try:
            X_processed = self._preprocess_features(X, is_training=False)
            return self.ensemble.predict(X_processed)
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'feature_columns') or self.feature_columns is None:
            raise NotFittedError("Model has not been trained yet. Call train() first.")
            
        try:
            X_processed = self._preprocess_features(X, is_training=False)
            
            model_predictions = []
            for name, model in self.models.items():
                pred = model.predict_proba(X_processed)
                model_predictions.append(pred)
            
            weights = [1.2, 1.1, 1.1, 1.0]
            weighted_pred = np.average(model_predictions, weights=weights, axis=0)
            
            temperature = 1.2
            scaled_pred = weighted_pred ** (1/temperature)
            scaled_pred = scaled_pred / scaled_pred.sum(axis=1, keepdims=True)
            
            return scaled_pred
            
        except Exception as e:
            logging.error(f"Error making probability predictions: {e}")
            raise
        
    def save(self, model_path: str, feature_columns_path: str) -> None:
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            for name, model in self.models.items():
                model_path_individual = str(Path(model_path).parent / f"{name}_model.joblib")
                joblib.dump(model, model_path_individual, compress=3)
            
            joblib.dump(self.ensemble, model_path, compress=3)
            joblib.dump(self.feature_columns, feature_columns_path)
            
            scaler_path = str(Path(model_path).parent / "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            logging.info(f"Model saved to {model_path}")
            logging.info(f"Feature columns saved to {feature_columns_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise
        
    @classmethod
    def load(cls, model_path: str, feature_columns_path: str) -> 'EnsemblePredictor':
        try:
            model = cls()
            
            for name in model.models.keys():
                model_path_individual = str(Path(model_path).parent / f"{name}_model.joblib")
                model.models[name] = joblib.load(model_path_individual)
            
            model.ensemble = joblib.load(model_path)
            model.feature_columns = joblib.load(feature_columns_path)
            
            scaler_path = str(Path(model_path).parent / "scaler.joblib")
            model.scaler = joblib.load(scaler_path)
            
            logging.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

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
            logging.info("New matches found, creating dataset and retraining model...")
            create_dataset()
            train_model_with_ensemble()
        elif not os.path.exists(MODEL_PATH):
            logging.info("No model found, creating dataset and training initial model...")
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
