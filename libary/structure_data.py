import os
import json
import pandas as pd
from glob import glob
from statistics import mean, stdev
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime, timedelta


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
        files = glob(os.path.join(self.input_dir, "*.json"))
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
