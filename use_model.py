import requests
import statistics
import pandas as pd
import joblib
from datetime import datetime, timedelta
import logging

MODEL_PATH = "./model/model.joblib"
FEATURES_PATH = "./model/feature_columns.joblib"

class TeamStatsFetcher:
    def __init__(self, team_slug):
        self.team_slug = team_slug
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.base_url = "https://api.bo3.gg/api/v1"
        today = datetime.today()
        six_months_ago = today - timedelta(days=180)
        self.date_from = six_months_ago.strftime('%Y-%m-%d')
        self.date_to = today.strftime('%Y-%m-%d')
        self.player_stats = []

    def fetch(self):
        team_data = self._get_json(f"/teams/{self.team_slug}?prefer_locale=en")
        players = [p for p in team_data.get("players", []) if not p.get("is_coach") and p.get("status") != 2]

        elo_ratings = []
        for player in players:
            slug = player.get("slug")
            rating = player.get("six_month_avg_rating") or 0
            stats = self._get_json(
                f"/players/{slug}/general_stats?filter%5Bstart_date_to%5D={self.date_to}&filter%5Bstart_date_from%5D={self.date_from}"
            )
            if not isinstance(stats, dict):
                stats = {}
            rounds = stats.get("rounds_count", 0)
            adr = stats.get("damage_sum", 0) / rounds if rounds else 0
            kills = stats.get("kills_sum", 0)
            deaths = stats.get("deaths_sum", 0)
            kd = kills / deaths if deaths else (1.0 if kills > 0 else 0)
            opening_sr = self._get_opening_kill_sr(slug)

            elo_rating = 1500
            elo_ratings.append(elo_rating)

            self.player_stats.append({
                "rating": rating,
                "adr": adr,
                "kd_ratio": kd,
                "opening_kill_success_rate": opening_sr,
                "elo_rating": elo_rating
            })

        elo_stats = self._calculate_elo_stats(elo_ratings)
        return {**self._aggregate_stats(), **elo_stats}

    def _calculate_elo_stats(self, elo_ratings):
        if not elo_ratings:
            return {
                "avg_elo": 1500,
                "elo_std": 0,
                "top2_elo": 1500,
                "elo_diff": 0
            }
        avg_elo = sum(elo_ratings) / len(elo_ratings)
        elo_std = statistics.stdev(elo_ratings) if len(elo_ratings) > 1 else 0
        top2_elo = sum(sorted(elo_ratings, reverse=True)[:2]) / 2 if len(elo_ratings) >= 2 else avg_elo
        return {
            "avg_elo": avg_elo,
            "elo_std": elo_std,
            "top2_elo": top2_elo,
            "elo_diff": max(elo_ratings) - min(elo_ratings)
        }

    def _aggregate_stats(self):
        if not self.player_stats:
            return {
                "avg_rating": 0,
                "top2_rating": 0,
                "rating_std": 0,
                "avg_adr": 0,
                "avg_kd_ratio": 0,
                "avg_opening_kill_sr": 0,
            }

        ratings = [p["rating"] for p in self.player_stats]
        top2_rating = 0
        if len(ratings) >= 2:
            top2_rating = sum(sorted(ratings, reverse=True)[:2]) / 2
        elif len(ratings) == 1:
            top2_rating = ratings[0]
        try:
            rating_std = statistics.stdev(ratings) if len(ratings) > 1 else 0
        except statistics.StatisticsError:
            rating_std = 0
        return {
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "top2_rating": top2_rating,
            "rating_std": rating_std,
            "avg_adr": sum(p["adr"] for p in self.player_stats) / len(self.player_stats) if self.player_stats else 0,
            "avg_kd_ratio": sum(p["kd_ratio"] for p in self.player_stats) / len(self.player_stats) if self.player_stats else 0,
            "avg_opening_kill_sr": sum(p["opening_kill_success_rate"] for p in self.player_stats) / len(self.player_stats) if self.player_stats else 0,
        }

    def _get_opening_kill_sr(self, slug):
        matches = self._get_json(f"/players/{slug}/matches_list_stats/general")
        rates = []
        for match in matches:
            for stat in match.get("stats", []):
                fk, fd = stat.get("first_kills", 0), stat.get("first_death", 0)
                total = fk + fd
                if total:
                    rates.append(fk / total)
        return sum(rates) / len(rates) if rates else 0

    def _get_json(self, endpoint):
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers)
            return response.json() if response.status_code == 200 else {}
        except Exception:
            return {}

class MapEncoder:
    MAPS = ["de_nuke", "de_inferno", "de_ancient", "de_dust2", "de_mirage", "de_train", "de_anubis"]

    @staticmethod
    def encode(selected_maps):
        return {f"map_{m}": int(m in selected_maps) for m in MapEncoder.MAPS}

class MatchPredictor:
    def __init__(self, model_path, feature_path, use_ensemble=True):
        if use_ensemble:
            from libary.train_model import EnsemblePredictor
            self.model = EnsemblePredictor.load(model_path, feature_path)
            unwanted_features = {'slug', 'scrape_date', 'team1_has_awper', 'team1_has_igl', 'team2_has_awper', 'team2_has_igl'}
            if hasattr(self.model, 'feature_columns') and self.model.feature_columns is not None:
                self.model.feature_columns = [f for f in self.model.feature_columns if f not in unwanted_features]
        else:
            self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(feature_path)

    def predict(self, team1_stats, team2_stats, map_encoding=None):
        current_date = datetime.now()

        row = {
            **{f"team1_{k}": v for k, v in team1_stats.items()},
            **{f"team2_{k}": v for k, v in team2_stats.items()},
            "year": current_date.year,
            "month": current_date.month,
            "day": current_date.day,
            "bo_type": 3
        }

        unwanted_features = {'team1_has_awper', 'team1_has_igl', 'team2_has_awper', 'team2_has_igl'}
        for feature in unwanted_features:
            if feature in row:
                del row[feature]

        if map_encoding:
            row.update(map_encoding)
        else:
            for map_name in MapEncoder.MAPS:
                row[f"map_{map_name}"] = 0

        df = pd.DataFrame([row])

        if hasattr(self.model, 'feature_columns') and self.model.feature_columns is not None:
            for feature in self.model.feature_columns:
                if feature not in df.columns:
                    df[feature] = 0
            df = df[self.model.feature_columns]
        else:
            df = df.reindex(columns=self.feature_columns)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)

        pred = self.model.predict(df)[0]
        proba = self.model.predict_proba(df)[0]
        return pred, proba

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict a CS2 match outcome.")
    parser.add_argument("team1", help="Slug of first team (e.g., natus-vincere)")
    parser.add_argument("team2", help="Slug of second team (e.g., g2-esports)")
    parser.add_argument("--maps", nargs="*", default=[], help="List of map slugs (e.g., de_inferno de_nuke)")
    args = parser.parse_args()

    fetcher1 = TeamStatsFetcher(args.team1)
    fetcher2 = TeamStatsFetcher(args.team2)
    team1_stats = fetcher1.fetch()
    team2_stats = fetcher2.fetch()

    map_encoding = MapEncoder.encode(args.maps)
    predictor = MatchPredictor(MODEL_PATH, FEATURES_PATH)
    _, proba = predictor.predict(team1_stats, team2_stats, map_encoding)
    print(f"{args.team1} win probability: {proba[1]:.4f}")
    print(f"{args.team2} win probability: {proba[0]:.4f}")

if __name__ == "__main__":
    main()
