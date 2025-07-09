import requests
import statistics
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from libary.player_stats_db import PlayerStatsCollector
from libary.csgoempire import CSGOEmpire

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
        self.stats_collector = PlayerStatsCollector()

    def fetch(self):
        team_data = self._get_json(f"/teams/{self.team_slug}?prefer_locale=en")
        players = [p for p in team_data.get("players", []) if not p.get("is_coach") and p.get("status") != 2]
        
        elo_ratings = []
        for player in players:
            slug = player.get("slug")
            elo_rating = self.stats_collector.get_player_elo(slug)
            elo_ratings.append(elo_rating)
            
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

class MatchRunner:
    def __init__(self, team1, team2, date):
        self.team1 = team1
        self.team2 = team2
        self.date = date
        self.stats_collector = PlayerStatsCollector()
        self.team1_players = []
        self.team2_players = []
        self.discord_webhook = "https://discord.com/api/webhooks/1381058355270914098/7t6ji9rQ5WhazkMiWrDVEm3UAfP023q9zIf7n7-6ykFwkrSiuFlWvdu_0YPQKkvNytpP"
        self.model = joblib.load(MODEL_PATH)
        self.feature_columns = joblib.load(FEATURES_PATH)
        
    def run(self, betting_data):
        try:
            self.team1_players = self._get_team_players(self.team1)
            self.team2_players = self._get_team_players(self.team2)
            
            if not self.team1_players or not self.team2_players:
                logging.error("Could not get team players")
                return False
                
            team1_odds = ""
            team2_odds = ""

            print(f"Running prediction for {self.team1} vs {self.team2} on {self.date}")
            #print(betting_data)
            for match in betting_data:
                #print(f"Checking match: {match['teams']['team1']} vs {match['teams']['team2']}")
                #print(f"Match date: {match['date']}")

                t_team1 = match['teams']['team1'].lower().split(" ")[0]
                t_team2 = match['teams']['team2'].lower().split(" ")[0]

                if t_team1 == self.team1.lower().split(" ")[0] and t_team2 == self.team2.lower().split(" ")[0]:
                    team1_odds = match['odds']['team1']
                    team2_odds = match['odds']['team2']
                    print(f"Found odds: {team1_odds} for {self.team1}, {team2_odds} for {self.team2}")
                    break


            team_stats = self._aggregate_stats()
            prediction = self._make_prediction(team_stats, team1_odds, team2_odds)
            self._save_prediction(prediction)
            
            return prediction
        except Exception as e:
            logging.error(f"Error running prediction: {e}")
            return False
            
    def _get_team_players(self, team_slug):
        try:
            team_data = self.stats_collector._get_json(f"/teams/{team_slug}")
            if not team_data or "players" not in team_data:
                return []
                
            return [player.get("slug") for player in team_data["players"] if player.get("slug")]
        except Exception as e:
            logging.error(f"Error getting team players: {e}")
            return []
            
    def _aggregate_stats(self):
        if not self.team1_players or not self.team2_players:
            return {}
            
        team1_stats = []
        team2_stats = []
        
        for slug in self.team1_players:
            stats = self.stats_collector.get_player_stats(slug)
            if stats:
                team1_stats.append(stats)
                
        for slug in self.team2_players:
            stats = self.stats_collector.get_player_stats(slug)
            if stats:
                team2_stats.append(stats)
        
        team1_avg = self._calculate_team_stats(team1_stats)
        team2_avg = self._calculate_team_stats(team2_stats)
        
        team1_map_stats = self._get_team_map_stats(self.team1_players)
        team2_map_stats = self._get_team_map_stats(self.team2_players)
        
        stats = {
            "team1_avg_rating": team1_avg.get("avg_rating", 0),
            "team1_top2_rating": self._get_top2_rating(team1_stats, "rating"),
            "team1_rating_std": self._get_std(team1_stats, "rating"),
            "team1_avg_adr": team1_avg.get("avg_adr", 0),
            "team1_avg_kd_ratio": team1_avg.get("avg_kd", 0),
            "team1_avg_map_rating": team1_map_stats.get("avg_rating", 0),
            "team1_best_map_rating": team1_map_stats.get("best_map_rating", 0),
            "team1_avg_kills": team1_map_stats.get("avg_kills", 0),
            
            "team2_avg_rating": team2_avg.get("avg_rating", 0),
            "team2_top2_rating": self._get_top2_rating(team2_stats, "rating"),
            "team2_rating_std": self._get_std(team2_stats, "rating"),
            "team2_avg_adr": team2_avg.get("avg_adr", 0),
            "team2_avg_kd_ratio": team2_avg.get("avg_kd", 0),
            "team2_avg_map_rating": team2_map_stats.get("avg_rating", 0),
            "team2_best_map_rating": team2_map_stats.get("best_map_rating", 0),
            "team2_avg_kills": team2_map_stats.get("avg_kills", 0),
            
            "map_de_ancient": 0,
            "map_de_anubis": 0,
            "map_de_dust2": 0,
            "map_de_inferno": 0,
            "map_de_mirage": 0,
            "map_de_nuke": 0,
            "map_de_train": 0,
            
            "year": datetime.now().year,
            "month": datetime.now().month,
            "day": datetime.now().day,
            
            "bo_type": 3,
            "scrape_date": datetime.now().strftime("%Y-%m-%d"),
            "slug": f"{self.team1}-vs-{self.team2}-{self.date}"
        }
        
        return stats
        
    def _get_top2_rating(self, stats, key):
        if not stats:
            return 0
        values = sorted([s.get(key, 0) for s in stats], reverse=True)
        if len(values) >= 2:
            return sum(values[:2]) / 2
        return values[0] if values else 0
        
    def _get_std(self, stats, key):
        if not stats:
            return 0
        values = [s.get(key, 0) for s in stats]
        if len(values) <= 1:
            return 0
        return statistics.stdev(values)
        
    def _get_team_map_stats(self, player_slugs):
        all_map_stats = []
        for slug in player_slugs:
            map_stats = self.stats_collector._get_map_specific_stats(slug)
            if map_stats and "maps" in map_stats:
                all_map_stats.extend(map_stats["maps"].values())
        
        if not all_map_stats:
            return {
                "avg_rating": 0,
                "best_map_rating": 0,
                "avg_kills": 0
            }
            
        return {
            "avg_rating": sum(m.get("rating", 0) for m in all_map_stats) / len(all_map_stats),
            "best_map_rating": max(m.get("rating", 0) for m in all_map_stats),
            "avg_kills": sum(m.get("average_kills", 0) for m in all_map_stats) / len(all_map_stats)
        }
        
    def _make_prediction(self, team_stats, team1_odds, team2_odds):
        try:
            data = pd.DataFrame([team_stats])
            
            for feature in self.feature_columns:
                if feature not in data.columns:
                    data[feature] = 0
            
            data = data[self.feature_columns]
            
            prediction = self.model.predict_proba(data)[0]
            
            prob_diff = abs(prediction[1] - prediction[0])
            
            if prob_diff > 0.3:
                confidence = "high"
            elif prob_diff > 0.15:
                confidence = "medium"
            else:
                confidence = "low"
            
            team1_prob = prediction[1]
            team2_prob = prediction[0]
            
            team1_rating = team_stats.get('team1_avg_rating', 0)
            team2_rating = team_stats.get('team2_avg_rating', 0)
            rating_diff = team1_rating - team2_rating
            
            rating_factor = 1 + (rating_diff * 0.5)
            
            if rating_diff > 0:
                team1_prob = min(0.95, team1_prob * rating_factor)
                team2_prob = max(0.05, team2_prob / rating_factor)
            else:
                team1_prob = max(0.05, team1_prob / rating_factor)
                team2_prob = min(0.95, team2_prob * rating_factor)
            
            total_prob = team1_prob + team2_prob
            team1_prob = team1_prob / total_prob
            team2_prob = team2_prob / total_prob
            
            return {
                'team1_win_probability': round(team1_prob, 4),
                'team2_win_probability': round(team2_prob, 4),
                'team1_odds': team1_odds,
                'team2_odds': team2_odds,
                'confidence': confidence,
                'team1_stats': {
                    'avg_rating': round(team_stats.get('team1_avg_rating', 0), 2),
                    'top2_rating': round(team_stats.get('team1_top2_rating', 0), 2),
                    'avg_adr': round(team_stats.get('team1_avg_adr', 0), 2),
                    'avg_kd_ratio': round(team_stats.get('team1_avg_kd_ratio', 0), 2)
                },
                'team2_stats': {
                    'avg_rating': round(team_stats.get('team2_avg_rating', 0), 2),
                    'top2_rating': round(team_stats.get('team2_top2_rating', 0), 2),
                    'avg_adr': round(team_stats.get('team2_avg_adr', 0), 2),
                    'avg_kd_ratio': round(team_stats.get('team2_avg_kd_ratio', 0), 2)
                }
            }
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None
            
    def _send_to_discord(self, prediction):
        if not prediction:
            return
            
        try:
            team1_win_prob = prediction["team1_win_probability"] * 100
            team2_win_prob = prediction["team2_win_probability"] * 100

            team1_stats = prediction["team1_stats"]
            team2_stats = prediction["team2_stats"]
            
            message = {
                "embeds": [{
                    "title": f"CS2 Match Prediction: {self.team1} vs {self.team2}",
                    "description": f"Match Date: {self.date}\nConfidence: {prediction['confidence'].upper()}",
                    "color": 0x00ff00,
                    "fields": [
                        {
                            "name": f"{self.team1} Analysis",
                            "value": (
                                f"Win Probability: {team1_win_prob:.1f}%\n"
                                f"Avg Rating: {team1_stats['avg_rating']}\n"
                                f"Top 2 Rating: {team1_stats['top2_rating']}\n"
                                f"Avg ADR: {team1_stats['avg_adr']}\n"
                                f"Avg K/D: {team1_stats['avg_kd_ratio']}"
                            ),
                            "inline": True
                        },
                        {
                            "name": f"{self.team2} Analysis",
                            "value": (
                                f"Win Probability: {team2_win_prob:.1f}%\n"
                                f"Avg Rating: {team2_stats['avg_rating']}\n"
                                f"Top 2 Rating: {team2_stats['top2_rating']}\n"
                                f"Avg ADR: {team2_stats['avg_adr']}\n"
                                f"Avg K/D: {team2_stats['avg_kd_ratio']}"
                            ),
                            "inline": True
                        }
                    ],
                    "footer": {
                        "text": "Prediction generated by CS2 Predictor"
                    }
                }]
            }
            
            response = requests.post(
                self.discord_webhook,
                json=message,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 204:
                logging.info("Prediction sent to Discord")
            else:
                logging.error(f"Failed to send prediction to Discord: {response.status_code}")
                
        except Exception as e:
            logging.error(f"Error sending prediction to Discord: {e}")
            
    def _save_prediction(self, prediction):
        if not prediction:
            return
            
        try:
            os.makedirs("./predictions", exist_ok=True)
            
            filename = f"./predictions/{self.team1}_vs_{self.team2}_{self.date}.json"
            with open(filename, 'w') as f:
                json.dump(prediction, f, indent=2)
                
            logging.info(f"Prediction saved to {filename}")
            
            self._send_to_discord(prediction)
            
            return filename
        except Exception as e:
            logging.error(f"Error saving prediction: {e}")

    def _calculate_team_stats(self, player_stats):
        if not player_stats:
            return {
                "avg_rating": 0,
                "avg_adr": 0,
                "avg_kd": 0,
                "avg_maps_played": 0
            }
            
        ratings = [s.get("rating", 0) for s in player_stats]
        adrs = [s.get("adr", 0) for s in player_stats]
        kds = [s.get("kd_ratio", 0) for s in player_stats]
        maps_played = [s.get("maps_played", 0) for s in player_stats]
        
        return {
            "avg_rating": sum(ratings) / len(ratings),
            "avg_adr": sum(adrs) / len(adrs),
            "avg_kd": sum(kds) / len(kds),
            "avg_maps_played": sum(maps_played) / len(maps_played)
        }

def predict_match(model, match_data):
    try:
        prediction = model.predict_proba(match_data)[0]
        return prediction[1], prediction[0]
    except Exception as e:
        logging.error(f"Error predicting match: {e}")
        return None, None