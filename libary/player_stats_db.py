import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

class PlayerStatsCollector:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.base_url = "https://api.bo3.gg/api/v1"
        self.db_path = "player_stats.db"
        self.elo_system = None
        self._init_db()
        
    def _init_db(self):
        try:
            db_exists = os.path.exists(self.db_path)
            
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            if not db_exists:
                logger.info(f"Creating new database file: {self.db_path}")
                
                self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS player_stats (
                        player_slug TEXT PRIMARY KEY,
                        player_id TEXT,
                        rating REAL,
                        maps_played INTEGER,
                        rounds_played INTEGER,
                        adr REAL,
                        kd_ratio REAL,
                        last_updated TEXT,
                        elo_rating REAL DEFAULT 1500,
                        last_match_date TEXT
                    )
                ''')
                
                self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS player_map_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        player_slug TEXT,
                        map_name TEXT,
                        maps_played INTEGER,
                        rating REAL,
                        rating_value REAL,
                        average_kills REAL,
                        average_damage REAL,
                        FOREIGN KEY (player_slug) REFERENCES player_stats(player_slug)
                    )
                ''')
                
                self.conn.commit()
                logger.info("Database tables created successfully")
            else:
                logger.info(f"Using existing database: {self.db_path}")
                
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def update_player_elo_from_match(self, match_data):
        if not match_data or "games" not in match_data:
            return

        try:
            match_date = match_data.get("start_date")
            if not match_date:
                return

            team1_id = match_data.get("team1_id")
            team2_id = match_data.get("team2_id")
            team1_score = match_data.get("team1_score", 0)
            team2_score = match_data.get("team2_score", 0)
            
            if not team1_id or not team2_id:
                return

            if team1_score > team2_score:
                winner_team_id = team1_id
            elif team2_score > team1_score:
                winner_team_id = team2_id
            else:
                return

            team1_players = self._get_team_players(team1_id)
            team2_players = self._get_team_players(team2_id)

            if not team1_players or not team2_players:
                return

            team1_won = winner_team_id == team1_id
            rating_change = self.elo_system.update_ratings(team1_players, team2_players, team1_won, match_data)

            for player_id in team1_players + team2_players:
                current_elo = self.get_player_elo(player_id)
                new_elo = current_elo + (rating_change if player_id in team1_players else -rating_change)
                self.update_player_elo_rating(player_id, new_elo, match_date)
                    
        except Exception as e:
            logger.error(f"Error updating ELO ratings: {str(e)}")
            return None

    def _get_team_players(self, team_id):
        try:
            team_data = self._get_json(f"/teams/{team_id}")
            if not team_data or "players" not in team_data:
                return []
                
            return [player.get("id") for player in team_data["players"] if player.get("id")]
        except Exception as e:
            logger.error(f"Error getting team players: {str(e)}")
            return []

    def update_player_elo_rating(self, player_id, new_elo, match_date):
        try:
            self.cursor.execute('''
                UPDATE player_stats 
                SET elo_rating = ?, last_match_date = ?
                WHERE player_id = ?
            ''', (new_elo, match_date, player_id))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating player ELO rating: {str(e)}")
            return None

    def get_player_elo(self, player_slug):
        if str(player_slug) in self.elo_system.player_ratings:
            return self.elo_system.player_ratings[str(player_slug)]
            
        self.cursor.execute('SELECT elo_rating FROM player_stats WHERE player_slug = ?', (player_slug,))
        result = self.cursor.fetchone()
        
        if result and result[0] is not None:
            self.elo_system.player_ratings[str(player_slug)] = result[0]
            return result[0]
            
        default_elo = 1500
        self.elo_system.player_ratings[str(player_slug)] = default_elo
        return default_elo

    def fetch_player_history(self, player_id, days_back=180):
        today = datetime.today()
        start_date = today - timedelta(days=days_back)
        
        player_data = self._get_json(f"/players/{player_id}")
        if not player_data:
            return None
            
        stats = self._get_json(
            f"/players/{player_id}/general_stats?filter%5Bstart_date_to%5D={today.strftime('%Y-%m-%d')}&filter%5Bstart_date_from%5D={start_date.strftime('%Y-%m-%d')}"
        )
        
        processed_stats = self._process_player_stats(player_data, stats, player_id)
        if processed_stats:
            self._store_player_stats(player_id, processed_stats)
        
        return processed_stats

    def _store_player_stats(self, player_slug, stats):
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO player_stats 
                (player_slug, player_id, rating, maps_played, rounds_played, adr, kd_ratio, last_updated, elo_rating, last_match_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player_slug,
                stats.get('player_id'),
                stats.get('rating', 0.0),
                stats.get('maps_played', 0),
                stats.get('rounds_played', 0),
                stats.get('adr', 0.0),
                stats.get('kd_ratio', 0.0),
                stats.get('last_updated'),
                stats.get('elo_rating', 1500),
                stats.get('last_match_date')
            ))
            
            if 'map_specific_stats' in stats and 'maps' in stats['map_specific_stats']:
                for map_name, map_stats in stats['map_specific_stats']['maps'].items():
                    self.cursor.execute("""
                        INSERT OR REPLACE INTO player_map_stats 
                        (player_slug, map_name, maps_played, rating, rating_value, average_kills, average_damage)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_slug,
                        map_name,
                        map_stats.get('maps_played', 0),
                        map_stats.get('rating', 0.0),
                        map_stats.get('rating_value', 0.0),
                        map_stats.get('average_kills', 0.0),
                        map_stats.get('average_damage', 0.0)
                    ))
            
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error storing player stats: {str(e)}")
            raise

    def _get_json(self, endpoint):
        try:
            response = requests.get(f"{self.base_url}{endpoint}", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {endpoint}: {str(e)}")
            return None

    def _process_player_stats(self, player_data, stats, player_id):
        if not isinstance(stats, dict):
            return None
            
        map_specific_stats = self._get_map_specific_stats(player_id)
        
        total_maps = sum(map_data.get("maps_played", 0) for map_data in map_specific_stats.get("maps", {}).values())
            
        processed_stats = {
            "player_id": player_data.get("id"),
            "rating": player_data.get("six_month_avg_rating", 0),
            "maps_played": total_maps,
            "rounds_played": stats.get("rounds_count", 0),
            "adr": stats.get("damage_sum", 0) / stats.get("rounds_count", 1),
            "kd_ratio": stats.get("kills_sum", 0) / max(stats.get("deaths_sum", 1), 1),
            "map_specific_stats": map_specific_stats,
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }
        
        return processed_stats

    def _get_map_specific_stats(self, player_id):
        if not player_id:
            return {}
            
        today = datetime.today()
        six_months_ago = today - timedelta(days=180)
        date_to = today.strftime('%Y-%m-%d')
        date_from = six_months_ago.strftime('%Y-%m-%d')
        
        url = f"/players/{player_id}/map_stats?filter%5Bbegin_at_to%5D={date_to}&filter%5Bbegin_at_from%5D={date_from}"
        map_stats = self._get_json(url)
        
        if not map_stats or not isinstance(map_stats, list):
            return {}
        
        processed_maps = {}
        for map_data in map_stats:
            map_name = map_data.get("map_name")
            if map_name:
                processed_maps[map_name] = {
                    "maps_played": map_data.get("maps_count", 0),
                    "rating": map_data.get("avg_player_rating", 0),
                    "rating_value": map_data.get("avg_player_rating_value", 0),
                    "average_kills": map_data.get("avg_kills", 0),
                    "average_damage": map_data.get("avg_damage", 0)
                }
        
        return {
            "maps": processed_maps,
            "best_map": max(processed_maps.items(), key=lambda x: x[1]["rating"])[0] if processed_maps else None,
            "worst_map": min(processed_maps.items(), key=lambda x: x[1]["rating"])[0] if processed_maps else None,
            "avg_rating": sum(m.get("rating", 0) for m in processed_maps.values()) / len(processed_maps) if processed_maps else 0,
            "avg_kills": sum(m.get("average_kills", 0) for m in processed_maps.values()) / len(processed_maps) if processed_maps else 0
        }

    def get_player_stats(self, player_slug):
        try:
            self.cursor.execute('''
                SELECT rating, maps_played, rounds_played, adr, kd_ratio
                FROM player_stats 
                WHERE player_slug = ?
            ''', (player_slug,))
            result = self.cursor.fetchone()
            
            if result:
                return {
                    "rating": result[0],
                    "maps_played": result[1],
                    "rounds_played": result[2],
                    "adr": result[3],
                    "kd_ratio": result[4]
                }
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting player stats: {str(e)}")
            return None

