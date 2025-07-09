import os
import json
import requests
import time
import logging
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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