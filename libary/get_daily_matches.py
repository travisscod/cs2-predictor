import requests
from datetime import datetime, timedelta

class MatchFetcher:
    def __init__(self):
        self.url = "https://api.bo3.gg/api/v2/matches/upcoming"
        self.headers = {
            "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "referer": "https://bo3.gg/matches/current"
        }
    
    def get_upcoming_matches(self, date=None):
        if date is None:
            tomorrow = datetime.now() + timedelta(days=0)
            date = tomorrow.strftime("%Y-%m-%d")
            
        params = {
            "date": date,
            "utc_offset": "7200",
            "filter[discipline_id][eq]": "1"
        }
        
        try:
            response = requests.get(
                self.url,
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None
    
    def get_match_urls(self, date=None):
        matches = self.get_upcoming_matches(date)
        if not matches or 'data' not in matches or 'tiers' not in matches['data']:
            return []
            
        all_matches = []
        if 'high_tier' in matches['data']['tiers']:
            all_matches.extend(matches['data']['tiers']['high_tier']['matches'])
        if 'low_tier' in matches['data']['tiers']:
            all_matches.extend(matches['data']['tiers']['low_tier']['matches'])
        print(f"Found {len(all_matches)} matches for date {date}")
        return [f"https://bo3.gg/matches/{match['slug']}" for match in all_matches]

    def get_match_urls_for_range(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime.now()
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date is None:
            end_date = start_date + timedelta(days=7)
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        urls = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            urls.extend(self.get_match_urls(date_str))
            current_date += timedelta(days=1)
        return urls

    def print_match_urls_for_range(self, start_date=None, end_date=None):
        urls = self.get_match_urls_for_range(start_date, end_date)
        for url in urls:
            print(f'"{url}",')

#MatchFetcher().print_match_urls_for_range()