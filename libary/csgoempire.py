import undetected_chromedriver as uc
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta

CSGO_EMPIRE_URL = "https://csgoempire.vip/"
BETTING_URL = "match-betting?bt-path=/counter-strike-109"

class CSGOEmpire:
    def __init__(self):
        self.driver = uc.Chrome(headless=False,use_subprocess=False)
        self.is_logged_in = False

    def login(self, username, password):
        self.driver.get(CSGO_EMPIRE_URL)
        sleep(5)
        self.driver.find_element(By.XPATH, '//*[@id="empire-header"]/div[2]/div/div[3]/div[2]/div[3]/button').click()
        sleep(2)
        
        mail = self.driver.find_element(By.XPATH, '//input[@placeholder="example@email.com"]')
        mail.click()
        sleep(2)
        mail.send_keys(username)
        sleep(2)
        
        passw = self.driver.find_element(By.XPATH, '//input[@placeholder="Enter Password"]')
        passw.click()
        sleep(2)
        passw.send_keys(password)
        sleep(2)

        self.driver.find_element(By.XPATH, '//button[@type="submit"]').click()
        
        WebDriverWait(self.driver, 120).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="empire-header"]/div[2]/div/div[3]/div[2]/div[4]/button'))
        )

        print("Login successful!")
        self.is_logged_in = True

    def go_to_betting(self, page):
        self.driver.get(CSGO_EMPIRE_URL + BETTING_URL + "?upcoming_page=" + page)
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#nav-1 > a.router-link-active.link-yellow-2.link.flex.items-center'))
        )

        print("Navigated to betting page successfully!")

        

    def _get_shadow_root(self):
         shadow_root = self.driver.execute_script("""
            function getAllShadowRoots(root = document) {
                const shadowRoots = [];

                function walk(node) {
                    if (node.shadowRoot) {
                    shadowRoots.push(node.shadowRoot);
                    // Recursively walk the shadow root's children
                    walk(node.shadowRoot);
                    }

                    for (const child of node.children) {
                    walk(child);
                    }
                }

                walk(root);
                return shadowRoots;
            }

            const allShadowRoots = getAllShadowRoots();
            return allShadowRoots[0];
        """)
         return shadow_root


    def _execute_shadow_script(self, script):
        shadow_root = self._get_shadow_root()
        #print("Shadow root found:", shadow_root)
        #print(script)
        return self.driver.execute_script(script, shadow_root)


    def scrape_betting_data(self):
        matches = []
        for i in range(0, 2):
            page = str(i)
            print(f"Scraping betting data for page {page}...")
            matches.extend(self._scrape_betting_data_for_page(page))
            sleep(2)
        return matches



    def _scrape_betting_data_for_page(self, page):
        self.go_to_betting(page)
        sleep(5)
       
        upcoming_title = self._execute_shadow_script("""
            return arguments[0].querySelectorAll("[data-editor-id=\\"blockTitle\\"]")[2];
        """)
        print("Upcoming Matches Title:", upcoming_title.text)
        if not upcoming_title or upcoming_title.text != "Upcoming":
            print("Upcoming matches title not found or does not match expected text.")
            return []
        
    
        odds_container = upcoming_title.find_element(By.XPATH, "..")
        parents = odds_container.find_elements(By.CSS_SELECTOR, 'div > div:nth-child(2) > div > div:nth-child(1) > div > div')
        matches = []
        listed_matches = []

        for element in parents:
            direct_children = element.find_elements(By.XPATH, './div')  
            if direct_children:
                #print("Direct children found:", len(direct_children))
                for child in direct_children:
                    if child.text.strip():
                        matches.append(child)
            else:
                #print("No direct children found for element:", element.text)
                continue
            #matches.extend(direct_children)  

 
        listed_matches = []
        
        for match in matches:
            match_text = match.text.split("\n")
            if len(match_text) < 10:
                continue

            #print("Match text:", match_text)
            _, title, date_text, team1, team2, _, _, odds1, _, odds2 = match_text
            if "Today, " in date_text:
                time_part = date_text.replace("Today, ", "").strip()
                date_str = datetime.today().strftime('%Y-%m-%d') + " " + time_part
                full_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            elif "Tomorrow, " in date_text:
                time_part = date_text.replace("Tomorrow, ", "").strip()
                date_str = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d') + " " + time_part
                full_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            else:
                full_date = date_text 

            match_data = {
                "title": title.strip(),
                "teams": {
                    "team1": team1.strip(),
                    "team2": team2.strip()
                },
                "odds": {
                    "team1": odds1.strip(),
                    "team2": odds2.strip()
                },
                "date": full_date.strftime("%Y-%m-%d %H:%M:%S") if isinstance(full_date, datetime) else full_date
            }
            listed_matches.append(match_data)
        return listed_matches

        #print("Total matches found:", len(listed_matches))
        #for match in listed_matches:
        #    print(f"Match: {match['title']}")
        #    print(f"Teams: {match['teams']['team1']} vs {match['teams']['team2']}")
        #    print(f"Odds: {match['odds']['team1']} vs {match['odds']['team2']}")
        #    print(f"Date: {match['date']}")
        #return listed_matches
    

            #print(match.text)
        """betting_data = []

        for match in matches:
            try:
                team1 = match.find_element(By.CSS_SELECTOR, '.team-1 .team-name').text
                team2 = match.find_element(By.CSS_SELECTOR, '.team-2 .team-name').text
                odds1 = match.find_element(By.CSS_SELECTOR, '.team-1 .odds').text
                odds2 = match.find_element(By.CSS_SELECTOR, '.team-2 .odds').text
                betting_data.append({
                    'team1': team1,
                    'team2': team2,
                    'odds1': float(odds1),
                    'odds2': float(odds2)
                })
            except Exception as e:
                print(f"Error scraping match data: {e}")

        return betting_data"""



    def close(self):
        self.driver.quit()