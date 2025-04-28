import requests
import random
import time
from datetime import datetime
import http.cookiejar

class HumanLikeRequester:
    def __init__(self):
        # Create a persistent session
        self.session = requests.Session()
        
        # Set up cookies
        self.cookie_jar = http.cookiejar.CookieJar()
        self.session.cookies = self.cookie_jar
        
        # Track visited pages
        self.visited_pages = []
        
        # Track last request time
        self.last_request_time = datetime.now()
        
        # Define user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36'
        ]
    
    def get(self, url):
        # Wait a random time between requests
        self._wait_random_time()
        
        # Create headers
        headers = self._create_headers()
        
        # Make the request
        response = self.session.get(url, headers=headers)
        
        # Update state
        self.last_request_time = datetime.now()
        self.visited_pages.append(url)
        
        # Keep list manageable
        if len(self.visited_pages) > 10:
            self.visited_pages.pop(0)
        
        return response
    
    def _wait_random_time(self):
        # Base wait (3-7 seconds)
        wait_time = random.uniform(3, 7)
        
        # Sometimes wait longer (10% chance)
        if random.random() < 0.1:
            wait_time = random.uniform(10, 20)
        
        # Rarely wait much longer (1% chance)
        if random.random() < 0.01:
            wait_time = random.uniform(30, 60)
        
        time.sleep(wait_time)
    
    def _create_headers(self):
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Add referer if we have visited pages
        if self.visited_pages:
            headers['Referer'] = random.choice(self.visited_pages)
        
        return headers