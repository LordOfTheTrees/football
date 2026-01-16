import random
import time
import sys
from datetime import datetime
import http.cookiejar

# Try to use curl_cffi for better TLS fingerprinting (impersonates Chrome)
# Falls back to requests if not available
try:
    from curl_cffi import requests
    HAS_CURL_CFFI = True
except ImportError:
    try:
        import requests
        HAS_CURL_CFFI = False
        print("[HumanLikeRequester] Warning: curl_cffi not installed. Using standard requests library.")
        print("[HumanLikeRequester] Install with: pip install curl_cffi")
        print("[HumanLikeRequester] This may result in Cloudflare blocks due to TLS fingerprinting.")
    except ImportError:
        raise ImportError("Neither curl_cffi nor requests is installed")

class HumanLikeRequester:
    def __init__(self):
        # Create a persistent session
        if HAS_CURL_CFFI:
            # Use curl_cffi with Chrome impersonation for better TLS fingerprinting
            self.session = requests.Session(impersonate="chrome120")
        else:
            self.session = requests.Session()
        
        # Set up cookies
        self.cookie_jar = http.cookiejar.CookieJar()
        self.session.cookies = self.cookie_jar
        
        # Track visited pages
        self.visited_pages = []
        
        # Track last request time
        self.last_request_time = datetime.now()
        
        # Track if we've initialized (visited homepage)
        self.initialized = False
        
        # Define user agents - Updated to Chrome 120+ (2024-2025)
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        ]
    
    def get(self, url):
        # Initialize session by visiting homepage first (helps with Cloudflare)
        if not self.initialized:
            print("[HumanLikeRequester] Initializing session with homepage visit...", flush=True)
            self._initialize_session()
            self.initialized = True
        
        # Wait a random time between requests (visible to user)
        self._wait_random_time()
        
        # Create headers with browser-like attributes
        headers = self._create_headers()
        
        # Log request details for validation
        print(f"[HumanLikeRequester] Making request to: {url}", flush=True)
        print(f"[HumanLikeRequester] User-Agent: {headers['User-Agent'][:50]}...", flush=True)
        if 'Referer' in headers:
            print(f"[HumanLikeRequester] Referer: {headers['Referer']}", flush=True)
        
        # Make the request with TLS impersonation if using curl_cffi
        try:
            if HAS_CURL_CFFI:
                # curl_cffi handles TLS fingerprinting automatically with impersonate
                response = self.session.get(url, headers=headers, allow_redirects=True, timeout=30)
            else:
                response = self.session.get(url, headers=headers, allow_redirects=True, timeout=30)
        except Exception as e:
            print(f"[HumanLikeRequester] Request error: {e}", flush=True)
            raise
        
        # Check if we got Cloudflare challenge
        if response.status_code == 403:
            response_preview = response.text[:500] if response.text else ""
            if 'Just a moment' in response_preview or 'challenge' in response_preview.lower():
                print("[HumanLikeRequester] ⚠️  Cloudflare challenge page detected (403)", flush=True)
                print("[HumanLikeRequester] This typically requires JavaScript execution to solve.", flush=True)
                print("[HumanLikeRequester] Consider using a headless browser (Selenium/Playwright) for Cloudflare-protected sites.", flush=True)
        
        # Update state
        self.last_request_time = datetime.now()
        self.visited_pages.append(url)
        
        # Keep list manageable
        if len(self.visited_pages) > 10:
            self.visited_pages.pop(0)
        
        print(f"[HumanLikeRequester] Response status: {response.status_code}", flush=True)
        return response
    
    def _initialize_session(self):
        """Visit homepage first to establish session and get Cloudflare cookies"""
        homepage_url = 'https://www.pro-football-reference.com/'
        
        # Create headers for first visit (no referer)
        user_agent = random.choice(self.user_agents)
        chrome_version = '120'
        if 'Chrome/' in user_agent:
            try:
                chrome_version = user_agent.split('Chrome/')[1].split('.')[0]
            except:
                chrome_version = '120'
        
        platform = 'Windows'
        if 'Macintosh' in user_agent:
            platform = 'macOS'
        elif 'Linux' in user_agent:
            platform = 'Linux'
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Sec-CH-UA': f'"Chromium";v="{chrome_version}", "Google Chrome";v="{chrome_version}", "Not A;Brand";v="99"',
            'Sec-CH-UA-Mobile': '?0',
            'Sec-CH-UA-Platform': f'"{platform}"',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'sec-ch-ua-platform-version': '"15.0.0"' if platform == 'Windows' else '"14.0.0"'
        }
        
        print(f"[HumanLikeRequester] Visiting homepage: {homepage_url}", flush=True)
        try:
            # Wait before first request
            time.sleep(random.uniform(2, 4))
            if HAS_CURL_CFFI:
                response = self.session.get(homepage_url, headers=headers, allow_redirects=True, timeout=30)
            else:
                response = self.session.get(homepage_url, headers=headers, allow_redirects=True, timeout=30)
            print(f"[HumanLikeRequester] Homepage response: {response.status_code}", flush=True)
            
            if response.status_code == 403:
                print("[HumanLikeRequester] ⚠️  Homepage also blocked by Cloudflare", flush=True)
                print("[HumanLikeRequester] Cloudflare is blocking automated requests. This may require:", flush=True)
                print("[HumanLikeRequester]   1. Using a headless browser (Selenium/Playwright)", flush=True)
                print("[HumanLikeRequester]   2. Manual cookie extraction from browser", flush=True)
                print("[HumanLikeRequester]   3. Different IP address or VPN", flush=True)
            elif response.status_code == 200:
                self.visited_pages.append(homepage_url)
                # Wait after successful initialization
                time.sleep(random.uniform(2, 4))
                print("[HumanLikeRequester] Session initialized successfully", flush=True)
        except Exception as e:
            print(f"[HumanLikeRequester] Warning: Homepage visit failed: {e}", flush=True)
    
    def _wait_random_time(self):
        # Base wait (3-7 seconds)
        wait_time = random.uniform(3, 7)
        
        # Sometimes wait longer (10% chance)
        if random.random() < 0.1:
            wait_time = random.uniform(10, 20)
        
        # Rarely wait much longer (1% chance)
        if random.random() < 0.01:
            wait_time = random.uniform(30, 60)
        
        print(f"[HumanLikeRequester] Waiting {wait_time:.2f} seconds before request...", flush=True)
        
        # Show progress during wait (update every second)
        elapsed = 0
        while elapsed < wait_time:
            remaining = wait_time - elapsed
            if remaining > 1:
                print(f"  ... {remaining:.1f}s remaining", end='\r', flush=True)
            time.sleep(min(1.0, remaining))
            elapsed += 1.0
        
        print(f"\n[HumanLikeRequester] Wait complete, making request...", flush=True)
    
    def _create_headers(self):
        # Select user agent
        user_agent = random.choice(self.user_agents)
        
        # Extract Chrome version from User-Agent for Sec-CH-UA
        chrome_version = '120'
        if 'Chrome/' in user_agent:
            try:
                chrome_version = user_agent.split('Chrome/')[1].split('.')[0]
            except:
                chrome_version = '120'
        
        # Determine if mobile from User-Agent
        is_mobile = 'Mobile' in user_agent
        platform = 'Windows'
        if 'Macintosh' in user_agent:
            platform = 'macOS'
        elif 'Linux' in user_agent:
            platform = 'Linux'
        elif 'Android' in user_agent:
            platform = 'Android'
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none' if not self.visited_pages else 'same-origin',
            'Sec-Fetch-User': '?1',
            # Chrome Client Hints (required for modern Chrome)
            'Sec-CH-UA': f'"Chromium";v="{chrome_version}", "Google Chrome";v="{chrome_version}", "Not A;Brand";v="99"',
            'Sec-CH-UA-Mobile': '?1' if is_mobile else '?0',
            'Sec-CH-UA-Platform': f'"{platform}"',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'sec-ch-ua-platform-version': '"15.0.0"' if platform == 'Windows' else '"14.0.0"'
        }
        
        # Add referer if we have visited pages (more browser-like)
        if self.visited_pages:
            headers['Referer'] = random.choice(self.visited_pages)
            headers['Sec-Fetch-Site'] = 'same-origin'
        else:
            # First request - no referer, but set proper Sec-Fetch-Site
            headers['Sec-Fetch-Site'] = 'none'
        
        return headers