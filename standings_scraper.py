import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import time
from datetime import datetime
from human_like_requester import HumanLikeRequester
import os
import random

def scrape_nfl_standings_year(year, requester=None):
    """
    Scrapes NFL standings for a given year from Pro Football Reference.
    Fixed version that handles different page structures across years.
    
    Args:
        year (int): The NFL season year to scrape
        requester (HumanLikeRequester): Optional requester instance
        
    Returns:
        DataFrame: Contains team standings data for that year
    """
    if requester is None:
        requester = HumanLikeRequester()
    
    url = f"https://www.pro-football-reference.com/years/{year}/"
    print(f"\nScraping standings for {year} season from: {url}")
    
    try:
        response = requester.get(url)
        
        if response.status_code != 200:
            print(f"Failed to fetch {year}. Status code: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize list to store all teams' data
        all_teams_data = []
        
        # Strategy 1: Look for the combined standings table (newer format)
        print("Looking for combined standings table...")
        combined_table = soup.find('table', {'id': 'standings'})
        
        if combined_table:
            print("Found combined standings table")
            all_teams_data.extend(parse_standings_table(combined_table, year))
        
        # Strategy 2: Look for separate AFC/NFC tables
        if not all_teams_data:
            print("Looking for separate AFC/NFC tables...")
            for conf in ['afc', 'nfc']:
                # Try different possible table IDs
                table_ids = [f'{conf}_standings', f'div_{conf}', f'all_{conf}']
                
                for table_id in table_ids:
                    table = soup.find('table', {'id': table_id})
                    if table:
                        print(f"Found {conf.upper()} table with ID: {table_id}")
                        all_teams_data.extend(parse_standings_table(table, year, conference=conf.upper()))
                        break
        
        # Strategy 3: Look in comments (PFR sometimes hides tables in comments)
        if not all_teams_data:
            print("Looking in HTML comments...")
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            
            for comment in comments:
                if 'standings' in str(comment).lower():
                    comment_soup = BeautifulSoup(str(comment), 'html.parser')
                    
                    # Look for standings tables in comments
                    comment_tables = comment_soup.find_all('table')
                    for table in comment_tables:
                        if table.get('id') and 'standings' in table.get('id').lower():
                            print(f"Found table in comment: {table.get('id')}")
                            all_teams_data.extend(parse_standings_table(table, year))
        
        # Strategy 4: Look for any table with team data (fallback)
        if not all_teams_data:
            print("Using fallback method - analyzing all tables...")
            tables = soup.find_all('table')
            
            for table in tables:
                # Check if table has team-like data
                rows = table.find_all('tr')
                if len(rows) < 4:  # Skip small tables
                    continue
                
                # Look for headers that suggest standings
                first_row = rows[0] if rows else None
                if first_row:
                    headers = first_row.find_all(['th', 'td'])
                    header_texts = [h.get_text(strip=True).lower() for h in headers]
                    
                    # Check if this looks like a standings table
                    if any(header in ['w', 'l', 'wins', 'losses', 'team', 'tm'] for header in header_texts):
                        print(f"Found potential standings table with headers: {header_texts[:5]}")
                        parsed_data = parse_standings_table(table, year)
                        if parsed_data:  # Only add if we got valid data
                            all_teams_data.extend(parsed_data)
                            break
        
        # Strategy 5: Parse from division tables (older format)
        if not all_teams_data:
            print("Looking for division-specific tables...")
            divisions = ['afc_east', 'afc_north', 'afc_south', 'afc_west', 
                        'nfc_east', 'nfc_north', 'nfc_south', 'nfc_west']
            
            for division in divisions:
                div_table = soup.find('table', {'id': division})
                if div_table:
                    print(f"Found division table: {division}")
                    all_teams_data.extend(parse_standings_table(div_table, year))
        
        if all_teams_data:
            # Remove duplicates based on team name
            seen_teams = set()
            unique_teams = []
            for team_data in all_teams_data:
                team_key = team_data['team'].lower().strip()
                if team_key not in seen_teams:
                    seen_teams.add(team_key)
                    unique_teams.append(team_data)
            
            df = pd.DataFrame(unique_teams)
            print(f"Successfully scraped {len(df)} teams for {year}")
            
            # Calculate additional metrics
            df['games_played'] = df['wins'] + df['losses'] + df['ties']
            df['win_pct'] = (df['wins'] + 0.5 * df['ties']) / df['games_played']
            df['win_pct'] = df['win_pct'].round(3)
            
            return df
        else:
            print(f"No standings data found for {year}")
            return None
            
    except Exception as e:
        print(f"Error scraping {year}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def parse_standings_table(table, year, conference=None):
    """
    Parse a standings table and extract team data.
    
    Args:
        table: BeautifulSoup table element
        year: Season year
        conference: Optional conference name
        
    Returns:
        List of team data dictionaries
    """
    teams_data = []
    
    try:
        rows = table.find_all('tr')
        
        for row in rows:
            # Skip header rows
            if not row.find('td'):
                continue
            
            # Look for team name - can be in th or td
            team_cell = row.find('th', {'data-stat': 'team'}) or row.find('td', {'data-stat': 'team'})
            if not team_cell:
                team_cell = row.find('th') or row.find('td')
            
            if team_cell:
                team_name = team_cell.get_text(strip=True)
                
                # Skip division headers and other non-team rows
                if any(keyword in team_name.lower() for keyword in 
                      ['east', 'west', 'north', 'south', 'central', 'division', 'conference']):
                    continue
                
                # Clean team name
                team_name = team_name.replace('*', '').replace('+', '').strip()
                
                if not team_name or len(team_name) < 2:
                    continue
                
                # Get wins - try multiple approaches
                wins = get_stat_value(row, ['wins', 'w'], default=0)
                losses = get_stat_value(row, ['losses', 'l'], default=0)
                ties = get_stat_value(row, ['ties', 't'], default=0)
                
                # Skip if we don't have basic W-L data
                if wins is None and losses is None:
                    continue
                
                team_data = {
                    'year': year,
                    'team': team_name,
                    'wins': wins if wins is not None else 0,
                    'losses': losses if losses is not None else 0,
                    'ties': ties if ties is not None else 0
                }
                
                # Add conference if provided
                if conference:
                    team_data['conference'] = conference
                
                # Get additional stats if available
                pf = get_stat_value(row, ['points_for', 'pf', 'pts_for'])
                pa = get_stat_value(row, ['points_against', 'pa', 'pts_against'])
                
                if pf is not None:
                    team_data['points_for'] = pf
                if pa is not None:
                    team_data['points_against'] = pa
                
                teams_data.append(team_data)
                
    except Exception as e:
        print(f"Error parsing table: {e}")
    
    return teams_data

def get_stat_value(row, stat_names, default=None):
    """
    Try to get a stat value from a row using multiple possible column names.
    
    Args:
        row: BeautifulSoup row element
        stat_names: List of possible data-stat attribute names
        default: Default value if not found
        
    Returns:
        Parsed integer value or default
    """
    for stat_name in stat_names:
        cell = row.find('td', {'data-stat': stat_name})
        if cell:
            try:
                value = cell.get_text(strip=True)
                if value and value.isdigit():
                    return int(value)
            except (ValueError, AttributeError):
                continue
    
    # Fallback: try to find by position if data-stat doesn't work
    if not stat_names:
        return default
        
    # For very old pages, try positional parsing
    cells = row.find_all('td')
    if len(cells) >= 3:  # Need at least team, wins, losses
        try:
            if 'w' in stat_names[0].lower():  # Looking for wins
                return int(cells[0].get_text(strip=True)) if cells[0].get_text(strip=True).isdigit() else default
            elif 'l' in stat_names[0].lower():  # Looking for losses  
                return int(cells[1].get_text(strip=True)) if cells[1].get_text(strip=True).isdigit() else default
        except (ValueError, IndexError):
            pass
    
    return default

def scrape_all_nfl_standings(start_year=2000, end_year=2024, save_individual=True):
    """
    Scrapes NFL standings for all years in the specified range.
    """
    # Create requester instance
    requester = HumanLikeRequester()
    
    # Create directory for standings data
    os.makedirs('standings_data', exist_ok=True)
    
    all_standings = []
    
    for year in range(start_year, end_year + 1):
        # Check if we already have this year's data
        csv_path = os.path.join('standings_data', f'standings_{year}.csv')
        
        if os.path.exists(csv_path):
            print(f"Loading existing data for {year}")
            try:
                year_df = pd.read_csv(csv_path)
                all_standings.append(year_df)
                continue
            except Exception as e:
                print(f"Error loading {csv_path}, will re-scrape: {e}")
        
        # Scrape the year
        year_df = scrape_nfl_standings_year(year, requester)
        
        if year_df is not None and not year_df.empty:
            # Save individual year if requested
            if save_individual:
                year_df.to_csv(csv_path, index=False)
                print(f"Saved {year} standings to {csv_path}")
            
            all_standings.append(year_df)
        else:
            print(f"Skipping {year} - no data retrieved")
        
        # Random wait between years (3-10 seconds)
        if year < end_year:  # Don't wait after the last year
            wait_time = random.uniform(3.0, 10.0)
            print(f"Waiting {wait_time:.1f} seconds before next request...")
            time.sleep(wait_time)
    
    # Combine all years
    if all_standings:
        combined_df = pd.concat(all_standings, ignore_index=True)
        
        # Sort by year and win percentage
        combined_df = combined_df.sort_values(['year', 'win_pct'], ascending=[True, False])
        
        # Save combined data
        combined_path = 'nfl_standings_2000_2024.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved combined standings to {combined_path}")
        print(f"Total records: {len(combined_df)}")
        
        return combined_df
    else:
        print("No standings data collected")
        return None

def debug_single_year(year=2000):
    """
    Debug function to inspect what's available on a single year's page
    """
    requester = HumanLikeRequester()
    url = f"https://www.pro-football-reference.com/years/{year}/"
    
    print(f"Debugging {year} page structure...")
    print(f"URL: {url}")
    
    response = requester.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    print(f"\nAll table IDs found:")
    tables = soup.find_all('table', id=True)
    for table in tables:
        table_id = table.get('id')
        rows = len(table.find_all('tr'))
        print(f"  - {table_id} ({rows} rows)")
        
        # Show first row headers for context
        first_row = table.find('tr')
        if first_row:
            headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
            print(f"    Headers: {headers[:6]}...")
    
    print(f"\nAll div IDs found:")
    divs = soup.find_all('div', id=True)[:20]  # First 20
    for div in divs:
        print(f"  - {div.get('id')}")

# Test the debug function
if __name__ == "__main__":
    # First, let's debug what's available on the 2000 page
    print("=== DEBUGGING 2000 PAGE ===")
    debug_single_year(2000)
    
    print("\n" + "=" * 50)
    print("=== TESTING SCRAPER ===")
    
    # Test scraping just one year first
    test_df = scrape_nfl_standings_year(2000)
    if test_df is not None:
        print(f"\nSuccess! Found {len(test_df)} teams for 2000:")
        print(test_df[['team', 'wins', 'losses', 'win_pct']].head(10))
    else:
        print("Still having issues with 2000...")