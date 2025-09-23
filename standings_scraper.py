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
    
    Args:
        year (int): The NFL season year to scrape
        requester (HumanLikeRequester): Optional requester instance
        
    Returns:
        DataFrame: Contains team standings data for that year
    """
    if requester is None:
        requester = HumanLikeRequester()
    
    url = f"https://www.pro-football-reference.com/years/{year}/"
    print(f"\nScraping standings for {year} season...")
    
    try:
        response = requester.get(url)
        
        if response.status_code != 200:
            print(f"Failed to fetch {year}. Status code: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize list to store all teams' data
        all_teams_data = []
        
        # Look for AFC and NFC standings tables
        # These are typically in divs with specific IDs
        for conference in ['AFC', 'NFC']:
            conf_div = soup.find('div', {'id': f'all_div_{conference}'})
            
            if not conf_div:
                # Try to find in comments (PFR sometimes hides tables in comments)
                comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    if f'div_{conference}' in str(comment):
                        comment_soup = BeautifulSoup(str(comment), 'html.parser')
                        conf_div = comment_soup.find('div', {'id': f'div_{conference}'})
                        if conf_div:
                            break
            
            # If still not found, try finding tables directly
            if not conf_div:
                # Look for table with conference standings
                tables = soup.find_all('table')
                for table in tables:
                    caption = table.find('caption')
                    if caption and conference in caption.text:
                        conf_div = table.parent
                        break
        
        # Alternative approach: find all team standings in the main tables
        standings_tables = soup.find_all('table', {'id': lambda x: x and 'standings' in x.lower() if x else False})
        
        if not standings_tables:
            # Try the main AFC/NFC tables
            for conf in ['AFC', 'NFC']:
                table = soup.find('table', {'id': f'{conf.lower()}_standings'})
                if not table:
                    table = soup.find('table', {'id': f'div_{conf}'})
                if table:
                    standings_tables.append(table)
        
        # If still no tables, look for them in a different way
        if not standings_tables:
            # Find all tables and look for ones with W, L, W-L% headers
            all_tables = soup.find_all('table')
            for table in all_tables:
                headers = table.find_all('th')
                header_texts = [h.get_text(strip=True) for h in headers]
                if 'W' in header_texts and 'L' in header_texts:
                    standings_tables.append(table)
        
        # Parse each standings table
        for table in standings_tables[:2]:  # AFC and NFC
            rows = table.find_all('tr')
            
            for row in rows:
                # Skip header rows
                if row.find('th', {'scope': 'row'}):
                    team_cell = row.find('th', {'data-stat': 'team'})
                    if not team_cell:
                        team_cell = row.find('th')
                    
                    if team_cell:
                        team_name = team_cell.get_text(strip=True)
                        
                        # Skip division headers
                        if any(div in team_name for div in ['East', 'West', 'North', 'South', 'Central']):
                            continue
                        
                        # Get wins and losses
                        wins_cell = row.find('td', {'data-stat': 'wins'})
                        losses_cell = row.find('td', {'data-stat': 'losses'})
                        ties_cell = row.find('td', {'data-stat': 'ties'})
                        
                        # Alternative data-stat names
                        if not wins_cell:
                            wins_cell = row.find('td', {'data-stat': 'W'})
                        if not losses_cell:
                            losses_cell = row.find('td', {'data-stat': 'L'})
                        if not ties_cell:
                            ties_cell = row.find('td', {'data-stat': 'T'})
                        
                        # If still not found, use position
                        if not wins_cell or not losses_cell:
                            cells = row.find_all('td')
                            if len(cells) >= 2:
                                wins_cell = cells[0]
                                losses_cell = cells[1]
                                if len(cells) >= 3:
                                    ties_cell = cells[2]
                        
                        if wins_cell and losses_cell:
                            team_data = {
                                'year': year,
                                'team': team_name.replace('*', '').replace('+', '').strip(),
                                'wins': int(wins_cell.get_text(strip=True)),
                                'losses': int(losses_cell.get_text(strip=True)),
                                'ties': int(ties_cell.get_text(strip=True)) if ties_cell and ties_cell.get_text(strip=True) else 0
                            }
                            
                            # Get additional stats if available
                            pf_cell = row.find('td', {'data-stat': 'points_for'})
                            pa_cell = row.find('td', {'data-stat': 'points_against'})
                            
                            if not pf_cell:
                                pf_cell = row.find('td', {'data-stat': 'PF'})
                            if not pa_cell:
                                pa_cell = row.find('td', {'data-stat': 'PA'})
                            
                            if pf_cell:
                                team_data['points_for'] = int(pf_cell.get_text(strip=True))
                            if pa_cell:
                                team_data['points_against'] = int(pa_cell.get_text(strip=True))
                            
                            all_teams_data.append(team_data)
        
        # Fallback: Parse the simplified standings from the main page
        if not all_teams_data:
            print(f"Using fallback method for {year}...")
            # Look for the standings in the main content area
            content = soup.find('div', {'id': 'content'})
            if content:
                # Find all team links and their records
                team_links = content.find_all('a', href=lambda x: x and '/teams/' in x)
                for link in team_links:
                    parent = link.parent
                    if parent:
                        text = parent.get_text()
                        # Look for pattern like "Team Name (W-L)"
                        import re
                        match = re.search(r'(\d+)-(\d+)(?:-(\d+))?', text)
                        if match:
                            team_data = {
                                'year': year,
                                'team': link.get_text(strip=True),
                                'wins': int(match.group(1)),
                                'losses': int(match.group(2)),
                                'ties': int(match.group(3)) if match.group(3) else 0
                            }
                            all_teams_data.append(team_data)
        
        if all_teams_data:
            df = pd.DataFrame(all_teams_data)
            print(f"Successfully scraped {len(df)} teams for {year}")
            return df
        else:
            print(f"No standings data found for {year}")
            return None
            
    except Exception as e:
        print(f"Error scraping {year}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def scrape_all_nfl_standings(start_year=2000, end_year=2024, save_individual=True):
    """
    Scrapes NFL standings for all years in the specified range.
    
    Args:
        start_year (int): First year to scrape
        end_year (int): Last year to scrape
        save_individual (bool): Whether to save individual year CSVs
        
    Returns:
        DataFrame: Combined standings for all years
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
            year_df = pd.read_csv(csv_path)
        else:
            # Scrape the year
            year_df = scrape_nfl_standings_year(year, requester)
            
            if year_df is not None and not year_df.empty:
                # Save individual year if requested
                if save_individual:
                    year_df.to_csv(csv_path, index=False)
                    print(f"Saved {year} standings to {csv_path}")
            
            # Random wait between years (3-10 seconds)
            wait_time = random.uniform(3.0, 10.0)
            print(f"Waiting {wait_time:.1f} seconds before next request...")
            time.sleep(wait_time)
        
        if year_df is not None:
            all_standings.append(year_df)
    
    # Combine all years
    if all_standings:
        combined_df = pd.concat(all_standings, ignore_index=True)
        
        # Calculate win percentage
        combined_df['games_played'] = combined_df['wins'] + combined_df['losses'] + combined_df['ties']
        combined_df['win_pct'] = (combined_df['wins'] + 0.5 * combined_df['ties']) / combined_df['games_played']
        combined_df['win_pct'] = combined_df['win_pct'].round(3)
        
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

def get_team_history(team_name, df=None):
    """
    Get the win-loss history for a specific team.
    
    Args:
        team_name (str): Name of the team
        df (DataFrame): Optional dataframe, will load from file if not provided
        
    Returns:
        DataFrame: Team's yearly records
    """
    if df is None:
        if os.path.exists('nfl_standings_2000_2024.csv'):
            df = pd.read_csv('nfl_standings_2000_2024.csv')
        else:
            print("No standings data found. Run scrape_all_nfl_standings() first.")
            return None
    
    team_df = df[df['team'].str.contains(team_name, case=False)]
    return team_df[['year', 'team', 'wins', 'losses', 'ties', 'win_pct']].sort_values('year')

if __name__ == "__main__":
    # Scrape all standings from 2000 to 2024
    print("Starting NFL standings scraper...")
    print("=" * 50)
    
    standings_df = scrape_all_nfl_standings(start_year=2000, end_year=2024)
    
    if standings_df is not None:
        print("\n" + "=" * 50)
        print("Scraping complete!")
        print(f"\nSample of data (first 5 teams from 2024):")
        print(standings_df[standings_df['year'] == 2024].head())
        
        # Example: Get Patriots history
        print("\n" + "=" * 50)
        print("Example: New England Patriots History")
        pats_history = get_team_history("Patriots", standings_df)
        if pats_history is not None:
            print(pats_history)