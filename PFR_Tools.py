import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from human_like_requester import HumanLikeRequester
import os
import random
import io

# Create a global requester instance
requester = HumanLikeRequester()

def random_sleep():
    """
    Pauses execution for a random duration between 3.0 and 7.0 seconds.
    
    This function helps avoid overwhelming the target website with too many
    rapid requests by simulating human-like browsing behavior.
    
    Returns:
        float: The actual wait time in seconds
    """
    wait_time = random.uniform(3.0, 7.0)
    print(f"Waiting for {wait_time:.2f} seconds...")
    time.sleep(wait_time)
    return wait_time

def extract_table_from_page(soup, table_type):
    """
    Extracts a specific table from the page, checking both visible tables and tables in comments.
    
    Args:
        soup (BeautifulSoup): Parsed HTML
        table_type (str): Type of table to extract ('passing', 'rushing', or 'advanced_passing')
    
    Returns:
        DataFrame or None: Extracted table as DataFrame, or None if not found
    """
    import pandas as pd
    import re
    from bs4 import Comment
    
    # Map table_type to likely table IDs
    table_id_patterns = {
        'passing': ['passing$', 'passing_', 'passing_regular_season', 'Passing'],
        'rushing': ['rushing$', 'rushing_', 'rushing_regular_season', 'Rushing'],
        'advanced_passing': ['advanced_passing$', 'passing_advanced', 'Advanced_Passing']
    }
    
    patterns = table_id_patterns.get(table_type, [])
    
    # Check visible tables
    for pattern in patterns:
        for tbl in soup.find_all('table'):
            if tbl.get('id') and re.search(pattern, tbl.get('id'), re.IGNORECASE):
                print(f"Found {table_type} table with ID: {tbl.get('id')}")
                try:
                    return pd.read_html(str(tbl))[0]
                except Exception as e:
                    print(f"Error reading {table_type} table: {e}")
    
    # Check tables within comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment_soup = BeautifulSoup(str(comment), 'html.parser')
        for tbl in comment_soup.find_all('table'):
            if tbl.get('id') and any(re.search(pattern, tbl.get('id'), re.IGNORECASE) for pattern in patterns):
                print(f"Found {table_type} table in comment with ID: {tbl.get('id')}")
                try:
                    return pd.read_html(str(tbl))[0]
                except Exception as e:
                    print(f"Error reading {table_type} table from comment: {e}")
            
    # Check for div containers that might have the tables
    for div in soup.find_all('div', class_=['table_container', 'overthrow table_container']):
        for tbl in div.find_all('table'):
            if tbl.get('id') and any(re.search(pattern, tbl.get('id'), re.IGNORECASE) for pattern in patterns):
                print(f"Found {table_type} table in div container with ID: {tbl.get('id')}")
                try:
                    return pd.read_html(str(tbl))[0]
                except Exception as e:
                    print(f"Error reading {table_type} table from div container: {e}")
    
    # If nothing is found, print information about all tables on the page
    if table_type == 'passing':  # Only do this once
        print("\nAll table IDs found on page:")
        for tbl in soup.find_all('table'):
            if tbl.get('id'):
                print(f"- {tbl.get('id')}")
        
        print("\nAll table IDs found in comments:")
        for comment in comments:
            comment_soup = BeautifulSoup(str(comment), 'html.parser')
            for tbl in comment_soup.find_all('table'):
                if tbl.get('id'):
                    print(f"- {tbl.get('id')}")
    
    return None

def inspect_player_tables(player_url):
    """
    A diagnostic function that inspects the structure of tables on a player's page.
    
    Args:
        player_url (str): URL to the player's page on Pro Football Reference
        
    Returns:
        None: This function just prints diagnostic information
    """
    
    print(f"Inspecting tables at: {player_url}")
    
    try:
        response = requests.get(player_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # List all tables
        print("\nAll tables found:")
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables on the page")
        
        for i, table in enumerate(tables):
            table_id = table.get('id', 'No ID')
            print(f"\nTable {i+1}: ID = {table_id}")
            
            # Print table structure
            print("  Structure:")
            
            # Check if table has a thead
            thead = table.find('thead')
            if thead:
                print("  - Found <thead> element")
                thead_rows = thead.find_all('tr')
                print(f"    - Contains {len(thead_rows)} header rows")
                
                # Print each header row
                for j, row in enumerate(thead_rows):
                    headers = row.find_all(['th', 'td'])
                    header_texts = [h.get_text(strip=True) for h in headers]
                    print(f"    - Header row {j+1}: {header_texts[:5]}... ({len(headers)} columns)")
            else:
                print("  - No <thead> element found")
            
            # Check if table has a tbody
            tbody = table.find('tbody')
            if tbody:
                print("  - Found <tbody> element")
                tbody_rows = tbody.find_all('tr')
                print(f"    - Contains {len(tbody_rows)} data rows")
                
                # Print first data row as example
                if tbody_rows:
                    first_row = tbody_rows[0]
                    cells = first_row.find_all(['th', 'td'])
                    cell_texts = [c.get_text(strip=True) for c in cells]
                    print(f"    - First row example: {cell_texts[:5]}... ({len(cells)} cells)")
                    
                    # Print data-stat attributes for the cells
                    data_stats = [c.get('data-stat', 'No data-stat') for c in cells]
                    print(f"    - data-stat attributes: {data_stats[:5]}...")
            else:
                print("  - No <tbody> element found")
            
        # Special analysis of passing table
        passing_table = soup.find('table', id='passing')
        if passing_table:
            print("\nDetailed analysis of passing table:")
            
            # Get all rows including header rows
            all_rows = passing_table.find_all('tr')
            print(f"Total rows in passing table: {len(all_rows)}")
            
            # Check first few rows
            for i, row in enumerate(all_rows[:5]):
                print(f"\nRow {i+1}:")
                
                # Check for th cells
                th_cells = row.find_all('th')
                td_cells = row.find_all('td')
                print(f"  TH cells: {len(th_cells)}, TD cells: {len(td_cells)}")
                
                if th_cells:
                    print(f"  TH cell values: {[h.get_text(strip=True) for h in th_cells]}")
                    print(f"  TH data-stat attributes: {[h.get('data-stat', 'None') for h in th_cells]}")
                
                if td_cells:
                    print(f"  TD cell values: {[d.get_text(strip=True) for d in td_cells[:5]]}...")
                    print(f"  TD data-stat attributes: {[d.get('data-stat', 'None') for d in td_cells[:5]]}...")
        
    except Exception as e:
        print(f"Error inspecting tables: {e}")
        import traceback
        traceback.print_exc()

# Example usage
if __name__ == "__main__":
    inspect_player_tables("https://www.pro-football-reference.com/players/A/AlleJo02.htm")

def get_draft_class(year):
    """
    Retrieves NFL draft class data for a specified year.
    
    This function scrapes the Pro Football Reference website to obtain
    draft information. It first checks if the data is cached locally as a CSV
    file before making a web request.
    
    Args:
        year (int): The NFL draft year to retrieve
        
    Returns:
        BeautifulSoup object: The parsed HTML table containing the draft class data,
                             or None if the data couldn't be retrieved
                             
    Notes:
        - Data is cached in the 'draft_data' directory as 'draft_class_{year}.csv'
        - Uses various fallback methods to locate the draft table on the page
        - Source URL format: https://www.pro-football-reference.com/years/{YEAR}/draft.htm
        - Example: https://www.pro-football-reference.com/years/2025/draft.htm
    """
    print(f"Scraping draft year: {year}")
    url = f"https://www.pro-football-reference.com/years/{year}/draft.htm"
    draft_table = None
    csv_file_path = os.path.join('draft_data', f'draft_class_{year}.csv')
    
    if os.path.exists(csv_file_path):
        print(f"Draft class for {year} already exists. Loading from CSV.")
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            draft_table = pd.read_csv(f)
        return draft_table

    try:
        response = requester.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')       
        
        # Print out all table ids to help debug
        tables = soup.find_all('table')
        #print(f"Found {len(tables)} tables on the page")
        #for i, table in enumerate(tables):
        #    table_id = table.get('id', 'No ID')
        #    print(f"Table {i+1} ID: {table_id}")
        # Try various possible table IDs

        possible_ids = ['drafts', 'draft', 'all_drafts', 'all_draft']
        for table_id in possible_ids:
            draft_table = soup.find('table', id=table_id)
            if draft_table:
                print(f"Found draft table with ID: {table_id}")
                break
        table_id = 'draft'

        # If still not found, try finding by class name
        if not draft_table:
            draft_table = soup.find('table', {'class': 'sortable stats_table'})
            if draft_table:
                print("Found draft table by class name since no ID found")
        
        # If still not found, use the first table
        if not draft_table and tables:
            draft_table = tables[0]
            print("Using first table on the page as draft table since no ID or class name found")

        else:
            print(f"Draft table not found for year: {year}")
    
    except requests.RequestException as e:
        print(f"No page found for draft year: {year}")
    
    os.makedirs('draft_data', exist_ok=True)  # Create directory if it doesn't exist
    if draft_table:
        # Save the draft table to a CSV file
        with open(csv_file_path, 'w', encoding='utf-8') as f:
            f.write(str(draft_table))
        print(f"Draft class for {year} saved to {csv_file_path}")
    else:
        print(f"No Draft table, so write to CSV failed for: {year}")
    
    return draft_table

def get_first_round_QBS (draft_class, year):
    """
    Extracts first-round quarterback selections from a draft class.
    
    This function parses the draft class HTML to identify quarterbacks selected
    in the first round of the specified draft year.
    
    Args:
        draft_class (BeautifulSoup object): The parsed HTML table containing draft data
        year (int): The NFL draft year
        
    Returns:
        list: A list of dictionaries containing information about each first-round QB:
              - 'name': Player's name
              - 'draft_year': Year drafted
              - 'draft_team': Team that drafted the player
              
    Notes:
        - Handles variations in HTML structure by trying multiple data-stat attributes
        - Skips header and spacer rows in the table
    """
    
    # Find all rows in the table
    rows = draft_class.find('tbody').find_all('tr')
    print(f"Found {len(rows)} rows in the draft table")
    first_round_qbs = []
    for row in rows:
        try:
            # Skip header rows or spacer rows
            if 'class' in row.attrs and ('thead' in row['class'] or 'divider' in row['class']):
                continue
            
            # Get round information - try various column names
            rnd_cell = None
            for data_stat in ['draft_round', 'round', 'rnd']:
                rnd_cell = row.find('td', {'data-stat': data_stat}) or row.find('th', {'data-stat': data_stat})
                if rnd_cell:
                    break
            
            # Check if we found a round cell and it's round 1
            if rnd_cell and '1' in rnd_cell.text.strip():
                # Find position - try various column names
                pos_cell = None
                for data_stat in ['pos', 'position', 'draft_position']:
                    pos_cell = row.find('td', {'data-stat': data_stat})
                    if pos_cell:
                        break
                
                # Check if this is a QB
                if pos_cell and 'QB' in pos_cell.text.strip():
                    # Get player name - try various column names
                    name_cell = None
                    for data_stat in ['player', 'name', 'draft_player']:
                        name_cell = row.find('td', {'data-stat': data_stat})
                        if name_cell:
                            break
                    
                    # Get team that drafted the player - try various column names
                    team_cell = None
                    for data_stat in ['team', 'draft_team', 'tm']:
                        team_cell = row.find('td', {'data-stat': data_stat})
                        if team_cell:
                            break
                    
                    if name_cell:
                        player_name = name_cell.text.strip()
                        # If there's a link, use that text as it's usually cleaner
                        if name_cell.find('a'):
                            player_name = name_cell.find('a').text.strip()
                        
                        team = team_cell.text.strip() if team_cell else "Unknown"
                        
                        print(f"Found first-round QB: {player_name} drafted by {team} in {year}")
                        first_round_qbs.append({
                            'name': player_name,
                            'draft_year': year,
                            'draft_team': team
                        })
        except Exception as inner_e:
            print(f"Error processing row in {year}: {str(inner_e)}")
    else:
        print(f"Draft table not found or has no tbody for year {year}")

#function to get a csv file of all players since 2000
def get_player_ids():
    """
    Creates a comprehensive CSV database of NFL player IDs since 2000.
    
    This function scrapes the Pro Football Reference fantasy stats pages for each year
    to build a database of player names and their corresponding unique IDs in the PFR system.
    
    Returns:
        None: The function saves the data directly to 'player_ids.csv'
        
    Notes:
        - Each record contains 'player_name', 'player_id', and 'year'
        - Uses HumanLikeRequester to respect server load
        - Includes pause times between requests to avoid rate limiting
    """
    current_year = datetime.now().year
    all_players = []
        
    for year in range(2000, current_year + 1):
        try:
            fantasy_year = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
            print(f"Fetching playerID data for {year}...")
            response = requester.get(fantasy_year)
            
            # Check if the request was successful
            if response.status_code != 200:
                print(f"Failed to fetch data for {year}. Status code: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Debug: Print all table IDs on the page
            tables = soup.find_all('table')
            print(f"Found {len(tables)} tables on the page for year {year}")
            if len(tables) > 0:
                print("Table IDs:", [table.get('id', 'No ID') for table in tables])
            
            player_table = soup.find('table', id='fantasy')
            if player_table:
                # Find all rows in the table
                rows = player_table.find('tbody').find_all('tr')
                print(f"Found {len(rows)} rows in the fantasy table for year {year}")
                
                for row in rows:
                    # Skip header rows or spacer rows
                    if 'class' in row.attrs and ('thead' in row['class'] or 'divider' in row['class']):
                        continue
                    
                    # Get player name and ID
                    name_cell = row.find('td', {'data-stat': 'player'})
                    if name_cell and name_cell.find('a'):
                        player_name = name_cell.find('a').text.strip()
                        player_link = name_cell.find('a').get('href', '')
                        
                        # Extract player ID from the link
                        # Example link: /players/B/BradTo00.htm
                        if '/players/' in player_link:
                            player_id = player_link.split('/')[-1].replace('.htm', '')
                            
                            all_players.append({
                                'player_name': player_name,
                                'player_id': player_id,
                                'year': year
                            })
            else:
                print(f"Fantasy table not found for year {year}")
            
            random_sleep()  # Sleep to avoid overwhelming the server
        
        except requests.RequestException as e:
            print(f"Error fetching data for year {year}: {e}")
        
        time.sleep(2)  # Sleep to avoid overwhelming the server

    # Convert the list to a DataFrame
    if all_players:
        df = pd.DataFrame(all_players)
        # Save the DataFrame to a CSV file
        df.to_csv('player_ids.csv', index=False)
        print(f"Saved {len(df)} player records to player_ids.csv")
    else:
        print("No player data found. CSV not created.")
    
    return None

def get_player_id(player_name):                
    """
    Retrieves a player's unique Pro Football Reference ID based on their name.
    
    Args:
        player_name (str): The player's full name
        
    Returns:
        str or None: The player's unique ID if found, None otherwise
        
    Notes:
        - Requires 'player_ids.csv' file to be present
        - Returns the first matching player ID if multiple matches exist
    """
    player_id = None
    player_ids_df = pd.read_csv('player_ids.csv')
    player_ids_df = player_ids_df[player_ids_df['player_name'] == player_name]
    if not player_ids_df.empty:
        player_id = player_ids_df.iloc[0]['player_id']
    return player_id

def update_qb_ids():
    """
    Updates the player IDs for all quarterbacks in the 'player_ids.csv' file.
    
    This function reads the existing player IDs from the CSV file, takes the list
     of first round QBs and then makes a second file of QB IDs for each one.
    
    Returns:
        None: The function creates a new CSV file '1st_rd_qb_ids.csv' with updated QB IDs.
    """
    qb_draft_df = pd.read_csv('first_round_qbs.csv')
    player_ids = pd.read_csv('player_ids.csv')
    qb_ids = pd.DataFrame(columns=['player_name', 'player_id', 'draft_year', 'draft_team'])
    for index, row in qb_draft_df.iterrows():
        qb_name = row['name']
        qb_draft_year = row['draft_year']
        qb_draft_team = row['draft_team']
        
        # Get the player ID from the player_ids DataFrame
        matching_players = player_ids[player_ids['player_name'] == qb_name]
        
        if not matching_players.empty:
            if len(matching_players) > 1:
                print(f"Multiple IDs found for {qb_name}: {matching_players['player_id'].tolist()}")
                # Look for the ID with the matching year
                matching_year_player = matching_players[matching_players['year'] == qb_draft_year]
                if not matching_year_player.empty:
                    player_id = matching_year_player.iloc[0]['player_id']
                    print(f"Found matching ID in draft year for {qb_name}: {player_id}")
                    qb_ids = pd.concat([qb_ids, pd.DataFrame({'player_name': [qb_name], 'player_id': [player_id], 'draft_year': [qb_draft_year], 'draft_team': [qb_draft_team]})], ignore_index=True)
                else:
                    # Use the first one if no matching year is found
                    player_id = matching_players.iloc[0]['player_id']
                    qb_ids = pd.concat([qb_ids, pd.DataFrame({'player_name': [qb_name], 'player_id': [player_id], 'draft_year': [qb_draft_year], 'draft_team': [qb_draft_team]})], ignore_index=True)
            else:
                player_id = matching_players.iloc[0]['player_id']
                qb_ids = pd.concat([qb_ids, pd.DataFrame({'player_name': [qb_name], 'player_id': [player_id], 'draft_year': [qb_draft_year], 'draft_team': [qb_draft_team]})], ignore_index=True)
        else:
            print(f"No ID found for first round qb: {qb_name} in master ID list")
    
    # Check if qb_ids is not empty before writing to CSV
    if not qb_ids.empty:
        csv_file_path = '1st_rd_qb_ids.csv'
        # Fix the CSV writing code - this had a syntax error too
        qb_ids.to_csv(csv_file_path, index=False)
        print(f"QB IDs saved to 1st_rd_qb_ids.csv")
    else:
        print(f"No QB IDs, so write to CSV failed for the QB specific IDs:")

    return True

def get_qb_seasons(qb_name, qb_id, draft_year=None, draft_team=None, debugging=False, overwrite=False):
    """
    Retrieves comprehensive career statistics for a quarterback.
    
    Args:
        qb_name (str): The quarterback's full name
        qb_id (str): The quarterback's unique Pro Football Reference ID
        draft_year (int, optional): Filter to include only seasons from this year onward
        draft_team (str, optional): The team that drafted the quarterback
        debugging (bool, optional): Whether to print debugging information
        
    Returns:
        DataFrame: A pandas DataFrame containing season-by-season statistics
    """
    
    current_year = datetime.now().year
    player_url = f"https://www.pro-football-reference.com/players/{qb_id[0]}/{qb_id}.htm"
    
    csv_file_path = os.path.join('QB_Data', f'{qb_id}.csv')
    if os.path.exists(csv_file_path) and not overwrite:
        print(f"Data for {qb_id} already exists. Loading from CSV.")
        return pd.read_csv(csv_file_path)
    
    print(f"Scraping player page: {player_url}")
    try:
        # Try to use requester if it exists, otherwise use requests directly
        try:
            response = requester.get(player_url)
        except NameError:
            response = requests.get(player_url)
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Print all table IDs for debugging
        if debugging:
            print("\nAll table IDs found on page:")
            for tbl in soup.find_all('table'):
                if tbl.get('id'):
                    print(f"- {tbl.get('id')}")
        
        # 1. Get passing table - this is our base table
        passing_table = soup.find('table', id='passing')
        if not passing_table:
            print(f"Passing table not found for {qb_name}")
            return None
        
        # Use pandas to parse the passing table
        print("Found passing table")
        passing_df = pd.read_html(io.StringIO(str(passing_table)))[0]
        
        # Print column names for debugging
        if debugging: 
            print(f"Passing table columns: {passing_df.columns.tolist()}")
        
        # Find the season column in passing table
        pass_year_col = None
        if 'Season' in passing_df.columns:
            pass_year_col = 'Season'
        elif 'season' in passing_df.columns:
            pass_year_col = 'season'
        elif 'Year' in passing_df.columns:
            pass_year_col = 'Year'
        else:
            # Try to guess based on data - usually first column
            if len(passing_df.columns) > 0:
                first_col = passing_df.columns[0]
                pass_year_col = first_col
                if debugging:
                    print(f"Using first column as season: {pass_year_col}")
        
        # 2. Get rushing table if available
        rushing_table = soup.find('table', id='rushing_and_receiving')
        rushing_df = None
        if rushing_table:
            print("Found rushing_and_receiving table")
            rushing_df = pd.read_html(io.StringIO(str(rushing_table)))[0]
            
            # Print column names for debugging
            if debugging: 
                print(f"Rushing table columns: {rushing_df.columns.tolist()}")
            
            # Handle multi-level columns if present
            if isinstance(rushing_df.columns, pd.MultiIndex):
                # Flatten the columns to single level
                rushing_df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in rushing_df.columns]
                if debugging:
                    print(f"Flattened rushing columns: {rushing_df.columns.tolist()}")
        
        # 3. Get advanced passing table if available
        adv_passing_table = soup.find('table', id='passing_advanced')
        adv_passing_df = None
        if adv_passing_table:
            print("Found passing_advanced table")
            adv_passing_df = pd.read_html(io.StringIO(str(adv_passing_table)))[0]
            
            # Print column names for debugging
            if debugging: 
                print(f"Advanced passing table columns: {adv_passing_df.columns.tolist()}")
            
            # Handle multi-level columns if present
            if isinstance(adv_passing_df.columns, pd.MultiIndex):
                # Flatten the columns to single level
                adv_passing_df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in adv_passing_df.columns]
                if debugging:
                    print(f"Flattened advanced passing columns: {adv_passing_df.columns.tolist()}")
        
        # Filter out career summary rows from passing DataFrame - these cause problems!
        if pass_year_col:
            # Identify rows that represent actual years (4-digit number)
            passing_df = passing_df[passing_df[pass_year_col].astype(str).str.match(r'^\d{4}$')]
            if debugging:
                print(f"Filtered passing seasons: {passing_df[pass_year_col].tolist()}")
        
        # Rename columns to standardize them
        passing_df.rename(columns={pass_year_col: 'season'}, inplace=True)
        
        # Start with prefixing the passing columns
        for col in passing_df.columns:
            if col != 'season' and col not in ['Team', 'Lg', 'Pos', 'G', 'GS', 'Age']:
                passing_df.rename(columns={col: f'Pass_{col}'}, inplace=True)
        
        # Now we can use pandas merge to join the tables on season
        final_df = passing_df.copy()
        
        # Find the season columns in the other tables
        rush_season_col = None
        adv_season_col = None
        
        # For rushing table
        if rushing_df is not None:
            # Look for likely season column names
            for col in rushing_df.columns:
                if 'Season' in col or 'season' in col:
                    rush_season_col = col
                    break
            
            # If not found, assume it's first column like in passing table
            if not rush_season_col and len(rushing_df.columns) > 0:
                rush_season_col = rushing_df.columns[0]
                if debugging:
                    print(f"Using first column as rushing season: {rush_season_col}")
            
            # Filter out summary rows
            if rush_season_col:
                # Convert to string and filter for 4-digit years
                rushing_df = rushing_df[rushing_df[rush_season_col].astype(str).str.match(r'^\d{4}$')]
                if debugging:
                    print(f"Filtered rushing seasons: {rushing_df[rush_season_col].tolist()}")
                
                # Rename to standard 'season'
                rushing_df.rename(columns={rush_season_col: 'season'}, inplace=True)
                
                # Prefix all other columns
                for col in rushing_df.columns:
                    if col != 'season' and col not in ['Team', 'Lg', 'Pos', 'G', 'GS', 'Age']:
                        rushing_df.rename(columns={col: f'Rush_{col}'}, inplace=True)
                
                # Merge with final_df
                if debugging:
                    print(f"Merging rushing data on season")
                    print(f"Final DF seasons: {final_df['season'].tolist()}")
                    print(f"Rushing DF seasons: {rushing_df['season'].tolist()}")
                
                # Ensure seasons are strings for merging
                final_df['season'] = final_df['season'].astype(str)
                rushing_df['season'] = rushing_df['season'].astype(str)
                
                final_df = pd.merge(final_df, rushing_df, on='season', how='left')
                
                if debugging:
                    print(f"After rushing merge, DataFrame shape: {final_df.shape}")
        
        # For advanced passing table
        if adv_passing_df is not None:
            # Look for likely season column names
            for col in adv_passing_df.columns:
                if 'Season' in col or 'season' in col:
                    adv_season_col = col
                    break
            
            # If not found, assume it's first column like in passing table
            if not adv_season_col and len(adv_passing_df.columns) > 0:
                adv_season_col = adv_passing_df.columns[0]
                if debugging:
                    print(f"Using first column as advanced passing season: {adv_season_col}")
            
            # Filter out summary rows
            if adv_season_col:
                # Convert to string and filter for 4-digit years
                adv_passing_df = adv_passing_df[adv_passing_df[adv_season_col].astype(str).str.match(r'^\d{4}$')]
                if debugging:
                    print(f"Filtered advanced passing seasons: {adv_passing_df[adv_season_col].tolist()}")
                
                # Rename to standard 'season'
                adv_passing_df.rename(columns={adv_season_col: 'season'}, inplace=True)
                
                # Prefix all other columns
                for col in adv_passing_df.columns:
                    if col != 'season' and col not in ['Team', 'Lg', 'Pos', 'G', 'GS', 'Age']:
                        adv_passing_df.rename(columns={col: f'AdvPass_{col}'}, inplace=True)
                
                # Merge with final_df
                if debugging:
                    print(f"Merging advanced passing data on season")
                    print(f"Final DF seasons: {final_df['season'].tolist()}")
                    print(f"Advanced passing DF seasons: {adv_passing_df['season'].tolist()}")
                
                # Ensure seasons are strings for merging
                final_df['season'] = final_df['season'].astype(str)
                adv_passing_df['season'] = adv_passing_df['season'].astype(str)
                
                final_df = pd.merge(final_df, adv_passing_df, on='season', how='left')
                
                if debugging:
                    print(f"After advanced passing merge, DataFrame shape: {final_df.shape}")
        
        # Add player metadata
        final_df['player_name'] = qb_name
        final_df['player_id'] = qb_id
        if draft_year:
            final_df['draft_year'] = draft_year
        if draft_team:
            final_df['draft_team'] = draft_team
        
        # Calculate total yards if possible
        passing_yds_col = next((col for col in final_df.columns if 'Yds' in col and 'Pass_' in col), None)
        rushing_yds_col = next((col for col in final_df.columns if 'Yds' in col and 'Rush_' in col), None)
        
        if passing_yds_col and rushing_yds_col:
            try:
                # Convert to numeric and calculate total
                final_df[passing_yds_col] = pd.to_numeric(final_df[passing_yds_col], errors='coerce').fillna(0)
                final_df[rushing_yds_col] = pd.to_numeric(final_df[rushing_yds_col], errors='coerce').fillna(0)
                final_df['total_yards'] = final_df[passing_yds_col] + final_df[rushing_yds_col]
                print(f"Calculated total_yards by adding {passing_yds_col} and {rushing_yds_col}")
            except Exception as e:
                print(f"Error calculating total yards: {e}")
        
        # Filter by year if needed
        if draft_year:
            final_df = final_df[final_df['season'].astype(int) >= int(draft_year)]
        
        # Clean up and save
        final_df = final_df.reset_index(drop=True)
        
        # Print debug info
        print(f"Final DataFrame shape: {final_df.shape}")
        
        # Save to CSV
        os.makedirs('QB_Data', exist_ok=True)
        if not final_df.empty:
            final_df.to_csv(csv_file_path, index=False)
            print(f"Data for {qb_name} ({qb_id}) saved to {csv_file_path}")
            print(f"Found {len(final_df)} seasons with {len(final_df.columns)} stats columns")
        else:
            print(f"No data to save for {qb_name}")
        
        return final_df
    
    except Exception as e:
        print(f"Error getting QB seasons for {qb_name} ({qb_id}): {e}")
        import traceback
        traceback.print_exc()
        return None

def pull_updated_QB_data(overwrite=False, target_year=None):
    """
    Retrieves statistics for all first-round quarterbacks and compiles them into two DataFrames:
    1. all_seasons_df - containing all seasons for all QBs
    2. best_seasons_df - containing only each QB's best season by total yards
    
    Args:
        overwrite (bool): If True, force re-scrape even if CSV exists (default: False)
        target_year (int, optional): If provided, only update QBs who played in this year
    
    Returns:
        bool: True if successful, False otherwise
    """
    if os.path.exists("1st_rd_qb_ids.csv"):  # Fixed the typo in filename (1sr → 1st)
        first_round_qbs = pd.read_csv("1st_rd_qb_ids.csv")
        all_seasons_df = pd.DataFrame()
        best_seasons_df = pd.DataFrame()
        
        # Loop through each QB in the first round
        for _, qb in first_round_qbs.iterrows():  # Use proper DataFrame iteration
            qb_stats = get_qb_seasons(
                qb_name=qb['player_name'], 
                qb_id=qb['player_id'], 
                draft_year=qb['draft_year'], 
                draft_team=qb['draft_team'],
                overwrite=overwrite
            )
            
            print(f"Found QB stats for: {qb['player_name']} ({qb['player_id']})")
            
            if qb_stats is not None and not qb_stats.empty:
                # Add to all seasons DataFrame
                if all_seasons_df.empty:
                    print("First QB stats found, creating DataFrame")
                    all_seasons_df = qb_stats.copy()
                else:
                    all_seasons_df = pd.concat([all_seasons_df, qb_stats], ignore_index=True)
                    print(f"Added all {qb['player_name']} stats to global DataFrame")
                
                # Find the best season for the QB (maximum total yards)
                if 'total_yards' in qb_stats.columns:
                    best_season = qb_stats.loc[qb_stats['total_yards'].idxmax()]
                    print(f"Best season for {qb['player_name']} ({qb['player_id']}): {best_season['season']} with {best_season['total_yards']} total yards")
                    
                    # Append the best season to the best_seasons_df DataFrame
                    best_seasons_df = pd.concat([best_seasons_df, pd.DataFrame([best_season])], ignore_index=True)
                else:
                    print(f"No total yards column found for {qb['player_name']}")
                    
            random_sleep()
            
        print("Iterated through the entire 1st round QB list")
        
        # Save the DataFrames to CSV files
        if not all_seasons_df.empty:
            all_seasons_df.to_csv("all_seasons_df.csv", index=False)
            print(f"Saved all seasons data with {len(all_seasons_df)} rows and {len(all_seasons_df.columns)} columns")
        else:
            print("No QB seasons data to save")
            
        if not best_seasons_df.empty:
            best_seasons_df.to_csv("best_seasons_df.csv", index=False)
            print(f"Saved best seasons data with {len(best_seasons_df)} rows")
        else:
            print("No best seasons data to save")
            
        return True
    else:
        print("1st round QB IDs file not found. Please run the script to generate it.")
        return False