import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from human_like_requester import HumanLikeRequester
import os
import random

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

# Function to get all seasons for a QB with comprehensive stats
def get_qb_seasons(qb_name, qb_id, draft_year=None, draft_team=None):
    """
    Retrieves comprehensive career statistics for a quarterback.
    
    This function scrapes passing, rushing, and advanced passing statistics 
    for each season of a quarterback's career from Pro Football Reference.
    
    Args:
        qb_name (str): The quarterback's full name
        qb_id (str): The quarterback's unique Pro Football Reference ID
        draft_year (int, optional): Filter to include only seasons from this year onward
        draft_team (str, optional): The team that drafted the quarterback
        
    Returns:
        DataFrame: A pandas DataFrame containing season-by-season statistics for the quarterback,
                  including:
                  - Basic passing stats (completions, attempts, yards, TDs, etc.)
                  - Rushing stats
                  - Advanced passing metrics
                  - Added total yards column (passing + rushing)
                  
    Notes:
        - Filters seasons to include only those from 2000 to the current year
        - Combines passing and rushing statistics into a single row per season
        - Drops rows with NaN values
        - Uses HumanLikeRequester to respect server load
    """
    current_year = datetime.now().year
    seasons = []
    Player_url = f"https://www.pro-football-reference.com/players/{qb_id[0]}/{qb_id}.htm"
    
    csv_file_path = os.path.join('QB_Data', f'{qb_id}.csv')
    if os.path.exists(csv_file_path):
        print(f"data for {qb_id} already exists. Loading from CSV.")
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            return pd.read_csv(f)
    
    print(f"Scraping player page: {Player_url}")
    try:
        response = requester.get(Player_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with the player's passing stats
        # Try various possible table IDs
        Passing_log_table = soup.find('table', id='Passing')
        
        if Passing_log_table:
            # Find all rows in the table
            rows = Passing_log_table.find('tbody').find_all('tr')
            print(f"Found {len(rows)} rows in the Passing table for {qb_name}")
            seasons.append(rows[0].find_all('th')) #should add the passing header row to the seasons list
            for row in rows:
                # Skip header rows or spacer rows
                if 'class' in row.attrs and ('thead' in row['class'] or 'divider' in row['class']):
                    continue
                
                year_cell = row.find('th', {'data-stat': 'season'})
                if year_cell:
                    year = year_cell.text.strip()
                    # Check if the year is in the range of interest
                    if draft_year and int(year) < draft_year:
                        continue
                    print(f"Processing passing season: {year} for {qb_name}")
                    # add the whole row to the seasons list
                    seasons.append(row)
        else:
            print(f"passing log table not found for {qb_name}")
        
        rushing_log_table = soup.find('table', id='Rushing')
        if rushing_log_table:
            # Find all rows in the table
            rows = rushing_log_table.find('tbody').find_all('tr')
            print(f"Found {len(rows)} rows in the Rushing table for {qb_name}")
            seasons.append('Rushing' + rows[0].find_all('th')[1:]) #should add the rushing header row to the seasons list
            for row in rows:
                # Skip header rows or spacer rows
                if 'class' in row.attrs and ('thead' in row['class'] or 'divider' in row['class']):
                    continue
                year_cell = row.find('th', {'data-stat': 'season'})
                if year_cell:
                    year = year_cell.text.strip()
                    # Check if the year is in the range of interest
                    if draft_year and int(year) < draft_year:
                        continue
                    print(f"Processing rushing season: {year} for {qb_name}")
                    if seasons.find(year) is not None:
                        # Check if the year already exists in the seasons list
                        # If it does, append the rushing stats to that entry
                        rushing_stats = row.find_all('td')
                        for stat in rushing_stats:
                            print(f"adding rushing season to passing season: {year} for {qb_name}")
                            seasons.find(year).append(stat.text.strip())
                    else:
                        print(f"Rushing season not found for {year} for {qb_name}")
                    # add the whole row to the seasons list
        else:
            print(f"Rushing log table not found for {qb_name}")
        
        advanced_passing_log_table = soup.find('table', id='Rushing')
        if advanced_passing_log_table:
            # Find all rows in the table
            rows = advanced_passing_log_table.find('tbody').find_all('tr')
            print(f"Found {len(rows)} rows in the Rushing table for {qb_name}")
            seasons.append(rows[0].find_all('th')) #should add the advanced passing header row to the seasons list
            for row in rows:
                # Skip header rows or spacer rows
                if 'class' in row.attrs and ('thead' in row['class'] or 'divider' in row['class']):
                    continue
                year_cell = row.find('th', {'data-stat': 'season'})
                if year_cell:
                    year = year_cell.text.strip()
                    # Check if the year is in the range of interest
                    if draft_year and int(year) < draft_year:
                        continue
                    print(f"Processing advanced passing season: {year} for {qb_name}")
                    if seasons.find(year) is not None:
                        # Check if the year already exists in the seasons list
                        # If it does, append the rushing stats to that entry
                        advanced_passing_stats = row.find_all('td')
                        for stat in advanced_passing_stats:
                            print(f"adding advanced passing season to passing season: {year} for {qb_name}")
                            seasons.find(year).append(stat.text.strip())
                    else:
                        print(f"advanced passing season not found for {year} for {qb_name}")
                    # add the whole row to the seasons list
        else:
            print(f"Advanced Passing log table not found for {qb_name}")
        seasons_df = pd.DataFrame(seasons)
        seasons_df.columns = seasons_df.iloc[0]  # Set the first row as the header
        seasons_df = seasons_df[1:]  # Remove the header row from the DataFrame
        seasons_df.reset_index(drop=True, inplace=True)  # Reset the index
        seasons_df['player_name'] = qb_name  # Add player name to the DataFrame
        seasons_df['draft_year'] = draft_year  # Add draft year to the DataFrame
        seasons_df['draft_team'] = draft_team  # Add draft team to the DataFrame
        seasons_df['total_yards'] = seasons_df['Yds'] + seasons_df['Rush Yds']  # Add total yards column
        seasons_df['season'] = seasons_df['season'].astype(int)  # Convert year to integer
        seasons_df = seasons_df[seasons_df['season'] >= 2000]  # Filter for years >= 2000
        seasons_df = seasons_df[seasons_df['season'] <= current_year]  # Filter for years <= current year
        seasons_df = seasons_df.dropna()  # Drop rows with NaN values
        seasons_df = seasons_df.reset_index(drop=True)  # Reset the index

        os.makedirs('QB_Data', exist_ok=True)  # Create directory if it doesn't exist
        if seasons_df is not None:
            # Save the seasons DataFrame to a CSV file
            with open(csv_file_path, 'w', encoding='utf-8') as f:
                f.writepd.DataFrame.to_csv(seasons_df, index=False)
            print(f"Data for {qb_id} saved to {csv_file_path}")
        else:
            print(f"QB data pull failed, so write to CSV failed for: {qb_name}, {qb_id}")

    except requests.RequestException as e:
        print(f"No page found for player: {qb_name}, {qb_id} - {e}")

    return seasons_df   