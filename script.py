import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from pro_football_reference_web_scraper import player_game_log as p
from datetime import datetime

# Function to get all first-round QBs since 2000
def get_first_round_qbs():
    first_round_qbs = []
    current_year = datetime.now().year
    
    # Process each draft year from 2000 to current year
    for year in range(2000, current_year):
        print(f"Processing draft year: {year}")
        url = f"https://www.pro-football-reference.com/years/{year}/draft.htm"
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Print out all table ids to help debug
            tables = soup.find_all('table')
            print(f"Found {len(tables)} tables on the page")
            for i, table in enumerate(tables):
                table_id = table.get('id', 'No ID')
                print(f"Table {i+1} ID: {table_id}")
            
            # Try various possible table IDs
            draft_table = None
            possible_ids = ['drafts', 'draft', 'all_drafts', 'all_draft']
            
            for table_id in possible_ids:
                draft_table = soup.find('table', id=table_id)
                if draft_table:
                    print(f"Found draft table with ID: {table_id}")
                    break
            
            # If still not found, try finding by class name
            if not draft_table:
                draft_table = soup.find('table', {'class': 'sortable stats_table'})
                if draft_table:
                    print("Found draft table by class name")
            
            # If still not found, use the first table
            if not draft_table and tables:
                draft_table = tables[0]
                print("Using first table on the page as draft table")
            
            if draft_table and draft_table.find('tbody'):
                # Find all rows in the table
                rows = draft_table.find('tbody').find_all('tr')
                print(f"Found {len(rows)} rows in the draft table")
                
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
                
            # Sleep to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"Error processing draft year {year}: {str(e)}")
    
    return first_round_qbs

#function to pair player name with their playerID
def get_player_id(player_name):
    # This function should return the player ID based on the player's name
    # For now, we will return a placeholder value
    # In a real implementation, you would query the database or API to get the player ID
    return player_name.replace(" ", "_").lower()  # Placeholder for player ID

# Function to get all seasons for a QB with comprehensive stats
def get_qb_seasons(qb_info):
    qb_name = qb_info['name']
    draft_year = qb_info['draft_year']
    draft_team = qb_info['draft_team']
    current_year = datetime.now().year
    seasons = []
    
    for year in range(draft_year, current_year):
        try:
            # Add delay to avoid rate limiting
            time.sleep(1.5)
            print(f"Trying to get stats for {qb_name} in {year}...")
            
            # Get game log for the season
            ###################################################################################################
                
        except Exception as e:
            print(f"Could not process {qb_name} for {year}: {e}")
    
    return seasons

# Main execution flow
def main():
    # Get list of first-round QBs since 2000
    print("Getting list of first-round QBs...")
    qbs = get_first_round_qbs()
    
    if not qbs:
        print("No first-round QBs found. Please check the web scraping logic.")
        return None, None
    
    print(f"Found {len(qbs)} QBs drafted in the first round since 2000")
    
    # Save the list of QBs to a CSV file
    qbs_df = pd.DataFrame(qbs)
    qbs_df.to_csv('first_round_qbs.csv', index=False)
    
    # Get stats for all seasons for each QB
    all_seasons = []
    
    for qb in qbs:
        print(f"Processing {qb['name']} drafted in {qb['draft_year']} by {qb['draft_team']}...")
        try:
            seasons = get_qb_seasons(qb)
            all_seasons.extend(seasons)
        except Exception as e:
            print(f"Error processing {qb['name']}: {str(e)}")
    
    if not all_seasons:
        print("No season data collected. Please check the player_game_log module.")
        return None, None
    
    # Create DataFrame with all seasons
    all_seasons_df = pd.DataFrame(all_seasons)
    
    # Create DataFrame with best season for each QB (based on total yards)
    best_seasons = []
    for name in all_seasons_df['name'].unique():
        qb_seasons = all_seasons_df[all_seasons_df['name'] == name]
        if not qb_seasons.empty:
            best_season = qb_seasons.loc[qb_seasons['total_yards'].idxmax()]
            best_seasons.append(best_season)
    
    best_seasons_df = pd.DataFrame(best_seasons)
    
    # Sort both DataFrames by total yards (descending)
    all_seasons_df = all_seasons_df.sort_values('total_yards', ascending=False)
    best_seasons_df = best_seasons_df.sort_values('total_yards', ascending=False)
    
    # Save to CSV
    all_seasons_df.to_csv('all_first_round_qb_seasons.csv', index=False)
    best_seasons_df.to_csv('best_first_round_qb_seasons.csv', index=False)
    
    print(f"Total seasons processed: {len(all_seasons_df)}")
    print(f"Best seasons saved for {len(best_seasons_df)} QBs")
    
    return all_seasons_df, best_seasons_df

# Run the main function
if __name__ == "__main__":
    try:
        all_seasons_df, best_seasons_df = main()
        
        if all_seasons_df is not None and best_seasons_df is not None:
            # Display top 10 from each DataFrame with more comprehensive stats
            print("\nTop 10 All-Time QB Seasons by Total Yards:")
            print(all_seasons_df.head(10)[[
                'name', 'season', 'team', 'games', 
                'completions', 'attempts', 'completion_pct', 'pass_yards', 'pass_tds', 'interceptions', 'qb_rating',
                'rush_attempts', 'rush_yards', 'rush_tds',
                'total_yards', 'total_tds', 'total_yards_per_game', 'draft_team'
            ]])
            
            print("\nTop 10 QBs by Best Season Total Yards:")
            print(best_seasons_df.head(10)[[
                'name', 'season', 'team', 'games', 
                'completions', 'attempts', 'completion_pct', 'pass_yards', 'pass_tds', 'interceptions', 'qb_rating',
                'rush_attempts', 'rush_yards', 'rush_tds',
                'total_yards', 'total_tds', 'total_yards_per_game', 'draft_team'
            ]])
    except Exception as e:
        print(f"An error occurred in the main execution: {str(e)}")