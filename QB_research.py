import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import PFR_Tools as PFR
import random
import os

def main():
    all_seasons_df = pd.DataFrame()
    best_seasons_df = pd.DataFrame()
    years = list(range(2000, current_year + 1))

    random.shuffle(years)
    # Get the current year
    current_year = datetime.now().year
    
    # Loop through each year from 2000 to the current year
    for year in years:
        try:
            PFR.random_sleep()
            # Get the draft class for the year
            draft_class = PFR.get_draft_class(year)
            
            if draft_class is not None:
                if os.path.exists(f"first_round_qbs.csv"):
                    first_round_qbs = pd.read_csv(f"first_round_qbs.csv")
                else:
                    # Get first round QBs from the draft class
                    first_round_qbs = PFR.get_first_round_QBS(draft_class, year)
                    if first_round_qbs:
                        # Process each QB in the first round
                        for qb in first_round_qbs:
                            PFR.random_sleep()
                            # Get the QB name and team
                            qb_name = qb.find('a').text.strip()
                            qb_team = qb.find('td', {'data-stat': 'team'}).text.strip()
                            qb_draft_team = qb.find('td', {'data-stat': 'draft_team'}).text.strip()
                            
                            # Get the QB stats for the season
                            qb_id = PFR.get_player_id(qb_name, year)
                            print(f"Processing {qb_name} ({qb_id}) drafted in year {year}...")
                            qb_stats = PFR.get_qb_seasons(qb_name, qb_id, year)
                            
                            if qb_stats is not None:
                                all_seasons_df = pd.concat([all_seasons_df, qb_stats], ignore_index=True)
                                total_yards = qb_stats[qb_stats.find('Yds')] + qb_stats[qb_stats.find('Rush Yds')]                            
                                # Check if this is the best season for the QB
                                if qb_stats['total_yards'].max() == qb_stats['total_yards']:
                                    best_seasons_df = pd.concat([best_seasons_df, qb_stats], ignore_index=True)
            
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            time.sleep(5)  # Wait before retrying
        except AttributeError as e:
            print(f"Attribute error: {str(e)}")
        except KeyError as e:
            print(f"Key error: {str(e)}")
        except ValueError as e:
            print(f"Value error: {str(e)}")
        except TypeError as e:
            print(f"Type error: {str(e)}")
        except pd.errors.EmptyDataError as e:
            print(f"Empty data error: {str(e)}")
        except pd.errors.ParserError as e:
            print(f"Parser error: {str(e)}")
        except IndexError as e:
            print(f"Index error: {str(e)}")
        except FileNotFoundError as e:
            print(f"File not found error: {str(e)}")
        except PermissionError as e:
            print(f"Permission error: {str(e)}")
        except TimeoutError as e:   
            print(f"Timeout error: {str(e)}")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {str(e)}")
        except requests.exceptions.TooManyRedirects as e:
            print(f"Too many redirects: {str(e)}")
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {str(e)}")
        except requests.exceptions.URLRequired as e:    
            print(f"URL required: {str(e)}")
        except requests.exceptions.MissingSchema as e:
            print(f"Missing schema: {str(e)}")
        except requests.exceptions.InvalidSchema as e:
            print(f"Invalid schema: {str(e)}")
        except requests.exceptions.InvalidURL as e:
            print(f"Invalid URL: {str(e)}")
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"Chunked encoding error: {str(e)}")
        except requests.exceptions.ContentDecodingError as e:   
            print(f"Content decoding error: {str(e)}")
        except requests.exceptions.StreamConsumedError as e:
            print(f"Stream consumed error: {str(e)}")
        except Exception as e:
            print(f"An error occurred while processing year {year}: {str(e)}")
    
    return all_seasons_df, best_seasons_df

# Run the main function
#main()

PFR.get_player_ids()