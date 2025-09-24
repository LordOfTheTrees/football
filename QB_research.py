import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import PFR_Tools as PFR
import random
import os
from IPython.display import display
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY

if __name__ == "__main__":
    # Uncomment these if you need to regenerate the data
    # PFR.get_player_ids()
    # PFR.update_qb_ids()
    # PFR.pull_updated_QB_data()
    # Initialize the client with your API key
    #client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    #response = client.messages.create(
    #model="claude-sonnet-4-20250514",  # Specify the model
    #max_tokens=1000,  # Maximum tokens in the response
    #temperature=0.7,  # Controls randomness (0-1)
    #system="You are a helpful assistant",  # Optional system prompt
    #messages=[
    #   {
    #        "role": "user",
    #        "content": "who is the best quarterback in the NFL based on sustained peak combined total yards per season?"
    #    }
    #]
    #)

    # Access the response
    #print(response.content[0].text)
    
    try:
        if os.path.exists("best_seasons_df.csv"):
            print("starting best QB seasons pull")
            df_QB_best_seasons = pd.read_csv("best_seasons_df.csv")
            #print( df_QB_best_seasons.shape, "columns: ", df_QB_best_seasons.columns)
            df_QB_best_seasons.sort_values(by=['total_yards'], ascending=[False], inplace=True)
            display(df_QB_best_seasons.head(5)[['player_name', 'draft_team', 'season', 'total_yards', 'AdvPass_Air Yards_IAY/PA', 'Pass_Cmp%', 'Pass_Rate', 'Pass_Yds', 'Rush_Rushing_Yds', 'Rush_Rushing_TD']])
    except Exception as e:
        print(f"Error running the best QB seasons pull: {e}")

    try:
        if os.path.exists("QB_contract_data.csv"):
            print(f"\nstarting most expensive QB contract data pull (apy as % of cap at signing)")
            df_QB_contract_data = pd.read_csv("QB_contract_data.csv")
            #print( df_QB_contract_data.shape, "columns: ", df_QB_contract_data.columns)
            #Player,Team,Year,Years,,Value,APY,Guaranteed,,APY as % Of,,Inflated,Inflated,Inflated
            #df_QB_contract_data = df_QB_contract_data[df_QB_contract_data['Year'] == '2024'] # take only the 2024 year
            df_QB_contract_data.sort_values(by=['APY as % Of'], ascending=[False], inplace=True)
            display(df_QB_contract_data.head(5)[['Player', 'Team', 'Year', 'APY as % Of', 'Value', 'Guaranteed']])
            print("\nlooking for Deshaun Watson contract details")
            display((df_QB_contract_data[df_QB_contract_data['Player']=="Deshaun Watson"])[['Player', 'Team', 'Year', 'APY as % Of', 'Value', 'Guaranteed']])
    except Exception as e:
        print(f"Error running the most expensive QB contracts pull: {e}")

    try:
        if os.path.exists("season_averages.csv"):
            print(f"\nstarting best season averages pull")
            df_season_averages = pd.read_csv("season_averages.csv")
            #print( df_season_averages.shape, "columns: ", df_season_averages.columns)
            #Year,Teams,PF,Total_Yards,Plays,Y/P,TO,FL,1stD,Pass_Cmp,Pass_Att,Pass_Yds,Pass_TD,Int,NY/A,Pass_1stD,Rush_Att,Rush_Yds,Rush_TD,Rush_Y/A,Rush_1stD,Penalties,Pen_Yds,1stPy,Drives,Drive_Sc%,Drive_TO%,Drive_Plays,Drive_Yds,Drive_Pts
            df_season_averages.sort_values(by=['Total_Yards'], ascending=[False], inplace=True)
            display(df_season_averages.head(5)[['Year', 'Y/P', 'Total_Yards', 'Pass_Yds', 'NY/A', 'Rush_Y/A']])
    except Exception as e:
        print(f"Error running the season averages pull: {e}")

    try:
        if os.path.exists("season_records.csv"):
            print(f"\nstarting best season records pull")
            df_season_records = pd.read_csv("season_records.csv")
            #print( df_season_records.shape, "columns: ", df_season_records.columns)
            #Rk,Season,Team,W,G,W,L,T,W-L%,Pts,PtsO,PtDif
            df_season_records.sort_values(by=['W-L%'], ascending=[False], inplace=True)
            display(df_season_records.head(5)[['Season', 'Team', 'PtDif', 'W-L%']])
    except Exception as e:
        print(f"Error running the season records pull: {e}")