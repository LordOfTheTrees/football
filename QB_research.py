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
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    response = client.messages.create(
    model="claude-3-7-sonnet-20250219",  # Use the latest model
    max_tokens=1000,  # Maximum tokens in the response
    temperature=0.7,  # Controls randomness (0-1)
    system="You are a helpful assistant",  # Optional system prompt
    messages=[
        {
            "role": "user",
            "content": "who is the best quarterback in the NFL based on sustained peak combined total yards per season?"
        }
    ]
    )

    # Access the response
    print(response.content[0].text)

       
    try:
        if os.path.exists("best_seasons_df.csv"):  # Fixed the typo in filename (1sr → 1st)
            df = pd.read_csv("best_seasons_df.csv")
            print(df.shape, "columns: ", df.columns)
            df.sort_values(by=['total_yards'], ascending=[False], inplace=True)
            display(df.head(5)[['player_name', 'draft_team', 'season', 'total_yards', 'AdvPass_Air Yards_IAY/PA', 'Pass_Cmp%', 'Pass_Rate', 'Pass_Yds', 'Rush_Rushing_Yds', 'Rush_Rushing_TD']])
    except Exception as e:
        print(f"Error running the script: {e}")