import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import PFR_Tools as PFR
import random
import os
from IPython.display import display

if __name__ == "__main__":
    # Uncomment these if you need to regenerate the data
    # PFR.get_player_ids()
    # PFR.update_qb_ids()
    # PFR.pull_updated_QB_data()
    
    try:
        if os.path.exists("best_seasons_df.csv"):  # Fixed the typo in filename (1sr → 1st)
            df = pd.read_csv("best_seasons_df.csv")
            print(df.shape, "columns: ", df.columns)
            df.sort_values(by=['total_yards'], ascending=[False], inplace=True)
            display(df.head(5)[['player_name', 'draft_team', 'season', 'total_yards', 'AdvPass_Air Yards_IAY/PA', 'Pass_Cmp%', 'Pass_Rate', 'Pass_Yds', 'Rush_Rushing_Yds', 'Rush_Rushing_TD']])
    except Exception as e:
        print(f"Error running the script: {e}")