from pro_football_reference_web_scraper import player_game_log as p
import pro_football_reference_web_scraper as pfr
from pro_football_reference_web_scraper import team_game_log as t

#print(pfr.get_player_game_log(player = 'Josh Allen', position = 'QB', season = 2022))

team_log = t.get_team_game_log(team = 'Buffalo Bills', season = 2022)
print(team_log)