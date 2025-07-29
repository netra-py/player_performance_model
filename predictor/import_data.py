import pandas as pd
import os
import sys
import django

# Step 1: Add project root to sys.path
# (assumes this script is in 'predictor/' and the project root is one level up)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Step 2: Set the DJANGO_SETTINGS_MODULE correctly
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "player_performance.settings")  # replace if your project name is different

# Step 3: Setup Django
django.setup()

# Step 4: Import models
from predictor.models import Teams, Players, Venue

# Step 5: Load data using pandas
src_path = os.path.join(project_root, 'src', 'raw')

players_path = os.path.join(src_path, 'players.csv')
players = pd.read_csv(players_path)
for _, row in players.iterrows():
    Players.objects.get_or_create(
        team_name=row['team'],
        player_name=row['striker']
    )

venue_path = os.path.join(src_path, 'venue.csv')
venue = pd.read_csv(venue_path)
for _, row in venue.iterrows():
    Venue.objects.get_or_create(
        venue=row['venue'],
    )

teams_path = os.path.join(src_path, 'teams.csv')
teams = pd.read_csv(teams_path)
for _, row in teams.iterrows():
    Teams.objects.get_or_create(
        team_name=row['batting_team'],
    )
