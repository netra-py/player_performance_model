from django.shortcuts import render
from .models import Teams, Players, Venue
from django.http import HttpResponse, JsonResponse
import numpy as np
import os
import pickle
import joblib
import pandas as pd
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

pkl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'models')

prep_pkl = os.path.join(pkl_path,'preprocessor.pkl')
with open(prep_pkl,'rb') as file:
    prep_model = joblib.load(file)

xgb_pkl = os.path.join(pkl_path,'XGBoost_cricket_model.pkl')
with open(xgb_pkl,'rb') as file:
    xgb_model = pickle.load(file)

def index(request):
    
    teams = Teams.objects.all()
    venues = Venue.objects.all()
    

    return render(request,'index.html',{'teams':teams,'venues':venues})


def get_players_by_team(request):
    team_name = request.GET.get('team_name')
    players = Players.objects.filter(team_name=team_name).values('player_name')
    return JsonResponse(list(players), safe=False)

@csrf_exempt
def final_prediction(request):
    if request.method == 'POST':

        innings = request.POST.get('innings')
        batting_team = request.POST.get('bat_team')
        bowling_team = request.POST.get('bowl_team')
        venue = request.POST.get('ven')
        striker = request.POST.get('player')
        avg_last_5 = request.POST.get('last5_avg')
        avg_runs = request.POST.get('car_avg')
        strike_rate = request.POST.get('car_sr')


        

        # Create DataFrame with column names exactly matching what preprocessor expects
        input_dict = {
            'innings': [innings],
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'venue': [venue],
            'striker': [striker],
            'avg_last_5': [float(avg_last_5)],
            'avg_runs': [float(avg_runs)],
            'strike_rate': [float(strike_rate)]
        }



        input_df = pd.DataFrame(input_dict)

        try:
            input_transformed = prep_model.transform(input_df)
            prediction = xgb_model.predict(input_transformed)
            # print(prediction[0])
            if prediction[0] == 1:
                result = 'Player will perform Well'
            else:
                result = 'Player will not perform Well'

            return JsonResponse({'result': result},safe=False)
        except Exception as e:
            return JsonResponse({'error': str(e)})

        


        