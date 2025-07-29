from django.urls import re_path,path
from predictor.views import index, get_players_by_team, final_prediction

urlpatterns = [
    path('', index, name='index'),
    path('get-players/', get_players_by_team, name='get_players'),
    path('final_prediction/', final_prediction, name='final_prediction'),
]