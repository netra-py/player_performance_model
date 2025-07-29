import pandas as pd

import os
import sys
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from utils import *


class data_ing():
    def __init__(self):
        pass

    def data_read_convert(self):
        # firstly read full data
        data = read_data('full_data.csv')

        # get over no and ball no
        data[['over','balls']] = data['ball'].astype(str).str.split('.',expand=True)
        data['over'] = data['over'].astype(int)
        data['balls'] = data['balls'].astype(int)
        data['over'] = data['over']+1

        # consider only 1st and 2nd innings
        data = data[data['innings'].isin([1,2])]
        data = data.reset_index(drop=True)

        # is_boundary: 1 if 4 or 6 0 else
        data['is_boundary'] = np.where(data['runs_off_bat'].isin([4,6]),1,0)

        data = data[['match_id','venue','innings','batting_team','bowling_team','striker','runs_off_bat','over','balls','is_boundary']]

        # get total runs and boundaries
        df = data.groupby(['match_id','innings','batting_team','bowling_team','venue','striker']).sum()[['runs_off_bat','is_boundary']].reset_index()

        # get total balls
        df2 = data.groupby(['match_id','innings','batting_team','bowling_team','venue','striker']).count()['balls'].reset_index()
        df = pd.merge(df,df2,on=['match_id','innings','batting_team','bowling_team','venue','striker'],how='left')
        # dataframe is ready for transformation

        # get career strike rate and average runs
        df_grp = df.groupby(['innings','batting_team','striker']).sum()[['runs_off_bat','is_boundary','balls']].reset_index()
        df_grp = df_grp.rename(columns={'runs_off_bat':'total_runs','is_boundary':'total_boundary','balls':'total_balls'})
        df_avg_runs = df[['innings','batting_team','striker','runs_off_bat']].groupby(['innings','batting_team','striker']).mean()['runs_off_bat'].reset_index()
        df_avg_runs = df_avg_runs.rename(columns={'runs_off_bat':'avg_runs'})
        df_grp = pd.merge(df_grp,df_avg_runs,on=['innings','batting_team','striker'],how='left')
        df_grp['strike_rate'] = round(df_grp['total_runs']/df_grp['total_balls']*100,4)
        df_grp = df_grp[['innings', 'batting_team', 'striker','avg_runs', 'strike_rate']]

        # get opposition wise average runs
        df_grp2 = df[['innings','batting_team','bowling_team','striker','runs_off_bat']].groupby(['innings','batting_team','bowling_team','striker']).mean()['runs_off_bat'].reset_index()
        df_grp2 = df_grp2.rename(columns={'runs_off_bat':'opposition_avg_runs'})

        dff_grp = pd.merge(df_grp2,df_grp,on=['innings','batting_team','striker'],how='left')

        # get last 5 innings average runs
        df = pd.merge(df,pd.DataFrame(df[['innings','batting_team','striker']].value_counts().reset_index()),on=['innings','batting_team','striker'],how='left')
        df = df[df['count']>=6]
        df = df.reset_index(drop=True)
        df = df.drop('count',axis=1)
        df = df.sort_values(['innings','batting_team','striker'],ascending=True)
        df = df.reset_index(drop=True)
        df_roll_mean = df.groupby(['innings','batting_team','striker'])['runs_off_bat'].apply(lambda x: x.shift(1).rolling(window=5,min_periods=1).mean()).reset_index()
        df_roll_mean = df_roll_mean.drop_duplicates(['innings','batting_team','striker'],keep='last')
        df_roll_mean = df_roll_mean.rename(columns={'runs_off_bat':'avg_last_5'})
        df_roll_mean = df_roll_mean[['innings','batting_team','striker','avg_last_5']].reset_index(drop=True)
        df = pd.merge(df,df_roll_mean,on=['innings','batting_team','striker'],how='left')

        # final dataframe
        final = pd.merge(df,dff_grp,on=['innings','batting_team','bowling_team','striker'],how='left')
        final = final[['innings', 'batting_team', 'bowling_team', 'venue','striker',
            'avg_last_5',
            'opposition_avg_runs', 'avg_runs', 'strike_rate']]
        final['player_performance'] = np.where(final['opposition_avg_runs']>=30,1,0)
        final = final.drop('opposition_avg_runs',axis=1)

        write_data(final,'data_for_model.csv')

        return final


        

