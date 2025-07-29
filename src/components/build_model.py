import pandas as pd

import os
import sys
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from utils import *
from exceptions import CustomException
from logger import logging
from data_ingestion_transformation import *

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib


class build_model():
    def __init__(self):
        self.features = ['innings','batting_team','bowling_team','venue','striker','avg_last_5','avg_runs','strike_rate']
        self.cat_features = ['batting_team','bowling_team','venue','striker']
        self.num_features = ['innings','avg_last_5','avg_runs','strike_rate']
        self.output = 'player_performance'
        self.folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def prepare_for_model(self):
        try:
            # read_data
            obj = data_ing()
            df = obj.data_read_convert()
            # print(df.head())
            
            # split data into train and test
            X = df[self.features]
            y = df[self.output]



            X_train, X_test, y_train , y_test  = train_test_split(X,y,test_size=0.3,random_state=45)

            # create a pipeline for scaling the data and converting categorical columns into numerical ones
            try:
                
                num_pipeline = Pipeline(
                    steps = [
                        ('imputer',SimpleImputer(strategy='median')),
                        ('scaler',StandardScaler())
                    ]
                )

                cat_pipeline = Pipeline(
                    steps = [
                        ('imputer',SimpleImputer(strategy='most_frequent')),
                        ('one_hot_encoder',OneHotEncoder()),
                        ('scaler',StandardScaler(with_mean=False))
                    ]
                )

                

                preprocessor = ColumnTransformer(
                    [
                        ('num_pipeline',num_pipeline,self.num_features),
                        ('cat_pipeline',cat_pipeline,self.cat_features)
                    ]
                )

                preprocessor.fit(X_train)
                joblib.dump(preprocessor, rf'{self.folder_path}\models\preprocessor.pkl')

                X_train = preprocessor.fit_transform(X_train)
                X_test = preprocessor.transform(X_test)
                logging.info('Numerical columns scaling completed')

                logging.info('Categorical columns encoding completed')

                

            except Exception as e:
                raise CustomException(e,sys)
            
            logging.info('Training and Testing datasets are prepared')
            return [X_train,y_train,X_test,y_test]

            
        except Exception as e:
            raise CustomException(e,sys)
        

