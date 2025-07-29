import pandas as pd
import numpy as np
import os

raw_path = rf'{os.path.dirname(os.path.abspath(__file__))}\raw'

def write_data(df,name):
    write_path = os.path.join(raw_path,name)
    df.to_csv(write_path,index=False)
    pass

def read_data(name):
    read_path = os.path.join(raw_path,name)
    df = pd.read_csv(read_path)
    return df

