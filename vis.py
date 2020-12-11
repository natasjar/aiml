# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:08:05 2020

First looks

@author: Natasja Hopkin S18121344
"""

import config
import pandas as pd
pd.set_option('display.max_columns', 14)
#use value_counts() from pandas library to find the distribution of unique values for each (or some)features

df = pd.read_csv(config.DATASET)

def vis():
    print(df.shape)
    print(df.info)
    print(df.describe(include='all')) #leaves lots of NaNs, but includes more useful data with non-numerics
    
    print()
    
    # for col in df.columns:
    #     print('\n', df[col].value_counts())
    print(df["is_promoted"].value_counts)
    
    print()
    
    print("Null value counts")
    print(df.isnull().sum()) #2409 in education and 4124 in previous year rating
    
    

vis()

    