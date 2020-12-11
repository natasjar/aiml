# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:31:36 2020

Data pre-processing

@author: Natasja Hopkin S18121344
"""

#to preprosess data
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
le = preprocessing.LabelEncoder()

#For over/under sampling to balance 
#conda install -c conda-forge imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours as ENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

import config

#import dataset
def df_conf():
    return pd.read_csv(config.DATASET)

def nulls(df):
    #as previous year rating nulls match with length_of_service = 1, we can give them averages
    #average taken from df.describe()
    df["previous_year_rating"].fillna(3.33, inplace = True)
    df["education"].fillna("Other", inplace = True)
    
    df = df.drop('employee_id', axis=1)
    return df.dropna()

#use LabelEncoder()from scikit-learnlibrary to convert string feature values to numbers
#transform string labels to numerical
def to_num(df):
    cols = [
        'department',
        'region',
        'education',
        'gender',
        'recruitment_channel'
        ]
    for col in (df.loc[:,cols]).columns:
            df[col] = le.fit_transform(df[col])
    return df

def pca_lda(df):
    features = df.drop('is_promoted', axis=1)
    target = df['is_promoted']
    
    #Standardizing the values
    features = StandardScaler().fit_transform(features)  
    
    #PCA
    pca = PCA(n_components=0.8)
    features = pca.fit_transform(features)
    
    lda = LDA()
    features = lda.fit(features, target).transform(features)
    
    #eigenvalues, eigenvectors and scree plot & principle compontent plot in graphs.py
    return features, target

    
def balance_smote(features, target):   
    #only balancing training data so test data doesn't overfit
    oversample = SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=config.RANDOM_STATE)
    undersample = ENN(n_neighbors=5)
    pipeline = make_pipeline(oversample, undersample)
    features, target = pipeline.fit_resample(features, target)
    
    return features, target

def balance_cut(features, target):
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=config.RANDOM_STATE)
    features, target = undersample.fit_resample(features, target)   
    
    return features, target

#returns fully preprocessed dataframe  
def get_df():
    df = df_conf()
    df = nulls(df)
    df = to_num(df)
    return df
