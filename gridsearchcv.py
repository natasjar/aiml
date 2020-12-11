# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:21:29 2020

GridSearchCV for tuning model hyperparameters

@author: Natasja Hopkin S18121344
"""

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import preprocessing as prep 


def ms():
    models = [
        MLPClassifier(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier()
        ]
    for m in models:
            print("\n", m)
            print(m.get_params().keys())
            

def gridsearch(params, estimator):
    df = prep.get_df()
    features, target = df.drop('is_promoted', axis=1), df["is_promoted"]
    features, target = prep.balance_smote(features, target)
    
    gs = GridSearchCV(
        estimator,
        params,
        cv = 3,
        verbose = 10,
        n_jobs = -1
        )
    gsr = gs.fit(features, target)
    #print(gs.cv_results_)
    print(gsr.best_score_)
    print(gsr.best_estimator_)
    print(gsr.best_params_)
    print()
    

def kn():
    grid_params = {
        'n_neighbors': [3,5,11,19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
        }
    gridsearch(grid_params, KNeighborsClassifier())
    
def dtc():
    params = {'criterion':['gini','entropy'],
              'max_depth':[3,7,9,11,13,15]}
    gridsearch(params, DecisionTreeClassifier())
    
def rfc():
    params = { 
        'n_estimators': [100, 500, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth':[3,7,9,11,13,15]
        }
    gridsearch(params, RandomForestClassifier())


def adabc():
    params = {
        'n_estimators': [10, 20, 50, 200]
        }
    gridsearch(params, AdaBoostClassifier())
    
def run_all():
    kn()
    dtc()
    rfc()
    adabc()
  