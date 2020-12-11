# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:32:19 2020

Config 

@author: Natasja Hopkin S18121344
"""

#importing algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

DATASET = "dataframes/train.csv"
RANDOM_STATE = 10

ALGORITHMS = [
    KNeighborsClassifier(),
    KNeighborsClassifier(metric='euclidean', n_neighbors=3, weights='distance'),
    
    DecisionTreeClassifier(),
    DecisionTreeClassifier(criterion='gini', max_depth=30),
    
    RandomForestClassifier(),
    RandomForestClassifier(max_depth=15, max_features='sqrt', n_estimators=100),
    
    AdaBoostClassifier(),
    AdaBoostClassifier(n_estimators=200),
    
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
    ]

#test size
TEST_SIZE = 0.33

#cv of cross val
CV = 10

