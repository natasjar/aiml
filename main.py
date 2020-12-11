# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:09:11 2020

Model

@author: Natasja Hopkin S18121344
"""

#to split datainto training and test datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#to calculate the accuracy score of the model
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import seaborn as sns
import pandas as pd
import time

import preprocessing as prep

#config settings
import config

def score(m, features_train, target_train, features_test, target_test):
    start = time.time()
    
    #print accuracy score   
    score = m.score(features_test, target_test)*100
    
    end = time.time()
    print("Accuracy Score: ", m)
    print("%.2f" %score)
    target_pred = m.predict(features_test)
    report = classification_report(target_test, target_pred)
    print(report)
    
    print("Time taken: %0.2f seconds \n" %(end-start))
    
def matrix(m, features_test, target_test):
    disp = plot_confusion_matrix(m, features_test, target_test, cmap=plt.cm.PuRd)
    disp.ax_.set_title(m)

def crossv(features, target):
    entries = []
    #for crossvalidation 
    features, target = prep.balance_cut(features, target)
    
    for m in config.ALGORITHMS:
        cv = config.CV
        start = time.time()
        scores = cross_val_score(m, features, target, cv=cv)
        end = time.time()
        print("cross-validated with %d folds: %s" %(cv, m))
        #print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2))
        print("Time taken: %0.2f seconds \n" %(end-start))
        
        for score in scores:
            entries.append((m, score))
    cv = pd.DataFrame(entries, columns=['model', 'score'])
    
    plt.xticks(rotation=90)
    sns.boxplot(y="score", x="model", data=cv)
    
def model():
    df = prep.get_df()
    
    #data slicing is passed from the pca() in preprocessing.py
    features, target = prep.pca_lda(df)

    #splitting into test and training 
    features_train, features_test, target_train, target_test = train_test_split(features,target, test_size = config.TEST_SIZE, random_state = config.RANDOM_STATE)
    
    #balancing training set
    features_train, target_train = prep.balance_smote(features_train, target_train)
    features_test, target_test = prep.balance_cut(features_test, target_test)
    
    for m in config.ALGORITHMS: 
        m.fit(features_train, target_train) 
        # score(m, features_train, target_train, features_test, target_test)
        matrix(m, features_test, target_test)

    # crossv(features, target)

def main():
    #make True for confusion matrix and cross validation scores to print
    all_start = time.time()
    model()
    
    all_end = time.time()
    print("Time taken overall: %0.2f seconds \n" %(all_end - all_start))

if __name__ == "__main__":
    main()
