# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:34:19 2020

Graphs and Visualisations 

@author: Natasja Hopkin S18121344
"""

import matplotlib.pyplot as plt
import preprocessing as prep
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator

import config 

df = prep.df_conf()
df = prep.nulls(df) #full dataset
df_s = df.sample(n=150, random_state=config.RANDOM_STATE) #150 sample for better visuals on scatter plots

def edu():
    c = df['education']
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(c, bins = 7)
    
    plt.title('Educational Backgrounds')
    plt.xlabel(c.name)
    plt.ylabel('#')
    plt.show()
    
def promo():
    c = df['is_promoted']
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(c, bins = 2)
    
    plt.title('Gets the promotion?')
    plt.xlabel(c.name)
    plt.ylabel('#')
    plt.show()

def scatter():
    c1 = df['length_of_service']
    c2 = df['avg_training_score']
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(c1,c2, c='coral')
    
    plt.title('')
    plt.xlabel(c1.name)
    plt.ylabel(c2.name)
    plt.show()
    
def dep_g():
    var = df.groupby(['department','gender']).is_promoted.sum()
    var.unstack().plot(kind='bar',stacked=True,  color=['red','blue'], grid=False)

def bp():
    c1 = df['age']
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    #Variable
    f = dict(markerfacecolor='r', marker='d')
    ax.boxplot(c1, vert=False, flierprops=f) #showfliers=False
    plt.title('Age of Employees (with fliers)')
    plt.xlabel(c1.name)
    
def pairplot():
    sns.pairplot(df)
    
def pca(df):
    df = prep.to_num(df)
    X = df.drop('is_promoted', axis=1)
    y = df['is_promoted']
    X = StandardScaler().fit_transform(X)  # Standardizing the values in X.
    
    pca = PCA(n_components=7)
    prin_comp = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = prin_comp)
    
    print(principalDf)
    
    print('\nEigenvalues \n%s' %pca.explained_variance_)
    print('Eigenvectors \n%s' %pca.components_)
    
    #scree plot
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(pca.explained_variance_)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, linewidth=1, color='r', alpha=0.5)
    plt.title('Scree Plot of PCA: Component Eigenvalues')
    plt.show()
    
    #PCA plot
    plt.figure()
    target_names = [0,1]
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(prin_comp[y == i, 0], prin_comp[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of HR dataset')
