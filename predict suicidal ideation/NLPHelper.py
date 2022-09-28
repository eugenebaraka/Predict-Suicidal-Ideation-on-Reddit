####################################################################################################
# Author: Eugene Baraka                                                                            #
# Date: 2022-09-09                                                                                 #
# Purpose: Helper functions for NLP model to predict suicidal ideation.                            #
####################################################################################################

####################################################################################################
#                             1. Import libraries                                                  #
####################################################################################################

## Basic data manipulation and visualization
from turtle import title
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Text analysis and preprocessing
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import math, re, string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
# from nltk.tokenize import word_tokenize
# from nltk.probability import FreqDist
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

## Machine Learning
from sklearn import preprocessing, model_selection, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline

from sklearn import preprocessing, model_selection, feature_extraction, linear_model, tree, neighbors, ensemble, svm, metrics
import xgboost as xgb

# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier


## Progress bar
from tqdm.auto import tqdm


####################################################################################################
#                             2. Text analysis                                                     #
####################################################################################################

# A. Univariate and Bivariate Visualizations

def plot_distributions(data, x, y = None, max_cat = 20, top = None, bins = None, figsize = (15, 9)):
    # Univariate Analysis

    if y is None: 
        fig, ax = plt.subplots(figsize = figsize)
        fig.suptitle(x, fontsize = 20)

        ## Plot categorical variables
        if data[x].nunique() <= max_cat:
            if top is None:
                data[x].value_counts().sort_values(by = "index").plot(kind = 'bar', legend = False, ax = ax).grid(axis = 'x')
            else: 
                data[x].value_counts().sort_values(by = "index").tail(top).plot(kind = 'bar', legend = False, ax = ax).grid(axis = 'x')

        ## Plot numerical variables
        else:
            sns.distplot(data[x], hist=True, kde=True, kde_kws={'shade':True}, ax=ax)
            ax.grid(True)
            ax.set(xlabel = None, yticklabels = [], yticks = [])

    # Bivariate analysis
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize = figsize)
        fig.suptitle(x, fontsize = 20)
        for i in data[y].unique():
            sns.distplot(data[data[y] == i][x], hist=True, kde=True, bins=bins, hist_kws={'alpha':0.8}, axlabel="", ax=ax[0])
            sns.distplot(data[data[y] == i][x], hist=False, kde=True, hist_kws={'shade':True}, axlabel="", ax=ax[1])

        ax[0].set(title = 'histogram')
        ax[0].grid(True)
        ax[0].legend(data[y].unique())
        ax[1].set(title = 'density')
        ax[1].grid(True)

    plt.show()


# B. Feature Extraction

def extract_lengths(data, col):
    
    tqdm.pandas()
    new_data = data.copy()
    new_data['word_count'] = new_data[col].progress_apply(lambda x: len(nltk.word_tokenize(str(x))))
    new_data['char_count'] = new_data[col].progress_apply(lambda x: sum(len(word) for word in nltk.word_tokenize(str(x))))
    new_data['sentence_count'] = new_data[col].progress_apply(lambda x: len(nltk.sent_tokenize(str(x))))
    new_data['avg_word_len'] = new_data['char_count']/new_data['word_count']
    new_data['avg_sent_len'] = new_data['word_count']/new_data['sentence_count']

    print("Characteristics of text", end= " ")
    print(new_data[['char_count', 'word_count', 'sentence_count', 'avg_word_len', 'avg_sent_len']].describe().T[['min', 'mean', 'max']])

    return new_data



# C. Sentiment Analysis 

def get_sentiment(data, col, algo = 'vader'):

    """
    Computing sentiment using Vader, or TextBlob
    """
    tqdm.pandas()
    new_data = data.copy()

    if algo == 'vader':
        sent = SentimentIntensityAnalyzer()
        new_data['sentiment'] = new_data[col].progress_apply(lambda x: sent.polarity_scores(str(x))['compound'])

    elif algo == 'textblob':
        new_data['sentiment'] = new_data[col].progress_apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    else:
        print("Please select a valid algorithm")

    print(data[['sentiment']].describe().T)

    return new_data