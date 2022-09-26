####################################################################################################
# Author: Eugene Baraka Baraka                                                                     #
# Date: 2022-09-09                                                                                 #
# Purpose: Helper functions for NLP model to predict suicidal ideation.                            #
####################################################################################################

## Basic data manipulation and visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Text analysis and preprocessing
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import math, re, string
import nltk
# from nltk.tokenize import word_tokenize
# from nltk.probability import FreqDist
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

## Machine Learning
from sklearn import preprocessing, model_selection, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipelin

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
