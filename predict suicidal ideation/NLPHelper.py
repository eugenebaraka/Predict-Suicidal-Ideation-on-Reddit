####################################################################################################
# Author: Eugene Baraka                                                                            #
# Date: 2022-09-09                                                                                 #
# Purpose: Helper functions for NLP model to predict suicidal ideation.                            #
####################################################################################################

####################################################################################################
#                             1. Import libraries                                                  #
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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from langdetect import detect, detect_langs
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
#                             2. Text analysis and Preprocessing                                                     #
####################################################################################################

# A. Univariate and Bivariate Visualizations

def plot_distributions(data, x, y = None, maxcat = 20, top_n = None, bins = None, figsize = (15, 9), title = None, xlabel = None, ylabel = None, normalize = True):
    
    # Univariate Analysis
    if y is None: 
        fig, ax = plt.subplots(figsize = figsize)
        fig.suptitle(title, fontsize = 20)

        ## Plot categorical variables
        if data[x].nunique() <= maxcat:
            if top_n is None:
                data[x].value_counts(normalize = normalize).plot(kind = "barh").grid('x')
                ax.set(ylabel = xlabel)
            else:  
                data[x].value_counts(normalize = normalize).sort_values(ascending = False).head(top_n).plot(kind = "barh").grid('x')
                ax.set(ylabel = xlabel)
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







import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)
text = 'This is an english text.'
doc = nlp(text)
print(doc._.language)



## Detect langauage
def detect_language(data, col):
    tqdm.pandas()

    def get_lang_detector(nlp, name):
        return LanguageDetector()

    nlp = spacy.load("en_core_web_sm")
    Language.factory("language_detector", func = get_lang_detector)
    nlp.add_pipe('language_detector', last = True)
    data['lang'] = data[col].progress_apply(lambda x: nlp(x)._.language['language'])
    






## Detect Language
# def detect_language(data, column):
#     tqdm.pandas()
#     data['lang'] = data[column].progress_apply(lambda x: TextBlob(x).detect_language())

# C. Sentiment Analysis 

def get_sentiment(data, col, algo = 'vader'):

    """
    Computing sentiment using Vader, or TextBlob
    """
    tqdm.pandas()
    new_data = data.copy()

    if algo == 'vader':
        sid = SentimentIntensityAnalyzer()
        new_data['sentiment'] = new_data[col].progress_apply(lambda x: sid.polarity_scores(str(x))['compound'])

    elif algo == 'textblob':
        new_data['sentiment'] = new_data[col].progress_apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    else:
        print("Please select a valid algorithm")

    print(data[['sentiment']].describe().T)

    return new_data


# D. Create stopwords

def stopwords_list(langs = ['english'], add_words = [], remove_words = []):
    
    for lang in langs:
        list_stopwords = set(nltk.corpus.stopwords.words(lang))

    list_stopwords = list_stopwords.union(add_words)
    list_stopwords = list(set(list_stopwords) - set(remove_words))

    return sorted(list(set(list_stopwords)))


# E. Define contractions and their expansion

## dictionary of common English contractions and their full meaning

"""" Dictionary created from https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions"""
contractions_dict = {
"ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
"'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
"didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
"hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he he will have",
"he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "how're": "how are",
"I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have",
"i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
"it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
"ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
"mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
"oughtn't": "ought not", "oughtn't've": "ought not have","shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
"she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
"shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "that'd": "that would",
"that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is",
"they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have","they're": "they are",
"they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
"we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
"what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
"who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
"won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have","y'all": "you all",
"y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
"you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
}


# F. Expand contractions in data


def expand_contractions(text: str, contractions_dict=contractions_dict) -> str:
    """
    Takes in a string and a dictionary of contractions and their expansions 
    Returns an expanded string. 
    """
    re_pattern = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    def replace(match):
        return contractions_dict[match.group(0)]

    return re_pattern.sub(replace, text)


# G. Text Preprocessing

def text_preprocessing(txt, rm_regex = None, punctuations = True, lower = True, contractions = True, list_stopwords = None, stem = True, lemma = False):
    ## Remove patterns from text
    if rm_regex is not None: 
        for regex in rm_regex:
            txt = re.sub(regex, "", txt)

    ## Remove punctuation and lower text
    # txt = txt.translate(str.maketrans('', '', string.punctuation)) if punctuations is True else txt
    txt = re.sub(r'[^\w\s]', '', txt)
    txt = txt.lower() if lower is True else txt

    ## Expand contractions
    tqdm.pandas()
    print("Expanding contractions...")
    print()
    txt = txt.progress_apply(lambda x: expand_contractions(x))

    ## Tokenize text
    if list_stopwords is not None:
        tokenized_txt = [w for w in txt.split() if w not in list_stopwords]
    else:
        print("Warning: No list of stopwords provided")
        tokenized_txt = txt.split()

    if stem is True & lemma is True:
        print("Warning: It is not recommended to both stem and lemmatize. Are you sure you want to continue?")
    ## Stemming
    if stem is True:
        porter_stemmer = nltk.stem.porter.PorterStemmer()
        tokenized_txt = [porter_stemmer.stem(word) for word in tokenized_txt]

    ## Lemmatization
    if lemma is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        tokenized_txt = [lem.lemmatize(word) for word in tokenized_txt]

    ## Join words again

    txt = " ".join(tokenized_txt)

    return txt