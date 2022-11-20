####################################################################################################
# Author: Eugene Baraka                                                                            #
# Date: 2022-09-09                                                                                 #
# Purpose: Helper functions for NLP model to predict suicidal ideation.                            #
####################################################################################################

# 1. Import libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import spacy
import re
import nltk
import contractions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn import preprocessing, model_selection, feature_extraction, linear_model, tree, neighbors, ensemble, svm, metrics, pipeline, decomposition
import xgboost as xgb
from tqdm.auto import tqdm
from langdetect import detect
import collections
import wordcloud
from tempfile import mkdtemp
from typing import Tuple
import copy as cp

# 2. Text Analysis and preprocessing

## Feature Extraction

# def ner_displacy(txt, ner=None, lst_tag_filter=None, title=None, serve=False):
#     ner = spacy.load("en_core_web_lg") if ner is None else ner
#     doc = ner(txt)
#     doc.user_data["title"] = title
#     if serve == True:
#         spacy.displacy.serve(doc, style="ent", options={"ents":lst_tag_filter})
#     else:
#         spacy.displacy.render(doc, style="ent", options={"ents":lst_tag_filter})


# def utils_ner_text(txt, ner=None, lst_tag_filter=None, grams_join="_"):
#     ## apply model
#     ner = spacy.load("en_core_web_lg") if ner is None else ner
#     entities = ner(txt).ents

#     ## tag text
#     tagged_txt = txt
#     for tag in entities:
#         if (lst_tag_filter is None) or (tag.label_ in lst_tag_filter):
#             try:
#                 tagged_txt = re.sub(tag.text, grams_join.join(tag.text.split()), tagged_txt) #it breaks with wild characters like *+
#             except Exception as e:
#                 continue

#     ## extract tags list
#     if lst_tag_filter is None:
#         lst_tags = [(tag.text, tag.label_) for tag in entities]  #list(set([(word.text, word.label_) for word in ner(x).ents]))
#     else: 
#         lst_tags = [(word.text, word.label_) for word in entities if word.label_ in lst_tag_filter]

#     return tagged_txt, lst_tags
        
        
# def utils_lst_count(lst, top=None):
#     dic_counter = collections.Counter()
#     for x in lst:
#         dic_counter[x] += 1
#     dic_counter = collections.OrderedDict(sorted(dic_counter.items(), key=lambda x: x[1], reverse=True))
#     lst_top = [ {key:value} for key,value in dic_counter.items() ]
#     if top is not None:
#         lst_top = lst_top[:top]
#     return lst_top

# def utils_ner_features(lst_dics_tuples, tag):
#     if len(lst_dics_tuples) > 0:
#         tag_type = []
#         for dic_tuples in lst_dics_tuples:
#             for tuple in dic_tuples:
#                 type, n = tuple[1], dic_tuples[tuple]
#                 tag_type = tag_type + [type]*n
#                 dic_counter = collections.Counter()
#                 for x in tag_type:
#                     dic_counter[x] += 1
#         return dic_counter[tag]   #pd.DataFrame([dic_counter])
#     else:
#         return 0


# def add_ner_spacy(dtf, column, ner=None, lst_tag_filter=None, grams_join="_", create_features=True):
#     tqdm.pandas()
#     ner = spacy.load("en_core_web_lg") if ner is None else ner
#     ## tag text and exctract tags
#     print("Tagging...")
#     dtf[[column+"_tagged", "tags"]] = dtf[[column]].progress_apply(lambda x: utils_ner_text(x[0], ner, lst_tag_filter, grams_join), 
#                                                           axis=1, result_type='expand')

#     ## put all tags in a column
#     print("Tags counting...")
#     dtf["tags"] = dtf["tags"].progress_apply(lambda x: utils_lst_count(x, top=None))
    
#     ## extract features
#     if create_features == True:
#         print("Creating features...")
#         ### features set
#         tags_set = []
#         for lst in dtf["tags"].tolist():
#             for dic in lst:
#                 for k in dic.keys():
#                     tags_set.append(k[1])
#         tags_set = list(set(tags_set))
#         ### create columns
#         for feature in tags_set:
#             dtf["tags_"+feature] = dtf["tags"].progress_apply(lambda x: utils_ner_features(x, feature))
#     return dtf

# def plot_tags(tags, top=30, figsize=(10,5)):   
#     tags_list = tags.sum()
#     map_lst = list(map(lambda x: list(x.keys())[0], tags_list))
#     dtf_tags = pd.DataFrame(map_lst, columns=['tag','type'])
#     dtf_tags["count"] = 1
#     dtf_tags = dtf_tags.groupby(['type','tag']).count().reset_index().sort_values("count", ascending=False)
#     fig, ax = plt.subplots(figsize=figsize)
#     fig.suptitle("Top frequent tags", fontsize=12)
#     sns.barplot(x="count", y="tag", hue="type", data=dtf_tags.iloc[:top,:], dodge=False, ax=ax)
#     ax.set(ylabel=None)
#     ax.grid(axis="x")
#     plt.show()
#     return dtf_tags

def extract_lengths(data, col):
    """Extract word, character, sentence, average word length, and 
    average sentence length for each row.
    
    Arguments:
        data (DataFrame): dataframe name
        col             : column name

    Returns: 
        new_data (DataFrame): data with appended length variables
    
    """
    tqdm.pandas()
    new_data = data.copy()
    new_data['word_count'] = new_data[col].progress_apply(lambda x: len(nltk.word_tokenize(str(x))))
    new_data['char_count'] = new_data[col].progress_apply(lambda x: sum(len(word) for word in nltk.word_tokenize(str(x))))
    new_data['sentence_count'] = new_data[col].progress_apply(lambda x: len(nltk.sent_tokenize(str(x))))
    new_data['avg_word_len'] = new_data['char_count']/new_data['word_count']
    new_data['avg_sent_len'] = new_data['word_count']/new_data['sentence_count']

    print("Characteristics of text:")
    print()
    print(new_data[['char_count', 'word_count', 'sentence_count', 'avg_word_len', 'avg_sent_len']].describe().T[['min', 'mean', 'max']])

    return new_data


## Univariate and Bivariate Visualizations

def plot_distributions(data, x, y = None, maxcat = 20, top_n = None, bins = None, figsize = (15, 9), title = None, xlabel = None, ylabel = None, normalize = True, xlog = False):
    """ Plot distributions of a variable.

    Arguments:
        data (DataFrame): dataframe name
        x               : column to display on the x axis
        y               : column to display on the y axix
        maxcat (int.)   : maximum number of categories to be in the x axis column
        top_n (int.)    : number of categories to plot
        bins (int.)     : number of bins in the barplot
        figsize (tuple) : size of the figure
        title (str.)    : title of the figure
        xlabel (str.)   : x axis label
        ylabel (str.)   : y axis label
        normalize (bool): whether to normalize x axis (by percentage) or not

    Returns:
        plot (matplotlib): histogram or barplot 
    
    """

    # Univariate Analysis
    if y is None: 
        fig, ax = plt.subplots(figsize = figsize)
        fig.suptitle(title, fontsize = 20)

        ## Plot categorical variables
        if data[x].nunique() <= maxcat:
            if top_n is None:
                data[x].value_counts(normalize = normalize).plot(kind = "barh").grid('x')
                ax.set(xlabel = xlabel)
            else:  
                data[x].value_counts(normalize = normalize).sort_values(ascending = False).head(top_n).plot(kind = "barh").grid('x')
                ax.set(xlabel = xlabel)
        ## Plot numerical variables
        else:
            sns.distplot(data[x], hist=True, kde=True, kde_kws={'shade':True}, ax=ax)
            ax.grid(True)
            ax.set(xlabel = None, yticklabels = [], yticks = [])

    # Bivariate analysis
    else:
        fig = plt.figure(figsize = figsize)
        # fig.suptitle(x, fontsize = 20)
        for i in data[y].unique():
            ax = sns.distplot(data[data[y] == i][x], hist=True, kde=True, bins=bins, hist_kws={'alpha':0.8}, axlabel="")
        
        ax.set(title = title)
        ax.grid(True)
        ax.legend(data[y].unique())
        
        if xlog is True:
            ax.set_xscale('log')
        # ax[1].set(title = 'density')
        # ax[1].grid(True)

    plt.show()


# D. Create stopwords

def stopwords_list(langs = ['english'], add_words = [], remove_words = []):
    """ List of stopwords to remove from the text.

    Arguments:
        langs (list): list of languages
        add_words (list): list of words to add
        remove_words (list): list of words to remove

    Returns:
        list_stopwords (list): sorted list of stopwords
    """
    
    for lang in langs:
        list_stopwords = set(nltk.corpus.stopwords.words(lang))

    list_stopwords = list_stopwords.union(add_words)
    list_stopwords = list(set(list_stopwords) - set(remove_words))

    return sorted(list(set(list_stopwords)))


# G. Text Preprocessing

def text_preprocessing(txt, rmv_regex = None,lower=True, stopwords = None):
    """Remove punctuations, numbers, stopwords, expand contractions, 
    convert text to lower case, and tokenize.

    Arguments:
        txt (string): text to be processed
        rmv_regex   : regex for patterns to remove from text
        lower (bool): whether to convert all text to lower case or not
        stopwords (list): stopwords to remove from txt

    Returns:
        txt (list): list of tokenized strings
    """
    if rmv_regex is not None:
        for regex in rmv_regex:
            txt = re.sub(regex, "", txt)
    txt = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', txt) # remove any url

    txt = re.sub(r'[^\w\s]', " ", txt) # remove punctuations
    txt = re.sub(r'[0-9]+', " ", txt)   # remove numbers
    txt = txt.lower() if lower is True else txt # lower case text
    txt = contractions.fix(txt) # expand contractions

    if stopwords is not None:
        tokenized_txt = [w for w in txt.split() if w not in stopwords]
    else:
        stopwords = stopwords_list()
        tokenized_txt = [w for w in txt.split() if w not in stopwords]

    txt = " ".join(tokenized_txt)

    return txt
    
def append_clean_text(data, column, rmv_regex = None, lower = True, stopwords = None, remove_na=True):
    """ Append cleaned text to the original dataset.
    Arguments:
        data (DataFrame): dataframe name
        column          : column name
        rmv_regex       : regex for pattern to be removed from text
        lower (bool)    : whether to convert all text to lower case or not
        stopwords (list): stopwords to remove from txt
        remove_na (bool): whether to remove NAs from the text

    Returns:
        dtf (DataFrame) : dataframe with appended clean text column
    """
    dtf = data.copy()
    tqdm.pandas()
    ## apply preprocess
    dtf = dtf[pd.notnull(dtf[column])]
    dtf[column+"_clean"] = dtf[column].progress_apply(lambda x: text_preprocessing(x, rmv_regex, lower, stopwords))
    
    ## residuals
    dtf["check"] = dtf[column+"_clean"].progress_apply(lambda x: len(x))
    if dtf["check"].min() == 0:
        print("--- found NAs ---")
        print(dtf[[column,column+"_clean"]][dtf["check"]==0].head())
        if remove_na is True:
            dtf = dtf[dtf["check"]>0] 
            
    return dtf.drop("check", axis=1)

## Sentiment Analysis
def add_sentiment(data, column, algo="vader", sentiment_range=(-1,1)):
    """Calculate sentiment analysis for text and add a column to the dataset.
    
    Arguments:
        data (DataFrame): dataframe name
        column          : column name
        algo (str.)     : sentiment analysis algorithm name
        sentiment_range : range of numbers for sentiment

    Returns: 
        dtf (DataFrame) : dataframe with appended clean text column
    """
    dtf = data.copy()
    tqdm.pandas()
    ## calculate sentiment
    if algo == "vader":
        vader = SentimentIntensityAnalyzer()
        dtf["sentiment"] = dtf[column].progress_apply(lambda x: vader.polarity_scores(x)["compound"])
    elif algo == "textblob":
        dtf["sentiment"] = dtf[column].progress_apply(lambda x: TextBlob(x).sentiment.polarity)
    ## rescaled
    if sentiment_range != (-1,1):
        dtf["sentiment"] = preprocessing.MinMaxScaler(feature_range=sentiment_range).fit_transform(dtf[["sentiment"]])
    print(dtf[['sentiment']].describe().T)
    return dtf

## Word Frequency
def word_freq(corpus, ngrams=[1,2,3], top=10, figsize=(10,7), title = None):
    """Find how frequent words are in the text and plot the frequency.
    
    Arguments:
        corpus (str.): text to evaluate word frequency on
        ngrams (list): number of n-grams to look for in corpus
        top (int.)   : number of top words to plot
        figsize (tuple): figure size
        title (str.)   : figure title

    Returns: 
        dtf_freq (DataFrame): dataframe containing frequency of words in each corpus
        """
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    ngrams = [ngrams] if type(ngrams) is int else ngrams
    
    ## calculate
    dtf_freq = pd.DataFrame()
    for n in ngrams:
        dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, n))
        dtf_n = pd.DataFrame(dic_words_freq.most_common(), columns=["word","freq"])
        dtf_n["ngrams"] = n
        dtf_freq = dtf_freq.append(dtf_n)
    dtf_freq["word"] = dtf_freq["word"].apply(lambda x: " ".join(string for string in x) )
    dtf_freq = dtf_freq.sort_values(["ngrams","freq"], ascending=[True,False])
    
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                data=dtf_freq.groupby('ngrams')["ngrams","freq","word"].head(top))
    ax.set(xlabel= "Word frequency")
    plt.suptitle(title)
    plt.title(f"Top {top} words per n-grams")
    plt.show()
    ax.grid(axis="x")
    return dtf_freq

def plot_wordcloud(corpus, max_words=150, max_font_size=35, figsize=(10,10)):
    """Plot word cloud for a corpus.
    
    Arguments: 
        corpus (str.)       : text to plot word cloud for
        max_words (int.)    : exlude words as freqent as this
        max_font_size (int.): the font size for the most frequent word
        figsize             : figure size

    Returns: 
        plot (matplotlib): wordcloud plot
    """
    wc = wordcloud.WordCloud(background_color='white', max_words=max_words, max_font_size=max_font_size)
    wc = wc.generate(str(corpus)) #if type(corpus) is not dict else wc.generate_from_frequencies(corpus)     
    fig = plt.figure(num=1, figsize=figsize)
    plt.axis('off')
    plt.imshow(wc, cmap=None)
    plt.show()

## Cross validation confusion matrix

def cross_validation_(model, kfold : model_selection.KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:

    model_ = cp.deepcopy(model)
    
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba


def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):

    report = metrics.classification_report(actual_classes, predicted_classes, labels=sorted_labels)
    matrix = metrics.confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    print("Classification Report:")
    print()
    print(report)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

    plt.show()


    