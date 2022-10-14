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
import spacy
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing, model_selection, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline
from sklearn import preprocessing, model_selection, feature_extraction, linear_model, tree, neighbors, ensemble, svm, metrics
import xgboost as xgb
from tqdm.auto import tqdm
from langdetect import detect
import collections

# 2. Text Analysis and preprocessing

## Feature Extraction

### NER (would be better to write a class for this??)

def ner_displacy(txt, ner=None, lst_tag_filter=None, title=None, serve=False):
    ner = spacy.load("en_core_web_lg") if ner is None else ner
    doc = ner(txt)
    doc.user_data["title"] = title
    if serve == True:
        spacy.displacy.serve(doc, style="ent", options={"ents":lst_tag_filter})
    else:
        spacy.displacy.render(doc, style="ent", options={"ents":lst_tag_filter})


def utils_ner_text(txt, ner=None, lst_tag_filter=None, grams_join="_"):
    ## apply model
    ner = spacy.load("en_core_web_lg") if ner is None else ner
    entities = ner(txt).ents

    ## tag text
    tagged_txt = txt
    for tag in entities:
        if (lst_tag_filter is None) or (tag.label_ in lst_tag_filter):
            try:
                tagged_txt = re.sub(tag.text, grams_join.join(tag.text.split()), tagged_txt) #it breaks with wild characters like *+
            except Exception as e:
                continue

    ## extract tags list
    if lst_tag_filter is None:
        lst_tags = [(tag.text, tag.label_) for tag in entities]  #list(set([(word.text, word.label_) for word in ner(x).ents]))
    else: 
        lst_tags = [(word.text, word.label_) for word in entities if word.label_ in lst_tag_filter]

    return tagged_txt, lst_tags
        
        
def utils_lst_count(lst, top=None):
    dic_counter = collections.Counter()
    for x in lst:
        dic_counter[x] += 1
    dic_counter = collections.OrderedDict(sorted(dic_counter.items(), key=lambda x: x[1], reverse=True))
    lst_top = [ {key:value} for key,value in dic_counter.items() ]
    if top is not None:
        lst_top = lst_top[:top]
    return lst_top

def utils_ner_features(lst_dics_tuples, tag):
    if len(lst_dics_tuples) > 0:
        tag_type = []
        for dic_tuples in lst_dics_tuples:
            for tuple in dic_tuples:
                type, n = tuple[1], dic_tuples[tuple]
                tag_type = tag_type + [type]*n
                dic_counter = collections.Counter()
                for x in tag_type:
                    dic_counter[x] += 1
        return dic_counter[tag]   #pd.DataFrame([dic_counter])
    else:
        return 0


def add_ner_spacy(dtf, column, ner=None, lst_tag_filter=None, grams_join="_", create_features=True):
    tqdm.pandas()
    ner = spacy.load("en_core_web_lg") if ner is None else ner
    ## tag text and exctract tags
    print("--- tagging ---")
    dtf[[column+"_tagged", "tags"]] = dtf[[column]].progress_apply(lambda x: utils_ner_text(x[0], ner, lst_tag_filter, grams_join), 
                                                          axis=1, result_type='expand')

    ## put all tags in a column
    print("--- counting tags ---")
    dtf["tags"] = dtf["tags"].progress_apply(lambda x: utils_lst_count(x, top=None))
    
    ## extract features
    if create_features == True:
        print("--- creating features ---")
        ### features set
        tags_set = []
        for lst in dtf["tags"].tolist():
            for dic in lst:
                for k in dic.keys():
                    tags_set.append(k[1])
        tags_set = list(set(tags_set))
        ### create columns
        for feature in tags_set:
            dtf["tags_"+feature] = dtf["tags"].progress_apply(lambda x: utils_ner_features(x, feature))
    return dtf

def plot_tags(tags, top=30, figsize=(10,5)):   
    tags_list = tags.sum()
    map_lst = list(map(lambda x: list(x.keys())[0], tags_list))
    dtf_tags = pd.DataFrame(map_lst, columns=['tag','type'])
    dtf_tags["count"] = 1
    dtf_tags = dtf_tags.groupby(['type','tag']).count().reset_index().sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("Top frequent tags", fontsize=12)
    sns.barplot(x="count", y="tag", hue="type", data=dtf_tags.iloc[:top,:], dodge=False, ax=ax)
    ax.set(ylabel=None)
    ax.grid(axis="x")
    plt.show()
    return dtf_tags

def extract_lengths(data, col):
    
    tqdm.pandas()
    new_data = data.copy()
    new_data['word_count'] = new_data[col].progress_apply(lambda x: len(nltk.word_tokenize(str(x))))
    new_data['char_count'] = new_data[col].progress_apply(lambda x: sum(len(word) for word in nltk.word_tokenize(str(x))))
    new_data['sentence_count'] = new_data[col].progress_apply(lambda x: len(nltk.sent_tokenize(str(x))))
    new_data['avg_word_len'] = new_data['char_count']/new_data['word_count']
    new_data['avg_sent_len'] = new_data['word_count']/new_data['sentence_count']

    print("Characteristics of text")
    print()
    print(new_data[['char_count', 'word_count', 'sentence_count', 'avg_word_len', 'avg_sent_len']].describe().T[['min', 'mean', 'max']])

    return new_data


## Univariate and Bivariate Visualizations

def plot_distributions(data, x, y = None, maxcat = 20, top_n = None, bins = None, figsize = (15, 9), title = None, xlabel = None, ylabel = None, normalize = True):
    
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
        print("Warning: No list of stopwords provided, so using default NLTK stopwords")
        stopwords = stopwords_list()
        tokenized_txt = [w for w in txt.split() if w not in stopwords]

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


def append_clean_text(data, column, rm_regex = None, punctuations = True, lower = True, contractions = True, list_stopwords = None, stem = True, lemma = False, remove_na=True):
    dtf = data.copy()

    ## apply preprocess
    dtf = dtf[ pd.notnull(dtf[column]) ]
    dtf[column+"_clean"] = dtf[column].apply(lambda x: text_preprocessing(x, rm_regex, punctuations, lower, contractions, list_stopwords, stem, lemma))
    
    ## residuals
    dtf["check"] = dtf[column+"_clean"].apply(lambda x: len(x))
    if dtf["check"].min() == 0:
        print("--- found NAs ---")
        print(dtf[[column,column+"_clean"]][dtf["check"]==0].head())
        if remove_na is True:
            dtf = dtf[dtf["check"]>0] 
            
    return dtf.drop("check", axis=1)