# Project checklist

## Supervised ML

1. Build a Twitter crawler or modify the already built crawler 
   - This will need a well-researched set of words that are most likely related to depression (find an existing dataset that is labelled and uses twitter data). 
   - Make sure you have at least a million data points 
   - word pool ([Janshisky et al. 2013](https://doi.org/10.1027/0227-5910/a000234)): 
   “suicidal; suicide; kill myself; my suicide note; my suicide letter; end my life; never wake up; can't go on; not worth living; ready to jump; sleep forever; want to die; be dead; better off without me; better off dead; suicide plan; suicide pact; tired of living; don't want to be here; die alone; go to sleep forever”

2. Label the data since some of tweets are not related to suicide 
   - This is a very hectic process  
   - May be use other research’s data (will I use this to train my model? Will I need to collect my own data in this case?) 
   - How big of a dataset would I need for training, validation, and testing? 

3. Data cleaning and preprocessing 
   - Deal with HTML text and other characters 
   - Convert all to lower case 
   - Any more steps for preprocessing? 

4. What models should I plan to use? May be do many of them and compare accuracy and applicability?  
   - Naïve Bayse 
   - Work my way up to neural networks (can use an already built CNN) 

### Steps
Train a model for data that's available 
Paper published in the Nature journal 
Classify whether a text is self-harm or not 
How companies' like FB and twitter solve that problem (self-harm) 
Novelty: train a model on available data, and scrape your own data and try to predict them... 


## Unsupervised ML or Semi-supervised

1. Semantic similarity measures; clustering
   - Is this possible for tweets? This would be a very feasible and easy way to categorize data (ex. Risk vs. No Risk)
   - May be manually label suicidal and non-suicidal for Risk category? This would save a lot of time
   - Then use NLP to predict suicidal ideation/thoughts

