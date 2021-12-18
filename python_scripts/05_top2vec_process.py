import pandas as pd
import os
import collections
import csv
import logging
import numpy as np
import datetime as datetime
import types
import pickle

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from top2vec import Top2Vec

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

find_topics = True
min_count = 25 # ignore words with total frequency less than this
speed = 'deep-learn' # can try 'deep-learn' for possible better embeddings but will take longer
# started deep-learn at 1pm Friday, still going at 6pm Monday start:01.10.21 13:17 finish: 05.10.21 10:15

if find_topics:
    # import lemmatised data
    #with open('data/df.pickle', 'rb') as f:
     #   data_articles = pickle.load(f)
    df_ind = pd.read_pickle('data/df_industry.pickle')

    #documents = [' '.join(article) for article in data_lemmatized]
    documents = list(df_ind['content_processed'])
    print(len(df_ind))
    print(len(documents))
    
    # Find topics
    # ~ 12.5 hours to run on lemmatised data
    model = Top2Vec(documents, workers=4, min_count=min_count, speed=speed)
    model.save('top2vec_deep_ind_consumer_staples_no_marketscreener.model')
else:
    #model = Top2Vec.load('top2vec.model')
    model = Top2Vec.load('top2vec_vocab_limit.model')

print(len(model.topic_words))
print(model._get_word_vectors().shape)