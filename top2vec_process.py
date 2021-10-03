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
min_count = 1000 # ignore words with total frequency less than this
speed = 'deep-learn' # can try 'deep-learn' for possible better embeddings but will take longer
# started deep-learn at 1pm, still going at 11am 2 days later

if find_topics:
    # import lemmatised data
    with open('data/data_lemmatized.pickle', 'rb') as f:
        data_lemmatized = pickle.load(f)
    
    data_lemmatized_str = [' '.join(article) for article in data_lemmatized]
    print(len(data_lemmatized))
    print(len(data_lemmatized_str))
    
    # Find topics
    # ~ 12.5 hours to run on lemmatised data
    documents = data_lemmatized_str
    model = Top2Vec(documents, workers=4, min_count=min_count, speed=speed)
    model.save('top2vec_vocab_limit_deep.model')
else:
    #model = Top2Vec.load('top2vec.model')
    model = Top2Vec.load('top2vec_vocab_limit.model')

print(len(model.topic_words))
print(model._get_word_vectors().shape)