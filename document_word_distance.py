# Get words in vocab - Takes ~2 hours per topic
import pandas as pd
import os
import collections
import csv
import logging
import numpy as np
import datetime as datetime
import types
import pickle
from tqdm import tqdm
from scipy import spatial

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from top2vec import Top2Vec

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def document_word_comparison(doc, vocab, topic):
    # remove out-of-vocabulary words
    doc = [word for word in doc.split() if word in vocab]
    doc_idx = np.where(np.isin(vocab, doc))[0]
    doc_vecs = vectors[doc_idx]
    cos_dist_doc = [spatial.distance.cosine(vec, topic) for vec in doc_vecs]
    return np.mean(cos_dist_doc)

def topic_vector(keyword, vocab, n_words):
    words_model, word_scores = model.similar_words(keywords=[keyword], num_words=n_words)
    words_model = np.append(keyword, words_model)
    words_idx = np.where(np.isin(vocab, words_model))[0]
    words_vecs = vectors[words_idx]
    return np.mean(words_vecs, axis=0)

model = Top2Vec.load('top2vec_vocab_limit_deep.model')

# Get words in vocab
vocab_length = len(model._get_word_vectors())
vocab = []
for n in range(vocab_length):
    vocab.append(model._index2word(n))
    
vectors = model._get_word_vectors()

#Technology
blockchain = topic_vector("blockchain", vocab, 5)
digitization = topic_vector("digitization", vocab, 5)
machine_learne = topic_vector("machine_learne", vocab, 5)
cloud = topic_vector("cloud", vocab, 5)
iot = topic_vector("iot", vocab, 5)

#Retail
store_closure = topic_vector("store_closure", vocab, 5)
delivery = topic_vector("delivery", vocab, 10)

#Airlines
redundancy = topic_vector("redundancy", vocab, 5)
costcutte = topic_vector("costcutte", vocab, 5)
flight = topic_vector("flight", vocab, 5)
airlines_costs = redundancy + costcutte + flight

#Other
supply_chain = topic_vector("supply_chain", vocab, 10)
shutdown = topic_vector("shutdown", vocab, 5)
outsourcing = topic_vector("outsourcing", vocab, 2)
workfromhome = topic_vector("workfromhome", vocab, 4)
diversification = topic_vector("diversification", vocab, 3)

topics_str = ['blockchain','digitization','machine_learne','cloud','iot','store_closure','delivery','redundancy','costcutte','flight','supply_chain','shutdown','outsourcing','workfromhome','diversification',]
topics_var = [blockchain ,digitization ,machine_learne ,cloud ,iot ,store_closure ,delivery ,redundancy ,costcutte ,flight ,supply_chain ,shutdown ,outsourcing ,workfromhome ,diversification]
topics = dict(zip(topics_str, topics_var))

calculate_doc_distance = True

if calculate_doc_distance:
    tqdm.pandas()
    df = pd.read_pickle('df_doc_embeddings.pickle')
    for key, value in topics.items():
        print(key + '_word')
        df[key + '_word'] = df['content_lemma'].progress_apply(document_word_comparison, args=(vocab, value,))
    df.to_pickle('df_doc_embeddings_word.pickle')
else:
    df = pd.read_pickle('df_doc_embeddings_word.pickle')