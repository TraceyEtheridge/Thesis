import pandas as pd
import numpy as np
import regex as re
from tqdm import tqdm
from nltk.tokenize import word_tokenize # must use this for collocations, spacy tokeniser seems incompatible when calcualting pmi score
import nltk
#nltk.download('punkt')
from nltk.collocations import *

def tokenise_dataframe(df):

    df = df.copy()
    df['tokens'] = ""
    col_index = df.columns.get_loc("tokens")
    
    for idx, article in enumerate(df['content_processed']):
        tokens = word_tokenize(article)
        df.iat[idx, col_index] = list(tokens)
    
    return df
    
def find_colocations(tokens, min_freq=1000, n_bigrams=2500):
    """
    min_freq: minimum number of occurances to be included
    n_bigrans: number of bigrams to return (rated from highest scored using pmi measure)
    """
    
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    #trigram_measures = nltk.collocations.TrigramAssocMeasures()

    finder_bigram = BigramCollocationFinder.from_documents(tokens)
    #finder_trigram = TrigramCollocationFinder.from_documents(tokens)
    
    finder_bigram.apply_freq_filter(min_freq)
    scores = finder_bigram.score_ngrams(bigram_measures.pmi)
    bigrams = finder_bigram.nbest(bigram_measures.pmi, n_bigrams)
    df_bigrams_freq = pd.DataFrame(finder_bigram.ngram_fd.items(), columns=['bigram', 'freq'])
    df_bigrams_freq['pmi_score'] = df_bigrams_freq['bigram'].apply(lambda x: [score[1] for score in scores if score[0] == x][0])

    return bigrams, df_bigrams_freq

def create_bigrams_article(text, bigrams):
    """
    :convert bigrams to one word for an individual article
    used within create_bigrams function
    this method does not allow for priority of bigram, i.e. mr_donald will be made and donald_trump will not
    """

    tokens = text.split(" ")
    tokens_with_bigrams = []
    idx = 0
    while idx < len(tokens) -1:
        if (tokens[idx], tokens[idx+1]) in bigrams:
            bigram = str(tokens[idx]) + '_' + str(tokens[idx+1])
            tokens_with_bigrams.append(bigram)
            idx += 2
        else:
            tokens_with_bigrams.append(tokens[idx])
            idx += 1
    if (tokens[-2], tokens[-1]) not in bigrams:
        tokens_with_bigrams.append(tokens[-1])

    text_clean = (" ").join(tokens_with_bigrams)
    
    return text_clean

def create_bigrams_article_v2(text, bigrams):
    """
    :convert bigrams to one word for an individual article
    used within create_bigrams function
    this method will create full connections bigger than bigrams, i.e. mr_donald_trump
    """

    for bigram in bigrams:
        bigram_str = bigram[0] + ' ' + bigram[1]
        bigram_str_sub = bigram[0] + '_' + bigram[1]
        text = re.sub(bigram_str, bigram_str_sub, text)
    
    return text
        
def create_bigrams(df, min_freq=1000, n_bigrams=2500):
    """
    :convert bigrams to one word for entire df
    """
    
    df = df.copy()
    col_index = df.columns.get_loc('content_processed')
    
    df_tokenised = tokenise_dataframe(df)
    print('df tokenised')
    tokens = df_tokenised['tokens']
    
    bigrams, df_bigrams_freq = find_colocations(tokens, min_freq, n_bigrams)
    print('bigrams found')
            
    for idx, content in tqdm(enumerate(df['content_processed'])):
        text = create_bigrams_article_v2(content, bigrams)
        df.iat[idx, col_index] = text
        #if idx % 50000 == 0:
         #   print(f'{idx} records processed')
            
    return bigrams, df, df_bigrams_freq


complete_preprocessing = True

if complete_preprocessing:
    # method 1 ~ 2.5 hours to run
    # method 2 ~ 12 hours to run
    df_processed = pd.read_pickle('./data/df_processed.pickle')
    bigrams, df_processed_bigrams, df_bigrams_freq = create_bigrams(df_processed)
    df_processed_bigrams.to_pickle('./data/df_processed_bigrams.pickle')
    
    df_bigrams = pd.DataFrame(bigrams, columns=['bigram_w1', 'bigram_w2'])
    df_bigrams['freq'] = ''
    for idx in range(len(df_bigrams)):
        bigram_check = (df_bigrams['bigram_w1'][idx], df_bigrams['bigram_w2'][idx])
        freq = df_bigrams_freq[df_bigrams_freq['bigram'] == bigram_check]['freq'].values[0]
        df_bigrams.loc[idx, 'freq'] = freq
    df_bigrams.to_csv('./data/bigrams/bigrams.csv', index=False)
    df_bigrams_freq.to_csv('./data/bigrams/df_bigrams_freq.csv', index=False)
    
else:
    df_processed_bigrams = pd.read_pickle('./data/df_processed_bigrams.pickle')
    df_bigrams = pd.read_csv('./data/bigrams/bigrams.csv')
    df_bigrams_freq = pd.read_csv('./data/bigrams/df_bigrams_freq.csv')