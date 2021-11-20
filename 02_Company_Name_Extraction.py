import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os
import logging
import types

# NER Imports
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import Counter

# Note to do - need to add time element

def log_newline(self, how_many_lines=1):
    file_handler = None
    if self.handlers:
        file_handler = self.handlers[0]

    # Switch formatter, output a blank line
    file_handler.setFormatter(self.blank_formatter)
    for i in range(how_many_lines):
        self.info('')

    # Switch back
    file_handler.setFormatter(self.default_formatter)

def logger_ner():
    
    log_file = os.path.join('./data/ner', 'ner.log')
    print('log file location: ', log_file)
    
    log_format= '%(asctime)s - %(levelname)s - [%(module)s]\t%(message)s'
    formatter = logging.Formatter(fmt=(log_format))
    
    fhandler = logging.FileHandler(log_file)
    fhandler.setFormatter(formatter)
    
    logger = logging.getLogger('ner')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fhandler)
    logger.default_formatter = formatter
    logger.blank_formatter = logging.Formatter(fmt="")
    logger.newline = types.MethodType(log_newline, logger)
    
    return logger
    


# Load company name stopwords
def load_stopwords(file_path, sheet_name):    
    org_stopwords = pd.read_excel(filepath_stopwords, sheet_name = sheet_name, header = None)
    org_stopwords = list(set(sorted(org_stopwords[0])))
    new_stopwords = ['company']
    org_stopwords = org_stopwords + new_stopwords
    org_stopwords = sorted([word.lower() for word in org_stopwords])
    print('number of phrases in org stopwords list: ', len(org_stopwords))
    return org_stopwords

def load_listed_companies(file_path, sheet_name, countries):
    companies_list = pd.read_excel(file_path, sheet_name = sheet_name)
    # Drop companies names that are 2 or 3 characters long
    companies_list = companies_list.drop(labels = companies_list[companies_list['name'].str.len() < 4].index)
    # Countries of interest
    companies_list = companies_list[companies_list['country'].isin(countries)]
    return companies_list


def match_companies(df, companies_list_common_words, companies_list, file_name='df_ner_temp'):
    
    df = df.copy()
    df['filtered_names'] = ''
    df['filtered_names_match'] = ''
    col_index_fn = df.columns.get_loc('filtered_names')
    col_index_fnm = df.columns.get_loc('filtered_names_match')
    #print_on = True
    print_on = False

    for idx, names in tqdm(enumerate(df['org_names'])):
        if print_on:
            print('\n', idx, names, 'labels: ', df.iat[idx,col_index_onl])
        filtered_names = []
        filtered_names_match = []
        for name in names:

            # Rule 1 - NER name only 1 word -> set of common words  -> similarity ratio on set
            if len(name.split(' ')) == 1:
                processor = lambda x: set(x.lower().split()) - set(companies_list_common_words)
                processor_2 = lambda x: x.lower()
                best_match = process.extractOne(query=name, choices=companies_list['name'], processor=processor, scorer=fuzz.token_sort_ratio, score_cutoff=98)
                #best_match = process.extractOne(query=name, choices=companies_list['name'], processor=processor_2, scorer=fuzz.ratio, score_cutoff=75)
                if best_match:
                    if print_on:
                        print('rule1: ', name, '-', best_match, best_match[1])
                    filtered_names.append(name)
                    filtered_names_match.append(best_match[0])            

            # Rule 2 - NER name longer than 1 word plus part of name in stopwords list -> similarity ratio on set
            elif set(name.split(' ')).intersection(org_stopwords):
                processor_2 = lambda x: x.lower()
                best_match = process.extractOne(query=name, choices=companies_list['name'], processor=processor_2, scorer=fuzz.token_set_ratio, score_cutoff=95)
                if best_match:
                    if print_on:
                        print('rule2: ', name, '-', best_match[0], best_match[1])
                    filtered_names.append(name)
                    filtered_names_match.append(best_match[0])

            else:
                best_match = process.extractOne(query=name, choices=companies_list['name'], scorer=fuzz.partial_ratio, score_cutoff=96)
                if best_match:
                    if print_on:
                        print(f'best match: {best_match}')
                        print('rule3: ', name, '-', best_match[0], best_match[1])
                    filtered_names.append(name)
                    filtered_names_match.append(best_match[0])

        df.iat[idx,col_index_fn] = sorted(list(set(filtered_names)))
        df.iat[idx,col_index_fnm] = sorted(list(set(filtered_names_match)))
    
    return df

def match_companies_titles(df, file_name='df_ner_temp'):
    
    df = df.copy()
    df['filtered_names_titles'] = ''
    df['filtered_names_titles_match'] = ''
    col_index_fn = df.columns.get_loc('filtered_names_titles')
    col_index_fnm = df.columns.get_loc('filtered_names_titles_match')
    #print_on = True
    print_on = False

    for idx, names in tqdm(enumerate(df['org_names_titles'])):
        if print_on:
            print('\n', idx, names, 'labels: ', df.iat[idx,col_index_onl])
        filtered_names = []
        filtered_names_match = []
        for name in names:

            # Rule 1 - NER name only 1 word -> set of common words  -> similarity ratio on set
            if len(name.split(' ')) == 1:
                processor = lambda x: set(x.lower().split()) - set(companies_list_common_words)
                processor_2 = lambda x: x.lower()
                best_match = process.extractOne(query=name, choices=companies_list['name'], processor=processor, scorer=fuzz.token_sort_ratio, score_cutoff=98)
                #best_match = process.extractOne(query=name, choices=companies_list['name'], processor=processor_2, scorer=fuzz.ratio, score_cutoff=75)
                if best_match:
                    if print_on:
                        print('rule1: ', name, '-', best_match, best_match[1])
                    filtered_names.append(name)
                    filtered_names_match.append(best_match[0])            

            # Rule 2 - NER name longer than 1 word plus part of name in stopwords list -> similarity ratio on set
            elif set(name.split(' ')).intersection(org_stopwords):
                processor_2 = lambda x: x.lower()
                best_match = process.extractOne(query=name, choices=companies_list['name'], processor=processor_2, scorer=fuzz.token_set_ratio, score_cutoff=95)
                if best_match:
                    if print_on:
                        print('rule2: ', name, '-', best_match[0], best_match[1])
                    filtered_names.append(name)
                    filtered_names_match.append(best_match[0])

            else:
                best_match = process.extractOne(query=name, choices=companies_list['name'], scorer=fuzz.partial_ratio, score_cutoff=96)
                if best_match:
                    if print_on:
                        print(f'best match: {best_match}')
                        print('rule3: ', name, '-', best_match[0], best_match[1])
                    filtered_names.append(name)
                    filtered_names_match.append(best_match[0])

        df.iat[idx,col_index_fn] = sorted(list(set(filtered_names)))
        df.iat[idx,col_index_fnm] = sorted(list(set(filtered_names_match)))
        
        #if idx % 5000 == 0:
         #   df.to_pickle('./data/' + file_name + '.pickle')
    
    return df

def process_ner(full_data, partial_data, first_pass, articles_to_process, logger):
    if full_data:
        df_ner_clean = pd.read_pickle('./data/ner/df_ner_clean_articles_titles_211120.pickle')
        df_ner_matched = match_companies(df_ner_clean, companies_list_common_words, companies_list)
        df_ner_matched.to_pickle('./data/ner/df_ner_matched.pickle')  
    elif partial_data:
        if first_pass:
            df_ner_matched = pd.read_pickle('./data/ner/df_ner_clean_articles_titles_211120.pickle') # only for first time
            start_index = 0
        else:
            df_ner_matched = pd.read_pickle('./data/ner/df_ner_matched_211120.pickle')
            start_index = df_ner_matched[df_ner_matched['filtered_names_match'].isna()].iloc[0].name
            print(f'start index name: {start_index}')
            start_index = df_ner_matched.index.get_loc(start_index)
            print(f'start index loc: {start_index}')
        end_index = start_index + articles_to_process
        print(f'end index: {end_index}')
        logger.info(f"start index: {start_index}, end_index: {end_index}")
        df_ner_matched_subset = df_ner_matched[start_index:end_index]
        df_ner_matched_subset = match_companies(df_ner_matched_subset, companies_list_common_words, companies_list)
        df_ner_matched_new = df_ner_matched[:start_index].append(df_ner_matched_subset).append(df_ner_matched[end_index:])
        df_ner_matched_new.to_pickle('./data/ner/df_ner_matched_211120.pickle')
    else:
        df_ner_matched = pd.read_pickle('./data/ner/df_ner_matched_211120.pickle')


def process_ner_titles(full_data, partial_data, first_pass, articles_to_process, logger):
    if full_data:
        df_ner_clean = pd.read_pickle('./data/ner/df_ner_clean_titles.pickle')
        df_ner_matched = match_companies(df_ner_clean)
        df_ner_matched.to_pickle('./data/ner/df_ner_matched_titles.pickle')  
    elif partial_data:
        if first_pass:
            df_ner_matched = pd.read_pickle('./data/ner/df_ner_clean_titles.pickle') # only for first time
            start_index = 0
        else:
            df_ner_matched = pd.read_pickle('./data/ner/df_ner_matched_titles_211030.pickle')
            start_index = df_ner_matched[df_ner_matched['filtered_names_titles_match'].isna()].iloc[0].name
            print(f'start index name: {start_index}')
            start_index = df_ner_matched.index.get_loc(start_index)
            print(f'start index loc: {start_index}')
        end_index = start_index + articles_to_process
        print(f'end index: {end_index}')
        logger.info(f"start index: {start_index}, end_index: {end_index}")
        df_ner_matched_subset = df_ner_matched[start_index:end_index]
        df_ner_matched_subset = match_companies_titles(df_ner_matched_subset)
        df_ner_matched_new = df_ner_matched[:start_index].append(df_ner_matched_subset).append(df_ner_matched[end_index:])
        df_ner_matched_new.to_pickle('./data/ner/df_ner_matched_titles_211030.pickle')
    else:
        df_ner_matched = pd.read_pickle('./data/ner/df_ner_matched_titles_211030.pickle')


logger = logger_ner()

# Import listed companies
filepath_companies = './data/ner/company_names_listed.xlsx'
sheet_name = 'company_names'
countries_included = ['United States', 'Canada', 'Australia', 'United Kingdom']

companies_list = load_listed_companies(filepath_companies, sheet_name, countries_included)

print(companies_list['country'].value_counts())
print(len(companies_list))

# Get list of most common words in company names, e.g. 'International', 'Company', 'Org'
# Used in matching formula
results = Counter()
companies_list['name'].str.lower().str.split().apply(results.update)
companies_list_common_words = sorted([k for k, v in results.items() if v > 99])

# Load stopwords
filepath_stopwords = './data/ner/company_stopwords.xlsx'
sheet_name = 'all'
org_stopwords = load_stopwords(filepath_stopwords, sheet_name)

# total processing will take 9 days so processed in stages
full_data = False
partial_data = True
articles_to_process = 5000
first_pass = False
num_runs = 10

for n in range(num_runs):
    print(f'run number {n + 1} of {num_runs}')
    logger.info(f"run number {n + 1} of {num_runs}")
    process_ner(full_data, partial_data, first_pass, articles_to_process, logger)
    #process_ner_titles(full_data, partial_data, first_pass, articles_to_process, logger)
