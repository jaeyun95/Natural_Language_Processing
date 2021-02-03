"""
data preprocessing code (popcorn data)
"""
import re
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
import os
import torchtext
from torchtext.data import get_tokenizer
import urllib.request

# set parameters
PATH = "./data/"
DATA_OUTPUT_PATH = "./preprocessing/"
TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TRAIN_CLEAN_DATA = 'train_clean.csv'
TEST_INPUT_DATA = 'test_input.npy'
TEST_ID_DATA = 'test_label.npy'
TEST_CLEAN_DATA = 'test_clean.csv'
DATA_CONFIGS = 'data_configs.json'
MAX_SEQUENCE_LENGTH = 174
TOKENIZER = get_tokenizer("basic_english")

def _preprocessing(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        clean_review = ' '.join(words)
    else:
        clean_review = ' '.join(words)
    return clean_review

def make_vocab(review_list):
    token_list = [TOKENIZER(sentence) for sentence in review_list]
    vocab = FreqDist(np.hstack(token_list))
    word_to_index = {word : index + 2 for index, word in enumerate(vocab)}
    word_to_index["<PAD>"] = 1
    word_to_index["<UNK>"] = 0
    return word_to_index

def text_sequence(review_list, word_to_index):
    encoded = []
    for line in review_list:
        temp = []
        for w in line:
            try:
                temp.append(word_to_index[w])
            except KeyError:
                temp.append(word_to_index['<UNK>'])
        if len(temp) < MAX_SEQUENCE_LENGTH:
            temp += [word_to_index['<PAD>']] * (MAX_SEQUENCE_LENGTH - len(line))
        elif len(temp) > MAX_SEQUENCE_LENGTH:
            temp = temp[:MAX_SEQUENCE_LENGTH]
        encoded.append(temp)
    return np.array(encoded)

def clean_reviews(data):
    clean_reviews = []
    for review in data['review']:
        clean_reviews.append(_preprocessing(review, True))
    return clean_reviews


#### preprocessing train data####
# load data file
train_data = pd.read_csv(PATH+'labeledTrainData.tsv/labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
# get clean reviews
clean_train_reviews = clean_reviews(train_data)
#get vocab dictionary
word_to_index = make_vocab(clean_train_reviews)
#get padded sentence
padded_train_sentence = text_sequence(clean_train_reviews, word_to_index)
#prepare saving data
train_df = {'review': clean_train_reviews, 'sentiment': train_data['sentiment']}
train_labels = np.array(train_data['sentiment'])

#### preprocessing test data####
# load data file
test_data = pd.read_csv(PATH+'testData.tsv/testData.tsv',header=0,delimiter='\t',quoting=3)
# get clean reviews
clean_test_reviews = clean_reviews(test_data)
#get padded sentence
padded_test_sentence = text_sequence(clean_test_reviews, word_to_index)
#prepare saving data
test_df = {'review': clean_test_reviews, 'id':test_data['id']}
test_id = np.array(test_data['id'])

#### save data ####
data_configs = {}
data_configs['vocab'] = word_to_index
data_configs['vocab_size'] = len(word_to_index)+1

if not os.path.exists(DATA_OUTPUT_PATH):
    os.makedirs(DATA_OUTPUT_PATH)

json.dump(data_configs, open(DATA_OUTPUT_PATH + DATA_CONFIGS,'w'),ensure_ascii=False)
json.dump(train_df, open(DATA_OUTPUT_PATH + TRAIN_CLEAN_DATA,'w'),ensure_ascii=False)
json.dump(test_df, open(DATA_OUTPUT_PATH + TEST_CLEAN_DATA,'w'),ensure_ascii=False)

np.save(open(DATA_OUTPUT_PATH + TRAIN_INPUT_DATA, 'wb'), padded_train_sentence)
np.save(open(DATA_OUTPUT_PATH + TRAIN_LABEL_DATA, 'wb'), train_labels)
np.save(open(DATA_OUTPUT_PATH + TEST_INPUT_DATA, 'wb'), padded_test_sentence)
np.save(open(DATA_OUTPUT_PATH + TEST_ID_DATA, 'wb'), test_id)
