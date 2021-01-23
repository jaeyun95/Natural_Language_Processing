#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import pandas as pd
import numpy
import json
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

PATH = "./data/"
train_data = pd.read_csv(PATH+'labeledTrainData.tsv/labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)

def preprocessing(review, remove_stopwords = False):
    review_text = BeautifulSoup(review, "html.parser").get_text()
    
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    
    words = review_text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        clean_review = ' '.join(words)
    else:
        clean_review = ' '.join(words)
    return clean_review
    
clean_train_reviews=[]
for review in train_data['review']:
    clean_train_reviews.append(preprocessing(review,True))

