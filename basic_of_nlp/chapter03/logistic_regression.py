"""
logistic regression
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import json

PATH = "./preprocessing/"
TRAIN_DATA = "train_clean.json"
with open(PATH+TRAIN_DATA, "r") as f:
    train_data = json.load(f)

reviews = train_data['review']
sentiments = list(train_data['sentiment'])
vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3), max_features=5000)

x = vectorizer.fit_transform(reviews)

RANDOM_SEED = 42
TEST_SPLIT = 0.2

y = np.array(sentiments)

x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED)

from sklearn.linear_model import LogisticRegression

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(x_train,y_train)
print("Accuracy: ",lgs.score(x_eval,y_eval))
