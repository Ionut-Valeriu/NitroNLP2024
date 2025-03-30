import re
import nltk
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
from string import punctuation
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("train.csv")

def nrAparitiiPunctuatie(line, punct):
    return str(line).count(punct)

df["class"] = df["class"].map({True: 1, False: 0})

for tag in punctuation:
    df[tag] = df['title'].apply(lambda x: str(x).count(tag))
    max_tag_id = df[tag].idxmax()
    # print(tag, " = ", df.loc[max_tag_id, tag], " on ", df.loc[max_tag_id, 'id'], " is ", df.loc[max_tag_id, 'class'])

X = df[ [
    'id',
    '!',
    'class'
] ]
y = df['!']

model = KNeighborsClassifier()
model.fit(X, y)
print(model.score(X, y))

# model = KNeighborsClassifier()
# print(model.fit(X, y))
# print(model.score(X, y))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# print(model.score(X_train, y_train))
# print(model.score(X_test, y_test))