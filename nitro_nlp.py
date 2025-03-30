import re
import nltk
import torch
import tiktoken
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

import random_seed_setter
random_seed_setter.set_random_seeds(seed_value=42)

most_common = [
    "strong",
    "decese",
    "Momente",
    "Udrea",
    "pacienți",
    "teste",
    "em",
    "Top",
    "sînt",
    "decît",
    "no",
    "cîteva",
    "atît",
    "încît",
    "decese",
    "Sursa",
    "CATEGORIA",
    "ăia",
    "Cînd",
    "Foto",
    "confirmate",
    "raportate",
    "Dată",
    ":)",
    "î",
    "COVID19",
    "ha"
]
    
numbers = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

df = pd.read_csv("train.csv")
dtest = pd.read_csv("test.csv")

def nrAparitiiPunctuatie(line, punct):
    return str(line).count(punct)

df["class"] = df["class"].map({True: 1, False: 0})

# df["capitalWord"] = df['con   t'].apply(lambda x: sum(set([str(x).count(word) for word in str(x).split() if word[0].isupper()])))
# dtest["capitalWord"] = dtest['content'].apply(lambda x: sum(set([str(x).count(word) for word in str(x).split() if word[0].isupper()])))

# df["caps"] = df['content'].apply(lambda x: sum([str(x).count(word.upper()) for word in str(x).split()]))
# dtest["caps"] = dtest['content'].apply(lambda x: sum([str(x).count(word.upper()) for word in str(x).split()]))

for tag in punctuation:
    df[tag] = df['content'].apply(lambda x: str(x).count(tag)) + df['title'].apply(lambda x: str(x).count(tag))
    dtest[tag] = dtest['content'].apply(lambda x: str(x).count(tag)) + dtest['title'].apply(lambda x: str(x).count(tag))
    
for tag in most_common:
    df[tag] =  df['content'].apply(lambda x: str(x).count(tag)) + df['title'].apply(lambda x: str(x).count(tag))
    dtest[tag] = dtest['content'].apply(lambda x: str(x).count(tag)) + dtest['title'].apply(lambda x: str(x).count(tag))

df["nrofnum"] = df['content'].apply(lambda x: sum([str(x).count(num) for num in numbers])) + df['title'].apply(lambda x: sum([str(x).count(num) for num in numbers]))
dtest["nrofnum"] = dtest['content'].apply(lambda x: sum([str(x).count(num) for num in numbers])) + dtest['title'].apply(lambda x: sum([str(x).count(num) for num in numbers]))

df["len"] = df['content'].apply(lambda x: len(str(x)))
dtest["len"] = dtest['content'].apply(lambda x: len(str(x)))
df["lenT"] = df['title'].apply(lambda x: len(str(x)))
dtest["lenT"] = dtest['title'].apply(lambda x: len(str(x)))

X = df[list(most_common) + list(punctuation) + ['len'] + ['lenT']]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(100*model.score(X_train, y_train))
print(100*model.score(X_test, y_test), "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(100*model.score(X_train, y_train))
print(100*model.score(X_test, y_test), "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(100*model.score(X_train, y_train))
print(100*model.score(X_test, y_test), "\n")

X_train = df[list(most_common) + list(punctuation) + ['len'] + ['lenT']]
X_test = dtest[list(most_common) + list(punctuation) + ['len'] + ['lenT']]
y_train = df['class']

model = RandomForestClassifier()
model.fit(X_train, y_train)
print(100*model.score(X_train, y_train))


dtest['class'] = model.predict(X_test)
print(dtest['class'].value_counts())

drez = dtest[[
    'id',
    'class'
]]

with open("rez.csv", "w") as  file:
    file.write("id,class\n")
    for index, row in drez.iterrows():
        file.write("\n" + str(row['id']) + "," + str(row['class']) + "\n")
