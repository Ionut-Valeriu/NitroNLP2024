import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec


def get_numbers_1(text):
  numbers = []
  for token in word_tokenize(text):
    if token.isdigit():
      numbers.append(token)
  return numbers

df = pd.read_csv("train.csv")

lastV = []
for name in df["title"]:
    titlu = get_numbers_1(name)
    lastV.append(titlu)
print(lastV)  
# word_tokenize(titlu)