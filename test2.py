import csv
import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from gensim.models import Word2Vec

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

lemmatizer = WordNetLemmatizer()

df = pd.read_csv('train.csv')

def get_numbers_1(text):
  numbers = []
  for token in word_tokenize(text):
    if token.isdigit():
      numbers.append(token)
  return numbers

def return_tokens(one_content):
    try :
        tokens = [lemmatizer.lemmatize((word)) for word in word_tokenize(one_content) if word not in punctuation]
        tokens = [token for token in tokens if token!="" and len(token)>1 and token not in stopwords.words("romanian")]
        return tokens
    except:
        return []




contents = df["title"].values
list_with_tokens = [return_tokens(content) for content in contents[:10]]
print(list_with_tokens)




# fDist = FreqDist()

# for token in tokens:
#     fDist[token]+=1

#print(fDist.most_common(10))
#print(len(fDist))

#print(ne_chunk(pos_tag(list(fDist.keys()))))


#text = [[word for word in tokens[i:i+20]] for i in range(0,len(tokens),+20)]

# text = [[word for word in tokens]]
# vectorizer = Word2Vec(text, min_count=1).wv
# print(vectorizer.key_to_index)
# [print(x) for x in vectorizer.most_similar("day")]
# #print(vectorizer.similarity ("human", "computer"))
