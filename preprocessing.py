import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
import joblib

dataset = pd.read_csv("Zadatak1.csv")
X = []
for i in range(len(dataset)):
    temp = dataset['text'][i]
    temp = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', temp, flags=re.MULTILINE)
    
    keyword = str(dataset['keyword'][i])
    if (keyword != 'nan'):
        temp += keyword
    X.append(temp)    

y = dataset['target'] 


lemmatizer = WordNetLemmatizer()

#funkcija koju koristi lemmatizer da bi pravilno radio
def get_wordnet_pos(word):
    
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
    }

    return tag_dict.get(tag, wordnet.NOUN)


tweets = []
for i in range(len(X)):
    tweet = str(X[i])
    tweet = " ".join(str(tweet).split('\\n'))

    tweet = re.sub('\W',' ', tweet)
    tweet = re.sub('\s[^a-zA-Z]+\s',' ', tweet)

    tweet = re.sub('\s+',' ', tweet)
    tweet = tweet.lower()

    tweet = " ".join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(tweet)])

    tweets.append(tweet)

X = np.array(tweets)
joblib.dump(X, 'X.joblib')
joblib.dump(y, 'y.joblib')



