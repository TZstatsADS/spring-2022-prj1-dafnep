
#basic
import string
import numpy as np
import pandas as pd
import multidict as multidict

#document processing
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import regex as re
import nltk.data
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer




# # Data processing
#pipeline for preprocessing the sentences each function is named appropriately and is defined separately 
def preprocess_text(text):
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = convert_to_lower(text)
    text = remove_white_space(text)
    trans_table={ord(c):None for c in string.punctuation+string.digits}
    text = tokenize(text, trans_table)
    return text

def convert_to_lower(text):
  return text.lower()

def remove_numbers(text):
  text = re.sub(r'\d+' , '', text)
  return text

def remove_punctuation(text):
     punctuations = '''!()[]{};«№»:'"\,`<>./?@=#$-(%^)+&[*_]~'''
     no_punct = "" 
     for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
     return no_punct

def remove_white_space(text):
  text = text.strip()
  return text

def tokenize(text, trans_table):
#we remove stop words both before and after lemmatization
    english_stemmer = PorterStemmer()
    text = text.lower()
    stops = stopwords.words('english')
    #additional stop words that we removed
    stops.extend(['go','sure','back','say','one','two','say','would','us', 'also' ,'dont' , 'another'                                                                                            
    'without','much','whose','therefor','first','within','yet','this', 'thing' , 'though' , 'may','often'])
    stop_words = set(stops)
    tokens= [world for world in nltk.word_tokenize(text.translate(trans_table)) if len(world)>1]
    tokens = [i for i in tokens if not i in stop_words]
    #lemmatizing words
    pos_map = {'J': 'a', 'N': 'n', 'R': 'r', 'V': 'v'}
    pos_tags_list = pos_tag(tokens)
    lemmatiser = WordNetLemmatizer()
    tokens = [lemmatiser.lemmatize(w.lower(), pos=pos_map.get(p[0], 'v')) for w, p in pos_tags_list]
    #again removal of stop words in the lemmatised text
    result = [i for i in tokens if not i in stop_words]
    return result