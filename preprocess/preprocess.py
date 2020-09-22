import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import TreebankWordTokenizer
import pandas as pd
import numpy as np
import string
import pickle
import json
import os
import re

'''
Code for processing articles extracted online
'''

def preprocess_singledoc(text):
    
    '''
    Function to preprocess documents. Preprocessing includes removing symbols in replace_with_space, removing any punctuation and removing stopwords. Returns tokenized text
    '''
    
    #variable to be used for text preprocessing function preprocess_text       
    replace_with_space = re.compile('[/(){}\[\]\|@,;]')
    symbols_to_remove = re.compile("[^a-z _]+")
    stop_words = set(stopwords.words('english'))
    added_stopwords = ['one','says','like', 'said', 'say', 'would', 'go', 'guardian', 'saga'], 
    stop_words = set(list(stop_words) + added_stopwords)

    #list where preprocessed text will be stored. 
    preprocessed_text = []
    tknzr = TreebankWordTokenizer()
    lmtzr = WordNetLemmatizer()
    #stemmer = PorterStemmer()
    
    text = text.lower()
    text = re.sub(replace_with_space, " ", text)
    text_tokens = tknzr.tokenize(text)
    text_tokens = [token for token in text_tokens if token not in stop_words]
    text_tokens = [lmtzr.lemmatize(token) for token in text_tokens]
    text = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(text_tokens)
    text = re.sub(symbols_to_remove, "", text)
    text_tokens = tknzr.tokenize(text)
        
    return text_tokens

def preprocess_text(list_text):
    
    '''
    Function to preprocess documents. Preprocessing includes removing symbols in replace_with_space, removing any punctuation and removing stopwords. Return a 2-d list of preprocessed text
    '''
    
    #variable to be used for text preprocessing function preprocess_text       
    replace_with_space = re.compile('[/(){}\[\]\|@,;]')
    symbols_to_remove = re.compile("[^a-z _]+")
    stop_words = set(stopwords.words('english'))
    added_stopwords = ['one','says','like', 'said', 'say', 'would', 'go']
    stop_words = set(list(stop_words) + added_stopwords)

    #list where preprocessed text will be stored. 
    preprocessed_text = []
    tknzr = TreebankWordTokenizer()
    lmtzr = WordNetLemmatizer()
    #stemmer = PorterStemmer()
    for sentence in list_text:
        text = sentence.lower()
        text = re.sub(replace_with_space, " ", text)
        text_tokens = tknzr.tokenize(text)
        text_tokens = [token for token in text_tokens if token not in stop_words]
        text_tokens = [lmtzr.lemmatize(token) for token in text_tokens]
        text = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(text_tokens)
        text = re.sub(symbols_to_remove, "", text)
        text_tokens = tknzr.tokenize(text)
        preprocessed_text.append(text_tokens)
        
    return preprocessed_text


file_bt = open('positive_body.pkl','rb')
bodyText= pickle.load(file_bt)
preprocessed_text= preprocess_text(bodyText)

#combining tokenized docs intp whole, untokenized but preprocessed doc
join_body =  sum(preprocessed_text, [])
#join_title = sum(title, [])

bt_whole = ' '.join(token for token in join_body)


'''
Saving preprocessed data into pickle files for later use
'''



file_body = open("positive_preprocessed_tknzd.pkl", "wb")



pickle.dump(preprocessed_text, file_body)

