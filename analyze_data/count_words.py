import nltk
import pickle
import numpy as	np
from sklearn.feature_extraction.text import TfidfVectorizer

'''
Simple code to see general word frequency in documents
'''
#extract prepocessed data from pickle files
title_file = open("titles_preprocessed.pkl", "rb")
text_file = open("body_preprocessed.pkl", "rb")

bodyText = pickle.load(text_file)
titles = pickle.load(title_file)

test = bodyText[:5]

def words_frequency(bodyText):
    words_counts = {}
    for document in bodyText:
        for word in document:
            if word in words_counts:
                words_counts[word] = words_counts[word] + 1
            else:
                words_counts[word] = 1
    return words_counts

titles_counts = words_frequency(titles)
bt_counts = words_frequency(bodyText)


most_common_title = sorted(titles_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_bt = sorted(bt_counts.items(), key=lambda x: x[1], reverse=True)[:10]
