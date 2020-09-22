import nltk
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#extract prepocessed data from pickle files

text_file = open("filepath", "rb")





documents = pickle.load(text_file)


tfidf_vec_text = TfidfVectorizer(tokenizer=lambda i:i, min_df = 5, max_df = 0.95,lowercase = False)
tfidf_text_features = tfidf_vec_text.fit_transform(documents)

#testing numpy sort tfidf
tfidf_feat_array = tfidf_text_features.toarray()

#numpy array of the top n weights (the rest is zero)
tfidf_trunc = np.zeros(tfidf_feat_array.shape)

#### most significant original work for project ####################################################################################################

'''
The follwoing code is for the purpose of creating a matrix of the top n most significant workds. The features in the tfidf matrix are sorted by documents.
The reason is to because we would like to derive the most significant words in each document. If we simply sum all the features associated to a word, we may have a large value for a word that is actually not
that significant. To avoid this, if we order the words by significance in each document and then place this word in its corresponding index in the zero matrix (tfidf_trunc) then we will have the top n most significant words per document. We then sort this matrix to get the most significant weights overall.  
'''
n = 50
for i_doc in range(tfidf_feat_array.shape[0]):
    #array of sorted indexes ordered by tfidf value
    sorted_per_doc = tfidf_feat_array[i_doc,:].argsort()[::-1]
    for i_best in range(n):
        tfidf_trunc[i_doc,sorted_per_doc[i_best]] = tfidf_feat_array[i_doc,sorted_per_doc[i_best]]

tfidf_text_vocab = np.array(tfidf_vec_text.get_feature_names())
tfidf_sorted_text_weights = np.argsort(tfidf_trunc.sum(axis=0))[::-1]
#weights sorted by most_sig
sorted_mostsig_weights = np.sort(tfidf_trunc.sum(axis=0))[::-1][:n]





#################################COUNTVECTORIZER##########################################################################
count_vec_text = CountVectorizer(tokenizer=lambda i:i, min_df = 5, max_df = 0.95, lowercase = False)
count_text_features = count_vec_text.fit_transform(documents)

count_text_vocab = np.array(count_vec_text.get_feature_names())
count_sorted_text_weights = np.argsort(count_text_features.toarray().sum(axis=0))[::-1]


#n assigned at top
#top_n_text_count = count_text_vocab[count_sorted_text_weights][:n]


#top_n_title_tfidf = tfidf_title_vocab[tfidf_sorted_title_weights][:n]


def make_wordcloud(top_n_words, sorted_weights, backgound_color = 'white', max_words = 50 , width = 400, height = 400, random_state=1):
    '''
    Generated word cloud from a list of top n number of words.
    Params:
        numpy array top_n_words: numpy_array of words
        string background color
        int max_no_words for word cloud
        int width, height and random_state
    '''
    #generate word cloud from most common words
    phrase_dict = {}
    for i in range(len(top_n_words)):
        phrase_dict[top_n_words[i]] = sorted_weights[i]
    #generate word cloud from most common words                                                                                                                                                                   
    #wc_text = ' '.join([word for word in top_n_words.tolist()])
    wc = WordCloud(background_color="white", max_words = max_words, width = width, height = height
                   ,random_state=random_state).generate_from_frequencies(phrase_dict)

    #plt.title("")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return phrase_dict

def make_barchart(vocab, weights, x_label='Weights', y_label='Vocabulary', align = 'center', alpha=0.5):
    '''
    Make bar chart of word frequency of top n words
    String x_label, y_label, title
    tuple vocab
    list weights
    String align
    float alpha
    '''
    y_pos = np.arange(len(vocab))

    plt.barh(y_pos, weights, align = align, alpha=alpha)
    plt.yticks(y_pos, vocab)
    plt.xlabel(x_label)
    #plt.ylabel(y_label)
    #plt.title(title)

    plt.show()


#calling make__bar chart with vocab, weights params given
top_n_text_count = count_text_vocab[count_sorted_text_weights][:n]
top_n_text_tfidf = tfidf_text_vocab[tfidf_sorted_text_weights][:n]

vocab_tfidf = top_n_text_tfidf.tolist()[:50]
weights_tfidf = np.sort(tfidf_trunc.sum(axis=0))[::-1][:50].tolist()
#title_tfidf = 'Top 15 words using tfidf'


vocab_count = top_n_text_count[:25].tolist()
weights_count = np.sort(count_text_features.toarray().sum(axis=0))[::-1][:25].tolist()
#title_count = 'Top 15 words using Count'

make_barchart(tuple(vocab_count), weights_count, x_label='Frequency')
make_barchart(tuple(vocab_tfidf), weights_tfidf)
#make_wordcloud(top_n_text_tfidf)




#generate wordCloud
phrase_dict = make_wordcloud(tfidf_text_vocab[tfidf_sorted_text_weights][:50], sorted_mostsig_weights[:50])

'''
a=[1,2,3]
b=[4,5,6] 
c = [a, b] 
with open("list1.txt", "w") as file:
    for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
'''

import pandas as pd

#make dict intp dataframe
s = pd.Series(phrase_dict, name = 'Weights Sum')
s.index.name = 'Tfidf'
s.reset_index()
s.to_csv(r'/media/francesita/Expansion Drive/Menopause/Figures/tfidf_unigrams.csv', index=True, header=True)

print(s)




import csv
f = open('pos_tweet_users.txt', 'w')

with f:
#    writer = csv.writer(f)
    for i in vocab_tfidf:
        f.write(i + '\n')

    
