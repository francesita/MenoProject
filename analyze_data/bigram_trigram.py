import nltk
from nltk.corpus import stopwords
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Extract preprocessed data from pickle files

text_file = open("filepath", "rb")


bodyText = pickle.load(text_file)





def detokenize(bodyText):
    bodyText_join = []
    for document in bodyText:
        text = [nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(document)]
        bodyText_join.append(text)
    return bodyText_join

#reassign bodyText
documents = detokenize(bodyText)

'''
def extract_tokens(dic):
                                                                                                                                                                                                              
   # Get tokens from dictionary                                                                                                                                                                                     
    
    tokens = []
    for key in dic:
        tokens.append(dic[key]['tokens'])
    return tokens



documents = extract_tokens(dic)
'''
bodyText= detokenize(documents)
#making bodyText 1d list
bodyText = np.array(bodyText).flatten().tolist()


tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(2,2), min_df =3)
tfidf_features = tfidf_vec.fit_transform(bodyText)

tfidf_vocab = np.array(tfidf_vec.get_feature_names())

def get_most_significant(tfidf_features, n):
    '''
    Gives most significant n n-grams per article
    n is int of most significant words
    tfidf_features is feature matrix 
    '''
    
    #tfidf feature array
    feature_array = tfidf_features.toarray()
    #numpy zero array of same shape as feature_array where we will fill most significant per article
    tfidf_trunc = np.zeros(feature_array.shape)
    
    for i_doc in range(feature_array.shape[0]):
        #gives us the sorted indexes of most sig words from largest to smallest 
        sorted_doc_array = feature_array[i_doc,:].argsort()[::-1]
        for i_best in range(n):
            tfidf_trunc[i_doc, sorted_doc_array[i_best]]=feature_array[i_doc, sorted_doc_array[i_best]]

    return tfidf_trunc

#n corresponds to an int - number of sig n_grams you want to find
n = 100

#numpy array of same shape as tfidf_features, but contains most significant 
most_sig = get_most_significant(tfidf_features, n)

#sum of most significant n-grams
sorted_mostsig_weightIndex = np.argsort(most_sig.sum(axis=0))[::-1]

#most_significant "vocabulary" or n_grams
most_sig_phrases = tfidf_vocab[sorted_mostsig_weightIndex][:n]

#weights sorted by most_sig
sorted_mostsig_weights = np.sort(most_sig.sum(axis=0))[::-1][:n]

print(n)
print(most_sig_phrases)
print(most_sig.shape)


def make_wordcloud(top_n_words, sorted_weights, backgound_color = 'white', max_words = 200 , width = 1500, height = 1000, random_state=1):
    '''           
    Generated word cloud from a list of top n number of words.                                                                                                                                     
    Params:                                                                                                                                                                                                       
        numpy array top_n_words: numpy_array of words                                                                                                                                                             
        string background color                                                                                                                                                                                   
        int max_no_words for word cloud                                                                                                                                                                           
        int width, height and random_state                                                                                                                                                                        
    '''
    phrase_dict = {}
    for i in range(len(top_n_words)):
        phrase_dict[top_n_words[i]] = sorted_weights[i]
    #generate word cloud from most common words                                                                                                                                                                   
    #wc_text = ' '.join([word for word in top_n_words.tolist()])
    wc = WordCloud(background_color="white", max_words = max_words, width = width, height = height
                   ,random_state=random_state).generate_from_frequencies(phrase_dict)

    plt.title("Most frequently occuring bigrams")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    return phrase_dict
#generate wordCloud
phrase_dict = make_wordcloud(most_sig_phrases[:100], sorted_mostsig_weights[:100])


import pandas as pd

#make dict intp dataframe
s = pd.Series(phrase_dict, name = 'Weights Sum')
s.index.name = 'Bigrams'
s.reset_index()
s.to_csv(r'filename', index=True, header=True)

print(s)
