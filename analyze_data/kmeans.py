from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import nltk 
import pickle

doc_file = open("/media/frances/Expansion Drive/Menopause/code/data/Saga_articles/body_preprocessed_tknzd.pkl", "rb")
documents = pickle.load(doc_file)
doc_file.close()

def detokenize(bodyText):
    bodyText_join = []
    for document in documents:
        text = [nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(document)]
        bodyText_join.append(text)
    return bodyText_join

#reassign bodyText
documents = detokenize(documents)

#making bodyText 1d list
documents = np.array(documents).flatten().tolist()

#token_pattern = r'(?u)\b[A-Za-z]+\b'
tfidf = TfidfVectorizer(min_df=3, lowercase=False, stop_words = 'english', ngram_range=(1,1))
tfidf_features = tfidf.fit_transform(documents)

tfidf_feat_array = tfidf_features.toarray()

#numpy array of the top n weights (the rest is zero)                                                                                                                                                             
tfidf_trunc = np.zeros(tfidf_feat_array.shape)

n_vocab = 300
for i_doc in range(tfidf_feat_array.shape[0]):
    '''
    Gets most significant vocabulary
    '''
    #array of sorted indexes ordered by tfidf value                                                                                                                                                                
    sorted_per_doc = tfidf_feat_array[i_doc,:].argsort()[::-1]
    for i_best in range(n_vocab):
        tfidf_trunc[i_doc,sorted_per_doc[i_best]] = tfidf_feat_array[i_doc,sorted_per_doc[i_best]]

        
tfidf_vocab = np.array(tfidf.get_feature_names())
tfidf_index_sorted_weights = np.argsort(tfidf_trunc.sum(axis=0))[::-1]
tfidf_sorted_weights = np.sort(tfidf_trunc.sum(axis=0))[::-1]

#Array of shape num_of_articles by size of most significant vocabulary, denoted by n
n = 100
most_sig_array = np.zeros((tfidf_trunc.shape[0],n))

for i_word in range(n):
    most_sig_array[:,i_word] = tfidf_feat_array[:,tfidf_index_sorted_weights[i_word]]



#Kmeans classifier

true_k = 3
kmeans = KMeans(n_clusters = true_k, init='k-means++', max_iter = 100, n_init=10)
kmeans.fit(most_sig_array)


labels = kmeans.predict(most_sig_array)


import plot_tools as pt
pt.plot1(labels)

ind0 = np.where(labels==0)[0]
ind1 = np.where(labels==1)[0]
ind2 = np.where(labels==2)[0]




def plot_clusters(w1,w2):
    ind0 = np.where(labels==0)[0]
    ind1 = np.where(labels==1)[0]
    ind2 = np.where(labels==2)[0]
    vocab_list = tfidf_vocab.tolist()
    try:
        w1 = int(w1)
        indword1 = w1
    except:
        #w1 is a string, not an integerer
        index1 = vocab_list.index(w1)
        indword1 = np.where(tfidf_index_sorted_weights==index1)[0][0]
    #
    try:
        w2 = int(w2)
        indword2 = w2
    except:
        #w2 is a string, not an integerer
        index2 = vocab_list.index(w2)
        indword2 = np.where(tfidf_index_sorted_weights==index2)[0][0]
    print(indword1,indword2)
    #
    pd = pt.Paradraw()
    pd.title = tfidf_vocab[tfidf_index_sorted_weights[indword1]] + '  ' + tfidf_vocab[tfidf_index_sorted_weights[indword2]]
    pd.x_label = tfidf_vocab[tfidf_index_sorted_weights[indword1]]
    pd.y_label = tfidf_vocab[tfidf_index_sorted_weights[indword2]]
    pd.colors = ['r','g','b']
    pd.markers = ['.','.','.']
    pd.marks = ['','','']
    pd.markeredge = True
    pd.markerssize = [8,8,8]
    pd.thickness = [18,18,18]
    pd.legend = ['cluster 1','cluster 2','cluster 3']

    datax = [most_sig_array[:,indword1][ind0],most_sig_array[:,indword1][ind1],most_sig_array[:,indword1][ind2]]
    datay = [most_sig_array[:,indword2][ind0],most_sig_array[:,indword2][ind1],most_sig_array[:,indword2][ind2]]
    pt.multiplot2(datax,datay,pd)



indword1 = 0
indword2 = 1
indword3 = 2


pd = pt.Paradraw()
pd.colors = ['r','g','b']
pd.markers = ['.','.','.']
pd.marks = ['','','']
pd.markeredge = True
pd.markerssize = [8,8,8]
pd.thickness = [18,18,18]
pd.legend = ['cluster 1','cluster 2','cluster 3']
pd.title = tfidf_vocab[tfidf_index_sorted_weights[indword1]] + '  ' + tfidf_vocab[tfidf_index_sorted_weights[indword2]]
pd.x_label = tfidf_vocab[tfidf_index_sorted_weights[indword1]]
pd.y_label = tfidf_vocab[tfidf_index_sorted_weights[indword2]]


datax = [most_sig_array[:,indword1][ind0],most_sig_array[:,indword1][ind1],most_sig_array[:,indword1][ind2]]
datay = [most_sig_array[:,indword2][ind0],most_sig_array[:,indword2][ind1],most_sig_array[:,indword2][ind2]]
dataz = [most_sig_array[:,indword3][ind0],most_sig_array[:,indword3][ind1],most_sig_array[:,indword3][ind2]]
pt.multiplot3(datax,datay,dataz,pd)


