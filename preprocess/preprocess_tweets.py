import pickle
import string
import re
import json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
"""
The following code helps in preprocessing the data contained in the pickel dictionary. I have already tokenized the tweets. I will now remove stop-words, lemmatize, temove punctuation and ignore usernames and hyperlinks. This will clean up my text so I can begin to use it on a bilstm model for training.
@author: Frances
@date 2020-07-01
"""
def load_json(input_file):
    f = open(input_file, "r")
    tweetObjects = json.load(f)
    return tweetObjects


def get_tweets(tweetObjects):
    tweets_unclean =[]
    for tweet in tweetObjects:
            tweets_unclean.append(tweet['tweet'].lower())
    return tweets_unclean

def tokenize(tweets_unclean):
    tknzr = TweetTokenizer(preserve_case = False, strip_handles=True, reduce_len=True)
    tokens=[]
    for tweet in tweets_unclean:
        tokens.append(tknzr.tokenize(tweet))
    return tokens


def get_authors(tweetObjects):
    '''
    Function creates a list of all the authors found in the tweets
    '''
    authors =[]
    for tweet in tweetObjects:
            if tweet['author'] not in authors:
                authors.append(tweet['author'])
    with open('tweet_authors.txt','w') as f:
         for author in authors:
            f.write(author + "\n")
    f.close()
    return authors


spanish_punctuation = set(['¿','¡',':','...',"'"])
def punctuation_rmv(token):
    clean_token = []
    for i in range(len(token)):
        if token[i] not in string.punctuation and token[i] not in spanish_punctuation:
            clean_token.append(token[i])
    return clean_token

"""
Function removes stopwords from token first in English 
"""

def stopword_rmv(token):
    stop_words = set(stopwords.words('english'))
    added_stopwords = ['one','says','like', 'said', 'say', 'would', 'go', 'im', 'wa', 'u','amp']
    stop_words = set(list(stop_words) + added_stopwords)
    #stopwords_sp = set(stopwords.words('spanish'))
    #clean_token_eng = []
    final_token = []
    
    for word in token:
        if word not in stop_words:
            final_token.append(word)

    return final_token
'''
remove contractions
returns tweets originally given but with contractions extended
'''
file = open("contractions.pkl", "rb")
cont_dic = pickle.load(file)
def contraction_ext(tokens):
    extended_token = []
    for token in tokens:
        if token in cont_dic.keys():
            token = cont_dic.get(token)
            extended_token.append(token)
        else:
            extended_token.append(token)
    return extended_token

#function that removes emojis from tweets
ehu = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
                          "]+", flags = re.UNICODE)
emoji_free_tweets=[]
def remove_emoji(tweets):
    for tweet in tweets:
       #nonascii chars replaces by space
        tweet = re.sub(r'[^\x00-\x7F]+','', tweet)
        tweet = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)",'', tweet) #gotten from stack-overflow to get rid of hashtags and remaining @ symbols
        tweet = re.sub(" \d+","",tweet)
#        tweet = ehu.sub(r'', tweet)
        emoji_free_tweets.append(tweet)
       #clean_tweets.append(tweet)
    return emoji_free_tweets


#function that calls all other preprocessing functions
'''
function preprocess tweets, takes dictionary as input and name of output file as input. 
it saves the preprocessed tweets to outputfile so it can be used by other python code
'''
def preprocess_tweets(tweetObjects, output_file_name):
    symbols_to_remove = re.compile("[^a-z _]+")
    remove_links = re.compile(r"http\S+")
    lmtzr = WordNetLemmatizer()
    tweets_list = get_tweets(tweetObjects)
	
	# removing unwanted symbols
    tweets= []
    clean_tweets = []
    for tweet in tweets_list:        
        tweet = re.sub(symbols_to_remove, "", tweet)
        tweet = re.sub(remove_links, "", tweet)
        tweets.append(tweet)
    print(len(tweets))
	# preprocessing continuation of tweets with sumbols and links removed
    tokens_list = tokenize(tweets)
    tokens_list = [[lmtzr.lemmatize(token) for token in tokens] for tokens in tokens_list]
    for token in tokens_list:
        clean_token = punctuation_rmv(token)
        #clean_token = contraction_ext(clean_token)
        clean_tweets.append( stopword_rmv(clean_token))
    output_dic(output_file_name, clean_tweets)

    return clean_tweets



#dumps new dictionary
def output_dic(file_name, tweets):
    write_file = open(file_name, "wb")
    pickle.dump(tweets, write_file)
    write_file.close()

input_file = '/media/frances/Expansion Drive/Menopause/code/Tweet_Collect/Menopause/Menopause_tweets/menopause_tweets_1.json'
tweetObjects = load_json(input_file)

def remove_duplicates(tweetObjects):
    tweets = []
    tweetIds = []
    for tweet in tweetObjects:
        if tweet['tweet_id'] not in tweetIds:
            tweetIds.append(tweet['tweet_id']) 
            tweets.append(tweet)
        else:
            continue
    return tweets
tweetObjects = remove_duplicates(tweetObjects)
clean_tweets = preprocess_tweets(tweetObjects, "body_preprocessed_tknzd.pkl")
authors = get_authors(tweetObjects)                                                                                                                 

