#Menopause Project

## Preprocessing Folder##
Contains the following files:

- contractions.pkl
    Pickle file containing contractions and their extended forms. This is used by the preprocess.py code

- prepocess.py
    Code containing the preprocessing functions used for articles

-preprocess_tweets.py
    Code for preprocessing tweets

- text_ext_json.py
    Code for converting extracted web articles to Json formatted styles. This is to have one format for later use. 

## Tweet_ext Folder
Contains the following files:

- tweet_collect.py
    Using tweepy, and a list of keywords this code collects tweets from Twitter.

-user_tweets.py
    Using tweepy and a list of twitter handles, collects all the tweets tweeted or retweeted by the users in the list. 

## Analyze_data Folder
Contains the following files:

- word_frequency.py
    tfidf is used to generate frequency of words. I developed a method to find the most significant words using the tfidf features. 
    Each document (i.e.  for each column of *M*), the values are sorted by decreasing value.  Only the top *n* (e.g. 100) features are kept, by replacing features not in the top *n* by 0.  The resulting matrix *M* is now sparse: for each row, only 100 are non zero. Most of them  are now 0, allowing me to keep the most important unigrams or words. This is because tfidf values of not-significant words are small, but when summed together over thousands of articles/tweets, the resulting value can be large.
For instance, the word *today* probably appears in many tweets. This means its *tfidf*, for each tweet, will be quite small (say 0.01 per tweet). 
But because there are over 62,000 tweets, the final sum would be around 620. 
Conversely, the word *creativity* does not appear on many tweets, but is very important when it is present: the tfifd value would be, say 1. But if it has appeared on only 200 tweets, the final sum would be 200, less than for the word *today* in the previously described scenario. Using my method, the tfidf associated to the word *today* for example, would be, most of the time replaced by 0 (assuming this is a word that occurs frequently in most of the documets). Summing all the values would lead to a number very close to 0. On the contrary, when the word *creativity* appears, its tfidf value, if really high for some documents, would remain unaffected.
This file also utilizes word cloud to generate a word cloud of n most significant terms. 

- bigram_trigram.py
    Does what has been described above, but for bi- and trigrams.

- count_words.py
    Looks at the frequency of words in a document

- kmeans.py
    Uses matrix of most significant features per document. Kmeans algorithm by sklearn is used to look at how one word clusters in relation to another word in all the documents. 
