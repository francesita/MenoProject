import tweepy
import pickle
import sys
import json


'''
Code using 
'''
#dic = {}

consumer_token = open( "private/.consumer_token" ).read().strip()
consumer_secret = open("private/.consumer_secret").read().strip()

access_token = open("private/.access_token").read().strip()
access_token_secret = open("private/.access_secret").read().strip()

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# this constructs API's instance                                                                                                                                                                                   
api = tweepy.API(auth, wait_on_rate_limit = True)


tweetObjects=[]
updated_dic_file = open("menopause_tweets_workuser.json", "w", encoding='utf-8')
user_file = open("workplace_user.txt", "r")


i = 0
def find_tweet(user_file):
    for line in user_file:
        user = line
        extract_tweet(user)
        json.dump(tweetObjects, updated_dic_file, indent=2)


def extract_tweet(user):
	print(user)
	try:
		for tweet in tweepy.Cursor(api.user_timeline, screen_name=user, tweet_mode= 'extended').items():			
			if hasattr(tweet, 'retweeted_status'):
				tweet_id = tweet.retweeted_status.id
				tweet_date = tweet.retweeted_status.created_at
				tweet_author = tweet.retweeted_status.author._json['screen_name']
				tweet_text = tweet.retweeted_status.full_text
				tweet_place = tweet.retweeted_status.author._json['location']
			else:
				tweet_id = tweet.id
				tweet_date = tweet.created_at
				tweet_author = tweet.author._json['screen_name']
				tweet_text = tweet.full_text
				tweet_place = tweet.author._json['location']


			#puttinh all info in a dic and appending to list tweetObject        		
			tweetObjects.append({
				'tweet_id': tweet_id,
				'author': tweet_author,
				'tweet': tweet_text,
				'location': tweet_place
			})		

	except tweepy.TweepError as e:
		print(e.reason)
		print(user)

find_tweet(user_file)

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

tweets = remove_duplicates(tweetObjects)
updated_dic_file1 = open("menopause_tweets_workuser1.json", "w", encoding='utf-8')
json.dump(tweets, updated_dic_file1, indent=2)
updated_dic_file.close()
updated_dic_file1.close()








