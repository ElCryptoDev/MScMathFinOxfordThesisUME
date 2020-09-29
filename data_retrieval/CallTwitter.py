from datetime import datetime
import tweepy
from tweepy import OAuthHandler
import pytz
import re
from textblob import TextBlob
import pandas as pd

def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\ / \ / \S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    '''
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    '''
    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet))
    # set sentiment
    polarity = analysis.sentiment.polarity

    return  polarity

def cal_polarity(row):
    """
    Calulates the polarity of a text with the textlob library
    """
    polarity = get_tweet_sentiment(row['TEXT'])

    return polarity

def cal_sentiment(row):
    """
    Calulates the sentiment from polarity
    """
    if row['POLARITY'] > 0:
        sentiment = 'positive'
    elif row['POLARITY'] == 0:
        sentiment = 'neutral'
    else:
        sentiment = 'negative'

    return sentiment

# Insert Twitter API credentials here
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# create OAuthHandler object
auth = OAuthHandler(consumer_key, consumer_secret)
# set access token and secret
auth.set_access_token(access_token, access_token_secret)
# create tweepy API object to fetch tweets
api = tweepy.API(auth)

search_words = list = [['ETH', 'eth OR etherum'],
                       ['BTC', 'btc OR (bitcoin -"bitcoin gold" -"bitcoin cash)"'],
                       ['XMR', 'xmr OR monero'],
                       ['XRP', 'xrp OR ripple'],
                       ['LTC', 'ltc OR litecoin'],
                       ['EOS', '(eos OR eosio) -canon -pokemon -widescreen -photo -camera -shot'],
                       ['BCH', 'bch OR "bitcoin cash"'],
                       ['ETC', '"$etc" OR "etherum classic" OR "eth_classic" OR "EthereumClassic"'], ['XLM', 'xlm OR stellar OR "stellar lumens"'], ['DOGE', 'doge OR "dogecoin"'],
                       ['DASH', 'dash (coin OR crypto OR currency OR cryptocurrency OR trading OR altcoin)'],
                       ['ZEC', '(zec (coin OR crypto OR currency OR cryptocurrency OR trading OR altcoin)) OR zcash'],
                       ['IOTA', 'iota OR miota'], ['NEO', 'neo (coin OR crypto OR cryptocurrency OR altcoin)'],
                       ['BTG', 'btg OR "bitcoin gold"'], ['TRX', 'trx OR tron'],
                       ['XVG', 'xvg OR (verge (coin OR crypto OR currency OR cryptocurrency OR trading OR altcoin)) OR vergecoin'],
                       ['VET', '(vet (coin OR crypto OR cryptocurrency OR altcoin)) OR vechain'],
                       ['QTUM', 'qtum -"Hairy Little Buggers" -"hlb"']]

currentDate = str(datetime.now(pytz.utc).date())
currentTime = datetime.now(pytz.utc)

col_names =  ['TOKEN', 'TIME', 'TEXT', 'POLARITY', 'SENTIMENT']
my_df  = pd.DataFrame(columns = col_names)

shouldContinue = True
for query in search_words:
    for tweets in tweepy.Cursor(api.search, q=query[1], count=1000, result_type="recent", include_entities=False, since = currentDate, lang='en', wait_on_rate_limit=True, tweet_mode='extended').items():
        tweet = tweets._json
        tweetTime = datetime.strptime(tweet["created_at"] , '%a %b %d %H:%M:%S %z %Y') # get the current time of the tweet
        interval = currentTime - tweetTime # subtract tweetTime from currentTime
        if interval.seconds <= 900: #get interval in seconds and use your time constraint in seconds (mine is 15 mins = 900secs)
            location = len(my_df)
            my_df.loc[location, 'TOKEN'] = query[0]
            my_df.loc[location, 'TIME'] = round(datetime.timestamp(tweetTime))
            my_df.loc[location, 'TEXT'] = tweet["full_text"]
        else:
            shouldContinue = False

        if not shouldContinue:
            break

my_df['POLARITY'] = my_df.apply(cal_polarity, axis=1)
my_df['SENTIMENT'] = my_df.apply(cal_sentiment, axis=1)

now = datetime.now(pytz.utc)
filename = 'cryptodata/coin_twitter_' + now.strftime("%Y-%m-%d") + '.csv'

with open(filename, 'a', encoding="utf-8") as f:
    my_df.to_csv(f, header=False, sep=',', index=False)