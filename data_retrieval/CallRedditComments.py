#!/usr/bin/python3 -u
import praw
import pandas as pd
import datetime
import pytz

# Input reddit api credentials here
reddit = praw.Reddit(client_id='',
                     client_secret='',
                     user_agent='',
                     username='',
                     password='')

dfRef = pd.read_csv('subreddits.csv', sep=',')
dfRefConcat = dfRef.apply(lambda x: '+'.join(x.dropna().values.tolist()), axis=0)
SearchString = dfRefConcat['SUBREDDIT']

subreddit = reddit.subreddit(SearchString)
for comment in subreddit.stream.comments():
    col_names =  ['TOKEN', 'TIME', 'TEXT']
    dfComment = pd.DataFrame(columns = col_names)
    location = len(dfComment)
    # Write data from comment in data frame
    try:
        dfComment.loc[location, 'TOKEN'] = dfRef.loc[dfRef['ID'] == comment.subreddit_id, 'TOKEN'].values[0]
    except:
        pass
    try:
        dfComment.loc[location, 'TIME'] = comment.created_utc
    except:
        pass
    try:
        dfComment.loc[location, 'TEXT'] = comment.body
    except:
        pass
    now = datetime.datetime.now(pytz.utc)
    filename = 'cryptodata/coin_reddit_comments_' + now.strftime("%Y-%m-%d") + '.csv'
    with open(filename, 'a') as f:
        dfComment.to_csv(f, header=False, index=False)
