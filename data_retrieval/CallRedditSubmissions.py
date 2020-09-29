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
for submission in subreddit.stream.submissions():
    col_names =  ['TOKEN', 'TIME', 'TITLE', 'TEXT']
    dfComment = pd.DataFrame(columns = col_names)
    location = len(dfComment)
    # Write data from submission in data frame
    try:
        dfComment.loc[location, 'TOKEN'] = dfRef.loc[dfRef['ID'] == submission.subreddit_id, 'TOKEN'].values[0]
    except:
        pass
    try:
        dfComment.loc[location, 'TIME'] = submission.created_utc
    except:
        pass
    try:
        dfComment.loc[location, 'TITLE'] = submission.title
    except:
        pass
    try:
        dfComment.loc[location, 'TEXT'] = submission.selftext
    except:
        pass
    now = datetime.datetime.now(pytz.utc)
    filename =  'cryptodata/coin_reddit_submissions_' + now.strftime("%Y-%m-%d") + '.csv'
    with open(filename, 'a') as f:
        dfComment.to_csv(f, header=False, index=False)

