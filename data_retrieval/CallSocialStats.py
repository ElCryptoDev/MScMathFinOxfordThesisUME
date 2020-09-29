#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cryptocompare_ume
import pandas as pd
import time
import datetime
import pytz
import api_key


list = ['ETH', 'BTC', 'XMR', 'XRP', 'LTC', 'EOS', 'BCH', 'ETC', 'XLM', 'DOGE', 'DASH', 'ZEC', 'MIOTA', 'NEO', 'BTG',
        'TRX', 'XVG', 'VET', 'QTUM']
list_id = [7605  # ETH
    , 1182  # BTC
    , 5038  # XMR
    , 5031  # XRP
    , 3808  # LTC
    , 166503  # EOS
    , 202330  # BCH
    , 5324  # ETC
    , 4614  # XLM
    , 4432  # DOGE
    , 3807  # DASH
    , 24854  # ZEC
    , 127356  # IOTA
    , 27368  # NEO
    , 347235  # BTG
    , 310829  # TRX
    , 4433  # XVG
    , 236131  # VET
    , 112392  # QTUM
    ]

dfCoins = pd.DataFrame([list, list_id])
dfCoins = dfCoins.transpose()
dfCoins.columns =['TOKEN', 'ID']


labels = ['TOKEN', 'TIME', 'CRYPTOCOMPARE_POSTS', 'CRYPTOCOMPARE_COMMENTS', 'CRYPTOCOMPARE_POINTS',
          'CRYPTOCOMPARE_FOLLOWERS',
          'TWITTER_STATUSES', 'TWITTER_FOLLOWERS', 'TWITTER_FAVOURITES', 'REDDIT_SUBSCRIBERS',
          'REDDIT_ACTIVE_USERS', 'REDDIT_COMMENTS_PER_HOUR', 'REDDIT_POSTS_PER_HOUR', 'REDDIT_POINTS',
          'FACEBOOK_TALKING_ABOUT', 'FACEBOOK_LIKES', 'FACEBOOK_POINTS', 'FIRST_CODEREPO_LAST_UPDATE',
          'FIRST_CODEREPO_LAST_PUSH', 'FIRST_CODEREPO_SUBSCRIBERS', 'CODEREPO_POINTS']

dfWork = pd.DataFrame(columns=labels)
dfWork['TOKEN'].apply(str)

DictSocial = [None] * len(dfCoins)
for row in dfCoins.itertuples(index=True, name=None):

    DictSocial[row[0]] = cryptocompare_ume.get_social_stats(row[2], api_key.api_key)
    #time.sleep(2)

    if DictSocial[row[0]]["Response"] == 'Success':
        try:
            dfWork.loc[row[0], 'TOKEN'] = row[1]
        except:
            pass
        try:
            dfWork.loc[row[0], 'TIME'] = round(time.time())
        except:
            pass
        try:
            dfWork.loc[row[0], 'CRYPTOCOMPARE_POSTS'] = DictSocial[row[0]]["Data"]["CryptoCompare"]["Posts"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'CRYPTOCOMPARE_COMMENTS'] = DictSocial[row[0]]["Data"]["CryptoCompare"]["Comments"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'CRYPTOCOMPARE_POINTS'] = DictSocial[row[0]]["Data"]["CryptoCompare"]["Points"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'CRYPTOCOMPARE_FOLLOWERS'] = DictSocial[row[0]]["Data"]["CryptoCompare"]["Followers"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'TWITTER_STATUSES'] = DictSocial[row[0]]["Data"]["Twitter"]["statuses"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'TWITTER_FOLLOWERS'] = DictSocial[row[0]]["Data"]["Twitter"]["followers"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'TWITTER_FAVOURITES'] = DictSocial[row[0]]["Data"]["Twitter"]["favourites"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'REDDIT_SUBSCRIBERS'] = DictSocial[row[0]]["Data"]["Reddit"]["subscribers"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'REDDIT_ACTIVE_USERS'] = DictSocial[row[0]]["Data"]["Reddit"]["active_users"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'REDDIT_COMMENTS_PER_HOUR'] = DictSocial[row[0]]["Data"]["Reddit"]["comments_per_hour"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'REDDIT_POSTS_PER_HOUR'] = DictSocial[row[0]]["Data"]["Reddit"]["posts_per_hour"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'REDDIT_POINTS'] = DictSocial[row[0]]["Data"]["Reddit"]["Points"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'FACEBOOK_TALKING_ABOUT'] = DictSocial[row[0]]["Data"]["Facebook"]["talking_about"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'FACEBOOK_LIKES'] = DictSocial[row[0]]["Data"]["Facebook"]["likes"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'FACEBOOK_POINTS'] = DictSocial[row[0]]["Data"]["Facebook"]["Points"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'FIRST_CODEREPO_LAST_UPDATE'] = DictSocial[row[0]]["Data"]["CodeRepository"]['List'][0]["last_update"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'FIRST_CODEREPO_LAST_PUSH'] = DictSocial[row[0]]["Data"]["CodeRepository"]['List'][0]["last_push"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'FIRST_CODEREPO_SUBSCRIBERS'] = DictSocial[row[0]]["Data"]["CodeRepository"]['List'][0]["subscribers"]
        except:
            pass
        try:
            dfWork.loc[row[0], 'CODEREPO_POINTS'] = DictSocial[row[0]]["Data"]["CodeRepository"]["Points"]
        except:
            pass


now = datetime.datetime.now(pytz.utc)
filename = 'cryptodata/coin_social_stats_' + now.strftime("%Y-%m-%d") + '.csv'
with open(filename, 'a') as f:
    dfWork.to_csv(f, header=False, index=False)
