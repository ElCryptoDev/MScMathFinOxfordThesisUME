#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cryptocompare_ume
import pandas as pd
import time
import datetime
import pytz
import api_key

# def realDate(row):
#
#   #  Calulates the date of epoch seconds
#   
#     realDate = datetime.datetime.fromtimestamp(row['LASTUPDATE']).strftime('%Y-%m-%d %H:%M:%S')
#     return realDate

# DictSocial = cryptocompare.get_social_stats(5038)

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
listCoins = dfCoins['TOKEN'].tolist()

test_call = cryptocompare_ume.get_price(listCoins, apikey = api_key.api_key, curr='USD',full=True)
dfs =[]
for listiter in listCoins:
    try:
        dfIter = pd.DataFrame.from_records(test_call["RAW"][listiter]["USD"], index=[0])
        dfIter["TOKEN"] = listiter
        dfs.append(dfIter)
    except:
        pass

dfAll = pd.concat(dfs, sort =True, ignore_index=True)
cols = ['TOKEN','LASTUPDATE','PRICE','LASTVOLUME','LASTVOLUMETO','MKTCAP','TOTALVOLUME24H', 'TOTALVOLUME24HTO', 'VOLUME24HOUR', 'VOLUME24HOURTO', 'VOLUMEDAY', 'VOLUMEDAYTO', 'VOLUMEHOUR', 'VOLUMEHOURTO']

dfPrice = dfAll[cols]
#dfSave.to_csv('coin_prices_usd_cccagg.csv', header = True, index=False)
now = datetime.datetime.now(pytz.utc)
filename = 'cryptodata/coin_prices_usd_cccagg_' + now.strftime("%Y-%m-%d") + '.csv'
with open(filename, 'a') as f:
    dfPrice.to_csv(f, header=False, index=False)


