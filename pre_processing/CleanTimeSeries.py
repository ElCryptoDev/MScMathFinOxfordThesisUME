import os
import config
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sentimentcal
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial


def parallelize_dataframe(df, func, column, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)

    func_new = partial(func, column)
    df = pd.concat(pool.map(func_new, df_split))

    pool.close()
    pool.join()
    return df

def sentical_textblob(column, df):
    df = df.apply(sentimentcal.cal_polarity, column=column, axis=1)
    return df

def sentical_vader(column, df):
    df = df.apply(sentimentcal.cal_sentiment_vader, column=column, axis=1)
    return df

def read_special(f):
    if time.strptime(f[-14:-4], "%Y-%m-%d") < cutdate or (time.strptime(f[-14:-4], "%Y-%m-%d") == cutdate and f[-16:-15] == 'g'):
        df_Price = pd.read_csv(f, header=None, sep=',', names=colnamesPrices)
        df_Price['TOTALVOLUME24H'] = None; df_Price['TOTALVOLUME24HTO'] = None; df_Price['VOLUME24HOUR'] = None
        df_Price['VOLUME24HOURTO'] = None; df_Price['VOLUMEDAY'] = None; df_Price['VOLUMEDAYTO'] = None
        df_Price['VOLUMEHOUR'] = None; df_Price['VOLUMEHOURTO'] = None
        return df_Price
    else:
        return pd.read_csv(f, header=None, sep=',', names=colnamesPricesNew)

if __name__ == '__main__':
    #pd.set_option('display.height', 1000)
    #pd.set_option('display.max_rows', 500)
    #pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option("display.max_columns", 20)
    np.set_printoptions(linewidth=1000)
    cpu_count = cpu_count()

    listPriceFilesLocal = glob.glob(config.storage_path + '\\Cryptodata' + '\\coin_prices*.csv')
    listStatsFilesLocal = glob.glob(config.storage_path + '\\Cryptodata' + '\\coin_social*.csv')
    listTwitterFilesLocal = glob.glob(config.storage_path + '\\Cryptodata' + '\\coin_twitter*.csv')
    listRedditSubmissionFilesLocal = glob.glob(config.storage_path + '\\Cryptodata' + '\\coin_reddit_submissions_*.csv')
    listRedditCommentFilesLocal = glob.glob(config.storage_path + '\\Cryptodata' + '\\coin_reddit_comments_*.csv')
    colnamesPrices = ['TOKEN','LASTUPDATE','PRICE','LASTVOLUME','LASTVOLUMETO','MKTCAP']
    colnamesPricesNew = ['TOKEN','LASTUPDATE','PRICE','LASTVOLUME','LASTVOLUMETO','MKTCAP','TOTALVOLUME24H', 'TOTALVOLUME24HTO', 'VOLUME24HOUR', 'VOLUME24HOURTO', 'VOLUMEDAY', 'VOLUMEDAYTO', 'VOLUMEHOUR', 'VOLUMEHOURTO']

    colnamesStats = ['TOKEN','TIME','CRYPTOCOMPARE_POSTS', 'CRYPTOCOMPARE_COMMENTS', 'CRYPTOCOMPARE_POINTS',
                     'CRYPTOCOMPARE_FOLLOWERS', 'TWITTER_STATUSES', 'TWITTER_FOLLOWERS', 'TWITTER_FAVOURITES', 'REDDIT_SUBSCRIBERS',
                     'REDDIT_ACTIVE_USERS', 'REDDIT_COMMENTS_PER_HOUR', 'REDDIT_POSTS_PER_HOUR', 'REDDIT_POINTS',
                     'FACEBOOK_TALKING_ABOUT', 'FACEBOOK_LIKES', 'FACEBOOK_POINTS', 'FIRST_CODEREPO_LAST_UPDATE',
                     'FIRST_CODEREPO_LAST_PUSH', 'FIRST_CODEREPO_SUBSCRIBERS', 'CODEREPO_POINTS']
    colnamesTwitter = ['TOKEN', 'TIME', 'TEXT', 'POLARITY', 'SENTIMENT']
    colnamesRedditSubmission = ['TOKEN', 'TIME', 'TITLE', 'TEXT']
    colnamesRedditComments = ['TOKEN', 'TIME', 'TEXT']

    # Reduced after adjustment due to api call limitation
    list = ['BTC', 'ETH', 'XMR', 'XRP', 'LTC', 'BCH', 'XLM', 'DASH', 'IOTA', 'TRX']

    for coin in list:
        print(coin)

        cutdate = time.strptime("2019-01-26", "%Y-%m-%d")
        print(time.strftime("%H:%M:%S %b %d %Y") + " Read data " + coin + "...")
        dfCoin = pd.concat((read_special(f) for f in listPriceFilesLocal), sort=False, axis=0, ignore_index=True)
        print(time.strftime("%H:%M:%S %b %d %Y") + " Read data " + coin + " complete")

        # Drop columns not needed
        dfCoin = dfCoin.drop(['VOLUME24HOUR', 'VOLUME24HOURTO', 'VOLUMEHOUR', 'VOLUMEHOURTO'], axis=1)

        if coin =='IOTA':
            dfCoin = dfCoin[(dfCoin.TOKEN == coin) | (dfCoin.TOKEN == 'MIOTA')]
            dfCoin.TOKEN = 'IOTA'
        else:
            dfCoin = dfCoin[dfCoin.TOKEN == coin]
        dfCoin = dfCoin.drop_duplicates(['LASTUPDATE'], keep='last')
        dfCoin = dfCoin.sort_values(axis=0, by= 'LASTUPDATE')
        dfCoin = dfCoin.set_index('LASTUPDATE')
        dfCoin.index = pd.to_datetime(dfCoin.index, unit='s', utc=True)

        dfSeries = dfCoin.asfreq('1Min', method='pad')
        dfSeries['LOW_10M'] = dfSeries['PRICE'].rolling('10Min').min()
        dfSeries['HIGH_10M'] = dfSeries['PRICE'].rolling('10Min').max()
        dfSeries['OPEN_10M'] = dfSeries['PRICE'].shift(1, freq='10Min')
        dfSeries['LOW_1H'] = dfSeries['PRICE'].rolling('60Min').min()
        dfSeries['HIGH_1H'] = dfSeries['PRICE'].rolling('60Min').max()
        dfSeries['OPEN_1H'] = dfSeries['PRICE'].shift(1, freq='60Min')
        dfSeries['LOW_1D'] = dfSeries['PRICE'].rolling('1D').min()
        dfSeries['HIGH_1D'] = dfSeries['PRICE'].rolling('1D').max()
        dfSeries['OPEN_1D'] = dfSeries['PRICE'].shift(1, freq='1D')

        # Convert daily cumulative volumes to total cumulative volume
        dfSeries['VOLUMETO_CUM'] = dfSeries['VOLUMEDAYTO'] + (
            dfSeries['VOLUMEDAYTO'].shift(1).where(dfSeries['VOLUMEDAYTO'].shift() > dfSeries['VOLUMEDAYTO'], 0.0)).cumsum()
        dfSeries['VOLUME_CUM'] = dfSeries['VOLUMEDAY'] + (
            dfSeries['VOLUMEDAY'].shift(1).where(dfSeries['VOLUMEDAY'].shift() > dfSeries['VOLUMEDAY'], 0.0)).cumsum()

        ########################################################################################################################
        # Resample to 10 min frequency
        ########################################################################################################################
        dfSeries = dfSeries.asfreq('10Min', method='pad')
        masterindex = dfSeries.index

        #dfSeries['RETURN_1M'] = dfSeries['PRICE'].pct_change(freq='1Min')
        dfSeries['RETURN_10M'] = dfSeries['PRICE'].pct_change(freq='10Min')
        dfSeries['RETURN_1H'] = dfSeries['PRICE'].pct_change(freq='60Min')
        dfSeries['RETURN_1D'] = dfSeries['PRICE'].pct_change(freq='1D')

        dfSeries[['VOLUME_10M', 'VOLUMETO_10M']] = dfSeries[['VOLUME_CUM', 'VOLUMETO_CUM']].diff(1)
        dfSeries[['VOLUME_1H', 'VOLUMETO_1H']] = dfSeries[['VOLUME_CUM', 'VOLUMETO_CUM']].diff(6)
        dfSeries[['VOLUME_1D', 'VOLUMETO_1D']] = dfSeries[['VOLUME_CUM', 'VOLUMETO_CUM']].diff(144)
        dfSeries = dfSeries.drop(['VOLUMEDAY', 'VOLUMEDAYTO', 'VOLUME_CUM', 'VOLUMETO_CUM'], axis=1)

        ########################################################################################################################
        # Load other time series
        ########################################################################################################################
        print(time.strftime("%H:%M:%S %b %d %Y") + " Read stats " + coin + "...")
        dfCoinStats = pd.concat(
            (pd.read_csv(f, header=None, sep=',', names=colnamesStats, engine="python") for f in listStatsFilesLocal),
            sort=False, axis=0, ignore_index=True)
        print(time.strftime("%H:%M:%S %b %d %Y") + " Read stats " + coin + " complete")
        if coin =='IOTA':
            dfCoinStats = dfCoinStats[(dfCoinStats.TOKEN == coin) | (dfCoinStats.TOKEN == 'MIOTA')]
            dfCoinStats.TOKEN = 'IOTA'
        else:
            dfCoinStats = dfCoinStats[dfCoinStats.TOKEN == coin]
        dfCoinStats = dfCoinStats.drop_duplicates()
        dfCoinStats = dfCoinStats.drop(['TOKEN', 'TWITTER_STATUSES', 'TWITTER_FOLLOWERS', 'TWITTER_FAVOURITES', 'FACEBOOK_TALKING_ABOUT',
                                        'FACEBOOK_LIKES', 'FACEBOOK_POINTS', 'FIRST_CODEREPO_LAST_UPDATE',
                                        'FIRST_CODEREPO_LAST_PUSH', 'FIRST_CODEREPO_SUBSCRIBERS', 'CODEREPO_POINTS'], axis=1)
        dfCoinStats = dfCoinStats.sort_values(axis=0, by= 'TIME')
        dfCoinStats = dfCoinStats.set_index(['TIME'])
        dfCoinStats.index = pd.to_datetime(dfCoinStats.index, unit='s', utc=True)

        mergeindex = dfSeries.index.union(dfCoinStats.index)
        dfCoinStatsReindex = dfCoinStats.reindex(mergeindex)
        dfCoinStatsReindex = dfCoinStatsReindex.fillna(method='pad')
        dfCoinStatsReindex = dfCoinStatsReindex.reindex(masterindex)

        dfMerge = dfSeries.merge(dfCoinStatsReindex, how = 'left', left_index=True, right_index=True)

        print(time.strftime("%H:%M:%S %b %d %Y") + " Read twitter " + coin + "...")
        dfTwitterCoin = pd.concat(
            (pd.read_csv(f, header=None, sep=',', names=colnamesTwitter, engine="python") for f in listTwitterFilesLocal),
            sort=False, axis=0, ignore_index=True)
        print(time.strftime("%H:%M:%S %b %d %Y") + " Read twitter " + coin + " complete")
        dfTwitterCoin = dfTwitterCoin[dfTwitterCoin.TOKEN == coin]
        dfTwitterCoin = dfTwitterCoin.drop_duplicates(['TIME', 'TEXT'])
        dfTwitterCoin = dfTwitterCoin.sort_values(axis=0, by= 'TIME')
        dfTwitterCoin['TEMP'] = dfTwitterCoin.groupby(['TIME']).cumcount()
        dfTwitterCoin['TIME'] = dfTwitterCoin['TIME'] + dfTwitterCoin['TEMP']/100
        dfTwitterCoin = dfTwitterCoin.drop(['TEMP'], axis=1)
        dfTwitterCoin = dfTwitterCoin.set_index(['TIME'])
        dfTwitterCoin.index = pd.to_datetime(dfTwitterCoin.index, unit='s', utc=True)

        # Recalculate sentiment
        start_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(start_time)) + " Sentimentcal textblob twitter " + coin + " ...")
        dfTwitterCoin = parallelize_dataframe(dfTwitterCoin, sentical_textblob, 'TEXT', n_cores=cpu_count)
        #dfTwitterCoin = dfTwitterCoin.apply(sentimentcal.cal_polarity, axis=1)
        end_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(end_time)) + " (duration " + time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)) + ") Sentimentcal textblob twitter " + coin + " complete")
        #dfTwitterCoin.head(50)
        #dfTwitterCoin['MEAN_POLARITY_TWITTER_1M'] = dfTwitterCoin['POLARITY'].rolling('1Min').mean()
        dfTwitterCoin['MEAN_POLARITY_TWITTER_10M'] = dfTwitterCoin['POLARITY'].rolling('10Min').mean()
        dfTwitterCoin['MEAN_POLARITY_TWITTER_1H'] = dfTwitterCoin['POLARITY'].rolling('60Min').mean()
        dfTwitterCoin['MEAN_POLARITY_TWITTER_1D'] = dfTwitterCoin['POLARITY'].rolling('1D').mean()
        #dfTwitterCoin['FREQ_TWITTER_1M'] = dfTwitterCoin['POLARITY'].rolling('1Min').count()
        dfTwitterCoin['FREQ_TWITTER_10M'] = dfTwitterCoin['POLARITY'].rolling('10Min').count()
        dfTwitterCoin['FREQ_TWITTER_1H'] = dfTwitterCoin['POLARITY'].rolling('60Min').count()
        dfTwitterCoin['FREQ_TWITTER_1D'] = dfTwitterCoin['POLARITY'].rolling('1D').count()

        start_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(start_time)) + " Sentimentcal vader twitter " + coin + " ...")
        dfTwitterCoin = parallelize_dataframe(dfTwitterCoin, sentical_vader, 'TEXT', n_cores=cpu_count)
        end_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(end_time)) + " (duration " + time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)) + ") Sentimentcal vader twitter " + coin + " complete")
        #dfTwitterCoin['MEAN_POLARITY_VADER_TWITTER_1M'] = dfTwitterCoin['VADER_COMPOUND'].rolling('1Min').mean()
        dfTwitterCoin['MEAN_POLARITY_VADER_TWITTER_10M'] = dfTwitterCoin['VADER_COMPOUND'].rolling('10Min').mean()
        dfTwitterCoin['MEAN_POLARITY_VADER_TWITTER_1H'] = dfTwitterCoin['VADER_COMPOUND'].rolling('60Min').mean()
        dfTwitterCoin['MEAN_POLARITY_VADER_TWITTER_1D'] = dfTwitterCoin['VADER_COMPOUND'].rolling('1D').mean()

        dfTwitterCoin = dfTwitterCoin.drop(['TOKEN', 'TEXT', 'SENTIMENT', 'POLARITY', 'VADER_COMPOUND'], axis=1)

        mergeindex = dfTwitterCoin.index.union(dfMerge.index)
        dfTwitterCoinReindex = dfTwitterCoin.reindex(mergeindex)
        dfTwitterCoinReindex = dfTwitterCoinReindex.fillna(method='pad')
        dfTwitterCoinReindex = dfTwitterCoinReindex.reindex(masterindex)

        dfMerge = dfMerge.merge(dfTwitterCoinReindex, how = 'left', left_index=True, right_index=True)

        # Reddit submissions
        print(time.strftime("%H:%M:%S %b %d %Y") + " Read reddit subs " + coin + " ...")
        dfRedditSubs = pd.concat(
            (pd.read_csv(f, header=None, sep=',', names=colnamesRedditSubmission, engine="c") for f in
             listRedditSubmissionFilesLocal),
            sort=False, axis=0, ignore_index=True)
        print(time.strftime("%H:%M:%S %b %d %Y") + " Read reddit subs " + coin + " complete")
        dfRedditSubs = dfRedditSubs[dfRedditSubs.TOKEN == coin]
        dfRedditSubs = dfRedditSubs.drop_duplicates(['TIME', 'TITLE'])
        dfRedditSubs = dfRedditSubs.sort_values(axis=0, by= 'TIME')

        dfRedditSubs['TEMP'] = dfRedditSubs.groupby(['TIME']).cumcount()
        dfRedditSubs['TIME'] = dfRedditSubs['TIME'] + dfRedditSubs['TEMP']/100
        dfRedditSubs = dfRedditSubs.drop(['TEMP'], axis=1)
        dfRedditSubs = dfRedditSubs.set_index(['TIME'])
        dfRedditSubs.index = pd.to_datetime(dfRedditSubs.index, unit='s', utc=True)

        #dfRedditSubs[dfRedditSubs['TIME'].duplicated(keep=False)]
        #dfRedditSubs.ix[5764]['TIME']

        # Calculate sentiment
        dfRedditSubs['TEXT'] = dfRedditSubs['TEXT'].fillna('')
        dfRedditSubs['MERGE_TEXT'] = dfRedditSubs['TITLE'] + ' ' + dfRedditSubs['TEXT']
        start_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(start_time)) + " Sentimentcal reddit subs " + coin + " ...")
        dfRedditSubs = parallelize_dataframe(dfRedditSubs, sentical_textblob, 'MERGE_TEXT', n_cores=cpu_count)
        #dfRedditSubs = dfRedditSubs.apply(sentimentcal.cal_polarity, column='MERGE_TEXT', axis=1)
        end_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(end_time)) + " (duration " + time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)) + ") Sentimentcal reddit subs " + coin + " complete")
        dfRedditSubs = dfRedditSubs.rename(columns={'POLARITY': 'REDDIT_SUB_POLARITY'})

        #dfRedditSubs['MEAN_POLARITY_REDDITSUB_1M'] = dfRedditSubs['REDDIT_SUB_POLARITY'].rolling('1Min').mean()
        dfRedditSubs['MEAN_POLARITY_REDDITSUB_10M'] = dfRedditSubs['REDDIT_SUB_POLARITY'].rolling('10Min').mean()
        dfRedditSubs['MEAN_POLARITY_REDDITSUB_1H'] = dfRedditSubs['REDDIT_SUB_POLARITY'].rolling('60Min').mean()
        dfRedditSubs['MEAN_POLARITY_REDDITSUB_1D'] = dfRedditSubs['REDDIT_SUB_POLARITY'].rolling('1D').mean()
        #dfRedditSubs['FREQ_REDDITSUB_1M'] = dfRedditSubs['REDDIT_SUB_POLARITY'].rolling('1Min').count()
        dfRedditSubs['FREQ_REDDITSUB_10M'] = dfRedditSubs['REDDIT_SUB_POLARITY'].rolling('10Min').count()
        dfRedditSubs['FREQ_REDDITSUB_1H'] = dfRedditSubs['REDDIT_SUB_POLARITY'].rolling('60Min').count()
        dfRedditSubs['FREQ_REDDITSUB_1D'] = dfRedditSubs['REDDIT_SUB_POLARITY'].rolling('1D').count()

        start_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(start_time)) + " Sentimentcal vader reddit subs " + coin + " ...")
        dfRedditSubs = parallelize_dataframe(dfRedditSubs, sentical_vader, 'MERGE_TEXT', n_cores=cpu_count)
        #dfRedditSubs = dfRedditSubs.apply(sentimentcal.cal_sentiment_vader, column='MERGE_TEXT', axis=1)
        end_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(end_time)) + " (duration " + time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)) + ") Sentimentcal vader reddit subs " + coin + " complete")
        #dfRedditSubs['MEAN_POLARITY_VADER_REDDITSUB_1M'] = dfRedditSubs['VADER_COMPOUND'].rolling('1Min').mean()
        dfRedditSubs['MEAN_POLARITY_VADER_REDDITSUB_10M'] = dfRedditSubs['VADER_COMPOUND'].rolling('10Min').mean()
        dfRedditSubs['MEAN_POLARITY_VADER_REDDITSUB_1H'] = dfRedditSubs['VADER_COMPOUND'].rolling('60Min').mean()
        dfRedditSubs['MEAN_POLARITY_VADER_REDDITSUB_1D'] = dfRedditSubs['VADER_COMPOUND'].rolling('1D').mean()

        dfRedditSubs = dfRedditSubs.drop(['TOKEN', 'TITLE', 'TEXT', 'MERGE_TEXT', 'REDDIT_SUB_POLARITY', 'VADER_COMPOUND'], axis=1)

        mergeindex = dfRedditSubs.index.union(dfMerge.index)

        dfRedditReindex = dfRedditSubs.reindex(mergeindex)
        dfRedditReindex = dfRedditReindex.fillna(method='pad')
        dfRedditReindex = dfRedditReindex.reindex(masterindex)

        dfMerge = dfMerge.merge(dfRedditReindex, how = 'left', left_index=True, right_index=True)

        # Reddit comments
        print(time.strftime("%H:%M:%S %b %d %Y") + " Read reddit comments " + coin + " ...")
        dfRedditComments = pd.concat(
            (pd.read_csv(f, header=None, sep=',', names=colnamesRedditComments, engine="c") for f in
             listRedditCommentFilesLocal),
            sort=False, axis=0, ignore_index=True)
        print(time.strftime("%H:%M:%S %b %d %Y") + " Read reddit comments " + coin + " complete")
        dfRedditComments = dfRedditComments[dfRedditComments.TOKEN == coin]
        dfRedditComments = dfRedditComments.drop_duplicates(['TIME', 'TEXT'])
        dfRedditComments = dfRedditComments.sort_values(axis=0, by= 'TIME')

        dfRedditComments['TEMP'] = dfRedditComments.groupby(['TIME']).cumcount()
        dfRedditComments['TIME'] = dfRedditComments['TIME'] + dfRedditComments['TEMP']/100
        dfRedditComments = dfRedditComments.drop(['TEMP'], axis=1)
        dfRedditComments = dfRedditComments.set_index(['TIME'])
        dfRedditComments.index = pd.to_datetime(dfRedditComments.index, unit='s', utc=True)

        # Calculate sentiment
        start_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(start_time)) + " Sentimentcal reddit comments " + coin + " ...")
        dfRedditComments = parallelize_dataframe(dfRedditComments, sentical_textblob, 'TEXT', n_cores=cpu_count)
        #dfRedditComments = dfRedditComments.apply(sentimentcal.cal_polarity, axis=1)
        end_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(end_time)) + " (duration " + time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)) + ") Sentimentcal reddit comments " + coin + " complete")
        dfRedditComments = dfRedditComments.rename(columns={'POLARITY': 'REDDIT_COMMENTS_POLARITY'})

        #dfRedditComments['MEAN_POLARITY_REDDITCOMMENTS_1M'] = dfRedditComments['REDDIT_COMMENTS_POLARITY'].rolling('1Min').mean()
        dfRedditComments['MEAN_POLARITY_REDDITCOMMENTS_10M'] = dfRedditComments['REDDIT_COMMENTS_POLARITY'].rolling('10Min').mean()
        dfRedditComments['MEAN_POLARITY_REDDITCOMMENTS_1H'] = dfRedditComments['REDDIT_COMMENTS_POLARITY'].rolling('60Min').mean()
        dfRedditComments['MEAN_POLARITY_REDDITCOMMENTS_1D'] = dfRedditComments['REDDIT_COMMENTS_POLARITY'].rolling('1D').mean()
        #dfRedditComments['FREQ_REDDITCOMMENTS_1M'] = dfRedditComments['REDDIT_COMMENTS_POLARITY'].rolling('1Min').count()
        dfRedditComments['FREQ_REDDITCOMMENTS_10M'] = dfRedditComments['REDDIT_COMMENTS_POLARITY'].rolling('10Min').count()
        dfRedditComments['FREQ_REDDITCOMMENTS_1H'] = dfRedditComments['REDDIT_COMMENTS_POLARITY'].rolling('60Min').count()
        dfRedditComments['FREQ_REDDITCOMMENTS_1D'] = dfRedditComments['REDDIT_COMMENTS_POLARITY'].rolling('1D').count()

        start_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(start_time)) + " Sentimentcal vader reddit comments " + coin + " ...")
        dfRedditComments = parallelize_dataframe(dfRedditComments, sentical_vader, 'TEXT', n_cores=cpu_count)
        #dfRedditComments = dfRedditComments.apply(sentimentcal.cal_sentiment_vader, axis=1)
        end_time = time.time()
        print(time.strftime("%H:%M:%S %b %d %Y", time.localtime(end_time)) + " (duration " + time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)) + ") Sentimentcal vaderreddit comments " + coin + " complete")
        #dfRedditComments['MEAN_POLARITY_VADER_REDDITCOMMENTS_1M'] = dfRedditComments['VADER_COMPOUND'].rolling('1Min').mean()
        dfRedditComments['MEAN_POLARITY_VADER_REDDITCOMMENTS_10M'] = dfRedditComments['VADER_COMPOUND'].rolling('10Min').mean()
        dfRedditComments['MEAN_POLARITY_VADER_REDDITCOMMENTS_1H'] = dfRedditComments['VADER_COMPOUND'].rolling('60Min').mean()
        dfRedditComments['MEAN_POLARITY_VADER_REDDITCOMMENTS_1D'] = dfRedditComments['VADER_COMPOUND'].rolling('1D').mean()

        dfRedditComments = dfRedditComments.drop(['TOKEN', 'TEXT', 'REDDIT_COMMENTS_POLARITY', 'VADER_COMPOUND'], axis=1)

        mergeindex = dfRedditComments.index.union(dfMerge.index)

        dfRedditCommentsReindex = dfRedditComments.reindex(mergeindex)
        dfRedditCommentsReindex = dfRedditCommentsReindex.fillna(method='pad')
        dfRedditCommentsReindex = dfRedditCommentsReindex.reindex(masterindex)

        dfMerge = dfMerge.merge(dfRedditCommentsReindex, how = 'left', left_index=True, right_index=True)


        # Calcualte percentage changes for selected variables
        list_cols_pct = ['CRYPTOCOMPARE_POSTS', 'CRYPTOCOMPARE_COMMENTS', 'CRYPTOCOMPARE_POINTS', 'CRYPTOCOMPARE_FOLLOWERS',
                         'REDDIT_SUBSCRIBERS', 'REDDIT_POINTS']
        list_cols_pct_10m = [i + '_10M' for i in list_cols_pct]
        list_cols_pct_1h = [i + '_1H' for i in list_cols_pct]
        list_cols_pct_1d = [i + '_1D' for i in list_cols_pct]

        dfMerge[list_cols_pct_10m] = dfMerge[list_cols_pct].pct_change(freq='10Min')
        dfMerge[list_cols_pct_1h] = dfMerge[list_cols_pct].pct_change(freq='60Min')
        dfMerge[list_cols_pct_1d] = dfMerge[list_cols_pct].pct_change(freq='1D')

        # Remove weekly seasonalities from selected variables
        list_cols_seasonal = ['FREQ_TWITTER_1H', 'FREQ_TWITTER_1D', 'FREQ_REDDITSUB_1H', 'FREQ_REDDITSUB_1D',
                              'FREQ_REDDITCOMMENTS_1H', 'FREQ_REDDITCOMMENTS_1D',]
        list_cols_seasonal_norm = [i + '_NORM' for i in list_cols_seasonal]
        dfMerge[list_cols_seasonal_norm] = dfMerge[list_cols_seasonal].diff(10080)

        reordered_colnames = ['TOKEN',
                              # Returns ####################################################################################
                              'RETURN_10M', 'RETURN_1H', 'RETURN_1D',

                              # Prices #####################################################################################
                              'PRICE', 'LOW_10M', 'HIGH_10M', 'OPEN_10M', 'LOW_1H', 'HIGH_1H', 'OPEN_1H', 'LOW_1D',
                              'HIGH_1D', 'OPEN_1D', 'MKTCAP',

                              # Volumes
                              'LASTVOLUME', 'LASTVOLUMETO', 'TOTALVOLUME24H', 'TOTALVOLUME24HTO', 'VOLUME_10M',
                              'VOLUMETO_10M', 'VOLUME_1H', 'VOLUMETO_1H', 'VOLUME_1D', 'VOLUMETO_1D',

                              # Cryptocompare ##############################################################################
                              'CRYPTOCOMPARE_POSTS', 'CRYPTOCOMPARE_COMMENTS', 'CRYPTOCOMPARE_POINTS',
                              'CRYPTOCOMPARE_FOLLOWERS',
                              ## 10min rolling
                              'CRYPTOCOMPARE_POSTS_10M', 'CRYPTOCOMPARE_COMMENTS_10M', 'CRYPTOCOMPARE_POINTS_10M',
                              'CRYPTOCOMPARE_FOLLOWERS_10M',
                              ## 1h rolling
                              'CRYPTOCOMPARE_POSTS_1H', 'CRYPTOCOMPARE_COMMENTS_1H', 'CRYPTOCOMPARE_POINTS_1H',
                              'CRYPTOCOMPARE_FOLLOWERS_1H',
                              ## 1d rolling
                              'CRYPTOCOMPARE_POSTS_1D', 'CRYPTOCOMPARE_COMMENTS_1D', 'CRYPTOCOMPARE_POINTS_1D',
                              'CRYPTOCOMPARE_FOLLOWERS_1D',

                              # Reddit #####################################################################################
                              ## Stats
                              'REDDIT_SUBSCRIBERS', 'REDDIT_ACTIVE_USERS', 'REDDIT_COMMENTS_PER_HOUR',
                              'REDDIT_POSTS_PER_HOUR', 'REDDIT_POINTS',
                              ## Rolling stats
                              'REDDIT_SUBSCRIBERS_10M', 'REDDIT_POINTS_10M',
                              'REDDIT_SUBSCRIBERS_1H', 'REDDIT_POINTS_1H',
                              'REDDIT_SUBSCRIBERS_1D', 'REDDIT_POINTS_1D',

                              ## Textblob subs
                              'MEAN_POLARITY_REDDITSUB_10M', 'MEAN_POLARITY_REDDITSUB_1H', 'MEAN_POLARITY_REDDITSUB_1D',
                              ## Vader subs
                              'MEAN_POLARITY_VADER_REDDITSUB_10M', 'MEAN_POLARITY_VADER_REDDITSUB_1H',
                              'MEAN_POLARITY_VADER_REDDITSUB_1D',
                              ## Volumes subs
                              'FREQ_REDDITSUB_10M', 'FREQ_REDDITSUB_1H', 'FREQ_REDDITSUB_1D',
                              ## Normalized volumes
                              'FREQ_REDDITSUB_1H_NORM', 'FREQ_REDDITSUB_1D_NORM',
                              ## Textblob comments
                              'MEAN_POLARITY_REDDITCOMMENTS_10M', 'MEAN_POLARITY_REDDITCOMMENTS_1H',
                              'MEAN_POLARITY_REDDITCOMMENTS_1D',
                              ## Vader comments
                              'MEAN_POLARITY_VADER_REDDITCOMMENTS_10M', 'MEAN_POLARITY_VADER_REDDITCOMMENTS_1H',
                              'MEAN_POLARITY_VADER_REDDITCOMMENTS_1D',
                              ## Volumes comments
                              'FREQ_REDDITCOMMENTS_10M', 'FREQ_REDDITCOMMENTS_1H', 'FREQ_REDDITCOMMENTS_1D',
                              ## Normalized volumes
                              'FREQ_REDDITCOMMENTS_1H_NORM', 'FREQ_REDDITCOMMENTS_1D_NORM',

                              # Twitter ####################################################################################
                              ## Textblob
                              'MEAN_POLARITY_TWITTER_10M', 'MEAN_POLARITY_TWITTER_1H', 'MEAN_POLARITY_TWITTER_1D',
                              ## Vader
                              'MEAN_POLARITY_VADER_TWITTER_10M', 'MEAN_POLARITY_VADER_TWITTER_1H',
                              'MEAN_POLARITY_VADER_TWITTER_1D',
                              ## Volume
                              'FREQ_TWITTER_10M', 'FREQ_TWITTER_1H', 'FREQ_TWITTER_1D',
                              ## Normalized volumes
                              'FREQ_TWITTER_1H_NORM', 'FREQ_TWITTER_1D_NORM']

        dfMerge = dfMerge[reordered_colnames]
        # print(time.strftime("%H:%M:%S %b %d %Y") + " Write 1m sample " + coin + " ...")
        # filename = config.storage_path + '\\Samples' + '\\sample_' + coin + '_1m.csv'
        # with open(filename, 'w') as f:
        #     dfMerge.to_csv(f, header=True, index=True)
        # print(time.strftime("%H:%M:%S %b %d %Y") + " Write 1m sample " + coin + "complete")

        #dfMerge1M = dfMerge.asfreq('10Min', method='pad')
        print(time.strftime("%H:%M:%S %b %d %Y") + " Write 10m sample " + coin + " ...")
        filename_train= config.storage_path + '\\Samples' + '\\sample_' + coin + '_10m_train.csv'
        filename_test = config.storage_path + '\\Samples' + '\\sample_' + coin + '_10m_test.csv'
        filename_val = config.storage_path + '\\Samples' + '\\sample_' + coin + '_10m_val.csv'
        dfMerge.loc[(dfMerge.index >= '2019-01-27 17:01:09') & (dfMerge.index <= '2019-05-23 16:31:09')].to_csv(filename_train, header=True, index=True)
        dfMerge.loc[(dfMerge.index >= '2019-11-09 16:31:09') & (dfMerge.index <= '2020-01-09 16:31:09')].to_csv(filename_test, header=True, index=True)
        dfMerge.loc[(dfMerge.index > '2020-01-09 16:31:09')].to_csv(filename_val, header=True, index=True)
        print(time.strftime("%H:%M:%S %b %d %Y") + " Write 10m sample " + coin + "complete")

        dfMerge = dfMerge.asfreq('60Min', method='pad')
        print(time.strftime("%H:%M:%S %b %d %Y") + " Write 1h sample " + coin + " ...")
        filename_train= config.storage_path + '\\Samples' + '\\sample_' + coin + '_1h_train.csv'
        filename_test = config.storage_path + '\\Samples' + '\\sample_' + coin + '_1h_test.csv'
        filename_val = config.storage_path + '\\Samples' + '\\sample_' + coin + '_1h_val.csv'
        dfMerge.loc[(dfMerge.index >= '2019-01-27 17:01:09') & (dfMerge.index <= '2019-05-23 16:31:09')].to_csv(filename_train, header=True, index=True)
        dfMerge.loc[(dfMerge.index >= '2019-11-09 16:31:09') & (dfMerge.index <= '2020-01-09 16:31:09')].to_csv(filename_test, header=True, index=True)
        dfMerge.loc[(dfMerge.index > '2020-01-09 16:31:09')].to_csv(filename_val, header=True, index=True)
        print(time.strftime("%H:%M:%S %b %d %Y") + " Write 1h sample " + coin + "complete")

        dfMerge = dfMerge.asfreq('1D', method='pad')
        print(time.strftime("%H:%M:%S %b %d %Y") + " Write 1d sample " + coin + " ...")
        filename_train= config.storage_path + '\\Samples' + '\\sample_' + coin + '_1d_train.csv'
        filename_test = config.storage_path + '\\Samples' + '\\sample_' + coin + '_1d_test.csv'
        filename_val = config.storage_path + '\\Samples' + '\\sample_' + coin + '_1d_val.csv'
        dfMerge.loc[(dfMerge.index >= '2019-01-27 17:01:09') & (dfMerge.index <= '2019-05-23 16:31:09')].to_csv(filename_train, header=True, index=True)
        dfMerge.loc[(dfMerge.index >= '2019-11-09 16:31:09') & (dfMerge.index <= '2020-01-09 16:31:09')].to_csv(filename_test, header=True, index=True)
        dfMerge.loc[(dfMerge.index > '2020-01-09 16:31:09')].to_csv(filename_val, header=True, index=True)
        print(time.strftime("%H:%M:%S %b %d %Y") + " Write 1d sample " + coin + "complete")

