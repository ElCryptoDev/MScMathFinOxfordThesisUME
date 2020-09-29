import os
import config
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np

# Set figure size
plt.rcParams["figure.figsize"] = (12,10)
plt.rcParams.update({'font.size': 18})
# Disable interactive mode to suppress plots
plt.ioff()


list = ['BTC', 'ETH', 'XMR', 'XRP', 'LTC', 'BCH', 'XLM', 'DASH', 'IOTA', 'TRX']

for coin in list:
    print("read samples " + coin)
    for sample in ['train', 'test', 'val']:
        listTS_10m = glob.glob(config.storage_path + '\\Samples' + '\\*' + coin +'*_10m_' + sample +'.csv')
        listTS_1d = glob.glob(config.storage_path + '\\Samples' + '\\*' + coin + '*_1d_' + sample +'.csv')
        df_ts_10m = pd.read_csv(listTS_10m[0])
        df_ts_1d = pd.read_csv(listTS_1d[0])
        print("read samples " + sample + " " + coin + " finished")

        columns_to_plot = ['RETURN_10M', 'RETURN_1H', 'RETURN_1D', 'PRICE',
                           'LOW_10M', 'HIGH_10M', 'OPEN_10M', 'LOW_1H', 'HIGH_1H', 'OPEN_1H',
                           'LOW_1D', 'HIGH_1D', 'OPEN_1D', 'MKTCAP', 'LASTVOLUME', 'LASTVOLUMETO',
                           'TOTALVOLUME24H', 'TOTALVOLUME24HTO', 'VOLUME_10M', 'VOLUMETO_10M',
                           'VOLUME_1H', 'VOLUMETO_1H', 'VOLUME_1D', 'VOLUMETO_1D',
                           'CRYPTOCOMPARE_POSTS', 'CRYPTOCOMPARE_COMMENTS', 'CRYPTOCOMPARE_POINTS',
                           'CRYPTOCOMPARE_FOLLOWERS', 'CRYPTOCOMPARE_POSTS_10M',
                           'CRYPTOCOMPARE_COMMENTS_10M', 'CRYPTOCOMPARE_POINTS_10M',
                           'CRYPTOCOMPARE_FOLLOWERS_10M', 'CRYPTOCOMPARE_POSTS_1H',
                           'CRYPTOCOMPARE_COMMENTS_1H', 'CRYPTOCOMPARE_POINTS_1H',
                           'CRYPTOCOMPARE_FOLLOWERS_1H', 'CRYPTOCOMPARE_POSTS_1D',
                           'CRYPTOCOMPARE_COMMENTS_1D', 'CRYPTOCOMPARE_POINTS_1D',
                           'CRYPTOCOMPARE_FOLLOWERS_1D', 'REDDIT_SUBSCRIBERS',
                           'REDDIT_ACTIVE_USERS', 'REDDIT_COMMENTS_PER_HOUR',
                           'REDDIT_POSTS_PER_HOUR', 'REDDIT_POINTS', 'REDDIT_SUBSCRIBERS_10M',
                           'REDDIT_POINTS_10M', 'REDDIT_SUBSCRIBERS_1H', 'REDDIT_POINTS_1H',
                           'REDDIT_SUBSCRIBERS_1D', 'REDDIT_POINTS_1D',
                           'MEAN_POLARITY_REDDITSUB_10M', 'MEAN_POLARITY_REDDITSUB_1H',
                           'MEAN_POLARITY_REDDITSUB_1D', 'MEAN_POLARITY_VADER_REDDITSUB_10M',
                           'MEAN_POLARITY_VADER_REDDITSUB_1H', 'MEAN_POLARITY_VADER_REDDITSUB_1D',
                           'FREQ_REDDITSUB_10M', 'FREQ_REDDITSUB_1H', 'FREQ_REDDITSUB_1D',
                           'FREQ_REDDITSUB_1H_NORM', 'FREQ_REDDITSUB_1D_NORM',
                           'MEAN_POLARITY_REDDITCOMMENTS_10M', 'MEAN_POLARITY_REDDITCOMMENTS_1H',
                           'MEAN_POLARITY_REDDITCOMMENTS_1D',
                           'MEAN_POLARITY_VADER_REDDITCOMMENTS_10M',
                           'MEAN_POLARITY_VADER_REDDITCOMMENTS_1H',
                           'MEAN_POLARITY_VADER_REDDITCOMMENTS_1D', 'FREQ_REDDITCOMMENTS_10M',
                           'FREQ_REDDITCOMMENTS_1H', 'FREQ_REDDITCOMMENTS_1D',
                           'FREQ_REDDITCOMMENTS_1H_NORM', 'FREQ_REDDITCOMMENTS_1D_NORM',
                           'MEAN_POLARITY_TWITTER_10M', 'MEAN_POLARITY_TWITTER_1H',
                           'MEAN_POLARITY_TWITTER_1D', 'MEAN_POLARITY_VADER_TWITTER_10M',
                           'MEAN_POLARITY_VADER_TWITTER_1H', 'MEAN_POLARITY_VADER_TWITTER_1D',
                           'FREQ_TWITTER_10M', 'FREQ_TWITTER_1H', 'FREQ_TWITTER_1D',
                           'FREQ_TWITTER_1H_NORM', 'FREQ_TWITTER_1D_NORM']

        df_ts_10m = df_ts_10m.set_index(['LASTUPDATE'])
        df_ts_10m.index = pd.to_datetime(df_ts_10m.index)
        df_ts_10m.index = df_ts_10m.index.to_pydatetime()

        df_ts_1d = df_ts_1d.set_index(['LASTUPDATE'])
        df_ts_1d.index = pd.to_datetime(df_ts_1d.index)
        df_ts_1d.index = df_ts_1d.index.to_pydatetime()

        chunk_size = 6
        for i in range(0, len(columns_to_plot), chunk_size):

            columns_to_plot_filter = columns_to_plot[i:i + chunk_size]

            print(coin + " full range plot vars "+ str(i) + ' to ' + str(i + chunk_size))
            # Full range for train set
            df_ts_1d.plot(y=columns_to_plot_filter, use_index=True, subplots = True)
            filename = config.storage_path + '\\Plots' + '\\sample_' + sample + '_' + coin + '_1d_vars_' + str(i) + '_to_' + str(i + chunk_size) + '_full.png'
            ax = plt.gca()
            plt.tight_layout()
            #plt.suptitle(coin + ' ' + sample + ' time series')

            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            ax.xaxis.set_minor_locator(mdates.DayLocator(15))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d-%m-%Y'))
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_horizontalalignment('center')

            #ax.set_title(coin + ' ' + sample + ' sample series')
            plt.savefig(filename, format='png')
            plt.close()

            if sample not in ['test', 'val']:
                print(coin + " detail plot vars "+ str(i) + ' to ' + str(i + chunk_size))
                # 1 day 2019-05-23 in detail
                df_ts_10m.loc[(df_ts_10m.index >= '2019-05-01 00:00:00') & (df_ts_10m.index <= '2019-05-02 00:00:00')].plot(y=columns_to_plot_filter, use_index=True, subplots = True)
                filename = config.storage_path + '\\Plots' + '\\sample_' + sample + '_' + coin + '_10m_vars_' + str(i) + '_to_' + str(i + chunk_size) + '_detail.png'
                ax = plt.gca()
                #plt.suptitle(coin +' intraday time series')
                plt.tight_layout()

                ax.xaxis.set_major_locator(mdates.HourLocator(0))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(0))
                ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

                for tick in ax.xaxis.get_minor_ticks():
                    tick.label1.set_horizontalalignment('right')
                    tick.label2.set_horizontalalignment('right')
                    #tick.set_pad(0)

                plt.savefig(filename, format='png')
                plt.close()
