import os
import config
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import norm

import seaborn as sn



# Set figure size
plt.rcParams["figure.figsize"] = (16,16)
# Disable interactive mode to suppress plots
#plt.ioff()
coin = 'BTC'
# Load 1 minute sample
print("read samples " + coin)
sample = 'train'
listTS_10m = glob.glob(config.storage_path + '\\Samples' + '\\*' + coin +'*_10m_' + sample +'.csv')
listTS_1h = glob.glob(config.storage_path + '\\Samples' + '\\sample_*' + coin + '*_1h_' + sample +'.csv')
listTS_1d = glob.glob(config.storage_path + '\\Samples' + '\\sample_*' + coin + '*_1d_' + sample + '.csv')
df_ts_10m = pd.read_csv(listTS_10m[0])
df_ts_1h = pd.read_csv(listTS_10m[0])
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
df_ts_10m['Price direction'] = np.where(df_ts_10m['RETURN_10M'] >= 0, 'Up', 'Down')

for col in columns_to_plot[3:]:
    ax = sn.boxplot(x='Price direction', y=col, data=df_ts_10m, orient='v')
    plt.tight_layout()
    filename = config.storage_path + '\\Analysis\\BoxPlots\\10m' + '\\sample_' + sample + '_10m' + '_' + coin + '_' + col + '_dist.png'
    plt.savefig(filename, format='png')
    plt.close()

df_ts_1h = df_ts_1h.set_index(['LASTUPDATE'])
df_ts_1h.index = pd.to_datetime(df_ts_1h.index)
df_ts_1h['Price direction'] = np.where(df_ts_1h['RETURN_1H']>= 0, 'Up', 'Down')

for col in columns_to_plot[3:]:
    ax = sn.boxplot(x='Price direction', y=col, data=df_ts_1h, orient='v')
    plt.tight_layout()
    filename = config.storage_path + '\\Analysis\\BoxPlots\\1h' + '\\sample_' + sample + '_1h'+ '_' + coin + '_' + col + '_dist.png'
    plt.savefig(filename, format='png')
    plt.close()


df_ts_1d = df_ts_1d.set_index(['LASTUPDATE'])
df_ts_1d.index = pd.to_datetime(df_ts_1d.index)
df_ts_1d['Price direction'] = np.where(df_ts_1d['RETURN_1D']>= 0, 'Up', 'Down')

for col in columns_to_plot[3:]:
    ax = sn.boxplot(x='Price direction', y=col, data=df_ts_1d, orient='v')
    plt.tight_layout()
    filename = config.storage_path + '\\Analysis\\BoxPlots\\1d' + '\\sample_' + sample +'_1d'+ '_' + coin + '_' + col + '_dist.png'
    plt.savefig(filename, format='png')
    plt.close()
