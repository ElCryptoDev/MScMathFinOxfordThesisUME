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

ax = sn.distplot(df_ts_10m[columns_to_plot[0]].values, kde=True, rug=True, kde_kws={'clip': (-0.03, 0.03)},
                 hist_kws={'range': (-0.03, 0.03)}, rug_kws={"alpha": 0.3, "height": 0.02},
                 fit_kws={'clip': (-0.03, 0.03)}, axlabel='RETURN_10M').set(
    title='Distribution of 10-minute return for ' + coin + ' ' + sample + ' sample')
plt.tight_layout()
filename = config.storage_path + '\\Analysis\\ReturnDist' + '\\sample_' + sample + '_' + coin + '_10m_ret_dist.png'
plt.savefig(filename, format='png')
plt.close()


df_ts_1h = df_ts_1h.set_index(['LASTUPDATE'])
df_ts_1h.index = pd.to_datetime(df_ts_1h.index)

ax = sn.distplot(df_ts_1h[columns_to_plot[1]].values, kde=True, rug=True, kde_kws={'clip': (-0.03, 0.03)},
                 hist_kws={'range': (-0.03, 0.03)}, rug_kws={"alpha": 0.3, "height": 0.02},
                 fit_kws={'clip': (-0.03, 0.03)}, axlabel='RETURN_1H').set(
    title='Distribution of 1-hour return for ' + coin + ' ' + sample + ' sample')
plt.tight_layout()
filename = config.storage_path + '\\Analysis\\ReturnDist' + '\\sample_' + sample + '_' + coin + '_1h_ret_dist.png'
plt.savefig(filename, format='png')
plt.close()

df_ts_1d = df_ts_1d.set_index(['LASTUPDATE'])
df_ts_1d.index = pd.to_datetime(df_ts_1d.index)


ax = sn.distplot(df_ts_1d[columns_to_plot[2]].values, kde=True, rug=True, kde_kws={'clip': (-0.03, 0.03)},
                 hist_kws={'range': (-0.03, 0.03)}, rug_kws={"alpha": 0.3, "height": 0.02},
                 fit_kws={'clip': (-0.03, 0.03)}, axlabel='RETURN_1D').set(
    title='Distribution of 1-day return for ' + coin + ' ' + sample + ' sample')
plt.tight_layout()
filename = config.storage_path + '\\Analysis\\ReturnDist' + '\\sample_' + sample + '_' + coin + '_1d_ret_dist.png'
plt.savefig(filename, format='png')
plt.close()
