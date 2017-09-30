# Author: Justin Dano
# FE550
# Assignment # 1

import pandas as pd
import matplotlib

matplotlib.use("Tkagg") # Needed to work for Ubuntu/Anaconda environment in Pycharm
import matplotlib.pyplot as plt
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.expand_frame_repr', False)

# For debugging
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

# Read CSV into Dataframes
bitstamp_df = pd.read_csv(filepath_or_buffer='/home/justin/Desktop/FALL_2017/FE550/fe550_lab1/bitstamp/xaj.csv.csv',
                          names=['timestamp', 'price', 'amount'])
kraken_df = pd.read_csv(filepath_or_buffer='/home/justin/Desktop/FALL_2017/FE550/fe550_lab1/kraken/xab.csv.csv.csv',
                        names=['timestamp', 'price', 'amount'])
coinbase_df = pd.read_csv(filepath_or_buffer='/home/justin/Desktop/FALL_2017/FE550/fe550_lab1/coinbase/xao.csv',
                          names=['timestamp', 'price', 'amount'])

# Convert Unix Timestamp to Date format (e.g. 1495803874 -> 2017-05-26 13:04:34)
bitstamp_df['timestamp'] = pd.to_datetime(bitstamp_df['timestamp'], unit='s')
kraken_df['timestamp'] = pd.to_datetime(kraken_df['timestamp'], unit='s')
coinbase_df['timestamp'] = pd.to_datetime(coinbase_df['timestamp'], unit='s')

# Set index to timestamp
bitstamp_df = bitstamp_df.set_index('timestamp')
kraken_df = kraken_df.set_index('timestamp')
coinbase_df = coinbase_df.set_index('timestamp')

# Re-sample datasets to minute-tick level. Done by taking the mean
# of all prices available for each second over their corresponding minute
bitstamp_sample_df = pd.DataFrame()
kraken_sample_df = pd.DataFrame()
coinbase_sample_df = pd.DataFrame()

bitstamp_sample_df['price'] = bitstamp_df.price.resample('60S').mean()
kraken_sample_df['price'] = kraken_df.price.resample('60S').mean()
coinbase_sample_df['price'] = coinbase_df.price.resample('60S').mean()


# Combine data to one dataframe, choose ds with fewest days (eliminates NA values)
price_comparison_df = coinbase_sample_df
price_comparison_df.columns = ['coinbase']
price_comparison_df['kraken'] = kraken_sample_df['price']
price_comparison_df['bitstamp'] = bitstamp_sample_df['price']

# Calcuate price differentials between exchanges
price_comparison_df['kb_spread'] = (price_comparison_df['kraken'] - price_comparison_df['bitstamp']).abs()
price_comparison_df['cb_spread'] = (price_comparison_df['coinbase'] - price_comparison_df['bitstamp']).abs()


# Plot a sample showing arbitrage opportunities
plot_sample = price_comparison_df[(price_comparison_df.index >= '2017-07-10 00:00')
                                  & (price_comparison_df.index <= '2017-07-10 12:00')]

plot_sample = plot_sample.dropna()

# First 12-hour interval of July 10th for Coinbase and Bitsamp
fig = plt.figure(figsize=(16, 12))
fig.add_subplot(211)
ax = plot_sample['coinbase'].plot()
plot_sample['bitstamp'].plot(ax=ax)

ax.fill_between(
    x=plot_sample.index,
    y1=plot_sample['coinbase'],
    y2=plot_sample['bitstamp'],
    where=plot_sample['cb_spread'] > 10,
    facecolor='lightgreen',
    label='arbitrage opportunity'
)
ax.legend()
ax.set_title('Coinbase/Bitstamp Arbitrage Opportunities')
ax.set_xlabel('July 10th Morning Hours')
ax.set_ylabel('USD/BTC')

# First 12-hour interval of July 10th for Kraken and Bitsamp
fig.add_subplot(212)
ax = plot_sample['kraken'].plot(color='#B9445D')
plot_sample['bitstamp'].plot(ax=ax)

ax.fill_between(
    x=plot_sample.index,
    y1=plot_sample['kraken'],
    y2=plot_sample['bitstamp'],
    where=plot_sample['kb_spread'] > 10,
    facecolor='lightgreen',
    label='arbitrage opportunity'
)
ax.legend()
ax.set_title('Kraken/Bitstamp Arbitrage Opportunities')
ax.set_xlabel('July 10th Morning Hours')
ax.set_ylabel('USD/BTC')

# Plot a sample showing arbitrage opportunities
plot_sample = price_comparison_df[(price_comparison_df.index >= '2017-07-10 07:00')
                                  & (price_comparison_df.index <= '2017-07-10 12:00')]

plot_sample = plot_sample.dropna()

fig = plt.figure(figsize=(16,8))
fig.add_subplot()
ax = plot_sample['kraken'].plot(color='#B9445D')
plot_sample['bitstamp'].plot(ax=ax)

ax.fill_between(
        x=plot_sample.index,
        y1=plot_sample['kraken'],
        y2=plot_sample['bitstamp'],
        where=plot_sample['kb_spread'] > 10,
        facecolor='lightgreen',
        label='arbitrage opportunity'
    )
ax.legend()
ax.set_title('Kraken/Bitstamp Arbitrage Opportunities')
ax.set_xlabel('July 10th 7:00am - 12:00pm')
ax.set_ylabel('USD/BTC')

# Plot a sample showing arbitrage opportunities
plot_sample2 = price_comparison_df[(price_comparison_df.index >= '2017-07-10 12:00')
                                  & (price_comparison_df.index <= '2017-07-10 23:59')]

plot_sample2 = plot_sample2.dropna()

# Second 12-hour interval of July 10th for Coinbase and Bitsamp
fig = plt.figure(figsize=(16,12))
fig.add_subplot(211)
ax = plot_sample2['coinbase'].plot()
plot_sample2['bitstamp'].plot(ax=ax)

ax.fill_between(
        x=plot_sample2.index,
        y1=plot_sample2['coinbase'],
        y2=plot_sample2['bitstamp'],
        where=plot_sample2['cb_spread'] > 10,
        facecolor='lightgreen',
        label='arbitrage opportunity'
    )
ax.legend()
ax.set_title('Coinbase/Bitstamp Arbitrage Opportunities')
ax.set_xlabel('July 10th Evening Hours')
ax.set_ylabel('USD/BTC')

# Second 12-hour interval of July 10th for Kraken and Bitsamp
fig.add_subplot(212)
ax = plot_sample2['kraken'].plot(color='#B9445D')
plot_sample2['bitstamp'].plot(ax=ax)

ax.fill_between(
        x=plot_sample2.index,
        y1=plot_sample2['kraken'],
        y2=plot_sample2['bitstamp'],
        where=plot_sample2['kb_spread'] > 10,
        facecolor='lightgreen',
        label='arbitrage opportunity'
    )
ax.legend()
ax.set_title('Kraken/Bitstamp Arbitrage Opportunities')
ax.set_xlabel('July 10th Evening Hours')
ax.set_ylabel('USD/BTC')

# Plot a sample showing arbitrage opportunities
plot_sample = price_comparison_df[(price_comparison_df.index >= '2017-07-10 12:00')
                                  & (price_comparison_df.index <= '2017-07-10 16:00')]

plot_sample = plot_sample.dropna()

fig = plt.figure(figsize=(16,8))
fig.add_subplot()
ax = plot_sample['kraken'].plot(color='#B9445D')
plot_sample['bitstamp'].plot(ax=ax)

ax.fill_between(
        x=plot_sample.index,
        y1=plot_sample['kraken'],
        y2=plot_sample['bitstamp'],
        where=plot_sample['kb_spread'] > 10,
        facecolor='lightgreen',
        label='arbitrage opportunity'
    )
ax.legend()
ax.set_title('Kraken/Bitstamp Arbitrage Opportunities')
ax.set_xlabel('July 10th 12:00pm - 4:00pm')
ax.set_ylabel('USD/BTC')