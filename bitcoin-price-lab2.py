# Import Modules and settings for styling
import pandas as pd
import matplotlib
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.mpl_style', 'default')
plt.rcParams['font.family'] = 'serif'

def read_data(exchange):
    """
    Read .csv data covering trades since at least 2016 for each exchange
    :param exchange: String
    :return: Dataframe
    """
    path = r'/home/justin/PycharmProjects/bitcoin_arbitrage/' + exchange + '_2016'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    csv_df = pd.DataFrame(columns=('timestamp', 'price', 'amount'))

    # Read in each CSV file for a given exchange
    for file in all_files:
        df = pd.read_csv(file, names=['timestamp', 'price', 'amount'], skiprows=1)

        # Manually drop last row, as read_csv does not support it with c-engine
        df.drop(df.index[len(df) - 1])
        csv_df = csv_df.append(df)

    # Convert Unix Timestamp to Date format (e.g. 1495803874 -> 2017-05-26 13:04:34)
    csv_df['timestamp'] = pd.to_datetime(csv_df['timestamp'], unit='s')
    csv_df = csv_df.sort_values(by='timestamp')

    # Remove nan values
    csv_df = csv_df.dropna()
    return csv_df.set_index('timestamp')

bitstamp_df = read_data('bitstamp')
kraken_df = read_data('kraken')
coinbase_df = read_data('coinbase')

def clean_data(df):
    """
    Resample data to the minute-tick level. Also take subset so to only look
    at trades for the year 2017
    :param df: dirty dataframe to be cleaned
    :return: cleaned dataframe
    """
    sampled_df = pd.DataFrame()
    sampled_df['price'] = df.price.resample('60S').mean()
    sampled_df = sampled_df[(sampled_df.index >= '2017-01-01 00:00')]
    return sampled_df

bitstamp_df = clean_data(bitstamp_df)
kraken_df = clean_data(kraken_df)
coinbase_df = clean_data(coinbase_df)

def calculate_spread(exchange1, exchange2, spread_name, ex1_col, ex2_col):
    """
    Calculate spreads for two exchanges
    :param exchange1: dataframe of first exchange
    :param exchange2: dataframe of second exchange
    :param spread_name: string to name column of spread
    :param ex1_col: string to name first exchange column price
    :param ex2_col: string to name second exchange column price
    :return: dataframe with spreads
    """
    spread_df = pd.DataFrame(index=exchange2.index, columns=[ex1_col, ex2_col, spread_name])
    spread_df[ex1_col] = exchange1['price']
    spread_df[ex2_col] = exchange2['price']
    spread_df[spread_name] = (spread_df[ex1_col] - spread_df[ex2_col]).abs()

    return spread_df

kb_df = calculate_spread(kraken_df, bitstamp_df,  'kb_spread', 'k', 'b')
cb_df = calculate_spread(coinbase_df, bitstamp_df, 'cb_spread', 'c', 'b')


def calculate_arbitrage_duration(df, spread_col_name):
    """
    First takes subset of dataframe with rows that are considered an arbitrage opportunity.
    Next take cumulative sum of each consecutive row to determine the duration of each opportunity.
    :param df: Dataframe with the spread calculated between two exchanges
    :param spread_col_name: The name of the column in [df] that contains the spread
    :return: dataframe with duration count for each arbitrage opportunity
    """
    # Find rows that are arbitrage opportunities. The assumption has been at least $10 USD
    arbitrage_df = df.loc[df[spread_col_name] >= 10]

    # Create temporary timestamp column since diff() function does not work on index
    arbitrage_df['timestamp'] = arbitrage_df.index

    # Determine which rows have consecutive timestamps
    bool_delta = (arbitrage_df.timestamp.diff() == pd.Timedelta('1 minute'))

    # Drop timestamp column now, as its no longer needed
    del arbitrage_df['timestamp']

    # Get cumulative sum for consecutive rows
    delta_count = bool_delta.cumsum()

    # Resets cumulative sum each time non-consecutive row appears
    arbitrage_df['arbitrage_duration'] = bool_delta.mul(delta_count) \
                                             .diff().where(lambda x: x < 0) \
                                             .ffill().add(delta_count, fill_value=0) + 1

    return arbitrage_df


cb_arbitrage = calculate_arbitrage_duration(cb_df, 'cb_spread')
kb_arbitrage = calculate_arbitrage_duration(kb_df, 'kb_spread')

# Plot distribution of opportunities across months
cb_sample_by_months = cb_arbitrage.loc[cb_arbitrage['arbitrage_duration'] == 1]
kb_sample_by_months = kb_arbitrage.loc[kb_arbitrage['arbitrage_duration'] == 1]

# Renaming titles for clarity
cb_sample_by_months= cb_sample_by_months.rename(columns={'arbitrage_duration': 'arbitrage_opportunities'})
kb_sample_by_months= kb_sample_by_months.rename(columns={'arbitrage_duration': 'arbitrage_opportunities'})

# Resamples data by Month
cb_sample_by_months = cb_sample_by_months.resample('M').count()
kb_sample_by_months = kb_sample_by_months.resample('M').count()

# Plotting number of opportunities by month (Bar Chart)
plot_df = pd.DataFrame(index=cb_sample_by_months.index)
plot_df['Coinbase/Bitstamp'] = cb_sample_by_months['arbitrage_opportunities']
plot_df['Kraken/Bitstamp'] = kb_sample_by_months['arbitrage_opportunities']

# Drop July as it was incomplete at time of data extraction
plot_df = plot_df[:-1]

#Plot Data
ax = plot_df.plot(kind='bar', figsize=(10,8))

# Format Visualization
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June'], rotation=0)
ax.legend(prop={'size': 12})
ax.set_title('Arbitrage Opportunities in 2017', fontsize=16)
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('# of Opportunities', fontsize=14);

# Sample by days
cb_sample_by_days = cb_arbitrage.loc[cb_arbitrage['arbitrage_duration'] == 1]
cb_sample_by_days = cb_sample_by_days.resample('D').count()
kb_sample_by_days = kb_arbitrage.loc[kb_arbitrage['arbitrage_duration'] == 1]
kb_sample_by_days = kb_sample_by_days.resample('D').count()

# Create two subplots (one for each cross)
f, axarr = plt.subplots(2, sharex=True, sharey=True, figsize=(12,8))

# Plot Data
ax = cb_sample_by_days['arbitrage_duration'].plot(ax=axarr[0], label='Coinbase/Bitstamp')
kb_sample_by_days['arbitrage_duration'].plot(ax=axarr[1], color='mediumpurple', label='Kraken/Bitstamp')

# Format Visualization
ax.set_title('Bitcoin Arbitrage Opportunities', fontsize=16)
ax.set_ylabel('# of Opportunities', fontsize=14)
axarr[1].set_ylabel('# of Opportunities', fontsize=14)
axarr[1].set_xlabel('Time', fontsize=14)
axarr[0].legend(loc='upper left', fontsize=12)
axarr[1].legend(loc='upper left', fontsize=12)
f.subplots_adjust(hspace=.1);


def extract_duration_df(df):
    reset_indicies = []

    # Get each tick data that resets to 1
    reset_rows = df.index[df['arbitrage_duration'] == 1]

    # Determine index in dataframe for each reset row
    for r in reset_rows:
        reset_indicies.append(df.index.get_loc(r))

    reset_indicies.pop(0)
    reset_indicies = np.asarray(reset_indicies)
    target_indicies = reset_indicies - 1

    # Return an that array shows the indexes in the dataframe that contain
    # the total time for a specific arbitrage opportunity
    return df.iloc[target_indicies]


cb_duration_df = extract_duration_df(cb_arbitrage)
kb_duration_df = extract_duration_df(kb_arbitrage)

# First analysis to cover different groupings over one hour
cb_duration_df1 = cb_duration_df.copy()
kb_duration_df1 = kb_duration_df.copy()

cb_duration_df1['cb_duration_category'] = pd.cut(cb_duration_df['arbitrage_duration'],
                                                 bins=[1, 5, 10, 15, 20, 30, 45, 60], include_lowest=True)
kb_duration_df1['kb_duration_category'] = pd.cut(kb_duration_df['arbitrage_duration'],
                                                 bins=[1, 5, 10, 15, 20, 30, 45, 60], include_lowest=True)

# Combine indicies of both dataframes
combine_df = pd.concat([cb_duration_df1, kb_duration_df1])

# Only concerned with duration column for visualization
plot_df = pd.DataFrame()
plot_df['kb_duration'] = combine_df['kb_duration_category']
plot_df['cb_duration'] = combine_df['cb_duration_category']
plot_df = plot_df.sort_index()

plot_df.head()

# Plot setup
width = 0.3
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

# Plot Data
plot_df['cb_duration'].value_counts().plot(kind='bar', color='C0', ax=ax, width=width, position=1, align='center', label='Coinbase/Bitstamp')
plot_df['kb_duration'].value_counts().plot(kind='bar', color='C1', ax=ax, width=width, position=0, align='center', label='Kraken/Bitstamp')

# Format Visualization
labels = ['1-5m', '5-10m', '10-15m', '15-20m','20-30m', '30-45m', '45-60m']
ax.set_xticklabels(labels, rotation=0)
ax.set_title('Durations of Arbitrage Opportunities for the Year 2017', fontsize=16)
ax.set_xlabel('Arbitrage Duration by Minutes', fontsize=14)
ax.set_ylabel('# of Occurences', fontsize=14)
ax.legend(prop={'size': 12})
ax.autoscale();

# Plot setup
width = 0.3
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

# Plot Data
plot_df['cb_duration'].value_counts().plot(kind='bar', color='C0', ax=ax, width=width, position=1, align='center', label='Coinbase/Bitstamp')
plot_df['kb_duration'].value_counts().plot(kind='bar', color='C1', ax=ax, width=width, position=0, align='center', label='Kraken/Bitstamp')

# Format Visualization
labels = ['1-5m', '5-10m', '10-15m', '15-20m','20-30m', '30-45m', '45-60m']
ax.set_xticklabels(labels, rotation=0)
ax.set_title('Durations of Arbitrage Opportunities for the Year 2017', fontsize=16)
ax.set_xlabel('Arbitrage Duration by Minutes', fontsize=14)
ax.set_ylabel('# of Occurences', fontsize=14)
ax.legend(prop={'size': 12})
ax.autoscale();

# Clean up the columns to avoid duplicates
cb_duration_df = cb_duration_df.rename(columns={"b": "cb", "arbitrage_duration": "cb_count"})
kb_duration_df = kb_duration_df.rename(columns={"b": "kb", "arbitrage_duration": "kb_count"})
plot_df = cb_duration_df
plot_df = pd.concat([cb_duration_df, kb_duration_df], axis=1)
plot_df.head()

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

# Plot Data
ax.plot(plot_df.index, plot_df.cb_count, "o", label='Coinbase/Bitstamp')
ax.plot(plot_df.index, plot_df.kb_count, "o", label='Kraken/Bitstamp')

# Format Visualization
ax.set_title('Durations of Arbitrage Opportunities for the Year 2017', fontsize=16)
ax.set_xlabel('Time', fontsize=14)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July'], rotation=0)
ax.set_ylabel('Duration in Minutes', fontsize=14)
ax.legend(prop={'size': 12}, loc='upper left');

scatter_cb = plot_df[(plot_df['cb_count'] < 100 )]
scatter_kb = plot_df[(plot_df['kb_count'] < 100 )]

f, axarr = plt.subplots(2, sharex=True, sharey=True, figsize=(12,8))

# Plot Data
axarr[0].plot(scatter_cb.index, scatter_cb.cb_count, 'o', label='Coinbase/Bitstamp')
axarr[1].plot(scatter_kb.index, scatter_kb.kb_count, "o", color='mediumpurple', label='Kraken/Bitstamp')

# Format Visualization
axarr[0].set_title('Durations of Arbitrage Opportunities for the Year 2017', fontsize=16)
axarr[1].set_xlabel('Time', fontsize=14)
axarr[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July'], rotation=0)
axarr[0].set_ylabel('Duration in Minutes', fontsize=14)
axarr[1].set_ylabel('Duration in Minutes', fontsize=14)
axarr[0].legend(loc='upper left', fontsize=12)
axarr[1].legend(loc='upper left', fontsize=12)
f.subplots_adjust(hspace=.1);