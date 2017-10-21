import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from bokeh.plotting import figure, show
from bokeh.io import gridplot
from bokeh.models import Span


def print_full(x):
    """
    For debugging
    :param x: Dataframe
    :return:
    """
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def clean_currency_data(curr_df, start_date, bitcoin_flag=False):
    """
    Takes a dataframe and performs data wrangling to prepare it for visualization
    :param curr_df: DataFrame - object to be cleaned
    :param start_date: String - Filter fiat currency data so it matches the available Bitcoin Data
    :param bitcoin_flag: Boolean - Bitcoin data uses different labels in the .csv data
    :return: DataFrame - An object that is suitable for calculating correlations
    """
    # Set Index as Date
    curr_df = curr_df.set_index('Date')

    # Convert index from string to Date
    curr_df.index = curr_df.index.to_datetime()

    # Select only subset of days based on available history of Bitcoin
    curr_df = curr_df.loc[start_date:]

    # Create new DataFrame, since we are only interested in the Adjusted Close
    temp_df = pd.DataFrame(index=curr_df.index, columns={'price'})

    if bitcoin_flag:
        temp_df['price'] = curr_df['Weighted Price']
    else:
        temp_df['price'] = curr_df['Adj Close']

    return temp_df


def clean_bitcoin_data(btc_df):
    """
    Cleans Bitcoin data (additional work needed compared to Fiat Currencies
    :param btc_df: DataFrame - OHLCV data of Bitcoin
    :return: DataFrame - Bitcoin data suitable for calculating correlations
    """

    # Reverse dataframe since .csv was originally date descending
    bitcoin_df = btc_df.iloc[::-1]

    # Get First date of available bitcoin data
    first_day = bitcoin_df['Date'].iloc[0]

    # Perform other techniques to clean data, such as setting index
    bitcoin_df = clean_currency_data(bitcoin_df, first_day, True)

    # Fill zero values with preceding price. This keeps the rolling correlation
    # from returning NaN values
    bitcoin_df['price'] = bitcoin_df['price'].replace(to_replace=0, method='ffill')

    return bitcoin_df, first_day


def calculate_log_returns(curr_df):
    """
    Takes a DataFrame and calculates the log of daily returns
    :param curr_df: DataFrame - Daily close price data
    :return: DataFrame - Daily close price data with daily log returns
    """
    curr_df['log_return'] = np.log(curr_df['price']) - np.log(curr_df.price.shift(1))
    return curr_df


def calculate_rolling_correlation(curr_df, btc_df):
    """
    Calculates the rolling correlation with a 1 year look back window
    :param curr_df: DataFrame - Fiat Currency dataframe with log returns
    :param btc_df: DataFrame - Bitcoin dataframe with log Returns
    :return: DataFrame - Combined dataframe with rolling correlation.
    """
    curr_df['btc_price'] = btc_df['price']
    curr_df['btc_log_return'] = btc_df['log_return']

    # 254 days is equivalent to one year (when excluding holidays and weekends)
    curr_df['corr'] = pd.rolling_corr(curr_df['log_return'], curr_df['btc_log_return'], window=254)
    curr_df = curr_df.dropna()
    return curr_df


def clean_and_calculate_corr(curr_df, btc_df, start_date):
    """
    Utility method that cleans the data for both Fiat and Cryptocurrencies, and calculates
    the rolling correlation between the two datasets
    :param curr_df: DataFrame - OHLCV of Fiat Currency
    :param btc_df: DataFrame - OHLCV of Bitcoin
    :param start_date: String - Date to initialize start of datasets
    :return: DataFrame - Combined dataframe with rolling correlation.
    """
    # Clean the data
    curr_df = clean_currency_data(curr_df, start_date)

    # Calculate log returns for the ETF
    curr_df = calculate_log_returns(curr_df)

    # Calculate rolling correlation for
    curr_df = calculate_rolling_correlation(curr_df, btc_df)

    return curr_df


def plot_grid(data_arr, title_arr):
    """
    Takes a list of Rolling Correlations and plots each in one grid
    :param data_arr: List [ DataFrames ] - Correlation dataframes to be plotted
    :param title_arr: List [ String ] - Title for each correlation
    """
    grid_arr = []
    p_height = 200
    p_width = 400

    # Horizontal line
    hline = Span(location=0, dimension='width', line_color='black', line_width=1)

    # Create figure and visualization for each country
    for i, curr in enumerate(data_arr):

        p = figure(plot_width=p_width, plot_height=p_height, title=title_arr[i],
                   x_axis_label='Time', x_axis_type='datetime',
                   y_axis_label='Correlation', y_range=(-0.5, 0.5))
        p.line(curr.index, curr['corr'], line_width=2)
        p.min_border_right = 40
        p.renderers.extend([hline])
        grid_arr.append(p)

    # Plots the Grid containing 6 visualizations
    gp = gridplot([[grid_arr[0], grid_arr[1]], [grid_arr[2], grid_arr[3]], [grid_arr[4], grid_arr[5]]])

    show(gp)

if __name__ == '__main__' :
    # From Bitstamp
    bitcoin_df = pd.read_csv('data/BCHARTS-BITSTAMPUSD.csv')
    # United States Greenback proxied with UUP ETF
    usd_df = pd.read_csv('data/UUP.csv')
    # China's Yuan Renmibni proxied with CYB ETF
    cny_df = pd.read_csv('data/CYB.csv')
    # Japanese Yen proxied with EWJ ETF
    jpy_df = pd.read_csv('data/EWJ.csv')
    # Australian Dollar proxied with EWA ETF
    aud_df = pd.read_csv('data/EWA.csv')
    # Euro proxied with FXE ETF
    eur_df = pd.read_csv('data/FXE.csv')
    # British Pound proxied with FXB ETF
    gbp_df = pd.read_csv('data/FXB.csv')
    # Swiss Franc proxied with FXF ETF
    chf_df = pd.read_csv('data/FXB.csv')

    # Clean Bitcoin data and get first trading day
    bitcoin_data, first_day = clean_bitcoin_data(bitcoin_df)
    bitcoin_df = calculate_log_returns(bitcoin_data)

    # Clean and calculate correlations for other currency ETFs
    usd_data = clean_and_calculate_corr(usd_df, bitcoin_df, first_day)
    cny_data = clean_and_calculate_corr(cny_df, bitcoin_df, first_day)
    jpy_data = clean_and_calculate_corr(jpy_df, bitcoin_df, first_day)
    aud_data = clean_and_calculate_corr(aud_df, bitcoin_df, first_day)
    eur_data = clean_and_calculate_corr(eur_df, bitcoin_df, first_day)
    gbp_data = clean_and_calculate_corr(gbp_df, bitcoin_df, first_day)

    currencies = [usd_data, cny_data, jpy_data, aud_data, eur_data, gbp_data]
    titles = ['United States', 'China', 'Japan', 'Australia', 'Europe', 'Great Britain']

    plot_grid(currencies, titles)

