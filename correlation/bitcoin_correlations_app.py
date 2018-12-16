import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from bokeh.plotting import figure, curdoc
from bokeh.models import Span
from bokeh.layouts import widgetbox, row
from bokeh.models.widgets import TextInput, Paragraph


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


def pull_web_data(ticker):
    """
    Takes ticker from user input and attempts to pull OHLCV data from Yahoo Finance. If the
    ticker is valid, the correlation will be calculated between the company and Bitcoin
    :param ticker: String - user input of ticker symbol
    :return: Either a DataFrame containing the rolling correlation or an error message
    """
    start = datetime.datetime(2011, 9, 13)
    end = datetime.datetime(2017, 10, 14)

    try:
        # Attempt to read in OHLCV data from Yahoo Finance
        data = web.DataReader(ticker, 'yahoo', start, end)

        # Create new DataFrame, since we are only interested in the Adjusted Close
        temp_df = pd.DataFrame(index=data.index, columns={'price'})
        temp_df['price'] = data['Adj Close']

        # Calculate log returns for the ETF
        temp_df = calculate_log_returns(temp_df)

        # Calculate rolling correlation for
        temp_df = calculate_rolling_correlation(temp_df, bitcoin_df)

    except:
        # Error assumes the ticker inputted by user could not be found
        error_msg = 'Error: Ticker %s does not exist in Yahoo!', ticker
        return error_msg

    return temp_df


def create_figure(user_input_ticker):
    """
    Takes the user input and attempts to create a rolling correlation line chart. If user input is invalid,
    The application will simply display an HTML Paragraph with an error message.
    :param user_input_ticker: String - Input from User
    :return: Either a Figure object of the visualization or an error message
    """

    # Attempt to get data
    user_data = pull_web_data(user_input_ticker)

    # Create visualization if data was returned
    if isinstance(user_data, pd.DataFrame):
        p_height = 400
        p_width = 800

        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=1)

        # Line chart with rolling correlation
        p = figure(plot_width=p_width, plot_height=p_height, title=user_input_ticker + ' and Bitcoins Correlation',
                   x_axis_label='Time', x_axis_type='datetime',
                   y_axis_label='Correlation', y_range=(-0.5, 0.5))

        p.line(user_data.index, user_data['corr'], line_width=2)
        p.renderers.extend([hline])
    else:
        # Data was not returned, so present an error message to the user
        p = Paragraph(text="""Ticker was not found in Yahoo!""", width=200, height=100)

    return p


def update(attr, old, new):
    """
    Utility method to update the user interface
    :param attr: String - The widget to be updated
    :param old: String - previous value (currently not used, except for debugging)
    :param new: String - New value inputted from user
    """
    layout.children[1] = create_figure(new)


# Clean Bitcoin data and get first trading day
bitcoin_df = pd.read_csv('data/BCHARTS-BITSTAMPUSD.csv')
bitcoin_data, first_day = clean_bitcoin_data(bitcoin_df)
bitcoin_df = calculate_log_returns(bitcoin_data)


# Controls based on UI with default value
text_input = TextInput(value="FXE", title="Choose Ticker:")

# Adds interaction to the text input by calling the update method
text_input.on_change('value', update)

# Updates the layout
controls = widgetbox([text_input], width=200)
p = create_figure('FXE')
layout = row(controls, p)

# Used for creating Webapp on local browser (Not Juypter)
curdoc().add_root(layout)
curdoc().title = "Bitcoin Correlation"


# To run this app use following command:  bokeh serve bitcoin_correlations_app.py
