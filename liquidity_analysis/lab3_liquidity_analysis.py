# Justin Dano <br>
# FE550 - Data Visualization Applications
# Cryptocurrency Arbitrage - Liquidity and the Order Book

# Import Modules and settings for styling
import urllib.request
import json
import time
import csv
import os
import warnings
import pandas as pd
import pprint as pp
import locale
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import LinearAxis, Range1d, ColumnDataSource, NumeralTickFormatter

locale.setlocale( locale.LC_ALL, '' )
warnings.filterwarnings('ignore')

chart_styling = {'axis_size':'12pt',
                 'title_size':'14pt',
                 'font':'times',
                 'legend_pos': 'top_right',
                 'legend_font': '8pt'}

#output_notebook()


def print_full(x):
    """
    For debugging
    :param x: Dataframe
    :return:
    """
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def write_to_csv(filename, data, directory):
    """
    Creates CSV file from order book data
    :param filename: name of csv file to be created
    :param data: list of prices/amounts
    """

    # Creates CSV file
    with open('order_book_data/' + str(directory) + '/' + filename, 'w') as csvfile:
        csvout = csv.writer(csvfile)
        for row in data:
            csvout.writerow(row)


def format_order_book_data_and_save(exchange, timestamp):
    """
    Formats order book data and saves into csv file
    :param exchange: String - name of exchange used in cryptowatch
    :param timestamp: String - timestamp of order book. used to create unique directory
    """

    # Assemble order book for CSV. The bids and asks are saved in different files.
    order_book = json.load(exchange[0])['result']
    asks = order_book['asks']
    bids = order_book['bids']

    csv_asks_file = exchange[1] + '_asks.csv'
    csv_bids_file = exchange[1] + '_bids.csv'

    write_to_csv(csv_asks_file, asks, timestamp)
    write_to_csv(csv_bids_file, bids, timestamp)


def get_order_book_data(list_of_exchanges):
    """
    Make HTTP Requests to pull order book data from Cryptowatch
    :param list_of_exchanges: list of different exchanges
    :return:
        exchange_responses: list of tuples (HTTP Response for Order book, name of exchange)
        timestamp: string of timestamp of data retrieval
    """

    exchange_responses = []

    for exchange in list_of_exchanges:
        # Read in data from Cryptowatch
        url = 'https://api.cryptowat.ch/markets/' + exchange + '/btcusd/orderbook'
        exchange_responses.append((urllib.request.urlopen(url), exchange))

    # Time will be off by a few microseconds. This analysis will assume they are equivalent for each exchange.
    timestamp = pd.to_datetime(time.time(), unit='s')

    return exchange_responses, timestamp


def read_data(exchange, time_extension):
    """
    Reads order book data saved in .csv files
    :param exchange: String - identifies the order books exchange
    :param time_extension: String - timestamp of order book. Used to create unique directory
    :return: Dataframe - Order book data
    """

    directory_path = r'/home/justin/PycharmProjects/bitcoin_arbitrage/order_book_data/' + time_extension
    # directory_path = r'your_path_to_submitted_data' + time_extension

    asks_file = directory_path + '/' + exchange + '_asks.csv'
    bids_file = directory_path + '/' + exchange + '_bids.csv'

    df_asks = pd.read_csv(asks_file, names=['ask_price', 'ask_volume'])
    df_bids = pd.read_csv(bids_file, names=['bid_price', 'bid_volume'])

    return pd.concat([df_bids, df_asks], axis=1)


def calculate_liquidity(order_book):
    """
    Calculates liquidity by taking cumulative sum of volume for bids and asks
    :param order_book: Dataframe - Order book with bids/asks
    :return: Dataframe - Order book with bids/asks and liquidity
    """
    order_book['ask_liquidity'] = order_book['ask_volume'].cumsum()
    order_book['bid_liquidity'] = order_book['bid_volume'].cumsum()

    # Reorder columns to align order book
    cols = ['bid_liquidity', 'bid_volume', 'bid_price', 'ask_price', 'ask_volume', 'ask_liquidity']

    return order_book[cols]


def generate_plot_coordinates(order_book, samples):
    """
    Generates the x and y coordinates used to create the shape of the order book.
    :param order_book: Dataframe - order book data to be visualized
    :param samples: Int - number of bids/asks to be plotted
    :return: Tuple - coordinates for the ask shape and bid shape
    """
    x_bid_coordinates = []
    y_bid_coordinates = []

    x_ask_coordinates = []
    y_ask_coordinates = []

    for i in range(samples):
        x_bid_coordinates.append(order_book['bid_price'][i])
        y_bid_coordinates.append(order_book['bid_liquidity'][i])

        x_ask_coordinates.append(order_book['ask_price'][i])
        y_ask_coordinates.append(order_book['ask_liquidity'][i])

    # Need to repeat first and last coordinate to create the bottom part of shape
    x_bid_coordinates = [x_bid_coordinates[0]] + x_bid_coordinates + [x_bid_coordinates[-1]]
    y_bid_coordinates = [0] + y_bid_coordinates + [0]

    x_ask_coordinates = [x_ask_coordinates[0]] + x_ask_coordinates + [x_ask_coordinates[-1]]
    y_ask_coordinates = [0] + y_ask_coordinates + [0]

    return x_bid_coordinates, x_ask_coordinates, y_bid_coordinates, y_ask_coordinates


def plot_order_book(order_book, order_count, exchange, ob_time, styling):
    """
    Creates a visualization of the Order Book Bids/Asks
    :param order_book: Dataframe - Order Book data
    :param exchange: String - Name of the order books exchange
    :param ob_time: String - Timestamp of order book
    :param styling: Dict - Styling parameters
    """

    # Get x/y coordinates for both bids and asks of order book
    order_book_coords = generate_plot_coordinates(order_book, order_count)

    # Create a new plot with a title and axis labels
    plot_title = ' Order Book for ' + str(exchange) + ' at ' + ob_time
    p = figure(title=plot_title,
               x_axis_label='USD/BTC',
               y_axis_label='Liquidity (BTC\'s)',
               plot_width=950,
               plot_height=500
               )

    # Create source for the data points, colors, and legend
    source = ColumnDataSource(dict(
        x_axis=[order_book_coords[0], order_book_coords[1]],
        y_axis=[order_book_coords[2], order_book_coords[3]],
        color=['green', 'red'],
        label=['Bids', 'Asks']
    ))

    # Plot order book with coordinates
    p.patches(xs='x_axis', ys='y_axis', color='color', legend='label', alpha=0.5, line_width=2, source=source)

    # Graph Formatting
    p.xaxis.formatter = NumeralTickFormatter(format="=$ 0,0[.]00")
    p.title.text_font_size = styling['title_size']
    p.xaxis.axis_label_text_font_size = styling['axis_size']
    p.yaxis.axis_label_text_font_size = styling['axis_size']

    p.title.text_font = styling['font']
    p.xaxis.axis_label_text_font = styling['font']
    p.yaxis.axis_label_text_font = styling['font']

    show(p)


def plot_dual_order_books(ob1, ob2, order_count, e1, e2, ob_time, x_scale, y_scale, styling, align_flag):
    """
    Creates a visualization of two order books on the same graph
    :param ob1: Dataframe - Order book from first exchange
    :param ob2: Dataframe - Order book from first exchange
    :param order_count: Int - number of orders to show in order book (max=99)
    :param e1: String - Name of exchange for ob1_coords
    :param e2: String - Name of exchange for ob2_coords
    :param ob_time: String - Timestamp
    :param x_scale: List of ints - parameter to define how the graph displays price
    :param y_scale: List of ints - parameter to define how the graph displays liquidity
    :param styling: Dict - Styling parameters
    :param align_flag: Bool - Determines whether vertical lines should be added or not
    """

    # Get x/y coordinates for both order books, including bids and asks up to {order_count}
    ob1_coords = generate_plot_coordinates(ob1, order_count)
    ob2_coords = generate_plot_coordinates(ob2, order_count)

    plot_title = ' Order Book for ' + str(e1) + ' and ' + str(e2) + ' at ' + ob_time

    # Create a new plot with a title and two axes
    p = figure(title=plot_title,
               x_axis_label='USD/BTC',
               y_axis_label=str(e1) + ' Liquidity (BTC\'s)',
               plot_width=950,
               plot_height=500,
               x_range=Range1d(x_scale[0], x_scale[1]),
               y_range=Range1d(y_scale[0], y_scale[1]),
               toolbar_location='above'
               )

    # Create second axis
    p.extra_y_ranges = {'ex2': Range1d(start=y_scale[1], end=y_scale[0])}

    # Adding the second axis to the plot.
    p.add_layout(LinearAxis(y_range_name="ex2", axis_label=str(e2) + ' Liquidity (BTC\'s)'), 'right')

    # Create source for first exchange, including coordinates, colors, and legend
    source1 = ColumnDataSource(dict(
        x_axis=[ob1_coords[0], ob1_coords[1]],
        y_axis=[ob1_coords[2], ob1_coords[3]],
        color=['lightskyblue', 'mediumpurple'],
        label=[str(e1) + ' Bids', str(e1) + ' Asks']
    ))

    # Create source for second exchange, including coordinates, colors, and legend
    source2 = ColumnDataSource(dict(
        x_axis=[ob2_coords[0], ob2_coords[1]],
        y_axis=[ob2_coords[2], ob2_coords[3]],
        color=['green', 'red'],
        label=[str(e2) + ' Bids', str(e2) + ' Asks']
    ))

    # Plot the first order book
    p.patches(xs='x_axis', ys='y_axis', color='color', legend='label',
              alpha=0.5, line_width=2, source=source1)

    # Plot the second order book
    p.patches(xs='x_axis', ys='y_axis', color='color', legend='label',
              alpha=0.5, line_width=2, y_range_name="ex2", source=source2)

    # Adds vertical lines to align the arbitrage
    if align_flag:
        left_vertical = (ob1_coords[1][0], (y_scale[0], y_scale[1]))
        p.line(left_vertical[0], left_vertical[1])
        right_vertical = (ob2_coords[0][0], (y_scale[0], y_scale[1]))
        p.line(right_vertical[0], right_vertical[1])

        # Graph Formatting
    p.xaxis.formatter = NumeralTickFormatter(format="=$ 0,0[.]00")
    p.title.text_font_size = styling['title_size']
    p.xaxis.axis_label_text_font_size = styling['axis_size']
    p.yaxis.axis_label_text_font_size = styling['axis_size']
    p.title.text_font = styling['font']
    p.xaxis.axis_label_text_font = styling['font']
    p.yaxis.axis_label_text_font = styling['font']
    p.legend.label_text_font = styling['font']
    p.legend.location = styling['legend_pos']
    p.legend.label_text_font_size = styling['legend_font']

    show(p)


def determine_arbitrage_orders(ob1, ob2):
    """
    Filters the orders that can be bough and sold for an arbitrage
    :param ob1: Dataframe - The first order book
    :param ob2: Dataframe - The second order book
    :return: Tuple of DataFrames - filtered by orders that con be arbitraged
    """

    # Determine which exchange is cheaper to buy
    low_exchange = pd.DataFrame()
    high_exchange = pd.DataFrame()

    if ob1['ask_price'][0] > ob2['ask_price'][0]:
        low_exchange = ob2[['ask_price', 'ask_volume', 'ask_liquidity']].copy()
        high_exchange = ob1[['bid_price', 'bid_volume', 'bid_liquidity']].copy()
    else:
        low_exchange = ob1[['ask_price', 'ask_volume', 'ask_liquidity']].copy()
        high_exchange = ob2[['bid_price', 'bid_volume', 'bid_liquidity']].copy()

    # Determine lowest ask from cheaper exchange
    lowest_ask = low_exchange['ask_price'][0]

    # Determine highest bid from other exchange
    highest_bid = high_exchange['bid_price'][0]

    # Get orders that can be arbitraged
    buy_orders = low_exchange[low_exchange['ask_price'] < highest_bid]
    sell_orders = high_exchange[high_exchange['bid_price'] > lowest_ask]

    return buy_orders, sell_orders


def calculate_profit(buy_orders, sell_orders):
    """
    Calculates and prints the total profit made from the arbitrage
    :param buy_orders: DataFrame - orders to buy
    :param sell_orders: DataFrame - orders to sell
    """

    # Determine weighted price of buy orders
    buy_orders['total_price'] = (buy_orders['ask_price'] * buy_orders['ask_volume']).cumsum()

    # Get last sell order since it will most likely be a partial
    last_sell_order = sell_orders.iloc[-1]
    sell_orders = sell_orders[:-1]

    # Calculate the liquidity to be used for the final order
    remaining_liquidity = buy_orders.iloc[-1]['ask_liquidity']
    for index, order in sell_orders.iterrows():
        remaining_liquidity -= order['bid_liquidity']

    sell_orders['total_price'] = (sell_orders['bid_price'] * sell_orders['bid_volume']).cumsum()

    # Determine weighted price of the final partial trade
    last_partial_order_price = last_sell_order['bid_price'] * remaining_liquidity

    # Total selling orders with the partial
    total_sell_price = sell_orders.iloc[-1]['total_price'] + last_partial_order_price
    total_buy_price = buy_orders.iloc[-1]['total_price']

    print('Total price paid for buying: ' + locale.currency(total_buy_price, grouping=True))
    print('Total price received for selling: ' + locale.currency(total_sell_price, grouping=True))

    arbitrage_profit = total_sell_price - total_buy_price
    print('Arbitrage profit: ' + locale.currency(arbitrage_profit, grouping=True))


if __name__ == '__main__':
    # Pull order book data from the web
    order_book_responses, eob_timestamp = get_order_book_data(['gdax', 'kraken', 'bitstamp'])

    # Creates unique Directory
    os.makedirs('order_book_data/' + str(eob_timestamp))

    for response in order_book_responses:
        format_order_book_data_and_save(response, eob_timestamp)

    # Pull order book data from CSV
    sample_time = '2017-10-02 16:29:21.338665'
    gdax_orderbook = read_data('gdax', sample_time)
    kraken_orderbook = read_data('kraken', sample_time)
    bitstamp_orderbook = read_data('bitstamp', sample_time)

    gdax_orderbook.head()

    # Calculate liquidity
    gdax_orderbook = calculate_liquidity(gdax_orderbook)
    kraken_orderbook = calculate_liquidity(kraken_orderbook)
    bitstamp_orderbook = calculate_liquidity(bitstamp_orderbook)

    gdax_orderbook.head()

    # Parameter to define the top bids/asks to visualize
    order_count = 99
    # Plot first 3 order books
    plot_order_book(bitstamp_orderbook, order_count, 'Bitstamp', sample_time, chart_styling)
    plot_order_book(kraken_orderbook, order_count, 'Kraken', sample_time, chart_styling)
    plot_order_book(gdax_orderbook, order_count, 'GDAX', sample_time, chart_styling)

    # Plot dual order books part 1
    order_count = 99
    x_scale = [4270, 4510]
    y_scale = [0, 500]
    chart_styling['legend_pos'] = 'bottom_right'

    plot_dual_order_books(gdax_orderbook, kraken_orderbook, order_count,
                          'GDAX', 'Kraken', sample_time, x_scale, y_scale, chart_styling, False)

    # Plot dual order books part 2
    plot_dual_order_books(gdax_orderbook, kraken_orderbook, order_count,
                          'GDAX', 'Kraken', sample_time, x_scale, y_scale, chart_styling, True)

    # Plot dual order books part 3
    order_count = 99
    x_scale = [4350, 4460]
    y_scale = [0, 300]
    chart_styling['legend_pos'] = 'bottom_left'

    plot_dual_order_books(gdax_orderbook, kraken_orderbook, order_count,
                          'GDAX', 'Kraken', sample_time, x_scale, y_scale, chart_styling, True)

    # Plot dual order books part 4
    order_count = 50
    x_scale = [4400, 4420]
    y_scale = [0, 100]

    chart_styling['legend_pos'] = 'bottom_right'

    plot_dual_order_books(gdax_orderbook, kraken_orderbook, order_count,
                          'GDAX', 'Kraken', sample_time, x_scale, y_scale, chart_styling, True)

    # Plot dual order books part 5
    order_count = 25
    x_scale = [4405, 4415]
    y_scale = [0, 30]
    chart_styling['legend_pos'] = 'bottom_right'

    plot_dual_order_books(gdax_orderbook, kraken_orderbook, order_count,
                          'GDAX', 'Kraken', sample_time, x_scale, y_scale, chart_styling, True)

    # Calculate arbitrage
    buy_orders, sell_orders = determine_arbitrage_orders(gdax_orderbook, kraken_orderbook)
    calculate_profit(buy_orders, sell_orders)