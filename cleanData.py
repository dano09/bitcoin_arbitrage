import pandas as pd


def get_bitcoin_data_by_date(start_date, end_date):
    """
    Reads in bitcoin data from coinbase (1.1 GB) and creates .csv with data
    based on the time ranges.

    ***Only needs to be done once for project
    :param start_date:
    :param end_date:
    :return:
    """
    bitcoin_data = pd.read_csv('coinbaseUSD.csv', names=['timestamp', 'price', 'amount'])
    bitcoin_data['timestamp'] = pd.to_datetime(bitcoin_data['timestamp'], unit='s')

    bitcoin_data = bitcoin_data[bitcoin_data['timestamp'] >= start_date]
    bitcoin_data = bitcoin_data[bitcoin_data['timestamp'] <= end_date]
    bitcoin_data = bitcoin_data.set_index('timestamp', drop=False)
    bitcoin_sampled_df = pd.DataFrame()

    # Take mean to get minute-level tick data
    bitcoin_sampled_df['price'] = bitcoin_data.price.resample('60S').mean()
    bitcoin_sampled_df['price'] = bitcoin_sampled_df.price.interpolate()

    bitcoin_sampled_df['amount'] = bitcoin_data.amount.resample('60S').sum()
    bitcoin_sampled_df['amount'] = bitcoin_sampled_df.amount.interpolate()

    bitcoin_sampled_df.to_csv('bitcoin_data_min_tick.csv')


def clean_news_data():
    news_data = pd.read_csv('final/news.csv')
    news_data.columns = ['author', 'description', 'popularity', 'published_at', 'source', 'title', 'url', 'url_image',
                         'nc', 'scraping_date']

    cols = ['published_at', 'author', 'source', 'title', 'description', 'url', 'url_image']
    news_data = news_data[cols]
    news_data = news_data.sort_values('published_at')
    news_data['published_at'] = pd.to_datetime(news_data['published_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # Truncate the seconds so it rounds to the minute
    news_data['published_at'] = news_data['published_at'].values.astype('<M8[m]')
    del news_data['url_image']
    news_data.to_csv('news_data_min_tick.csv', index=False)



news_start_date = '2017-09-20'
news_end_date = '2017-12-03'
get_bitcoin_data_by_date(news_start_date, news_end_date)
clean_news_data()
