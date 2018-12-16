"""
Pulls news data from newsAPI
https://newsapi.org/account
email: jdano0914@gmail.com
API KEY: 085f96bfd37f4e0aa6181662a3a5b371

References:
https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html
https://newsapi.org/
"""


import requests
import pandas as pd
from datetime import datetime
from functools import reduce


def mapping():
    # A dictionary mapping each source id (from the list displayed above) to the corresponding news category
    d = {}
    response = requests.get('https://newsapi.org/v1/sources?language=en')
    response = response.json()
    for s in response['sources']:
        d[s['id']] = s['category']
    return d


def determine_category(source, m):
    # Helper function for parsing news JSON
    try:
        return m[source]
    except:
        return 'NC'


def clean_data(path):
    # Does some initial cleaning before converting to csvytest
    data = pd.read_csv(path)
    data = data.drop_duplicates('url')
    data.to_csv(path, index=False)


def get_source_name(source):
    # Helper function for parsing news JSON
    return source['name']


def get_news_from_date(date):
    key = '085f96bfd37f4e0aa6181662a3a5b371'
    topic = 'bitcoin'
    from_dt = date
    sort_by = 'popularity'  # Popularity is based specifically on the source, not the article
    url = 'https://newsapi.org/v2/everything?q={0}&from={1}&language=en&sortBy={2}&page={3}&apiKey={4}'
    responses = []
    page_counter = 1
    popularity_counter = 1
    page_flag = True
    # API using pagination on results
    while page_flag:
        # attempt to get all data for current page
        try:
            u = url.format(topic, from_dt, sort_by, page_counter, key)
            response = requests.get(u)
            r = response.json()
            if len(r['articles']) > 0:
                # do some preliminary data wrangling for each article
                for article in r['articles']:
                    source_name = get_source_name(article['source'])
                    article['source'] = source_name
                    article['popularity'] = popularity_counter
                    responses.append(r)
                    popularity_counter += 1
                page_counter += 1
            else:
                page_flag = False

        except KeyError as e:
            print(e)
            page_flag = False
        except requests.exceptions.Timeout:
            page_counter += 1
        except requests.exceptions.RequestException as e:
            print(e)
            page_flag = False

    if len(responses) > 0:
        # Transform the JSON-like structure of the news articles into a csv-like structure (dataframe)
        news = pd.DataFrame(reduce(lambda x, y: x + y, map(lambda r: r['articles'], responses)))
        news = news.dropna()
        news = news.drop_duplicates()
        d = mapping()
        news['category'] = news['source'].map(lambda s: determine_category(s, d))
        news['scraping_date'] = datetime.now()

        # Save dataframe as .csv file
        with open('/home/justin/anaconda3/envs/crypto_trading/news.csv', 'a') as f:
            news.to_csv(f, header=False, encoding='utf-8', index=False)

        clean_data('/home/justin/anaconda3/envs/crypto_trading/news.csv')


if __name__ == '__main__':
    get_news_from_date('2017-8-17')
