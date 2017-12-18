import nltk
from nltk.corpus import stopwords
import pandas as pd
import urllib.request
import re
import numpy as np
import _pickle as pickle
pd.set_option('display.width', 5000)


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def generate_full_corpus(news_df):
    """ NOT USED - HERE FOR REFERENCE
    Creates the vocabulary and feature vectors for naive bayes scraping the web
    :param news_df: Dataframe of either positive or negative sentiment
    :return: List of Dictionaries, each containing a timestamp as a key, and a set of unique words
    """
    articles = {}
    for index, row in news_df.iterrows():
        url = row['url']

        # Attempt to Scrape article from web
        try:
            page = urllib.request.urlopen(url)
            page_data = page.read().decode(page.headers.get_content_charset())

            # Take words only in paragraph elements
            paragraphs = re.findall("<p>(.*?)</p>", str(page_data))
            words = set()

            # Combine all words in paragraph to a unique set
            for p in paragraphs:
                regex = re.compile('[^a-zA-Z]')
                words |= set(regex.sub(' ', p).split())

            # Include the title and description to corpus
            words |= set(row['description'].split())
            words |= set(row['title'].split())
            articles[index] = words

        except Exception as e:
            print(e)
            print('Error message: {} \n Parsing article from: {}'.format(e, row['url']))
    return articles


def generate_corpus(news_df):
    """
    Creates the vocabulary and feature vectors for naive bayes using just title and description
    :param news_df: Dataframe of either positive or negative sentiment
    :return: List of Dictionaries, each containing a timestamp as a key, and a set of unique words
    """
    articles = {}
    for index, row in news_df.iterrows():
        words = set()
        # Clean up the string, removing special characters
        description = row['description'].lower()
        description = re.sub('[^A-Za-z0-9 ]+', '', description)

        title = row['title'].lower()
        title = re.sub('[^A-Za-z0-9 ]+', '', title)

        # Combine words into one set
        words |= set(description.split())
        words |= set(title.split())
        articles[index] = words

    return articles


def classify_articles(articles_df, crypto_df, time_offset):
    """
    Determines if the price of the cryptocurrency increased or decreased
    since the time the article is published, and the time_offset
    :param articles_df:
    :param crypto_df:
    :param time_offset: Minutes
    :return: articles_df (with additional columns sentiment)
    """
    # Determine if the returns are positive or negative
    crypto_df['next_price'] = crypto_df.price.shift(-time_offset)
    crypto_df['sentiment'] = np.sign(crypto_df.next_price - crypto_df.price)

    articles_df = pd.concat([articles_df, crypto_df], axis=1, join_axes=[articles_df.index])

    # Remove unwanted columns
    del articles_df['timestamp']

    return articles_df.dropna()


def define_vocab(pos_articles, neg_articles):
    """
    Defines the vocabulary for all words in training data
    :param pos_articles:
    :param neg_articles:
    :return:
    """
    vocabulary = set()
    for key, value in pos_articles.items():
        vocabulary = vocabulary.union(value)

    for key, value in neg_articles.items():
        vocabulary = vocabulary.union(value)

    # Now Remove stop words
    stop_words= set(stopwords.words("english"))
    filtered_vocab = [word for word in vocabulary if word not in stop_words]

    # Remove non-english words
    english_words = set(nltk.corpus.words.words())
    filtered_vocab = [word for word in filtered_vocab if word in english_words]

    return filtered_vocab


def extract_features(news_article):
    """
    Method used in NLTK to determine features of a news article
    :param news_article: List of words from a specific article
    :return: feature vector from article
    """
    review_words = set(news_article)
    features = {}
    for word in vocabulary:
        features[word] = (word in review_words)
    return features


def get_trained_naive_bayes_classifier(extract_features, training_data):
    """
    Creates the Naives Bayes classifier
    :param extract_features: Method used to create feature vectores
    :param training_data:  Data used to create classifier
    :return:
    """
    # Extract Training Features to build classifier
    trainingFeatures = nltk.classify.apply_features(extract_features, training_data)

    # Train the Classifier with Naive Bayes
    trained_nb_classifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

    return trained_nb_classifier


def naive_bayes_sentiment_calculator(article):
    """
    Wrapper function to classify an article using the Naive Bayes Classifier
    :param article: Article to classify
    :return: The sentiment of the article
    """
    problem_features = extract_features(article)
    return trained_nb_classifier.classify(problem_features)


def update_test_dataframe(naive_bayes_sentiment_calculator, news_df, corpus):
    """
    Use classifier on the test data. Returns results os a Dataframe
    :param naive_bayes_sentiment_calculator:
    :param news_df:
    :param corpus:
    :return:
    """
    news_df['predicted'] = None
    for index, row in news_df.iterrows():
        news_df.loc[index, 'predicted'] = 1.0 if naive_bayes_sentiment_calculator(list(corpus[index])) == 'positive' else -1.0

    return news_df


def get_test_review_sentiments(naive_bayes_sentiment_calculator, test_pos_data, test_neg_data):
    """
    Helper function to determine the accuracy of the Naive Bayes Classifier
    :param naive_bayes_sentiment_calculator:
    :param test_pos_data:
    :param test_neg_data:
    :return:
    """
    test_neg_results = [naive_bayes_sentiment_calculator(article_and_label[0]) for article_and_label in test_neg_data]
    test_pos_results = [naive_bayes_sentiment_calculator(article_and_label[0]) for article_and_label in test_pos_data]

    label_to_num = {'positive': 1, 'negative': -1}
    numeric_neg_results = [label_to_num[x] for x in test_neg_results]
    numeric_pos_results = [label_to_num[x] for x in test_pos_results]

    return {'results-on-positive': numeric_pos_results, 'results-on-negative': numeric_neg_results}


def run_diagnostics(article_results):
    """
    Computes the accuracy statistics of the Naive Bayes Classifier
    :param article_results:
    :return:
    """
    positive_articles_results = article_results['results-on-positive']
    negative_articles_results = article_results['results-on-negative']

    num_true_pos = sum(x > 0 for x in positive_articles_results)
    num_true_neg = sum(x < 0 for x in negative_articles_results)

    pct_true_pos = float(num_true_pos) / len(positive_articles_results)
    pct_true_neg = float(num_true_neg) / len(negative_articles_results)
    total_accurate = num_true_pos + num_true_neg
    total = len(positive_articles_results) + len(negative_articles_results)

    accuracy_on_pos = pct_true_pos * 100
    accuracy_on_neg = pct_true_neg * 100
    overall_accuracy = (total_accurate * 100) / total
    print('Number of Positive articles correctly predicted is: {}  \nTotal number of positive articles is: {}'
          .format(num_true_pos, len(positive_articles_results)))
    print('Number of Negative articles correctly predicted is: {} \nTotal number of negative articles is: {}'
          .format(num_true_neg, len(negative_articles_results)))
    print('Accuracy on positive news is: {0:.2f}%'.format(accuracy_on_pos))
    print('Accuracy on negative news is: {0:.2f}%'.format(accuracy_on_neg))
    print('Total accuracy of Naive Bayes Classifier is: {0:.2f}%'.format(overall_accuracy))


def save_training_set(news):
    training_news = news[:news.shape[0] // 2]
    training_news.to_csv('training_news.csv', index=False)

if __name__ == '__main__':

    ''' This flag is used to create the classifier.
        When set to False, it will extract the the nb_classifier.p object instead.
        This is done for performance as the classifier takes several minutes to complete.'''
    generate_classifier_flag = True

    ''' This parameter determines how the naive bayes classifier classifies the news sentiment.
        It works by comparing the price at t + time_offset_minutes
        where t = the time the article is published'''
    time_offset_minutes = 1

    # Read in .csv data for News and Bitcoin Prices
    news_data = pd.read_csv('final/news_data_min_tick.csv')
    news_data['published_at'] = pd.to_datetime(news_data['published_at'])
    news_data = news_data.set_index('published_at', drop=False)

    bitcoin_data = pd.read_csv('final/bitcoin_data_min_tick.csv')
    bitcoin_data['timestamp'] = pd.to_datetime(bitcoin_data['timestamp'])
    bitcoin_data = bitcoin_data.set_index('timestamp', drop=False)

    # Determine sentiment of news articles (either +1 or -1) by change in price at time_offset_minutes
    classified_news = classify_articles(news_data, bitcoin_data, time_offset_minutes)

    # Save off training set for Visualization
    save_training_set(classified_news)

    # Split news by into binary classification of either positive sentiment or negative sentiment
    pos_news = classified_news.loc[classified_news['sentiment'] == 1]
    pos_news = pos_news.reset_index(drop=True)
    neg_news = classified_news.loc[classified_news['sentiment'] == -1]
    neg_news = neg_news.reset_index(drop=True)

    # Split news into Training and Testing Data
    pos_news_training = pos_news[:pos_news.shape[0] // 2]
    pos_news_testing = pos_news[pos_news.shape[0] // 2:]
    neg_news_training = neg_news[:neg_news.shape[0] // 2]
    neg_news_testing = neg_news[neg_news.shape[0] // 2:]

    # Create Corpus by finding unique set of words in each article
    pos_corpus_training = generate_corpus(pos_news_training)
    pos_corpus_testing = generate_corpus(pos_news_testing)
    neg_corpus_training = generate_corpus(neg_news_training)
    neg_corpus_testing = generate_corpus(neg_news_testing)

    # Define vocabulary for the naive bayes classifier. Make sure to only use training corpus
    vocabulary = define_vocab(pos_corpus_training, neg_corpus_training)

    # Data formatting for NLTK -- need list of tuples (review, label)
    training_data = []
    for key, value in pos_corpus_training.items():
        training_data.append((list(value), 'positive'))

    for key, value in neg_corpus_training.items():
        training_data.append((list(value), 'negative'))

    # Setup testing data for NLTK -- need list of tuples (review, label)
    # Split here so we can determine accuracy of both positive and negative predictions
    test_pos_data = []
    test_neg_data = []
    for key, value in pos_corpus_testing.items():
        test_pos_data.append((list(value), 'positive'))

    for key, value in neg_corpus_testing.items():
        test_neg_data.append((list(value), 'negative'))

    # Create Naive Bayes Classifier
    trained_nb_classifier = None

    # Either create classifier or pull the serialized classifier from file storage. If it is the first run,
    # the classifier must be generated
    if generate_classifier_flag:
        print('Training Naive Bayes Classifier. This will take some time.')
        trained_nb_classifier = get_trained_naive_bayes_classifier(extract_features, training_data)
        # Save classifier for later use
        fileHandle = open('nb_classifier.p', 'wb')
        pickle.dump(trained_nb_classifier, fileHandle, -1)
        fileHandle.close()
    else:
        fileHandle = open('nb_classifier.p', 'rb')
        trained_nb_classifier = pickle.load(fileHandle)

    # Create dataframe with predicted values. Save as .CSV for visualization
    print('\nPredicting results on positive news sentiment. This will take some time')
    predicted_pos_data = update_test_dataframe(naive_bayes_sentiment_calculator, pos_news_testing, pos_corpus_testing)
    print('\nPredicting results on positive news sentiment. This will take some time')
    predicted_neg_data = update_test_dataframe(naive_bayes_sentiment_calculator, neg_news_testing, neg_corpus_testing)
    predicted_news = pd.concat([predicted_pos_data, predicted_neg_data])
    predicted_news.to_csv('final/news_sentiment_predictions.csv', index=False)

    # Run these to determine accuracy of results
    run_accuracy_test = True
    if run_accuracy_test:
        test_results = get_test_review_sentiments(naive_bayes_sentiment_calculator, test_pos_data, test_neg_data)
        run_diagnostics(test_results)

        fileHandle = open('final/nbc_results.p', 'wb')
        pickle.dump(test_results, fileHandle, -1)
        fileHandle.close()



