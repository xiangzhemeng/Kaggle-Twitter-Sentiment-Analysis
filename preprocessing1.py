import nltk
import re
import pandas as pd
import numpy as np
import pickle
import multiprocessing
from multiprocessing import Pool

num_partitions = multiprocessing.cpu_count()
num_cores = multiprocessing.cpu_count()

positive_word_library = list(set(open('positive-words.txt', encoding = "ISO-8859-1").read().split()))
negative_word_library = list(set(open('negative-words.txt', encoding = "ISO-8859-1").read().split()))

def expand_contraction_word(tweets):
    tweets = tweets.str.replace('n\'t', ' not', case=False)
    tweets = tweets.str.replace('i\'m', 'i am', case=False)
    tweets = tweets.str.replace('\'re', ' are', case=False)
    tweets = tweets.str.replace('it\'s', 'it is', case=False)
    tweets = tweets.str.replace('that\'s', 'that is', case=False)
    tweets = tweets.str.replace('\'ll', ' will', case=False)
    tweets = tweets.str.replace('\'l', ' will', case=False)
    tweets = tweets.str.replace('\'ve', ' have', case=False)
    tweets = tweets.str.replace('\'d', ' would', case=False)
    tweets = tweets.str.replace('he\'s', 'he is', case=False)
    tweets = tweets.str.replace('what\'s', 'what is', case=False)
    tweets = tweets.str.replace('who\'s', 'who is', case=False)
    tweets = tweets.str.replace('\'s', '', case=False)

    for punctuation in ['!', '?', '.']:
        regex = '['+ punctuation + ']' + "+"
        tweets = tweets.str.replace(regex, punctuation + ' <repeat> ', case=False)

    return tweets

def emphasize_sentiment_words(tweet):
    t = []
    for w in tweet.split():
        if w in positive_word_library:
            t.append('<positive> ' + w)
        elif w in negative_word_library:
            t.append('<negative> ' + w)
        else:
            t.append(w)
    newTweet = " ".join(t)
    return newTweet

def filter_digits(tweet):
    t = []
    for w in tweet.split():
        # If w is a number
        try:
            float(w)
            t.append("<number>")
        # Otherwise
        except:
            t.append(w)
    newTweet = " ".join(t)
    return newTweet

def tweets_preprocessing(tweets):
    tweets['reviewText'] = expand_contraction_word(tweets['reviewText'])
    print('Contraction words expansion finished!')

    tweets['reviewText'] = tweets.apply(lambda tweet: emphasize_sentiment_words(tweet['reviewText']), axis=1)
    print('Sentiment words emphasizing finished!')

    tweets['reviewText'] = tweets.apply(lambda tweet: filter_digits(tweet['reviewText']), axis=1)
    print('Number to hashtag <number> transformation finished!')

    print('ALL DONE!')

    return tweets

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def multiply_columns(data):
    data = tweets_preprocessing(data)
    return data

df_kindle_book_review = pd.read_pickle('origin.pkl')
df_kindle_book_review['rate'] = df_kindle_book_review['rate'].apply(lambda x: 4 if x in [1,2,3] else x)
df_train = df_kindle_book_review.iloc[:900000]
df_train = parallelize_dataframe(df_train, multiply_columns)
df_train.to_pickle('train_after_preprocess_1.pkl')
