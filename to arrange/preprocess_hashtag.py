import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool
from segment import Analyzer
import regex as re

num_partitions = multiprocessing.cpu_count()
num_cores = multiprocessing.cpu_count()

e = Analyzer(file_dir='en')

def clean_hashtag(text):
    words = []
    tag_list = extract_hashtag(text)
    for tag in tag_list:
        words += split_hashtag_to_words(tag)

    if len(words):
        return (" ".join(words)).strip()
    else:
        return ""


def extract_hashtag(text):
    hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
    return hash_list


def split_hashtag_to_words(tag):
    word_list = [w for w in e.segment(tag[1:]) if len(w) > 3]
    return word_list

def clean(tweet):
    tweet = tweet + ' ' + clean_hashtag(tweet)
    return tweet.strip().lower()

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def multiply_columns(data):
    data['tweet'] = data['tweet'].apply(lambda x: clean(x))
    return data

X_train = pd.read_pickle("train_tweets_after_preprocess_cnn_spelling_corr.pkl")
X_train = parallelize_dataframe(X_train, multiply_columns)
X_train.to_pickle("train_tweets_after_preprocess_cnn_new2.pkl")
print("train preprocessing finished!")

X_train = pd.read_pickle("test_tweets_after_preprocess_spelling_corr.pkl")
X_train = parallelize_dataframe(X_train, multiply_columns)
X_train.to_pickle("test_tweets_after_preprocess_cnn_new2.pkl")
print("train preprocessing finished!")
