import pandas as pd
import numpy as np
import itertools
import enchant
import multiprocessing
from multiprocessing import Pool
from segment import Analyzer
import regex as re

num_partitions = multiprocessing.cpu_count()
num_cores = multiprocessing.cpu_count()

e = Analyzer(file_dir='en')

dict = {}
corpus1 = open('tweet_typo_corpus.txt', 'rb')
for word in corpus1:
    word = word.decode('utf8')
    word = word.split()
    dict[word[0]] = word[1]
corpus1.close()
corpus2 = open('text_norm.txt', 'rb')
for word in corpus2:
    word = word.decode('utf8')
    word = word.split()
    dict[word[1]] = word[3]
corpus2.close()
corpus3 = open('dico.txt', 'rb')
for word in corpus3:
    word = word.decode('utf8')
    word = word.split()
    dict[word[0]] = word[1]
corpus3.close()


def remove_repetitions(tweet):
    dict_us = enchant.Dict('en_US')
    tweet = tweet.split()
    for i in range(len(tweet)):
        tweet[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i])).replace('#', '')
        if len(tweet[i])>0:
            if not dict_us.check(tweet[i]):
                tweet[i] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i])).replace('#', '')
    tweet = ' '.join(tweet)
    return tweet

def spelling_correction(tweet):
    tweet = tweet.split()
    for i in range(len(tweet)):
        if tweet[i] in dict.keys():
            tweet[i] = dict[tweet[i]]
    tweet = ' '.join(tweet)
    return tweet

def filter_digits(tweet):
    t = []
    for w in tweet.split():
        try:
            float(w)
            t.append("")
        except:
            t.append(w)
    newTweet = " ".join(t)
    return newTweet


def clean(tweet):

    tweet = re.sub(r"i\'m","i am", tweet)
    tweet = re.sub(r"\'re", "are", tweet)
    tweet = re.sub(r"he\'s", "he is", tweet)
    tweet = re.sub(r"it\'s", "it is", tweet)
    tweet = re.sub(r"that\'s", "that is", tweet)
    tweet = re.sub(r"who\'s", "who is", tweet)
    tweet = re.sub(r"what\'s", "what is", tweet)
    tweet = re.sub(r"n\'t", "not", tweet)
    tweet = re.sub(r"\'ve", "have", tweet)
    tweet = re.sub(r"\'d", "would", tweet)
    tweet = re.sub(r"\'ll", "will", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\.", " \. ", tweet)
    tweet = re.sub(r"\(", " \( ", tweet)
    tweet = re.sub(r"\)", " \) ", tweet)
    tweet = re.sub(r"\?", " \? ", tweet)
    tweet = remove_repetitions(tweet)
    tweet = spelling_correction(tweet)
    tweet = filter_digits(tweet)

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


X_test = pd.read_pickle("test_origin.pkl")
X_test = parallelize_dataframe(X_test, multiply_columns)
X_test.to_pickle("test_tweets_after_preprocess_8.pkl")
print("test preprocessing finished!")

X_train = pd.read_pickle("train_origin.pkl")
X_train = parallelize_dataframe(X_train, multiply_columns)
X_train.to_pickle("train_tweets_after_preprocess_cnn_8.pkl")
print("train preprocessing finished!")
