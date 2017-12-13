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

def emoji_transformation(tweet):
    loves = ["<3", "♥"]
    smilefaces = []
    sadfaces = []
    neutralfaces = []

    eyes = ["8",":","=",";"]
    nose = ["'","`","-",r"\\"]
    for e in eyes:
        for n in nose:
            for s in ["\)", "d", "]", "}","p"]:
                smilefaces.append(e+n+s)
                smilefaces.append(e+s)
            for s in ["\(", "\[", "{"]:
                sadfaces.append(e+n+s)
                sadfaces.append(e+s)
            for s in ["\|", "\/", r"\\"]:
                neutralfaces.append(e+n+s)
                neutralfaces.append(e+s)
            #reversed
            for s in ["\(", "\[", "{"]:
                smilefaces.append(s+n+e)
                smilefaces.append(s+e)
            for s in ["\)", "\]", "}"]:
                sadfaces.append(s+n+e)
                sadfaces.append(s+e)
            for s in ["\|", "\/", r"\\"]:
                neutralfaces.append(s+n+e)
                neutralfaces.append(s+e)

    smilefaces = list(set(smilefaces))
    sadfaces = list(set(sadfaces))
    neutralfaces = list(set(neutralfaces))

    t = []
    for w in tweet.split():
        if w in loves:
            t.append("<love>")
        elif w in smilefaces:
            t.append("<happy>")
        elif w in neutralfaces:
            t.append("<neutral>")
        elif w in sadfaces:
            t.append("<sad>")
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
            t.append("")
        # Otherwise
        except:
            t.append(w)
    newTweet = " ".join(t)
    return newTweet


def clean(tweet):

    tweet = re.sub(r"\'s", "is", tweet)
    tweet = re.sub(r"\'ve", "have", tweet)
    tweet = re.sub(r"n\'t", "not", tweet)
    tweet = re.sub(r"\'re", "are", tweet)
    #tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", "will", tweet)
    re.sub(r"<user>", "", tweet)
    tweet = re.sub(r"\.", "", tweet)
    tweet = re.sub(r",", "", tweet)
    tweet = re.sub(r"!", "", tweet)
    tweet = re.sub(r"\(", "", tweet)
    tweet = re.sub(r"\)", "", tweet)
    tweet = re.sub(r"\?", "", tweet)
    #tweet = re.sub(r"\s{2,}", " ", tweet)
    #tweet = remove_repetitions(tweet)
    #tweet = spelling_correction(tweet)
    #tweet = emoji_transformation(tweet)
    #tweet = filter_digits(tweet)

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


X_test = pd.read_pickle("test_tweets_after_preprocess_new4.pkl")
X_test = parallelize_dataframe(X_test, multiply_columns)
X_test.to_pickle("test_tweets_after_preprocess_6.pkl")
print("test preprocessing finished!")

X_train = pd.read_pickle("train_tweets_after_preprocess_cnn_new4.pkl")
X_train = parallelize_dataframe(X_train, multiply_columns)
X_train.to_pickle("train_tweets_after_preprocess_cnn_6.pkl")
print("train preprocessing finished!")
