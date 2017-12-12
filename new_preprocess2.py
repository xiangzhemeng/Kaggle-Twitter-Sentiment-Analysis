import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool

num_partitions = multiprocessing.cpu_count()
num_cores = multiprocessing.cpu_count()

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
            t.append("<number>")
        # Otherwise
        except:
            t.append(w)
    newTweet = " ".join(t)
    return newTweet

def clean(tweet):
    tweet = emoji_transformation(tweet)
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


X_test = pd.read_pickle("test_tweets_after_preprocess_new4.pkl")
X_test = parallelize_dataframe(X_test, multiply_columns)
X_test.to_pickle("test_tweets_after_preprocess_new5.pkl")
print("test preprocessing finished!")

X_train = pd.read_pickle("train_tweets_after_preprocess_cnn_new4.pkl")
X_train = parallelize_dataframe(X_train, multiply_columns)
X_train.to_pickle("train_tweets_after_preprocess_cnn_new5.pkl")
print("train preprocessing finished!")
