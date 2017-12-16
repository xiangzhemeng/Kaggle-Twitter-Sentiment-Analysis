import pandas as pd
import numpy as np
import regex as re
from segment import Analyzer
import multiprocessing
from multiprocessing import Pool

np.random.seed(0)

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

def remove_number(tweet):
    new_tweet = []
    for word in tweet.split():
        try:
            float(word)
            new_tweet.append("")
        except:
            new_tweet.append(word)
    return " ".join(new_tweet)

dict = {}

corpus1 = open('data/dictionary/tweet_typo_corpus.txt', 'rb')
for term in corpus1:
    term = term.decode('utf8').split()
    dict[term[0]] = term[1]

corpus2 = open('data/dictionary/text_correction.txt', 'rb')
for term in corpus2:
    term = term.decode('utf8').split()
    dict[term[1]] = term[3]

def spelling_correction(tweet):
    tweet = tweet.split()
    for idx in range(len(tweet)):
        if tweet[idx] in dict.keys():
            tweet[idx] = dict[tweet[idx]]
    tweet = ' '.join(tweet)
    return tweet

def clean(tweet):
    tweet = re.sub(r"i\'m", "i am", tweet)
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

    tweet = tweet + ' ' + clean_hashtag(tweet)
    tweet = remove_number(tweet)
    tweet = spelling_correction(tweet)

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

def runn_preprocessing():
    print("Test data preprocessing start!")
    X_test = pd.read_pickle("data/pickles/test_origin.pkl")
    X_test = parallelize_dataframe(X_test, multiply_columns)
    X_test.to_pickle("test_clean.pkl")
    print("Test data preprocessing finish!")

    print("Train data preprocessing start!")
    X_train = pd.read_pickle("data/pickles/train_origin.pkl")
    X_train = parallelize_dataframe(X_train, multiply_columns)
    X_train.to_pickle("train_clean.pkl")
    print("Train data preprocessing finish!")

if __name__ == "__main__":
    runn_preprocessing()
