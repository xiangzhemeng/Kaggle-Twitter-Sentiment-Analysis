import pandas as pd
import fasttext

def fast_text(tweets, test_tweets):

    tweets_labeled = tweets.copy()
    tweets_labeled['sentiment'] = tweets['sentiment'].apply(lambda row: '__label__'+str(row))

    create_fasttext_train_file_supervised(tweets_labeled)
    classifier = fasttext.supervised('fasttext_train_supervised.txt', 'fasttext_model_supervised',label_prefix='__label__',epoch=50,dim=300,ws=5,lr=0.05)

    test_tweets = change_test_dataframe_to_list(test_tweets)
    labels = classifier.predict(test_tweets)
    labels = [int(value) for label in labels for value in label]

    return labels

def change_test_dataframe_to_list(test_tweets):
    test_tweets_list = []
    for t in test_tweets['tweet']:
        test_tweets_list.append(t)

    return test_tweets_list

def create_fasttext_train_file_supervised(tweets):
    f = open('fasttext_train_supervised.txt','w')
    for tweet, sentiment in zip(tweets['tweet'], tweets['sentiment']):
        f.write((tweet.rstrip() + ' ' + sentiment + '\n'))
    f.close()

def create_fasttext_train_file(tweets):
    f = open('fasttext_train.txt','w')
    for tweet in tweets['tweet']:
        f.write((tweet.rstrip() + '\n'))
    f.close()
