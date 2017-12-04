import pandas as pd
import fasttext

def fast_text(tweets, test_tweets):
    tweets['sentiment'] = tweets['sentiment'].apply(lambda row: '__label__'+str(row))

    create_fasttext_train_file(tweets)
    classifier = fasttext.supervised('fasttext_train.txt', 'fasttext_model',label_prefix='__label__',epoch = 50,dim = 200,ws = 5,lr = 0.05)

    test_tweets = change_test_dataframe_to_list(test_tweets)
    labels = classifier.predict(test_tweets)
    labels = [int(value) for label in labels for value in label]

    return labels

def create_fasttext_train_file(tweets):
    f = open('fasttext_train.txt','w')
    for tweet, sentiment in zip(tweets['tweet'], tweets['sentiment']):
        f.write((tweet.rstrip() + ' ' + sentiment + '\n'))
    f.close()

def change_test_dataframe_to_list(test_tweets):
    test_tweets_list = []
    for t in test_tweets['tweet']:
        test_tweets_list.append(t)

    return test_tweets_list
