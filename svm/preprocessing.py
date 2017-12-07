import nltk
import re
import pandas as pd
import numpy as np
import pickle

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
    tweets['tweet'] = expand_contraction_word(tweets['tweet'])
    print('Contraction words expansion finished!')

    tweets['tweet'] = tweets.apply(lambda tweet: emoji_transformation(tweet['tweet']), axis=1)
    print('Emoji to hashtag transformation finished!')

    tweets['tweet'] = tweets.apply(lambda tweet: emphasize_sentiment_words(tweet['tweet']), axis=1)
    print('Sentiment words emphasizing finished!')

    tweets['tweet'] = tweets.apply(lambda tweet: filter_digits(tweet['tweet']), axis=1)
    print('Number to hashtag <number> transformation finished!')

    print('ALL DONE!')

    return tweets
