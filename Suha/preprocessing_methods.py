import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import nltk
import re
import csv
import pandas as pd
import numpy as np
import pickle
import os
import io
import itertools
import re
from collections import Counter
import requests
from bs4 import BeautifulSoup
import string
from ekphrasis.classes import segmenter, spellcorrect
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import binascii
import fasttext

pickle_in = open("slang.pickle","rb")
slang_dict = pickle.load(pickle_in)
slang_set = set(slang_dict.keys())

pickle_in = open("words.pickle","rb")
words_set = pickle.load(pickle_in)

pickle_in = open("contractions.pickle","rb")
contractions = pickle.load(pickle_in)

pickle_in = open("all_words.pickle","rb")
all_words = pickle.load(pickle_in)

pickle_in = open("emoji_sets.pickle","rb")
emoji_sets = pickle.load(pickle_in)

pickle_in = open("stop_words.pickle","rb")
stop_words = pickle.load(pickle_in)

positive_word_set = set(open('positive-words.txt', encoding = "ISO-8859-1").read().split())
negative_word_set = set(open('negative-words.txt', encoding = "ISO-8859-1").read().split())

redundant_set = set(["'t", "'m", "'re", "'s", "'ll", "'l", "'ve", "'d"])

def standardize_repetitive(tweet):
    word_list = list()
    for word in tweet.split():
        if word not in slang_set and word not in words_set:
            standart_word = ''.join(c for c, _ in itertools.groupby(word))
            if standart_word in slang_set or standart_word in words_set:
                word_list.append(standart_word)
            else:
                word_list.append(word)
        else:
            word_list.append(word)
    return " ".join(word_list)

def replace_slang(tweet):
    word_list = list()
    for word in tweet.split():
        if word not in words_set and word in slang_set:
            word_list.append(slang_dict[word])
        else:
            word_list.append(word)
    return " ".join(word_list)

def replace_contraction(tweet):
    word_list = list()
    for word in tweet.split():
        if word in contractions:
            word_list.append(contractions[word])
        else:
            word_list.append(word)
    return " ".join(word_list)

def replace_emoji(tweet):
    word_list = list()
    for word in tweet.split():
        if word in emoji_sets['positive']:
            word_list.append("<positive>")
        elif word in emoji_sets['negative']:
            word_list.append("<negative>")
        else:
            word_list.append(word)
    return " ".join(word_list)

seg = segmenter.Segmenter(corpus="twitter")
def word_segmentation(tweet):
    word_list = list()
    for word in tweet.split():
        if word not in words_set and word not in slang_set:
            word_list.append(seg.segment(word.strip("#")))
        else:
             word_list.append(word)
    return " ".join(word_list)

sp = spellcorrect.SpellCorrector(corpus="english")
def spell_correction(tweet):
    word_list = list()
    for word in tweet.split():
        if word not in words_set and word not in slang_set:
            word_list.append(sp.correct(word))
        else:
             word_list.append(word)
    return " ".join(word_list)

def remove_punctiotions(tweet):
    char_list = list()
    punctuation = string.punctuation
    for char in tweet:
        if char not in punctuation:
            char_list.append(char)
        else:
            char_list.append(" ")
    return "".join(char_list)

def emphasize_sentiment_words(tweet):
    word_list = list()
    for word in tweet.split():
        if word in positive_word_set:
            word_list.append('<positive> ' + word)
        elif word in negative_word_set:
            word_list.append('<negative> ' + word)
        else:
            word_list.append(word)
    return " ".join(word_list)

def filter_digits(tweet):
    word_list = list()
    for word in tweet.split():
        try:
            float(word)
            word_list.append("<number>")
        except:
            word_list.append(word)
    return " ".join(word_list)

def is_word(tweet):
    word_list = list()
    for word in tweet.split():
        if word in all_words:
            word_list.append(word)
    return " ".join(word_list)

def clean_stopwords(tweet):
    word_list = list()
    for word in tweet.split():
        if word not in stop_words:
            word_list.append(word)
    return " ".join(word_list)

def clean_one_length(tweet):
    word_list = list()
    for word in tweet.split():
        if len(word) > 0:
            word_list.append(word)
    return " ".join(word_list)

def clean_redundant(tweet):
    word_list = list()
    for word in tweet.split():
        if word not in redundant_set:
            word_list.append(word)
    return " ".join(word_list)
