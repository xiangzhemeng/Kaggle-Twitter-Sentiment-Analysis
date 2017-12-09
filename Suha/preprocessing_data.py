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
from fast_text import fast_text
from sklearn import linear_model
from sklearn.metrics import accuracy_score

## CREATE SLANG DICTIONARY
page = requests.get("https://www.netlingo.com/acronyms.php")
soup = BeautifulSoup(page.content, 'html.parser')

slang_dict = dict()
for i in soup.find_all('li'):
    if 'span' in str(i):
        words = str(i).split(">")
        slang_dict[words[3].split('</a')[0].lower().split(" or")[0]] =\
            words[5].split('</li')[0].split(" -or-")[0].split(",")[0].lower()

# Corrections
slang_dict['oomf'] = 'one of my friends'
slang_dict['...'] = '<unfinished>'
slang_dict['aw'] = "cute"
slang_dict['hm'] = "thinking"
slang_dict['um'] = "not sure"
slang_dict["outta"] = "out of"
slang_dict["rt"] = "<retweet>"
slang_dict["yeah"] = "yes"
slang_dict["congrats"] = "congratulation"
slang_dict["grats"] = "congratulation"
slang_dict["gonna"] = "going to"
slang_dict["wanna"] = "want to"
slang_dict["gotta"] = "have to"
slang_dict["awhh"] = "cute"
slang_dict["yeah"] = "yes"
slang_dict["em"] = "them"
slang_dict["pic"] = "picture"
slang_dict["!"] = "!"
slang_dict["?"] = "?"
slang_dict["nah"] = "no"
slang_dict["yup"] = "yes"
slang_dict["ima"] = "i am going to"
slang_dict["imma"] = "i am going to"

pickle_out = open("slang.pickle","wb")
pickle.dump(slang_dict, pickle_out)
pickle_out.close()

## CREATE CONTRACTIONS DICTIONARY
# I removed the real dictionary that I've used before saving as Pickle file
# since it was too long to display

# Corrections
contractions["can't"] = "can not"
contractions["dont"] = "do not"
contractions["im"] = "I am"

pickle_out = open("contractions.pickle","wb")
pickle.dump(contractions, pickle_out)
pickle_out.close()

## CREATE BASIC WORDS SET
words_set = set()
fptr = open("words.txt")
for i in fptr.readlines():
    words_set.add(i[:-1].lower())

# Corrections
words_set.remove('u')
words_set.remove('')

pickle_out = open("words.pickle","wb")
pickle.dump(words_set, pickle_out)
pickle_out.close()

## CREATE AN EXTENSIVE WORD SET
path = "words/"
all_words = set()
for dir,subdir,files in os.walk(path):
    for file in files:
        with io.open(path+file,'r',encoding='ISO-8859-1') as f:
            a = f.read()
            for words in a.split(","):
                for word in words.split("\n"):
                    all_words.add(word)

all_words = set(list(all_words)[1:])
all_words.add("<positive>")
all_words.add("<negative>")
all_words.add("<number>")
all_words.add("<url>")
all_words.add("<user>")
all_words.add("<neutral>")

pickle_out = open("all_words.pickle","wb")
pickle.dump(all_words, pickle_out)
pickle_out.close()

## CREATE EMOJI SET
positive = ["<3", "♥"]
negative = []
emoji_sets = dict()

eyes = ["8",":","=",";","x",">","<"]
nose = ["'","`","-",">","<"]
for e in eyes:
    for n in nose:
        for s in [")", "d", "]", "}","p",">","b","j","v"]:
            positive.append(e+n+s)
            positive.append(e+s)
        for s in ["(", "[", "{","<","f","t","c","|", "/","\\","l","o","i"]:
            negative.append(e+n+s)
            negative.append(e+s)
        for s in ["(", "[", "{","<","d","v","c"]:
            positive.append(s+n+e)
            positive.append(s+e)
        for s in [")", "]", "}",">","b","p","|", "/","\\","l","o","i"]:
            negative.append(s+n+e)
            negative.append(s+e)

emoji_sets['positive'] = set(positive)
emoji_sets['negative'] = set(negative)

pickle_out = open("emoji_sets.pickle","wb")
pickle.dump(emoji_sets, pickle_out)
pickle_out.close()

## CREATE STOP WORDS SET
path = "stopwords/"
stop_words = set()
for dir,subdir,files in os.walk(path):
    for file in files:
        with io.open(path+file,'r',encoding='ISO-8859-1') as f:
            a = f.read()
            for words in a.split(","):
                for word in words.split("\n"):
                    stop_words.add(word)

stop_words = set(list(stop_words)[1:])

pickle_out = open("stop_words.pickle","wb")
pickle.dump(stop_words, pickle_out)
pickle_out.close()
