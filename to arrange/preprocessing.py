import itertools
import enchant

dict = {}
corpus = open('tweet_typo_corpus.txt', 'rb')
for word in corpus:
    word = word.decode('utf8')
    word = word.split()
    dict[word[1]] = word[3]
corpus.close()

def remove_repetitions(tweet):
    dict_us = enchant.Dict('en_US')
    tweet=tweet.split()
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

def clean(tweet):
    tweet = remove_repetitions(tweet)
    tweet = spelling_correction(tweet)
    return tweet.strip().lower()

