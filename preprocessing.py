import itertools

dico = {}
corpus = open('tweet_typo_corpus.txt', 'rb')
for word in corpus:
    word = word.decode('utf8')
    word = word.split()
    dico[word[1]] = word[3]
corpus.close()

def remove_repetitions(tweet):
    tweet=tweet.split()
    for i in range(len(tweet)):
        tweet[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i])).replace('#', '')
        if len(tweet[i])>0:
            if not d.check(tweet[i]):
                tweet[i] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i])).replace('#', '')
    tweet = ' '.join(tweet)
    return tweet

def spelling_correction(tweet):
    tweet = tweet.split()
    for i in range(len(tweet)):
        if tweet[i] in dico.keys():
            tweet[i] = dico[tweet[i]]
    tweet = ' '.join(tweet)
    return tweet

def clean(tweet):
    tweet = remove_repetitions(tweet)
    tweet = spelling_correction(tweet)
    return tweet.strip().lower()

