import pandas as pd
import numpy as np
import regex as re
from segmenter import Analyzer
import multiprocessing
from multiprocessing import Pool
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from preprocess_setting import *


np.random.seed(0)

# Loading current instance resource
num_partitions = multiprocessing.cpu_count()
num_cores = multiprocessing.cpu_count()



def abbreviation_replacement(text):
    text = re.sub(r"i\'m", "i am", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"he\'s", "he is", text)
    text = re.sub(r"it\'s", "it is", text)
    text = re.sub(r"that\'s", "that is", text)
    text = re.sub(r"who\'s", "who is", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"n\'t", "not", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\.", " \. ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    return text


def emoji_translation(text):
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
    for w in text.split():
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
    newText = " ".join(t)
    return newText


def emphaszie_punctuation(text):
    for punctuation in ['!', '?', '.']:
        regex = r'[' + punctuation + ']' + "+"
        text = re.sub(regex, punctuation + ' <repeat> ', text)
    return text


# Emphasize positive and negative words

# Loading positive and negative words from dictionary
positive_word_library = list(set(open('data/dictionary/positive-words.txt', encoding = "ISO-8859-1").read().split()))
negative_word_library = list(set(open('data/dictionary/negative-words.txt', encoding = "ISO-8859-1").read().split()))

def emphasize_pos_and_neg_words(text):
    t = []
    for w in text.split():
        if w in positive_word_library:
            t.append('<positive> ' + w)
        elif w in negative_word_library:
            t.append('<negative> ' + w)
        else:
            t.append(w)
    newTweet = " ".join(t)
    return newTweet
# End


# Clean hashtag part


# Initialize  Analyzer
e = Analyzer()

def extract_hashtag(text):
    hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
    return hash_list

def split_hashtag_to_words(tag):
    word_list = [w for w in e.segment(tag[1:]) if len(w) > 3]
    return word_list

def clean_hashtag(text):
    words = []
    tag_list = extract_hashtag(text)
    for tag in tag_list:
        words += split_hashtag_to_words(tag)
    if len(words):
        return (" ".join(words)).strip()
    else:
        return ""
    return word_list
# End

def remove_number(text):
    new_tweet = []
    for word in text.split():
        try:
            float(word)
            new_tweet.append("")
        except:
            new_tweet.append(word)
    return " ".join(new_tweet)



# Spelling corretion part

# Loading the dictaionary from metadata
dict = {}

corpus1 = open('data/dictionary/tweet_typo_corpus.txt', 'rb')
for term in corpus1:
    term = term.decode('utf8').split()
    dict[term[0]] = term[1]

corpus2 = open('data/dictionary/text_correction.txt', 'rb')
for term in corpus2:
    term = term.decode('utf8').split()
    dict[term[1]] = term[3]


def spelling_correction(text):
    text = text.split()
    for idx in range(len(text)):
        if text[idx] in dict.keys():
            text[idx] = dict[text[idx]]
    text = ' '.join(text)
    return text
# End



# Loading stopwords list from NLTK
stoplist = stopwords.words('english')

def remove_stopwords(text):
    tokens = text.split()
    for word in tokens:
        if word in stoplist:
            tokens.remove(word)
    return ' '.join(tokens)



# Initialize NLTK function
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()


#  Lemmatization
def lemmatize_word(w):
    try:
        x = lemma.lemmatize(w).lower()
        return x
    except Exception as e:
        return w


def lemmatize_sentence(text):
    x = [lemmatize_word(t) for t in text.split()]
    return " ".join(x)
# End


# Stemming
def stemming_word(w):
    return stemmer.stem(w)


def stemming_sentence(text):
    x = [stemming_word(t) for t in text.split()]
    return " ".join(x)
# End



# Combine all preprocessing method
def preprocessing_all_method(tweet):

    if opt_abbreviation_replacement:
        tweet = abbreviation_replacement(tweet)

    if opt_emphaszie_punctuation:
        tweet = emphaszie_punctuation(tweet)

    if opt_clean_hastag:
        tweet = tweet + ' ' + clean_hashtag(tweet)

    if opt_emoji_translation:
        tweet = emoji_translation(tweet)

    if opt_remvoe_number:
        tweet = remove_number(tweet)

    if opt_emphasize_pos_and_neg_words:
        tweet = emphasize_pos_and_neg_words(tweet)

    if opt_spelling_correction:
        tweet = spelling_correction(tweet)

    if opt_remove_stopwords:
        tweet = remove_stopwords(tweet)

    if opt_stemming_sentence:
        tweet = stemming_sentence(text)

    if opt_lemmatize_sentence:
        tweet = lemmatize_sentence(tweet)

    return tweet.strip().lower()


# Paralelize Pandas datafames
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def multiply_columns(data):
    data['tweet'] = data['tweet'].apply(lambda x: preprocessing_all_method(x))
    return data


# Main function of data_preprocessing module
def run_preprocessing():
    print(" == Enter into preprocessing step ==")
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
    run_preprocessing()
