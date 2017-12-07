from options import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from preprocessing import *

def init_tfidf_vectorizer():
  """
  DESCRIPTION:
          Initializes the tfidf vectorizer by taking the parameters from options.py
  """
  print_dict_settings(algorithm['options']['TFIDF'], msg='tf-idf Vectorizer settings\n')

  if algorithm['options']['TFIDF']['number_of_stopwords'] != None:
      algorithm['options']['TFIDF']['number_of_stopwords'] = find_stopwords(number_of_stopwords=algorithm['options']['TFIDF']['number_of_stopwords'])
  if algorithm['options']['TFIDF']['tokenizer'] != None:
      algorithm['options']['TFIDF']['tokenizer'] = LemmaTokenizer()

  return TfidfVectorizer(
     min_df = algorithm['options']['TFIDF']['min_df'],
     max_df = algorithm['options']['TFIDF']['max_df'],
     sublinear_tf = algorithm['options']['TFIDF']['sublinear_tf'],
     use_idf = algorithm['options']['TFIDF']['use_idf'],
     stop_words = algorithm['options']['TFIDF']['number_of_stopwords'],
     tokenizer = algorithm['options']['TFIDF']['tokenizer'],
     ngram_range = algorithm['options']['TFIDF']['ngram_range'],
     max_features = algorithm['options']['TFIDF']['max_features']
  )

def load_vectorizer(tweets, test_tweets):
  """
  DESCRIPTION:
        If there exists a cached tfidf file then it is loaded and returned. Otherwisem a new vectorizer is fittied
        by the gived data.
  INPUT:
        tweets: Dataframe of a set of tweets
        test_tweets: Dataframe of a set of test_tweets
  OUTPUT:
        train_reptweets: TFIDF representation of train tweets (dimensions: |train tweets| x |vocabulary|)
        test_reptweets: TFIDF representation of test tweets (dimensions: |train tweets| x |vocabulary|)
  """
  import os.path
  if(os.path.exists(TFIDF_TRAIN_FILE) and os.path.exists(TFIDF_TRAIN_FILE)):
    f = open(TFIDF_TRAIN_FILE,'rb')
    train_reptweets = pickle.load(f)
    f = open(TFIDF_TEST_FILE,'rb')
    test_reptweets = pickle.load(f)
  else:
    tfidf = init_tfidf_vectorizer()
    train_reptweets = tfidf.fit_transform(tweets['tweet'])
    if algorithm['options']['TFIDF']['cache_tfidf']:
      f = open(TFIDF_TRAIN_FILE,'wb')
      pickle.dump(train_reptweets, f)
    test_reptweets = tfidf.transform(test_tweets['tweet'])
    if algorithm['options']['TFIDF']['cache_tfidf']:
      f = open(TFIDF_TEST_FILE,'wb')
      pickle.dump(test_reptweets, f)

  return train_reptweets, test_reptweets

def init_count_vectorizer():
  pass
