import pandas as pd
import csv
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

train_tweets = pd.read_pickle('train_tweets_after_preprocess.pkl')
test_tweets = pd.read_pickle('test_tweets_after_preprocess.pkl')

def load_vectorizer(train_tweets, test_tweets):
    tfidf = TfidfVectorizer(min_df=1, max_df=1, sublinear_tf=True, use_idf=True, ngram_range=(1, 4))

    train_reptweets = tfidf.fit_transform(train_tweets['tweet'])
    f = open('train_reptweets.pkl', 'wb')
    # Save dictionary train_reptweets into pickle file f
    pickle.dump(train_reptweets, f)

    test_reptweets = tfidf.transform(test_tweets['tweet'])
    f = open('test_reptweets.pkl', 'wb')
    pickle.dump(test_reptweets, f)
    return train_reptweets, test_reptweets

def create_csv_file(results,filepath):
    with open(filepath, 'w') as file:
        fieldnames = ['Id', 'Prediction']
        writeFile = csv.DictWriter(file, delimiter=",", fieldnames=fieldnames)
        writeFile.writeheader()
        id_ = 1
        for result in results:
            writeFile.writerow({'Id':int(id_),'Prediction':result})
            id_ += 1

train_reptweets, test_reptweets = load_vectorizer(train_tweets, test_tweets)
print("Load vectorizer finished!")

# Initialization
clf = svm.LinearSVC(max_iter=10000,intercept_scaling=1,loss='squared_hinge')

# Model training
clf.fit(train_reptweets, train_tweets['sentiment'])
print("Model training finished!")

# Prediction
pred = clf.predict(test_reptweets)
print("Prediction finished!")

# Generating csv file
create_csv_file(pred,'svm_submission.csv')

