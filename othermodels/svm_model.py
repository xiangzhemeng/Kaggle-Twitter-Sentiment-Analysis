import pandas as pd
import pickle
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

train_tweets = pd.read_pickle('train_tweets_after_preprocess.pkl')
test_tweets = pd.read_pickle('test_tweets_after_preprocess.pkl')

# Transform data into tfidf vector
def transform_tfidf_vector(train_tweets, test_tweets):
    tfidf = TfidfVectorizer(min_df=1, max_df=1, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))

    train_reptweets = tfidf.fit_transform(train_tweets['tweet'])
    pickle.dump(train_reptweets, open('train_reptweets.pkl', 'wb'))

    test_reptweets = tfidf.transform(test_tweets['tweet'])
    pickle.dump(test_reptweets, open('test_reptweets.pkl', 'wb'))
    return train_reptweets, test_reptweets

train_reptweets, test_reptweets = transform_tfidf_vector(train_tweets, test_tweets)
print("Transform data into tfidf vertor finished!")

# Build SVM Model
model = svm.LinearSVC(max_iter = 10000, intercept_scaling = 1, loss = 'squared_hinge')
# Fit Model
model.fit(train_reptweets, train_tweets['sentiment'])
print("Model training finished!")

# Prediction
pred = model.predict(test_reptweets)
print("Prediction finished!")

# Generating csv file
with open('svm_submission.csv', 'w') as file:
    fieldnames = ['Id', 'Prediction']
    writeFile = csv.DictWriter(file, delimiter=",", fieldnames=fieldnames)
    writeFile.writeheader()
    idx = 1
    for x in pred:
        writeFile.writerow({'Id': int(idx), 'Prediction': x})
        idx += 1
