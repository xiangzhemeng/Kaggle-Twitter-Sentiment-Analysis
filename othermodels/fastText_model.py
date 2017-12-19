import pandas as pd
import numpy as np
import csv
import fasttext

train_tweets = pd.read_pickle('train_tweets_after_preprocess.pkl')
test_tweets = pd.read_pickle('test_tweets_after_preprocess.pkl')
train_tweets['sentiment'] = train_tweets['sentiment'].apply(lambda row: 'label_' + str(row))

# Create fastText train file
f = open('fasttext_train.txt', 'w')
for tweet, sentiment in zip(train_tweets['tweet'], train_tweets['sentiment']):
    f.write((tweet.rstrip() + ' ' + sentiment + '\n'))

# Train model
model = fasttext.supervised('fasttext_train.txt', 'fasttext_model', label_prefix='label_', epoch=20, dim=200)

# Prediction
test_tweets = np.array(test_tweets['tweet'])
pred = model.predict(test_tweets)
pred = [int(value) for x in pred for value in x]
pred = np.array(pred)

with open('ft_submission.csv', 'w') as file:
    fieldnames = ['Id', 'Prediction']
    writeFile = csv.DictWriter(file, delimiter=",", fieldnames=fieldnames)
    writeFile.writeheader()
    idx = 1
    for x in pred:
        writeFile.writerow({'Id':int(idx),'Prediction':x})
        idx += 1