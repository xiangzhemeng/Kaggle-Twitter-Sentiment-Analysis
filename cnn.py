import pandas as pd
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer

X_train = pd.read_pickle("train_with_preprocessing.p")
#X_train = X_train['clean_tweets']

X_test = pd.read_pickle("test_with_preprocessing.p")
#X_test = X_test['clean_tweets']

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
max_features = len(word_index)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print('Tokenization finished!')

# Shuffle training dataset
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
train_sequences = X_train[indices]
y = np.array(int(X_train.shape[0]/2) * [0] + int(X_train.shape[0]/2) * [1])
y = y[indices]

# CNN model
model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=X_train.shape[1]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("Build model finished!")


model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)
print("Fit model finished!")

y_pred = model.predict_proba(X_test)
print("Prediction finished!")

y_pred = 1 - 2 * y_pred
with open('cnn_submission.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1, r2 in zip(np.arange(1,10001), y_pred):
        writer.writerow({'Id': int(r1), 'Prediction': int(r2)})