import pandas as pd
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

X_train = pd.read_pickle("train_with_preprocessing.p")
X_test = pd.read_pickle("test_with_preprocessing.p")

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
max_features = len(word_index)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
print('Tokenization finished!')

train_sequences = sequence.pad_sequences(train_sequences, maxlen=30)
test_sequences = sequence.pad_sequences(test_sequences, maxlen=30)
print('train_sequences shape:', train_sequences.shape)
print('test_sequences shape:', test_sequences.shape)

# Shuffle training dataset
indices = np.arange(train_sequences.shape[0])
np.random.shuffle(indices)
train_sequences = train_sequences[indices]
y = np.array(int(2500000/2) * [0] + int(2500000/2) * [1])
y = y[indices]

# CNN model
model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=train_sequences.shape[1]))
model.add(Conv1D(padding="same", kernel_size=3, filters=32, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("Build model finished!")


model.fit(train_sequences, y, validation_split=0.1, epochs=1, batch_size=128, verbose=0, shuffle=True)
print("Fit model finished!")

y_pred = model.predict_proba(test_sequences)
print("Prediction finished!")
print("length="+len(y_pred))

y_pred = 1 - 2 * y_pred
with open('cnn_submission.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1, r2 in zip(np.arange(1,10001), y_pred):
        writer.writerow({'Id': int(r1), 'Prediction': int(r2)})