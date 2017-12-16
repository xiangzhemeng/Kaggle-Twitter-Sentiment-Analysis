import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

np.random.seed(0)

X_train = pd.read_pickle("train_rank1.pkl")
X_train = np.array(X_train['tweet'])
X_test = pd.read_pickle("test_rank1.pkl")
X_test = np.array(X_test['tweet'])
print("Data loading finished!")

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
max_features = len(word_index)
print('Found %s unique tokens.' % max_features)

train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
print('Tokenization finished!')

print('Pad sequences (samples x time)')
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

np.random.seed(1)

### Model 1 ###
print("Model1 start!")
model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=train_sequences.shape[1]))
model.add(Conv1D(padding="same", kernel_size=3, filters=32, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("Build model finished!")

model.fit(train_sequences, y, validation_split=0.1, epochs=1, batch_size=128, verbose=1, shuffle=True)
print("Fit model finished!")

train = model.predict(train_sequences, batch_size=128)
test = model.predict(test_sequences)
pickle.dump(train, open('train_model1.txt', 'wb'))
pickle.dump(test, open('test_model1.txt', 'wb'))
print("Model1 finished!")

np.random.seed(2)

### Model 2 ###
print("Model2 start!")
model = Sequential()
model.add(Embedding(max_features+1, 20, input_length=train_sequences.shape[1]))
model.add(Conv1D(padding="same", kernel_size=3, filters=32, activation="relu"))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("Build model finished!")

model.fit(train_sequences, y, validation_split=0.1, epochs=1, batch_size=128, verbose=1, shuffle=True)

print("Fit model finished!")

train = model.predict(train_sequences, batch_size=128)
test = model.predict(test_sequences)
pickle.dump(train, open('train_model2.txt', 'wb'))
pickle.dump(test, open('test_model2.txt', 'wb'))
print("Model2 finished!")

np.random.seed(3)

### Model 3 ###
print("Model3 start!")
model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=train_sequences.shape[1]))
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("Build model finished!")

model.fit(train_sequences, y, validation_split=0.1, epochs=1, batch_size=128, verbose=1, shuffle=True)

print("Fit model finished!")

train = model.predict(train_sequences, batch_size=128)
test = model.predict(test_sequences)
pickle.dump(train, open('train_model3.txt', 'wb'))
pickle.dump(test, open('test_model3.txt', 'wb'))
print("Model3 finished!")

np.random.seed(4)

### Model 4 ###
print("Model4 start!")
model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=train_sequences.shape[1]))
model.add(Conv1D(padding="same", kernel_size=3, filters=32, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("Build model finished!")

model.fit(train_sequences, y, validation_split=0.1, epochs=1, batch_size=128, verbose=1, shuffle=True)

print("Fit model finished!")

train = model.predict(train_sequences, batch_size=128)
test = model.predict(test_sequences)
pickle.dump(train, open('train_model4.txt', 'wb'))
pickle.dump(test, open('test_model4.txt', 'wb'))
print("Model4 finished!")

