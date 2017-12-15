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
#from keras.callbacks import EarlyStopping

X_train = pd.read_pickle("train_tweets_after_preprocess_cnn_7.pkl")
X_train = np.array(X_train['tweet'])
X_test = pd.read_pickle("test_tweets_after_preprocess_7.pkl")
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

#earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)

# CNN Model1
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

y_pred1 = model.predict(test_sequences)
print("Model1 finished!")

pickle.dump(y_pred1, open('model1.txt', 'wb'))

# CNN Model2
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

y_pred2 = model.predict(test_sequences)
print("Model2 finished!")

pickle.dump(y_pred2, open('model2.txt', 'wb'))

# CNN Model3
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

y_pred3 = model.predict(test_sequences)
print("Model3 finished!")

pickle.dump(y_pred3, open('model3.txt', 'wb'))

# CNN Model4
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

y_pred4 = model.predict(test_sequences)
print("Model4 finished!")

pickle.dump(y_pred4, open('model4.txt', 'wb'))

