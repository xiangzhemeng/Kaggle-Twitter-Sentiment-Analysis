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
#from keras.callbacks import EarlyStopping

def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list)-ngram_range+1):
            for ngram_value in range(2, ngram_range+1):
                ngram = tuple(new_list[i:i+ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

X_train = pd.read_pickle("train_tweets_after_preprocess_cnn_6.pkl")
X_train = np.array(X_train['tweet'])
X_test = pd.read_pickle("test_tweets_after_preprocess_6.pkl")
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

maxlen = 60
ngram_range = 2
print('Adding {}-gram features'.format(ngram_range))

ngram_set = set()
for input_list in train_sequences:
    for i in range(2, ngram_range + 1):
        set_of_ngram = create_ngram_set(input_list, ngram_value=i)
        ngram_set.update(set_of_ngram)

start_index = max_features + 1
token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
indice_token = {token_indice[k]: k for k in token_indice}

max_features = np.max(list(indice_token.keys())) + 1

train_sequences = add_ngram(train_sequences, token_indice, ngram_range)
test_sequences = add_ngram(test_sequences, token_indice, ngram_range)
print('Average train sequence length: {}'.format(np.mean(list(map(len, train_sequences)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, test_sequences)), dtype=int)))

print('Pad sequences (samples x time)')
train_sequences = sequence.pad_sequences(train_sequences, maxlen=60)
test_sequences = sequence.pad_sequences(test_sequences, maxlen=60)
print('train_sequences shape:', train_sequences.shape)
print('test_sequences shape:', test_sequences.shape)

# Shuffle training dataset
indices = np.arange(train_sequences.shape[0])
np.random.shuffle(indices)
train_sequences = train_sequences[indices]
y = np.array(int(2500000/2) * [0] + int(2500000/2) * [1])
y = y[indices]

#earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)

# CNN Model
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

y_pred_origin = model.predict(test_sequences)
print("Prediction finished!")

y_pred = []
for x in y_pred_origin:
    if x[0] > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

y_pred = 1 - 2 * np.array(y_pred)

with open('cnn_model1_bigram.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1, r2 in zip(np.arange(1,10001), y_pred):
        writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
print("Submission file generated!")