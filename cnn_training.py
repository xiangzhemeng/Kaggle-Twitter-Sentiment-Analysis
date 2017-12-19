import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import EarlyStopping


# Main function of cnn training

def run_neural_network():
    print(" == Enter into CNN training step ==")

    np.random.seed(0)

    x_train = pd.read_pickle("data/pickles/train_after_preprocess.pkl")
    x_train = np.array(x_train['tweet'])

    x_test = pd.read_pickle("data/pickles/test_after_preprocess.pkl")
    x_test = np.array(x_test['tweet'])

    y = np.array(int(2500000 / 2) * [0] + int(2500000 / 2) * [1])
    print("Data loading finish!")

    # Tokenization
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(x_train)

    # Turn x_train into sequence form
    sequence_train = tokenizer.texts_to_sequences(x_train)
    # Turn x_test into sequence form
    sequence_test = tokenizer.texts_to_sequences(x_test)

    # Transform sequence_train into into a 2D Numpy array
    sequence_train = sequence.pad_sequences(sequence_train, maxlen = 30)
    # Transform sequence_test into into a 2D Numpy array
    sequence_test = sequence.pad_sequences(sequence_test, maxlen = 30)

    # Affect input dimension
    input_dim = len(tokenizer.word_index) + 1
    input_length = sequence_train.shape[1]
    print("Tokenization finish!")

    # Shuffle training dataset
    new_index = np.arange(sequence_train.shape[0])
    np.random.shuffle(new_index)
    sequence_train = sequence_train[new_index]
    y = y[new_index]
    print("Data shuffling finish!")

    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)


    ### Model 1 ###
    print("Build model1!")
    np.random.seed(1)
    model = Sequential()
    model.add(Embedding(input_dim, 50, input_length = input_length))
    model.add(Conv1D(padding = "same", kernel_size = 3, filters = 32, activation = "relu"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(250, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print("Fit model1!")
    model.fit(sequence_train, y, validation_split = 0.1, epochs = 10, batch_size = 128, verbose = 1, shuffle = True, callbacks = [earlyStopping])

    print("Generate prediction!")
    train_model1 = model.predict(sequence_train, batch_size = 128)
    pickle.dump(train_model1, open('data/xgboost/train_model1.txt', 'wb'))
    test_model1 = model.predict(sequence_test)
    pickle.dump(test_model1, open('data/xgboost/test_model1.txt', 'wb'))
    print("Model1 finished!")


    ### Model 2 ###
    print("Build model2!")
    np.random.seed(2)
    model = Sequential()
    model.add(Embedding(input_dim, 50, input_length = input_length))
    model.add(LSTM(100, recurrent_dropout = 0.2, dropout = 0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print("Fit model2!")
    model.fit(sequence_train, y, validation_split = 0.1, epochs = 10, batch_size = 128, verbose = 1, shuffle = True, callbacks = [earlyStopping])

    print("Generate prediction!")
    train_model2 = model.predict(sequence_train, batch_size = 128)
    pickle.dump(train_model2, open('data/xgboost/train_model2.txt', 'wb'))
    test_model2 = model.predict(sequence_test)
    pickle.dump(test_model2, open('data/xgboost/test_model2.txt', 'wb'))
    print("Model2 finished!")


    ### Model 3 ###
    print("Build model1!")
    np.random.seed(3)
    model = Sequential()
    model.add(Embedding(input_dim, 50, input_length = input_length))
    model.add(Conv1D(padding = "same", kernel_size = 3, filters = 32, activation = "relu"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(LSTM(100))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print("Fit model3!")
    model.fit(sequence_train, y, validation_split = 0.1, epochs = 10, batch_size = 128, verbose = 1, shuffle = True, callbacks = [earlyStopping])

    print("Generate prediction!")
    train_model3= model.predict(sequence_train, batch_size = 128)
    pickle.dump(train_model3, open('data/xgboost/train_model3.txt', 'wb'))
    test_model3 = model.predict(sequence_test)
    pickle.dump(test_model3, open('data/xgboost/test_model3.txt', 'wb'))
    print("Model3 finished!")


if __name__ == "__main__":
    run_neural_network()
