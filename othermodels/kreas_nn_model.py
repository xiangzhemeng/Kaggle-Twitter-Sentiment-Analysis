import pandas as pd
import numpy as np
import csv
import keras

train_tweets = pd.read_pickle('train_tweets_after_preprocess.pkl')
test_tweets = pd.read_pickle('test_tweets_after_preprocess.pkl')

train_labels = [[0, 1] if label == 1 else [1, 0] for label in np.array(train_tweets['sentiment'])]
test_labels = [[0, 1] if label == 1 else [1, 0] for label in np.array(test_tweets['sentiment'])]

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=64, activation='relu', input_dim=vector_length))
model.add(keras.layers.Dense(units=2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.05, momentum=0.9, nesterov=True))

model.fit(np.array(train_features), np.array(train_labels), epochs=150, batch_size=128)

pred = np.argmax(model.predict(np.array(test_features), batch_size=128), axis=1)

def create_csv_file(results,filepath):
    with open(filepath, 'w') as file:
        fieldnames = ['Id', 'Prediction']
        writeFile = csv.DictWriter(file, delimiter=",", fieldnames=fieldnames)
        writeFile.writeheader()
        id_ = 1
        for result in results:
            writeFile.writerow({'Id':int(id_),'Prediction':result})
            id_ += 1

create_csv_file(pred,'kreas_nn_submission.csv')
