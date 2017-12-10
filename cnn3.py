import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from sklearn.externals import joblib

train_sequences = joblib.load('train_features_FT_50.sav')
test_sequences = joblib.load('test_feature_FT_50.sav')
y = joblib.load('train_labels.sav')

# CNN model
model = Sequential()
model.add(Conv1D(padding="same", kernel_size=3, filters=32, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("Build model finished!")

model.fit(train_sequences, y, validation_split=0.1, epochs=1, batch_size=128, verbose=1, shuffle=True)
print("Fit model finished!")

y_pred_origin = model.predict_proba(test_sequences)
print("Prediction finished!")

print(y_pred_origin)
y_pred = []
for x in y_pred_origin:
    if x[0] > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
print(y_pred)

y_pred = 1 - 2 * np.array(y_pred)
with open('cnn_submission3.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1, r2 in zip(np.arange(1,10001), y_pred):
        writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
print("Submission file generated!")