import numpy as np
import pickle
import csv
import xgboost as xgb

np.random.seed(0)

train_model1 = pickle.load(open("train_model/train_model1.txt", "rb"))
train_model2 = pickle.load(open("train_model/train_model2.txt", "rb"))
train_model3 = pickle.load(open("train_model/train_model3.txt", "rb"))

test_model1 = pickle.load(open("test_model/test_model1.txt", "rb"))
test_model2 = pickle.load(open("test_model/test_model2.txt", "rb"))
test_model3 = pickle.load(open("test_model/test_model3.txt", "rb"))


x_train = np.hstack((train_model1, train_model2, train_model3))
x_test = np.hstack((test_model1, test_model2, test_model3))
y = np.array(1250000 * [0] + 1250000 * [1])
np.random.shuffle(y)

model = xgb.XGBClassifier().fit(x_train, y)

prediction = model.predict(x_test)
# 0 --> 1 & 1 --> -1
prediction = 1 - 2 * prediction

with open('prediction.csv', 'w') as file:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(file, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1, r2 in zip(np.arange(1, 10001), prediction):
        writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
