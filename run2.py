import numpy as np
import pickle
import csv
import xgboost as xgb

train1 = pickle.load(open("train/train_model1.txt", "rb"))
train3 = pickle.load(open("train/train_model3.txt", "rb"))
train4 = pickle.load(open("train/train_model4.txt", "rb"))

test1 = pickle.load(open("test/test_model1.txt", "rb"))
test3 = pickle.load(open("test/test_model3.txt", "rb"))
test4 = pickle.load(open("test/test_model4.txt", "rb"))


train = np.hstack((train1, train3, train4))
test = np.hstack((test1, test3, test4))
y = np.array(int(2500000/2) * [0] + int(2500000/2) * [1])
np.random.seed(0)
np.random.shuffle(y)


model = xgb.XGBClassifier().fit(train, y)

y_pred = model.predict(test)
y_pred = 1 - 2 * y_pred

with open('run_submission6.csv', 'w') as file:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(file, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1, r2 in zip(np.arange(1,10001), y_pred):
        writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
