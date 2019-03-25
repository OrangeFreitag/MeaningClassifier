import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import math
from keras.utils.vis_utils import plot_model
import uuid
from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths, get_outputs_path

def clearY(y):
    clean_input = np.array([]).reshape(0, 1)
    for data in y:
        pos1 = data[0]
        pos2 = data[1]
        pos3 = data[2]
        if  pos1 == 1 and pos2 == 0 and pos3 ==0:
                clean_input = np.vstack((clean_input, [1]))
        else:
                clean_input = np.vstack((clean_input, [0]))
    return clean_input


 experiment = Experiment()

train_x = np.loadtxt('/data/shared-task/vec_train_x.csv' ,delimiter=',',usecols=range(11)[1:])
train_y = clearY(np.loadtxt('/data/shared-task/vec_train_y.csv', delimiter=',',usecols=range(4)[1:]))

dev_test_x = np.loadtxt('/data/shared-task/vec_test_x.csv', delimiter=',',usecols=range(11)[1:])
dev_test_y = np.loadtxt('/data/shared-task/vec_test_y.csv', delimiter=',',usecols=range(4)[1:])

st2_test_x = np.loadtxt('/data/shared-task/vec_st2_test_x.csv', delimiter=',',usecols=range(11)[1:])
st2_test_y = np.loadtxt('/data/shared-task/vec_st2_test_y.csv', delimiter=',',usecols=range(4)[1:])

seed = 7
np.random.seed(seed)

sc = StandardScaler()
scaled_train_x = sc.fit_transform(train_x)
scaled_dev_test_x = sc.transform(dev_test_x)
scaled_st2_test_x = sc.transform(st2_test_x)

#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(64, activation='relu', input_dim=10))
classifier.add(Dropout(0.2))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

metrics = classifier.fit(scaled_train_x, train_y, batch_size = 150, epochs = 600, validation_split=0.1)

experiment.log_metrics(loss=metrics.history['loss'],
                       accuracy=metrics.history['accuracy'],
                       precision=metrics.history['precision'])