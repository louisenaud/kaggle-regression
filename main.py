"""
Project:    
File:       main.py
Created by: louise
On:         6/8/17
At:         2:09 PM
"""

import numpy as np
import pandas as pd
import matplotlib
import os
import matplotlib.pyplot as plt
from scipy.stats import skew

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold, StratifiedKFold

from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import tensorflow

# For tensorboard use
ROOT_DIR = '/tmp/tfboard'
#os.makedirs(ROOT_DIR)
OUTPUT_MODEL_FILE_NAME = os.path.join(ROOT_DIR, 'tf.ckpt')

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# load dataset
train = pd.read_csv("/home/louise/Documents/datasets/kaggle/housing market/train.csv")
test = pd.read_csv("/home/louise/Documents/datasets/kaggle/housing market/test.csv")
dataset = train.values

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

## Keras test
# First test
X_train_sc = StandardScaler().fit_transform(X_train)
X_tr, X_val, y_tr, y_val = train_test_split(X_train_sc, y, random_state=3)
print X_tr.shape
model = Sequential()
model.add(Dense(1, input_dim=X_train.shape[1], W_regularizer=l1(0.001)))

model.compile(loss="mse", optimizer="adam")
model.summary()
hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val))
pd.Series(model.predict(X_val)[:, 0]).hist()
scores = model.evaluate(X_val, y_val, verbose=0)
print "Score = ", scores


# Second test

# define the model
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model_2 = larger_model()
tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0,
          write_graph=True, write_images=True)
tbCallBack.set_model(model_2)
hist2 = model_2.fit(X_tr, y_tr, validation_data=(X_val, y_val), callbacks=[tbCallBack])
scores = model_2.evaluate(X_val, y_val, verbose=0)
print scores



# define wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(2*X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(X_train.shape[1], input_dim=2*X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model_3 = wider_model()
hist3 = model_3.fit(X_tr, y_tr, validation_data=(X_val, y_val))
scores = model_3.evaluate(X_val, y_val, verbose=0)
print scores
mse = mean_squared_error(model.predict(X_val)[:, 0], y_val)
print "Model 1 : ", mse
mse = mean_squared_error(model_2.predict(X_val)[:, 0], y_val)
print "Model 2 : ", mse
mse = mean_squared_error(model_3.predict(X_val)[:, 0], y_val)
print "Model 3 : ", mse

# Plot Results
plt.figure()
x = range(0, len(X_val))
plt.plot(x, model.predict(X_val)[:, 0], label="Model 1")
plt.plot(x, model_2.predict(X_val)[:, 0], label="Model 2")
plt.plot(x, model_3.predict(X_val)[:, 0], label="Model 3")
plt.plot(x, y_val, label="GT")
plt.title("Keras Models")
plt.legend()
plt.show()
# # define 10-fold cross validation test harness
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# cvscores = []
# for train, test in kfold.split(X_train, y):
#     # create model
#     model = wider_model()
#
#     model.fit(all_data[:train], y[:train], epochs=150, batch_size=10, verbose=0)
#     # evaluate the model
#     scores = model.evaluate(all_data[:test], y[:test], verbose=0)
# 0
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



