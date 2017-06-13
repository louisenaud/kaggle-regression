"""
Project:    
File:       main.py
Created by: louise
On:         6/8/17
At:         2:09 PM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import KFold, StratifiedKFold


def simple_model():
    """
    Simple one Layer Network to estimate the sale price of the kaggle regression dataset.
    This model was tested in the very good Kaggle Kernel: https://www.kaggle.com/apapiu/regularized-linear-models/notebook/notebook
    :return: keras model
    """
    # Create model
    model = Sequential()
    model.add(Dense(1, input_dim=X_train.shape[1], W_regularizer=l1(0.001)))
    # Compile model
    model.compile(loss="mse", optimizer="adam")
    return model

def larger_model():
    """
    Creates a larger model with 5 layers. The width of the layers are: k -> k -> k -> 6 -> 1. 
    :return: Keras model.
    """
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

def wider_model():
    """
    Creates a Keras model that is wider than the dimension of the feature space. 
    The layers width are like this:
    k -> 2*k -> k -> k -> 1.
    :return: Keras model.
    """
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

def custom_model():
    """
    Model with a L1 regularization term in the loss function for the dense layers to prevent overfitting.
    :return: Keras model.
    """
    # create model
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu', W_regularizer=l1(0.1)))
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu', W_regularizer=l1(0.1)))
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu',
                    W_regularizer=l1(0.1)))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == '__main__':
    # Data preprocessing; from https://www.kaggle.com/apapiu/regularized-linear-models/notebook/notebook

    # loading data
    train = pd.read_csv("/home/louise/Documents/datasets/kaggle/housing market/train.csv")
    test = pd.read_csv("/home/louise/Documents/datasets/kaggle/housing market/test.csv")
    dataset = train.values

    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))

    # log transform the target:
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
    #y_test = test.SalePrice

    # Scale the data:
    X_train_sc = StandardScaler().fit_transform(X_train)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_sc, y, random_state=10)

    ## Keras test
    # First model
    model = simple_model()
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val))
    scores = model.evaluate(X_val, y_val, verbose=0)
    print "Score = ", scores

    # Second model
    model_2 = larger_model()
    # Create tensorflow call back to view the tensorboard associated with this model.
    tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0,
              write_graph=True, write_images=True)
    tbCallBack.set_model(model_2)
    hist2 = model_2.fit(X_tr, y_tr, validation_data=(X_val, y_val), callbacks=[tbCallBack])
    scores = model_2.evaluate(X_val, y_val, verbose=0)
    print scores

    # Third model
    model_3 = wider_model()
    hist3 = model_3.fit(X_tr, y_tr, validation_data=(X_val, y_val))
    scores = model_3.evaluate(X_val, y_val, verbose=0)
    print scores

    # Custom model
    model_4 = custom_model()
    hist4 = model_4.fit(X_tr, y_tr, validation_data=(X_val, y_val))
    scores = model_4.evaluate(X_val, y_val, verbose=1)
    print scores

    # Print MSE for each model
    print "Model 1 MSE: ", mean_squared_error(model.predict(X_val)[:, 0], y_val)
    print "Model 2 MSE: ", mean_squared_error(model_2.predict(X_val)[:, 0], y_val)
    print "Model 3 MSE: ", mean_squared_error(model_3.predict(X_val)[:, 0], y_val)
    print "Model 4 MSE: ", mean_squared_error(model_4.predict(X_val)[:, 0], y_val)

    # Print R^2 for each model
    print "Model 1 R^2: ", r2_score(model.predict(X_val)[:, 0], y_val)
    print "Model 2 R^2: ", r2_score(model_2.predict(X_val)[:, 0], y_val)
    print "Model 3 R^2: ", r2_score(model_3.predict(X_val)[:, 0], y_val)
    print "Model 4 R^2: ", r2_score(model_4.predict(X_val)[:, 0], y_val)

    # Plot Results
    plt.figure()
    x = range(0, len(X_val))
    plt.plot(x, model.predict(X_val)[:, 0], label="Model 1")
    plt.plot(x, model_2.predict(X_val)[:, 0], label="Model 2")
    plt.plot(x, model_3.predict(X_val)[:, 0], label="Model 3")
    plt.plot(x, model_4.predict(X_val)[:, 0], label="Model 4")
    plt.xlabel("Houses")
    plt.ylabel("Real and Predicted Prices")
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



