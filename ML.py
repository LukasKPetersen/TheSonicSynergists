import scipy.io 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import model_selection, linear_model
import pickle
from helper import rlr_validate, train_neural_net
import torch

#%% Data define
mat = scipy.io.loadmat('training.mat')
data = mat["data"]

y = np.load('training_labels.npy')
X = data.reshape((data.shape[0], -1)) # reshape to 2D array

N, M = X.shape

#%% Model 1
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.999) # % of data located to test

# # Modeling
# model = DecisionTreeClassifier()
# model.fit(X_train,y_train)

# # Predictions
# p = model.predict(X_test)
# print(p)

# # Accuracy of model 1
# score = accuracy_score(y_test,p)
# print(score)

#%% Model 2
# classifier = SVC()

# parameters = [{'gamma': [0.01], 'C': [1, 10]}]

# grid_search = GridSearchCV(classifier, parameters)

# grid_search.fit(X_train, y_train)

# # Accuracy of model 2
# best_estimator = grid_search.best_estimator_

# y_prediction = best_estimator.predict(X_test)

# score = accuracy_score(y_prediction, y_test)

# print('{}% of samples were correctly classified'.format(str(score * 100)))

#pickle.dump(best_estimator, open('./model.p', 'wb'))

#%% Model 3
# # Normalize input data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train model
# model = MLPClassifier(hidden_layer_sizes=(100,50,60), max_iter=1000, alpha=0.2,
#                       activation="relu", solver="adam", random_state=42)
# model.fit(X_train, y_train)

# # Evaluate model
# accuracy = model.score(X_test, y_test)

# print("Accuracy:", accuracy)

#%% Model 4: Linear Model

# Using K-fold training method
#  - - - - - - - - - - - - - - - - - - -
# |   t e s t   |       t r a i n       |
#  - - - - - - - - - - - - - - - - - - -

K = 5

CV = model_selection.KFold(K, shuffle=True)

for i, (train_index, test_index) in enumerate(CV.split(X,y)):
    # define the train data and the test data
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    print('Test- and train data was defined')

    # we find the optimal regularization parameter
    lambdas = np.power(10.,range(-1,5)) # we set the range
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, 10)
    print('Optimal value for lambda = ' + str(opt_lambda))
    
    # TODO: find the correct way of applying the lambda weights
    
    # initialize the model
    linearModel = linear_model.LinearRegression().fit(X_train, y_train)
    print('Model was created')
    
    # prediction
    y_pred = linearModel.predict(X_test)
    
    # estimate error rate
    error_rate = sum((y_test - y_pred) ** 2) / N
    print('Error rate for the linear model: ' + str(error_rate))
