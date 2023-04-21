import scipy.io 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pickle
from pprint import pprint

#%% Data define
mat = scipy.io.loadmat('training.mat')
data = mat["data"]

y = np.load('training_labels.npy')
X = data.reshape((data.shape[0], -1)) # reshape to 2D array

# N, M = x.shape

# for i in range(0, N):
#     min_val = np.min(x[i, :]) # = -11
#     max_val = np.max(x[i, :]) # = -1
    
#     thresh = min_val + (min_val - max_val) * 0.3 * (-1)
    
#     for j in range(0, M):
#         if x[i, j] < thresh:
#             x[i, j] = -80


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # % of data located to test

# #%% Modeling
# model = DecisionTreeClassifier()
# model.fit(X_train,y_train)

# #%% Predictions
# p = model.predict(X_test)
# print(p)

# #%% Accuracy of model
# score = accuracy_score(y_test,p)
# print(score)

# #%% Modeling type 2
# classifier = SVC()

# parameters = [{'gamma': [0.01], 'C': [1, 10]}]

# grid_search = GridSearchCV(classifier, parameters)

# grid_search.fit(X_train, y_train)

# #%% Accuracy of model 2

# best_estimator = grid_search.best_estimator_

# y_prediction = best_estimator.predict(X_test)

# score = accuracy_score(y_prediction, y_test)

# print('{}% of samples were correctly classified'.format(str(score * 100)))

#pickle.dump(best_estimator, open('./model.p', 'wb'))

#%% Model 3
# Normalize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = MLPClassifier(hidden_layer_sizes=(100,50,60), max_iter=1000, alpha=0.1,
                      activation="relu", solver="adam", random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)



# #%% Model 4

# model_pipeline = []

# #model_pipeline.append(LogisticRegression(solver='liblinear'))
# model_pipeline.append(SVC())
# model_pipeline.append(KNeighborsClassifier())
# model_pipeline.append(DecisionTreeClassifier())
# model_pipeline.append(RandomForestClassifier())
# model_pipeline.append(GaussianNB())

# #%% Model 4 run

# #model_list = ['LR','SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes'] 

# model_list = ['SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes'] 
# acc_list = []
# auc_list = []
# cm_list = []

# for model in model_pipeline:
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc_list.append(metrics.accuracy_score(y_test, y_pred))
#     #fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred) 
#     # auc_list.append(round(metrics.auc(fpr, tpr),2)) 
#     cm_list.append(confusion_matrix(y_test, y_pred))
    
    
#     # result_df = pd.DataFrame({'Model':model_list, 'Accuracy': acc_list, 'AUC': auc_list})
# result_df = pd.DataFrame({'Model':model_list, 'Accuracy': acc_list})
# print(result_df)
    

