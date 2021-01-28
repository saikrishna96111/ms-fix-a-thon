# from azureml.core import Workspace
# ws = Workspace.from_config(path=".file-path/ws_config.json")

import numpy as np
import pandas as pd
import pickle
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
dataset = pd.read_csv('heart_data.csv')
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state =0)
# svc_scores=[]

# from sklearn.externals import joblib
# joblib.dump(lr, 'model.pkl')
# print("Model dumped!")

# lr = joblib.load('model.pkl')

# model_columns = list(x.columns)
# joblib.dump(model_columns, 'model_columns.pkl')
# print("Models columns dumped!")

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train, y_train)
    pickle.dump(svc_classifier,open('model.pkl','wb'))
    model = pickle.load(open('model.pkl','rb'))
    # svc_scores.append(svc_classifier.score(X_test, y_test))
    #take input and give the predicted output as json
# print("The score for Support Vector Classifier is {}% with {} kernel.".format(svc_scores[0]*100, 'linear'))

