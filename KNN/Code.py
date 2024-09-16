import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
dataset = pd.read_csv('D:/PythonProject/KNN/Assignment_1/adult.csv', encoding='gbk')
# clean the dataset which contains "?" in csv
dataset.replace('?', np.nan, inplace=True)
cleaned_dataset = dataset.dropna()
# select features in csv which will be used in ML
data = cleaned_dataset[['age',
                        'workclass',
                        'education',
                        'marital_status',
                        'occupation',
                        'hours-per-week']]
target = cleaned_dataset['income']
# transform data to int or float
data['workclass'] = pd.factorize(data['workclass'])[0]
data['education'] = pd.factorize(data['education'])[0]
data['marital_status'] = pd.factorize(data['marital_status'])[0]
data['occupation'] = pd.factorize(data['occupation'])[0]
# spilt data
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
# using KNN
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
# knn.predict(x_test)
print(knn.score(x_test, y_test))