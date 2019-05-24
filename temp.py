# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


dataset = pd.read_csv("dataset - Dataset.csv")
x = dataset.iloc[:,0:8].values
y = dataset.iloc[:,8].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:,0] = labelencoder_x_1.fit_transform(x[:,0])
labelencoder_x_2 = LabelEncoder()
x[:,1] = labelencoder_x_2.fit_transform(x[:,1])
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()
#x = x[:,0: ]



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=0)



print("----------")

from xgboost import XGBRegressor
regressor = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.4, max_depth = 5, alpha = 10, n_estimators = 10)
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)


from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(x_test,y_pred)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))

print("************")

input_dataset = pd.read_csv("Input.csv")
x_input = input_dataset.iloc[:,:].values

labelencoder_input_1 = LabelEncoder()
x_input[:,0] = labelencoder_input_1.fit_transform(x_input[:,0])
labelencoder_input_2 = LabelEncoder()
x_input[:,1] = labelencoder_input_2.fit_transform(x_input[:,1])
onehotencoder = OneHotEncoder(categorical_features=[1])
x_input = onehotencoder.fit_transform(x_input).toarray()


y_results = regressor.predict(x_input)
