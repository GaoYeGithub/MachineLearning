import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from google.colab import drive
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import gdown

# let's load in our data using the pd.read_excel() function
data = pd.read_excel('student_preprocessing.xlsx')

# now let's take a look at our data by printing out the first 5 samples using the .head() function
data.head()

X = data.drop(columns = 'grades')

Y = data['grades']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

print("X train: ", np.array(X_train).shape)
print("X train: ", np.array(X_test).shape)
print("X train: ", np.array(Y_train).shape)
print("X train: ", np.array(Y_test).shape)



#print("X test: ", X_test.shape)
#print("Y train: ", Y_train.shape)
#print("Y test: ", Y_test.shape)


regression_model = LinearRegression()

regression_model.fit(X_train, Y_train)

pred = regression_model.predict(X_test)


print(mean_absolute_error(Y_test, pred)) #added print statement :) should work now! 