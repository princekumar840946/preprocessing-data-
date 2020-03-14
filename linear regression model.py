import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv("salary_data.csv")
x=dataset.iloc[: , :-1].values
y=dataset.iloc[: , 1].values

#splitting the data set into trainig data and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#fitting simple regressor into training dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the set results
y_pred=regressor.predict(x_test)

#visualising the prediction training dataset

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("salary vs experience(training set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()


#visualisig the predicted test_data
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("salary vs experience(test data)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()
