#Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#no need to split into training and test set bcoz dataset is small(not enough info) and we need 
#accurate prediction of negotiation of salaries-so we need max info as possible

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting linear regression to the dataset #to compare both linear(as ref) and polynomial reg models
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
#fitting polynomial regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures #PolynomialFeatures is the class and poly_reg is 
poly_reg = PolynomialFeatures(degree = 4)     #the object of the class-transforms matrix of features x(one indep var)into
x_poly = poly_reg.fit_transform(x)            #a new matrix of features containing x1,x1^2,etc
lin_reg2 = LinearRegression()   # to fit x_poly into the lin reg model
lin_reg2.fit(x_poly, y) 

#visualizing the linear regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')  
plt.title('truth or bluff (linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
#visualizing the polynomial regression results#for higher resolution and smoother curve
x_grid = np.arange(min(x), max(x), 0.1) #this will give a vector, so we use shape func to get matrix
x_grid = x_grid.reshape(len(x_grid),1)  
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')  
plt.title('truth or bluff (Polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predicting a new result with linear regression
lin_reg.predict(6.5)
#predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))