import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#############################
# I SLR Baby steps
#############################
x 		=	np.array([1,2,3,4])
y 		=	np.array([2,4,6,8])

n 		=	len(x)# equals len(y)

x_prom	=	np.mean(x)
y_prom	=	np.mean(y)

sxx=	np.sum(((1-x_prom)*(1-x_prom)) + ((2-x_prom)*(2-x_prom)) + ((3-x_prom)*(3-x_prom)) + ((4-x_prom)*(4-x_prom)))

b1_est 	=	np.sum(((x-x_prom)/sxx)*y)
b0_est	=	y_prom - (b1_est*x_prom)


# special libraries
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m, c)

#plot the results
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
plt.show()



#############################
# II SLR for adults
#############################
#more sophisticated libraries
from sklearn import linear_model
lreg = linear_model.LinearRegression() 
lreg.fit(x, y) 


#----------------------------
# Code source: Jaques Grobler
# License: BSD 3 clause

#----------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

import code; code.interact(local=locals())



