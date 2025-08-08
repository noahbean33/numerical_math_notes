# -*- coding: utf-8 -*-
# Auto-generated from 'LA_leastSquares_example1.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# #     COURSE: Linear algebra: theory and implementation
# ##    SECTION: Least-squares
# ###      VIDEO: Least-squares example 1
# 
# #### Instructor: sincxpress.com
# ##### Course url: https://www.udemy.com/course/linear-algebra-theory-and-implementation/?couponCode=202110

# In [ ]
import numpy as np
import matplotlib.pyplot as plt

# In [ ]
# data
data = np.array([[-4,0,-3,1,2,8,5,8]]).T
N    = len(data)

# design matrix
X = np.ones([N,1])
# fit the model
b = np.linalg.solve(X.T@X,X.T@data)

# compare against the mean
m = np.mean(data)

# print the results
print(b,m)

# In [ ]
# compute the model-predicted values
yHat = X@b

# plot data and model prediction
plt.plot(np.arange(1,N+1),data,'bs-',label='Data')
plt.plot(np.arange(1,N+1),yHat,'ro--',label='Model pred.')

plt.legend()
plt.show()

# In [ ]

# new design matrix
X = np.array([np.arange(0,N)]).T

# fit the model
b = np.linalg.solve(X.T@X,X.T@data)

# compute the model-predicted values
yHat = X@b

# plot data and model prediction
plt.plot(np.arange(1,N+1),data,'bs-',label='Data')
plt.plot(np.arange(1,N+1),yHat,'ro--',label='Model pred.')

plt.legend()
plt.show()

# In [ ]

# design matrix
X = np.concatenate( [np.ones([N,1]),np.array([np.arange(0,N)]).T],axis=1)
# fit the model
b = np.linalg.solve(X.T@X,X.T@data)

# compute the model-predicted values
yHat = X@b

# plot data and model prediction
plt.plot(np.arange(1,N+1),data,'bs-',label='Data')
plt.plot(np.arange(1,N+1),yHat,'ro--',label='Model pred.')

plt.legend()
plt.show()

# In [ ]
## now with nonlinearity in the design matrix

# design matrix
X = np.concatenate( [np.ones([N,1]),np.array([np.arange(0,N)**2]).T],axis=1)
# fit the model
b = np.linalg.solve(X.T@X,X.T@data)

# compute the model-predicted values
yHat = X@b

# plot data and model prediction
plt.plot(np.arange(1,N+1),data,'bs-',label='Data')
plt.plot(np.arange(1,N+1),yHat,'ro--',label='Model pred.')

plt.legend()
plt.show()

# In [ ]

