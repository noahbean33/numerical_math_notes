# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_intuition_inverseDiff.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration and applications
# ## SECTION: Intuition for integration
# ### LECTURE: Integration as "inverse differentiation"
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202506

# In [ ]
import numpy as np
import matplotlib.pyplot as plt

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Discrete differences (approx of derivative)

# In [ ]
# x-axis grid on which to evaluate the function
x = np.linspace(-1,4,301)

# define a function
fx = x**2

# find the x-axis coordinate of x=0
zeroIdx = np.argmin(abs(x-0))

# visualize the function
plt.plot(x,fx,'ks',markerfacecolor='w',alpha=.4,markersize=10,linewidth=2)
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.show()

# In [ ]
# difference (discrete derivative)
dx = x[1] - x[0]
df = np.diff(fx) / dx

# visualize the derivative
plt.plot(x[:-1],df,'ks',markerfacecolor='w',alpha=.4,markersize=10,linewidth=2)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.show()

# In [ ]

# %% [markdown]
# # About the cumulative sum

# In [ ]
# a brief aside on the cumulative sum

v = np.arange(10)
print('The vector:')
print(v)

# take the sum
regularSum = np.sum(v)
print('')
print('"Regular" sum:')
print(regularSum)

# cumulative sum via for-loop
cumulativeSum = np.zeros(len(v),dtype=int)
for i in range(len(v)):
  cumulativeSum[i] = np.sum( v[:i+1] )

print('')
print('Cumulative sum via for-loop:')
print(cumulativeSum)


# cumulative sum via function
cumulativeSumF = np.cumsum( v )
print('')
print('Cumulative sum via formula:')
print(cumulativeSumF)

# In [ ]

# %% [markdown]
# # Approximation of the integral using cumulative sum

# In [ ]
# cumulative sum (discrete integral)
idf = np.cumsum(df) * dx
idf -= idf[zeroIdx] # normalize so that idf(0)=0
idf += fx[zeroIdx]  # then add constant from original function

# visualize the integral
plt.plot(x[:-1],idf,'ks',markerfacecolor='w',alpha=.4,markersize=10,linewidth=2,label='Integral of df')

# and plot the original function on top
plt.plot(x,fx,'m',linewidth=3,label='f(x)')

plt.legend()
plt.xlabel('x')
plt.ylabel(r"y = $\int f' dx$")
plt.show()

# In [ ]

