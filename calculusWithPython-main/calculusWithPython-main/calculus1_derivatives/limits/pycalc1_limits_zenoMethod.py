# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_limits_zenoMethod.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Limits
# ### LECTURE: CodeChallenge: Limits via Zeno's paradox
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# better image resolution
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# In [ ]

# %% [markdown]
# # Exercise 1: implement the function

# In [ ]
# a function for the function
def fx(u):
  return np.cos(u**2)**2 + np.pi

xx = np.linspace(-2.1,2.1,201)

plt.plot(xx,fx(xx))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Approximate the limit via Zeno's paradox

# In [ ]
# target value (limit)
a = 1

# starting x-axis values
x0 = np.array([a-1,a+1])


# initialize
iterations = 10
limitvals = np.zeros((iterations,2))
xAxisvals = np.zeros((iterations,2))


# run the zeno's limit method in a for-loop
for i in range(iterations):

  # compute and store x0,y=f(x0)
  limitvals[i,:] = fx(x0)
  xAxisvals[i,:] = x0

  # update x-values (could this line be above the previous?)
  x0 = (x0+a)/2

# In [ ]
# print out in a table
print('Limit from the left:')
print(np.vstack((xAxisvals[:,0],limitvals[:,0])).T)

print(' ')
print('Limit from the right:')
print(np.vstack((xAxisvals[:,1],limitvals[:,1])).T)

print(' ')
print(f'Function value at x={a}')
print(fx(a))

# In [ ]

# %% [markdown]
# # Exercise 3: visualize the results

# In [ ]
# and plot
plt.plot(xx,fx(xx),'k')
plt.plot([a,a],[np.pi,1+np.pi],'k--',linewidth=.2)
plt.plot(xAxisvals,limitvals,'o',markerfacecolor='w')
# plt.xlim([a-1,a+1]) # optional zoom in
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'f({a}) = {fx(a)}')
plt.show()

# In [ ]

