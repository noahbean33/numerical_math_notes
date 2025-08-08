# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_applications_GradientDescent.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Applications
# ### LECTURE: CodeChallenge: Gradient descent
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [ ]
# import all necessary modules
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# In [ ]

# %% [markdown]
# # Exercise 1: The function and its derivative

# In [ ]
# function (as a function)
def fx(x):
  return 3*x**2 - 3*x + 4

# derivative function
def deriv(x):
  return 6*x - 3

# In [ ]
# plot the function and its derivative

# define a range for x
xx = np.linspace(-2,2,2001)

# plotting
plt.plot(xx,fx(xx), xx,deriv(xx))
plt.xlim(xx[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['y','dy'])
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Gradient descent

# In [ ]
# random starting point
localmin = np.random.choice(xx,1)
print(f'Started at  x = {localmin[0]:.3f}')

# learning parameters
learning_rate = .01
training_epochs = 100

# run through training
for i in range(training_epochs):
  grad = deriv(localmin)
  localmin = localmin - learning_rate*grad

print(f'Finished at x = {localmin[0]:.3f}')

# In [ ]
# plot the results

plt.plot(xx,fx(xx), xx,deriv(xx))
plt.plot(localmin,deriv(localmin),'ro')
plt.plot(localmin,fx(localmin),'ro')

plt.xlim(xx[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)','df','f(x) min'])
plt.title('Empirical local minimum: %s'%localmin[0])
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Plot the descent

# In [ ]
# Note: I don't discuss this in the video, 
# but it's insightful to modify the parameters
# and observe the effects on the results. For example,
# fix the localmin and adjust the learning rate and epochs!

# fix the starting point
localmin = -1.5

# learning parameters
learning_rate = .05
training_epochs = 40

# run through training and store all the results
modelparams = np.zeros((training_epochs,2))
for i in range(training_epochs):
  grad = deriv(localmin)
  localmin = localmin - learning_rate*grad
  modelparams[i,0] = localmin
  modelparams[i,1] = grad

# In [ ]
# plot the results

# plot the function
plt.plot(xx,fx(xx), xx,deriv(xx))

# plot the progression of points
for i,(xi,dx) in enumerate(modelparams):
  plt.plot(xi,fx(xi),'o',color=[(i/training_epochs)**(1/2),.2,.2])
  plt.plot(xi,deriv(xi),'o',color=[(i/training_epochs)**(1/2)*.7+.3,.1,(i/training_epochs)**(1/2)])


# adjust some plotting aspects
plt.xlim(xx[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)','df'])
plt.title('Empirical local minimum: %s'%modelparams[-1,0])
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Now with sympy

# In [ ]
x = sym.var('x')

# define the function and its derivative
fx = sym.sin(x) * (x-2)**2
df = sym.diff(fx,x)

# quick plot
sym.plot(fx,(x,0,5*sym.pi));

# In [ ]
# lambdify the functions (easier for plotting)
fx_lam = sym.lambdify(x,fx)
df_lam = sym.lambdify(x,df)

# a nicer-looking plot
xx = np.linspace(0,5*np.pi,2001)

_,axs = plt.subplots(2,1,figsize=(8,6))
axs[0].plot(xx,fx_lam(xx),linewidth=3)
axs[0].set_xlim(xx[[0,-1]])
axs[0].grid()
axs[0].set_title(f'f(x) = ${sym.latex(fx)}$')

axs[1].plot(xx,df_lam(xx),linewidth=3)
axs[1].plot(xx[[0,-1]],[0,0],'k--')
axs[1].set_xlim(xx[[0,-1]])
axs[1].grid()
axs[1].set_title(f"f'(x) = ${sym.latex(df)}$")

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 5: Gradient descent in sympy

# In [ ]
# random starting point in x domain
localmin = np.random.choice(xx,1)[0]

# learning parameters
learning_rate = .05
training_epochs = 40


# setup the plot with functions
plt.figure(figsize=(8,6))
plt.plot(xx,fx_lam(xx),label='f(x)')
plt.plot(xx,df_lam(xx),label="f'(x)")


# run through training and plot immediately
for i in range(training_epochs):
  
  # implement gradient descent
  grad = df.subs(x,localmin)
  localmin = localmin - learning_rate*grad

  # plot the current result
  plt.plot(localmin,fx.subs(x,localmin),'o',color=[(i/training_epochs)**(1/2),.2,.2])
  plt.plot(localmin,df.subs(x,localmin),'o',color=[(i/training_epochs)**(1/2)*.7+.3,.1,(i/training_epochs)**(1/2)])


plt.grid()
plt.xlabel('x')
plt.ylabel("f(x) or f'(x)")
plt.xlim(xx[[0,-1]])
plt.legend()
plt.title('Empirical local minimum: %s'%localmin)
plt.show()

# In [ ]

