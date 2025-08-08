# -*- coding: utf-8 -*-
# Auto-generated from '3-calculus-i.ipynb' on 2025-08-08T15:22:56
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/3-calculus-i.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Calculus I: Limits & Derivatives

# %% [markdown]
# This topic, *Calculus I: Limits & Derivatives*, introduces the mathematical field of calculus -- the study of rates of change -- from the ground up. It is essential because computing derivatives via differentiation is the basis of optimizing most machine learning algorithms, including those used in deep learning such as backpropagation and stochastic gradient descent. 
# 
# Through the measured exposition of theory paired with interactive examples, you’ll develop a working understanding of how calculus is used to compute limits and differentiate functions. You’ll also learn how to apply automatic differentiation within the popular TensorFlow 2 and PyTorch machine learning libraries. The content covered in this class is itself foundational for several other topics in the *Machine Learning Foundations* series, especially *Calculus II* and *Optimization*.

# %% [markdown]
# Over the course of studying this topic, you'll: 
# 
# * Develop an understanding of what’s going on beneath the hood of machine learning algorithms, including those used for deep learning. 
# * Be able to more intimately grasp the details of machine learning papers as well as many of the other subjects that underlie ML, including partial-derivative calculus, statistics and optimization algorithms. 
# * Compute the derivatives of functions, including by using AutoDiff in the popular TensorFlow 2 and PyTorch libraries.

# %% [markdown]
# **Note that this Jupyter notebook is not intended to stand alone. It is the companion code to a lecture or to videos from Jon Krohn's [Machine Learning Foundations](https://github.com/jonkrohn/ML-foundations) series, which offer detail on the following:**
# 
# *Segment 1: Limits*
# 
# * What Calculus Is
# * A Brief History of Calculus
# * The Method of Exhaustion 
# * Calculating Limits 
# 
# *Segment 2: Computing Derivatives with Differentiation*
# * The Delta Method
# * The Differentiation Equation
# * Derivative Notation
# * The Power Rule
# * The Constant Multiple Rule
# * The Sum Rule
# * The Product Rule
# * The Quotient Rule
# * The Chain Rule
# 
# *Segment 3: Automatic Differentiation*
# * AutoDiff with PyTorch
# * AutoDiff with TensorFlow 2
# * Machine Learning via Differentiation 
# * Cost (or Loss) Functions
# * The Future: Differentiable Programming 

# %% [markdown]
# ## Segment 1: Limits

# %% [markdown]
# ### The Calculus of Infinitesimals

# In [1]
import numpy as np
import matplotlib.pyplot as plt

# In [2]
x = np.linspace(-10, 10, 10000) # start, finish, n points

# %% [markdown]
# If $y = x^2 + 2x + 2$: 

# In [3]
y = x**2 + 2*x + 2

# In [4]
fig, ax = plt.subplots()
_ = ax.plot(x,y)

# %% [markdown]
# * There are no straight lines on the curve. 
# * If we zoom in _infinitely_ close, however, we observe curves that _approach_ lines. 
# * This enables us to find a slope $m$ (tangent) anywhere on the curve, including to identify where $m = 0$: 

# In [5]
fig, ax = plt.subplots()
ax.set_xlim([-2, -0])
ax.set_ylim([0, 2])
_ = ax.plot(x,y)

# In [6]
fig, ax = plt.subplots()
ax.set_xlim([-1.5, -0.5])
ax.set_ylim([0.5, 1.5])
_ = ax.plot(x,y)

# In [7]
fig, ax = plt.subplots()
ax.set_xlim([-1.1, -0.9])
ax.set_ylim([0.9, 1.1])
_ = ax.plot(x,y)

# In [8]
fig, ax = plt.subplots()
ax.set_xlim([-1.01, -0.99])
ax.set_ylim([0.99, 1.01])
_ = ax.plot(x,y)

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ### Limits

# In [9]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.xlim(-5, 10)
plt.ylim(-10, 80)
plt.axvline(x=5, color='purple', linestyle='--')
plt.axhline(y=37, color='purple', linestyle='--')
_ = ax.plot(x,y)

# %% [markdown]
# $$\lim_{x \to 1} \frac{x^2 - 1}{x - 1}$$

# In [10]
def my_fxn(my_x):
    my_y = (my_x**2 - 1)/(my_x - 1)
    return my_y

# In [11]
my_fxn(2)

# In [12]
# Uncommenting the following line results in a 'division by zero' error:
# my_fxn(1)

# In [13]
my_fxn(0.9)

# In [14]
my_fxn(0.999)

# In [15]
my_fxn(1.1)

# In [16]
my_fxn(1.001)

# In [17]
y = my_fxn(x)

# In [18]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.axvline(x=1, color='purple', linestyle='--')
plt.axhline(y=2, color='purple', linestyle='--')
_ = ax.plot(x,y)

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# $$\lim_{x \to 0} \frac{\text{sin } x}{x}$$

# In [19]
def sin_fxn(my_x):
    my_y = np.sin(my_x)/my_x
    return my_y

# In [20]
# Uncommenting the following line results in a 'division by zero' error:
# y = sin_fxn(0)

# In [21]
sin_fxn(0.1)

# In [22]
sin_fxn(0.001)

# In [23]
sin_fxn(-0.1)

# In [24]
sin_fxn(-0.001)

# In [25]
y = sin_fxn(x)

# In [26]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.xlim(-10, 10)
plt.ylim(-1, 2)
plt.axvline(x=0, color='purple', linestyle='--')
plt.axhline(y=1, color='purple', linestyle='--')
_ = ax.plot(x,y)

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# $$ \lim_{x \to \infty} \frac{25}{x} $$

# In [27]
def inf_fxn(my_x):
    my_y = 25/my_x
    return my_y

# In [28]
inf_fxn(1e3)

# In [29]
inf_fxn(1e6)

# In [30]
y = inf_fxn(x)

# In [31]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.xlim(-10, 10)
plt.ylim(-300, 300)
_ = ax.plot(x, y)

# In [32]
left_x = x[x<0]
right_x = x[x>0]

# In [33]
left_y = inf_fxn(left_x)
right_y = inf_fxn(right_x)

# In [34]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.xlim(-10, 10)
plt.ylim(-300, 300)
ax.plot(left_x, left_y, c='C0')
_ = ax.plot(right_x, right_y, c='C0')

# %% [markdown]
# **Exercises:**
# 
# Evaluate the limits below using techniques from the slides or above.
# 
# 1. $$ \lim_{x \to 0} \frac{x^2-1}{x-1} $$
# 2. $$ \lim_{x \to -5} \frac{x^2-25}{x+5} $$
# 3. $$ \lim_{x \to 4} \frac{x^2 -2x -8}{x-4} $$
# 4. $$ \lim_{x \to -\infty} \frac{25}{x} $$
# 5. $$ \lim_{x \to 0} \frac{25}{x} $$

# %% [markdown]
# FYI: While not necessary for ML nor for this *ML Foundations* curriculum, the `SymPy` [symbolic mathematics library](https://www.sympy.org/en/index.html) includes a `limits()` method. You can read about applying it to evaluate limits of expressions [here](https://www.geeksforgeeks.org/python-sympy-limit-method/).

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ## Segment 2: Computing Derivatives with Differentiation

# %% [markdown]
# Let's bring back our ol' buddy $y = x^2 + 2x + 2$:

# In [35]
def f(my_x):
    my_y = my_x**2 + 2*my_x + 2
    return my_y

# In [36]
y = f(x)

# In [37]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
_ = ax.plot(x,y)

# %% [markdown]
# Let's identify the slope where, say, $x = 2$.

# %% [markdown]
# First, let's determine what $y$ is: 

# In [38]
f(2)

# %% [markdown]
# Cool. Let's call this point $P$, which is located at (2, 10):

# In [39]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.scatter(2, 10) # new
_ = ax.plot(x,y)

# %% [markdown]
# The _delta method_ uses the difference between two points to calculate slope. To illustrate this, let's define another point, $Q$ where, say, $x = 5$.

# In [40]
f(5)

# In [41]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.scatter(2, 10)
plt.scatter(5, 37, c = 'orange', zorder=3) # new
_ = ax.plot(x,y)

# %% [markdown]
# To find the slope $m$ between points $P$ and $Q$: 
# $$m = \frac{\text{change in }y}{\text{change in }x} = \frac{\Delta y}{\Delta x} = \frac{y_2 - y_1}{x_2 - x_1} = \frac{37-10}{5-2} = \frac{27}{3} = 9$$

# In [42]
m = (37-10)/(5-2)
m

# %% [markdown]
# To plot the line that passes through $P$ and $Q$, we can rearrange the equation of a line $y = mx + b$ to solve for $b$: 
# $$b = y - mx$$

# In [43]
b = 37-m*5
b

# In [44]
line_y = m*x + b

# In [45]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.scatter(2, 10)
plt.scatter(5, 37, c='orange', zorder=3)
plt.ylim(-5, 150) # new
plt.plot(x, line_y, c='orange') # new
_ = ax.plot(x,y)

# %% [markdown]
# The closer $Q$ becomes to $P$, the closer the slope $m$ comes to being the true tangent of the point $P$. Let's demonstrate this with another point $Q$ at $x = 2.1$.

# %% [markdown]
# Previously, our $\Delta x$ between $Q$ and $P$ was equal to 3. Now it is much smaller: $$\Delta x = x_2 - x_1 = 2.1 - 2 = 0.1 $$

# In [46]
f(2.1)

# In [47]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.scatter(2, 10)
plt.scatter(2.1, 10.61, c = 'orange', zorder=3)
_ = ax.plot(x,y)

# In [48]
m = (10.61-10)/(2.1-2)
m

# In [49]
b = 10.61-m*2.1
b

# In [50]
line_y = m*x + b

# In [51]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.scatter(2, 10)
plt.scatter(2.1, 10.61, c='orange', zorder=3)
plt.ylim(-5, 150)
plt.plot(x, line_y, c='orange', zorder=3)
_ = ax.plot(x,y)

# %% [markdown]
# The closer $Q$ becomes to $P$ (i.e., $\Delta x$ approaches 0), the clearer it becomes that the slope $m$ at point $P$ = (2, 10) is equal to 6.

# %% [markdown]
# Let's make $\Delta x$ extremely small, 0.000001, to illustrate this:

# In [52]
delta_x = 0.000001
delta_x

# In [53]
x1 = 2
y1 = 10

# %% [markdown]
# Rearranging $\Delta x = x_2 - x_1$, we can calculate $x_2$ for our point $Q$, which is now extremely close to $P$: 
# $$x_2 = x_1 + \Delta x$$

# In [54]
x2 = x1 + delta_x
x2

# %% [markdown]
# $y_2$ for our point $Q$ can be obtained with the usual function $f(x)$: 
# $$y_2 = f(x_2)$$

# In [55]
y2 = f(x2)
y2

# %% [markdown]
# To find the slope $m$, we continue to use $$m = \frac{\Delta y}{\Delta x} = \frac{y_2 - y_1}{x_2 - x_1}$$

# In [56]
m = (y2 - y1)/(x2 - x1)
m

# %% [markdown]
# Boom! Using the delta method, we've shown that at point $P$, the slope of the curve is 6. 

# %% [markdown]
# **Exercise**: Using the delta method, find the slope of the tangent where $x = -1$.

# %% [markdown]
# **Spoiler alert! The solution's below.**

# In [57]
x1 = -1

# In [58]
y1 = f(x1)
y1

# %% [markdown]
# Point $P$ is located at (-1, 1)

# In [59]
delta_x

# In [60]
x2 = x1 + delta_x
x2

# In [61]
y2 = f(x2)
y2

# %% [markdown]
# Quick aside: Pertinent to defining differentiation as an equation, an alternative way to calculate $y_2$ is $f(x + \Delta x)$

# In [62]
y2 = f(x1 + delta_x)
y2

# %% [markdown]
# Point $Q$ is at (-0.999999, 1.000000000001), extremely close to $P$.

# In [63]
m = (y2-y1)/(x2-x1)
m

# %% [markdown]
# Therefore, as $x_2$ becomes infinitely close to $x_1$, it becomes clear that the slope $m$ at $x_1 = -1$ is equal to zero. Let's plot it out: 

# In [64]
b = y2-m*x2
b

# In [65]
line_y = m*x + b

# In [66]
fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.scatter(x1, y1)
plt.scatter(x2, y2, c='orange', zorder=3)
plt.ylim(-5, 150)
plt.plot(x, line_y, c='orange', zorder=3)
_ = ax.plot(x,y)

# %% [markdown]
# As $Q$ becomes infinitely close to $P$:
# * $x_2$ - $x_1$ approaches 0
# * In other words, $\Delta x$ approaches 0
# * This can be denoted as $\Delta x \to 0$

# %% [markdown]
# Using the delta method, we've derived the definition of differentiation from first principles. The derivative of $y$ (denoted $dy$) with respect to $x$ (denoted $dx$) can be represented as: 
# $$\frac{dy}{dx} = \lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x}$$

# %% [markdown]
# Expanding $\Delta y$ out to $y_2 - y_1$: 
# $$\frac{dy}{dx} = \lim_{\Delta x \to 0} \frac{y_2 - y_1}{\Delta x}$$

# %% [markdown]
# Finally, replacing $y_1$ with $f(x)$ and replacing $y_2$ with $f(x + \Delta x)$, we obtain a common representation of differentiation:
# $$\frac{dy}{dx} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

# %% [markdown]
# Let's observe the differentiation equation in action: 

# In [67]
def diff_demo(my_f, my_x, my_delta):
    return (my_f(my_x + my_delta) - my_f(my_x)) / my_delta

# In [68]
deltas = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

# In [69]
for delta in deltas:
    print(diff_demo(f, 2, delta))

# In [70]
for delta in deltas:
    print(diff_demo(f, -1, delta))

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ## Segment 3: Automatic Differentiation

# %% [markdown]
# **TensorFlow** and **PyTorch** are the two most popular automatic differentiation libraries.

# %% [markdown]
# Let's use them to calculate $dy/dx$ at $x = 5$ where: 

# %% [markdown]
# $$y = x^2$$

# %% [markdown]
# $$ \frac{dy}{dx} = 2x = 2(5) = 10 $$

# %% [markdown]
# ### Autodiff with PyTorch

# In [71]
import torch

# In [72]
x = torch.tensor(5.0)

# In [73]
x

# In [74]
x.requires_grad_() # contagiously track gradients through forward pass

# In [75]
y = x**2

# In [76]
y.backward() # use autodiff

# In [77]
x.grad

# %% [markdown]
# ### Autodiff with TensorFlow

# In [78]
import tensorflow as tf

# In [79]
x = tf.Variable(5.0)

# In [80]
with tf.GradientTape() as t:
    t.watch(x) # track forward pass
    y = x**2

# In [81]
t.gradient(y, x) # use autodiff

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# As usual, PyTorch feels more intuitive and pythonic than TensorFlow. See the standalone [*Regression in PyTorch*](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/regression-in-pytorch.ipynb) notebook for an example of autodiff paired with gradient descent in order to fit a simple regression line.

