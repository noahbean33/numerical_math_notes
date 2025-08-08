# -*- coding: utf-8 -*-
# Auto-generated from '8-optimization.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/8-optimization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Optimization

# %% [markdown]
# This class, *Optimization*, is the eighth of eight classes in the *Machine Learning Foundations* series. It builds upon the material from each of the other classes in the series -- on linear algebra, calculus, probability, statistics, and algorithms -- in order to provide a detailed introduction to training machine learning models. 
# 
# Through the measured exposition of theory paired with interactive examples, you’ll develop a working understanding of all of the essential theory behind the ubiquitous gradient descent approach to optimization as well as how to apply it yourself — both at a granular, matrix operations level and a quick, abstract level — with TensorFlow and PyTorch. You’ll also learn about the latest optimizers, such as Adam and Nadam, that are widely-used for training deep neural networks. 
# 
# Over the course of studying this topic, you'll:
# 
# * Discover how the statistical and machine learning approaches to optimization differ, and why you would select one or the other for a given problem you’re solving.
# * Understand exactly how the extremely versatile (stochastic) gradient descent optimization algorithm works, including how to apply it
# * Get acquainted with the “fancy” optimizers that are available for advanced machine learning approaches (e.g., deep learning) and when you should consider using them.
# 
# Note that this Jupyter notebook is not intended to stand alone. It is the companion code to a lecture or to videos from Jon Krohn's [Machine Learning Foundations](https://github.com/jonkrohn/ML-foundations) series, which offer detail on the following:
# 
# *Segment 1: The Machine Learning Approach to Optimization*
# 
# * The Statistical Approach to Regression: Ordinary Least Squares
# * When Statistical Approaches to Optimization Break Down
# * The Machine Learning Solution 
# 
# *Segment 2: Gradient Descent*
# 
# * Objective Functions
# * Cost / Loss / Error Functions
# * Minimizing Cost with Gradient Descent
# * Learning Rate
# * Critical Points, incl. Saddle Points
# * Gradient Descent from Scratch with PyTorch
# * The Global Minimum and Local Minima
# * Mini-Batches and Stochastic Gradient Descent (SGD)
# * Learning Rate Scheduling
# * Maximizing Reward with Gradient Ascent
# 
# *Segment 3: Fancy Deep Learning Optimizers*
# 
# * A Layer of Artificial Neurons in PyTorch
# * Jacobian Matrices
# * Hessian Matrices and Second-Order Optimization
# * Momentum
# * Nesterov Momentum
# * AdaGrad
# * AdaDelta 
# * RMSProp
# * Adam 
# * Nadam
# * Training a Deep Neural Net
# * Resources for Further Study

# In [18]
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Segment 1: Optimization Approaches

# %% [markdown]
# Refer to the slides and the *Ordinary Least Squares* section of the [*Intro to Stats* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/6-statistics.ipynb).

# %% [markdown]
# ## Segment 2: Gradient Descent

# %% [markdown]
# ### Cost Functions

# %% [markdown]
# Fundamentally, for some instance $i$, we'd like to quantify the difference between the "correct" target model output $y_i$ and the model's predicted output $\hat{y}_i$. A first idea might be to take the simple difference: $$\Delta y_i = \hat{y}_i - y_i$$

# %% [markdown]
# For computational efficiency (we'll explore this later in the segment), in ML we seldom consider the cost associated with a single instance $i$. Instead, we typically consider several instances simultaneously, in which case calculating the simple difference is largely  ineffective because positive and negative $\Delta y_i$ values cancel out. E.g., consider a situation where: 
# 
# * $\Delta y_1 = \hat{y}_1 - y_1$ = 7 - 2 = 5
# * $\Delta y_2 = \hat{y}_2 - y_2$ = 3 - 8 = -5
# 
# On an individual-instance basis, there are differences between the predicted and target outputs, indicating the model could be improved. Despite this, the total cost ($\Sigma \Delta y_i = \Delta y_1 + \Delta y_2 = 5-5$) is zero and therefore the mean cost ($\frac{\Sigma{\Delta y_i}}{n} = \frac{0}{2}$) is also zero, erroneously implying a perfect model fit.

# %% [markdown]
# #### Mean Absolute Error

# %% [markdown]
# A straightforward resolution to the simple-difference shortcoming is to take the absolute value of the difference. I.e., using the same contrived values as above: 
# 
# * $|\Delta y_1| = |\hat{y}_1 - y_1|$ = |7 - 2| = 5
# * $|\Delta y_2| = |\hat{y}_2 - y_2|$ = |3 - 8| = 5
# 
# In this case:
# 
# * The total error is ten: $\Sigma |\Delta y_i| = |\Delta y_1| + |\Delta y_2| = 5+5$ 
# * The **mean absolute error** (MAE) is five: $\frac{\Sigma{|\Delta y_i}|}{n} = \frac{10}{2}$

# %% [markdown]
# #### Mean Squared Error

# %% [markdown]
# While MAE satisfactorily quantifies cost across multiple instances, for reasons we'll numerate in a moment, it is much more common in ML to use **mean *squared* error** (MSE). 
# 
# Let's use the contrived values again to show how MSE is calculated. As covered in [*Calculus II*](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/4-calculus-ii.ipynb), the individual-instance variant of MSE is *squared error* or *quadratic cost*, which -- as with absolute error -- dictates that the cost for each instance $i$ must be $\geq 0$: 
# 
# * $(\hat{y}_1 - y_1)^2 = (7 - 2)^2 = 5^2 = 25$
# * $(\hat{y}_2 - y_2)^2 = (3 - 8)^2 = (-5)^2 = 25$
# 
# As suggested by its name, MSE is the average: $$C = \frac{1}{n} \sum_{i=1}^n (\hat{y_i}-y_i)^2 = \frac{1}{2} (25 + 25) = \frac{50}{2} = 25 $$
# 
# Since cost for each instance $i$ must be $\geq 0$, MSE (as with MAE) must therefore also be $\geq 0$. In addition: 
# 
# 1. The partial derivative of the MSE $C$ can be efficiently computed w.r.t. model parameters, providing a *gradient* of cost, $\nabla C$ (the primary focus of [*Calculus II*](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/4-calculus-ii.ipynb) is deriving $\nabla C$). Adjusting model parameters, we can *descend* $\nabla C$ and thereby minimize $C$ (the mechanics of which are the primary focus of *Optimization*).
# 2. Compared to MAE, MSE is relatively tolerant of small $\Delta y_i$ and intolerant of large $\Delta y_i$, a characteristic that tends to lead to better-fitting models.

# In [19]
delta_y = np.linspace(-5, 5, 1000)

# In [20]
abs_error = np.abs(delta_y)

# In [21]
sq_error = delta_y**2

# In [22]
fig, ax = plt.subplots()

plt.plot(delta_y, abs_error)
plt.plot(delta_y, sq_error)
ax.axhline(c='lightgray')

plt.xlabel('Delta y')
plt.ylabel('Error')

ax.set_xlim(-4.2, 4.2)
ax.set_ylim(-1, 17)
_ = ax.legend(['Absolute', 'Squared'])

# %% [markdown]
# There are other cost functions out there (e.g., [cross-entropy cost](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/cross_entropy_cost.ipynb) is the typical choice for a deep learning classifier), but since MSE is the most common, including being the default option for regression models, it will be our focus for the remainder of *ML Foundations*.

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ### Minimizing Cost with Gradient Descent

# %% [markdown]
# Refer to the slides and the [*Gradient Descent from Scratch* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/gradient-descent-from-scratch.ipynb).

# %% [markdown]
# ### Critical Points

# %% [markdown]
# #### Minimum

# In [23]
x = np.linspace(-10, 10, 1000)

# %% [markdown]
# If $y = x^2 + 3x + 4$...

# In [24]
y_min = x**2 + 3*x + 4

# %% [markdown]
# ...then $y' = 2x + 3$. 
# 
# Critical point is located where $y' = 0$, so where $2x + 3 = 0$.
# 
# Rearranging to solve for $x$:
# $$2x = -3$$
# $$x = \frac{-3}{2} = -1.5$$
# 
# At which point, $y = x^2 + 3x + 4 = (-1.5)^2 + 3(-1.5) + 4 = 1.75$
# 
# Thus, the critical point is located at (-1.5, 1.75).

# In [25]
fig, ax = plt.subplots()

plt.scatter(-1.5, 1.75, c='orange')
plt.axhline(y=1.75, c='orange', linestyle='--')

plt.axvline(x=0, c='lightgray')
plt.axhline(y=0, c='lightgray')

ax.set_xlim(-5, 2)
ax.set_ylim(-1, 15)

_ = ax.plot(x, y_min)

# %% [markdown]
# #### Maximum

# %% [markdown]
# If $y = -x^2 + 3x + 4$...

# In [26]
y_max = -x**2 + 3*x + 4

# %% [markdown]
# ...then $y' = -2x + 3$. 
# 
# Critical point is located where $y' = 0$, so where $-2x + 3 = 0$.
# 
# Rearranging to solve for $x$:
# $$-2x = -3$$
# $$x = \frac{-3}{-2} = 1.5$$
# 
# At which point, $y = -x^2 + 3x + 4 = -(1.5)^2 + 3(1.5) + 4 = 6.25$
# 
# Thus, the critical point is located at (1.5, 6.25).

# In [27]
fig, ax = plt.subplots()

plt.scatter(1.5, 6.25, c='orange')
plt.axhline(y=6.25, c='orange', linestyle='--')

plt.axvline(x=0, c='lightgray')
plt.axhline(y=0, c='lightgray')

ax.set_xlim(-2, 5)
ax.set_ylim(-5, 10)

_ = ax.plot(x, y_max)

# %% [markdown]
# #### Saddle point

# %% [markdown]
# If $y = x^3 + 6$...

# In [28]
y_sp = x**3 + 6

# %% [markdown]
# ...then $y' = 3x^2$. 
# 
# Critical point is located where $y' = 0$, so where $3x^2 = 0$.
# 
# Rearranging to solve for $x$:
# $$x^2 = \frac{0}{3} = 0$$
# $$x = \sqrt{0} = 0$$
# 
# At which point, $y = x^3 + 6 = (0)^3 + 6 = 6$
# 
# Thus, the critical point is located at (0, 6).

# In [29]
fig, ax = plt.subplots()

plt.scatter(0, 6, c='orange')
plt.axhline(y=6, c='orange', linestyle='--')

plt.axvline(x=0, c='lightgray')
plt.axhline(y=0, c='lightgray')

ax.set_xlim(-2, 2)
ax.set_ylim(-1, 10)

_ = ax.plot(x, y_sp)

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ### Stochastic Gradient Descent

# %% [markdown]
# Refer to the slides and the [*SGD from Scratch* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/SGD-from-scratch.ipynb).

# %% [markdown]
# ### Learning Rate Scheduling

# %% [markdown]
# Refer to the slides and the [*Learning Rate Scheduling* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/learning-rate-scheduling.ipynb).

# %% [markdown]
# ## Segment 3: Fancy Deep Learning Optimizers

# %% [markdown]
# ### Jacobian and Hessian Matrices

# %% [markdown]
# A *gradient* holds the partial derivatives of a function with vector input and scalar output. That is, $\boldsymbol{f}: \mathbb{R}^n \rightarrow \mathbb{R}$.

# %% [markdown]
# A **Jacobian matrix** holds the partial derivatives of a function with vector input *and vector output*. That is, if $\boldsymbol{f}: \mathbb{R}^m \rightarrow \mathbb{R}^n$. 
# 
# E.g., in the [*Artificial Neurons* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/artificial-neurons.ipynb), $m = 784$ and $n = 128$. In this case: 
# * The Jacobian matrix $\boldsymbol{J}$ has 128 rows and 784 columns.
# * Each element of $\boldsymbol{J}$ contains the partial derivative of a particular output element with respect to a particular input element: $J_{i,j} = \frac{\partial}{\partial x_j} f(\boldsymbol{x})_i$

# %% [markdown]
# Note that a *generalized Jacobian* holds the partial derivatives of a function with an input tensor of any dimensionality and an output tensor of any dimensionality.

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# A **Hessian matrix** is the Jacobian matrix of the gradient. Its elements hold all of the possible second derivatives of some function $f(\boldsymbol{x})$: 
# $$ \boldsymbol{H}(f)(\boldsymbol{x})_{i,j} = \frac{\partial^2}{\partial x_i \partial x_j} f(\boldsymbol{x}) $$

# %% [markdown]
# Otherwise, refer to the slides and the following notebooks: 
# 
# * [*Artificial Neurons*](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/artificial-neurons.ipynb)
# * [*Deep Net in TensorFlow* (1.x)](https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/deep_net_in_tensorflow.ipynb)
# * [*Deep Convolutional Net in TensorFlow* (1.x)](https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/lenet_in_tensorflow.ipynb)
# * [*Deep Convolutional Net in TensorFlow* (2.x, with Keras)](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/lenet_in_tensorflow.ipynb)
# * [*Deep Net in PyTorch*](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/deep_net_in_pytorch.ipynb)

