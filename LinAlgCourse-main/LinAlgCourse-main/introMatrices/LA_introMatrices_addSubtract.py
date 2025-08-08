# -*- coding: utf-8 -*-
# Auto-generated from 'LA_introMatrices_addSubtract.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# #     COURSE: Linear algebra: theory and implementation
# ##    SECTION: Introduction to matrices
# ###      VIDEO: Matrix addition and subtraction
# 
# #### Instructor: sincxpress.com
# ##### Course url: https://www.udemy.com/course/linear-algebra-theory-and-implementation/?couponCode=202110

# In [ ]
import numpy as np

# In [ ]
# create random matrices
A = np.random.randn(5,4)
B = np.random.randn(5,3)
C = np.random.randn(5,4)

# try to add them
A+B
A+C

# In [ ]
# "shifting" a matrix
l = .03 # lambda
N = 5  # size of square matrix
D = np.random.randn(N,N) # can only shift a square matrix

Ds = D + l*np.eye(N)
print(D), print(' '), print(Ds)

