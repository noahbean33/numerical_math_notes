# -*- coding: utf-8 -*-
# Auto-generated from 'LA_inverse_conceptUses.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# #     COURSE: Linear algebra: theory and implementation
# ##    SECTION: Matrix inverse
# ###      VIDEO: Concept and uses of the inverse
# 
# #### Instructor: sincxpress.com
# ##### Course url: https://www.udemy.com/course/linear-algebra-theory-and-implementation/?couponCode=202110

# In [ ]
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

# In [ ]
# size of square matrix
m = 3

# generate random matrix
A = np.random.randn(m,m)

# compute its inverse
Ainv = np.linalg.inv(A)

# and check the multiplication
idm = A@Ainv

# print the matrix. Note the computer rounding errors on the off-diagonals
print(idm)

# In [ ]
# show in an image
plt.subplot(131)
plt.imshow(A)
plt.title('Matrix A')

plt.subplot(132)
plt.imshow(Ainv)
plt.title('Matrix $A^{-1}$')

plt.subplot(133)
plt.imshow(idm)
plt.title('AA$^{-1}$')

plt.show()

