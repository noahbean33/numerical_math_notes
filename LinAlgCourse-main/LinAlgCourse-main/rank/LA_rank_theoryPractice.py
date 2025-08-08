# -*- coding: utf-8 -*-
# Auto-generated from 'LA_rank_theoryPractice.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# #     COURSE: Linear algebra: theory and implementation
# ##    SECTION: Matrix rank
# ###      VIDEO: Computing rank: theory and practice
# 
# #### Instructor: sincxpress.com
# ##### Course url: https://www.udemy.com/course/linear-algebra-theory-and-implementation/?couponCode=202110

# In [ ]
import numpy as np
import matplotlib.pyplot as plt
import math

# In [ ]
# make a matrix
m = 4
n = 6

# create a random matrix
A = np.random.randn(m,n)

# what is the largest possible rank?
ra = np.linalg.matrix_rank(A)
print('rank = ' + str(ra))

# set last column to be repeat of penultimate column
B = A
B[:,-1] = B[:,-2]

rb = np.linalg.matrix_rank(B)
# print('rank = ' + str(rb))

# In [ ]
## adding noise to a rank-deficient matrix

# square for convenience
A = np.round( 10*np.random.randn(m,m) )

# reduce the rank
A[:,-1] = A[:,-2]

# noise level
noiseamp = .001

# add the noise
B = A + noiseamp*np.random.randn(m,m)

print('rank (w/o noise) = ' + str(np.linalg.matrix_rank(A)))
print('rank (with noise) = ' + str(np.linalg.matrix_rank(B)))

