# -*- coding: utf-8 -*-
# Auto-generated from 'LA_matrixMults_standardMatrixMult.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# #     COURSE: Linear algebra: theory and implementation
# ##    SECTION: Matrix multiplications
# ###      VIDEO: Standard matrix multiplication, parts 1 & 2
# 
# #### Instructor: sincxpress.com
# ##### Course url: https://www.udemy.com/course/linear-algebra-theory-and-implementation/?couponCode=202110

# In [ ]
import numpy as np
import matplotlib.pyplot as plt
import math

# In [ ]
## rules for multiplication validity

m = 4
n = 3
k = 6

# make some matrices
A = np.random.randn(m,n)
B = np.random.randn(n,k)
C = np.random.randn(m,k)

# test which multiplications are valid.
# Think of your answer first, then test.
np.matmul(A,B)
np.matmul(A,A)
np.matmul(A.T,C)
np.matmul(B,B.T)
np.matmul(np.matrix.transpose(B),B)
np.matmul(B,C)
np.matmul(C,B)
np.matmul(C.T,B)
np.matmul(C,B.T)

