# -*- coding: utf-8 -*-
# Auto-generated from 'LA_vectors_dotProduct.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# #     COURSE: Linear algebra: theory and implementation
# ##    SECTION: Vectors
# ###      VIDEO: Vector-vector multiplication: the dot product
# 
# #### Instructor: sincxpress.com
# ##### Course url: https://www.udemy.com/course/linear-algebra-theory-and-implementation/?couponCode=202110

# In [ ]
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# In [ ]
## many ways to compute the dot product

v1 = np.array([ 1, 2, 3, 4, 5, 6 ])
v2 = np.array([ 0, -4,  -3, 6, 5 ])

# method 1
dp1 = sum( np.multiply(v1,v2) )

# method 2
dp2 = np.dot( v1,v2 )

# method 3
dp3 = np.matmul( v1,v2 )

# method 4
dp4 = 0  # initialize

# loop over elements
for i in range(len(v1)):
    
    # multiply corresponding element and sum
    dp4 = dp4 + v1[i]*v2[i]


print(dp1,dp2,dp3,dp4)

