# -*- coding: utf-8 -*-
# Auto-generated from 'LA_projorth_R2.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# #     COURSE: Linear algebra: theory and implementation
# ##    SECTION: Projections and orthogonalization
# ### VIDEO: Projections in R^2
# 
# #### Instructor: sincxpress.com
# ##### Course url: https://www.udemy.com/course/linear-algebra-theory-and-implementation/?couponCode=202110

# In [1]
import numpy as np
import matplotlib.pyplot as plt

# In [ ]

# point b
b = np.array([4,1])

# line a
a = np.array([ 2, 5 ])

# beta
beta = (a.T@b) / (a.T@a)

# draw!
plt.plot(b[0],b[1],'ko',label='b')
plt.plot([0, a[0]],[0, a[1]],'b',label='a')

# now plot projection line
plt.plot([b[0], beta*a[0]],[b[1], beta*a[1]],'r--',label=r'b-$\beta$a')
plt.axis('square')
plt.grid()
plt.legend()
plt.axis((-6, 6, -6, 6))

plt.show()

