# -*- coding: utf-8 -*-
# Auto-generated from 'LA_svd_TheSVD.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# #     COURSE: Linear algebra: theory and implementation
# ##    SECTION: Singular value decomposition
# ###     VIDEO: Singular value decomposition
# 
# #### Instructor: sincxpress.com
# ##### Course url: https://www.udemy.com/course/linear-algebra-theory-and-implementation/?couponCode=202110

# In [ ]
import numpy as np
import matplotlib.pyplot as plt

# In [ ]
# the matrix
A = [ [3,0,5], [8,1,3] ]

# Note differences from MATLAB: S is vector, V is V'
U,S,V = np.linalg.svd(A)

print(U), print(' ')
print(S), print(' ')
print(V)

# In [ ]
plt.subplot(141)
plt.imshow(A)
plt.title('A')
plt.axis('off')

plt.subplot(142)
plt.imshow(U)
plt.title('U')
plt.axis('off')

plt.subplot(143)
plt.imshow(np.diag(S))
plt.title('$\Sigma$')
plt.axis('off')

plt.subplot(144)
plt.imshow(V)
plt.title('$V^T$')
plt.axis('off')

plt.show()

# In [ ]

