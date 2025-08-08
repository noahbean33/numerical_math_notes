# -*- coding: utf-8 -*-
# Auto-generated from 'LA_introMatrices_transpose.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# #     COURSE: Linear algebra: theory and implementation
# ##    SECTION: Introduction to matrices
# ###      VIDEO: Transpose
# 
# #### Instructor: sincxpress.com
# ##### Course url: https://www.udemy.com/course/linear-algebra-theory-and-implementation/?couponCode=202110

# In [ ]
import numpy as np

# In [ ]
M = np.array([ [1,2,3],
               [2,3,4] ])

print(M), print('')
print(M.T), print('') # one transpose
print(M.T.T), print('') # double-transpose returns the original matrix

# can also use the function transpose
print(np.transpose(M))

# In [ ]
# warning! be careful when using complex matrices
C = np.array([ [4+1j , 3 , 2-4j] ])

print(C), print('')
print(C.T), print('')
print(np.transpose(C)), print('')

# Note: In MATLAB, the transpose is the Hermitian transpose; 
#       in Python, you need to call the Hermitian explicitly by first converting from an array into a matrix
print(C.conjugate().T) # note the sign flips!
# Another note: the code I used in the video will soon be depreciated; use the above line instead.

