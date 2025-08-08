# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geometry_calculartFaceVase.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in geometry
# ### LECTURE: Calculart: A vase, two faces, or just math?
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import Math
import scipy.integrate as spi

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Step 1: Draw Rubin's illusion

# In [ ]
# xy coordinates
y = np.array([1,0.981328,0.975104,0.973029,0.96888,0.960581,0.952282,0.943983,0.93361,0.925311,0.917012,0.906639,0.892116,0.875519,0.860996,0.844398,0.817427,0.804979,0.788382,0.746888,0.73029,0.717842,0.688797,0.6639,0.647303,0.624481,0.605809,0.589212,0.562241,0.545643,0.529046,0.506224,0.495851,0.481328,0.477178,0.462656,0.452282,0.43361,0.423237,0.410788,0.404564,0.39834,0.394191,0.390041,0.387967,0.383817,0.377593,0.363071,0.350622,0.342324,0.325726,0.311203,0.302905,0.275934,0.261411,0.246888,0.23029,0.211618,0.19917,0.190871,0.174274,0.16805,0.157676,0.149378,0.141079,0.13278,0.126556,0.116183,0.103734,0.0871369,0.0726141,0.060166,0.0435685,0.026971,0.0124481,0])
xR = np.array([1, 0.96094299, 0.91765732, 0.87437165, 0.80078537, 0.7574997 , 0.7272, 0.6969003 , 0.65794329, 0.62764268, 0.58868567, 0.55405732, 0.50644299, 0.46748597, 0.42852896, 0.38957104, 0.33762896, 0.31598567, 0.3029997 , 0.28135732, 0.2727    , 0.2727    , 0.28135732, 0.28568597, 0.27702866, 0.25971403, 0.22941433, 0.20344329, 0.13851433, 0.1168714 , 0.09522857, 0.0909    , 0.1168714 , 0.13851433, 0.14717146, 0.16448573, 0.1688143 , 0.14717146, 0.1298571 , 0.12552854, 0.13851433, 0.15150003, 0.16448573, 0.17314287, 0.17314287, 0.16448573, 0.15150003, 0.14717146, 0.16015716, 0.17747143, 0.20777104, 0.22508567, 0.22508567, 0.2120997 , 0.19911463, 0.1818    , 0.17314287, 0.1818    , 0.2120997 , 0.23374299, 0.32464299, 0.36792866, 0.44584268, 0.51077165, 0.56704329, 0.61898537, 0.6665997 , 0.70122896, 0.73585732, 0.77914299, 0.82242866, 0.8483997 , 0.87004299, 0.89601403, 0.91332866,0.93064329])
xL = np.array([-1, -0.96094299, -0.91765732, -0.87437165, -0.80078537,-0.7574997 , -0.7272 , -0.6969003 , -0.65794329, -0.62764268,-0.58868567, -0.55405732, -0.50644299, -0.46748597, -0.42852896,-0.38957104, -0.33762896, -0.31598567, -0.3029997 , -0.28135732,-0.2727    , -0.2727    , -0.28135732, -0.28568597, -0.27702866,-0.25971403, -0.22941433, -0.20344329, -0.13851433, -0.1168714 ,-0.09522857, -0.0909    , -0.1168714 , -0.13851433, -0.14717146,-0.16448573, -0.1688143 , -0.14717146, -0.1298571 , -0.12552854,-0.13851433, -0.15150003, -0.16448573, -0.17314287, -0.17314287,-0.16448573, -0.15150003, -0.14717146, -0.16015716, -0.17747143,-0.20777104, -0.22508567, -0.22508567, -0.2120997 , -0.19911463,-0.1818    , -0.17314287, -0.1818    , -0.2120997 , -0.23374299,-0.32464299, -0.36792866, -0.44584268, -0.51077165, -0.56704329,-0.61898537, -0.6665997 , -0.70122896, -0.73585732, -0.77914299,-0.82242866, -0.8483997 , -0.87004299, -0.89601403, -0.91332866,-0.93064329])

# In [ ]

# In [ ]

# In [ ]

# In [ ]

# In [ ]

# In [ ]

# In [ ]

# In [ ]
# Giving plenty of space here so you don't have to peak :P

# In [ ]

# In [ ]

# In [ ]

# In [ ]

# In [ ]

# In [ ]

# In [ ]
# draw the graph
plt.figure(figsize=(8,10))
plt.plot(xL,y,'k-')
plt.plot(xR,y,'k-')
plt.fill_between(xR,y,color='k',alpha=.4)
plt.fill_between(xL,y,color='k',alpha=.4)
plt.show()

# In [ ]

# %% [markdown]
# # Step 2: Calculate the three sub-areas

# In [ ]
# drawn again with flipping the axes
plt.figure(figsize=(10,8))
plt.plot(y,xL,'k.-',markersize=10)
plt.plot(y,xR,'k.-',markersize=10)

plt.fill_between(y,xR,xL,color='k',alpha=.4)
plt.fill_between(y,xL,-np.ones(len(y)),color='r',alpha=.4)
plt.fill_between(y,xR,np.ones(len(y)),color='b',alpha=.2)

plt.show()

# In [ ]
# now compute the area. consider each individual slice to be a trapezoid.
# area of a trapezoid is (a+b)/2*h
areaVase = 0
areaLeft = 0
areaRigt = 0

for idx in range(1,len(y)):

  # calculate each part of the area formula
  h = y[idx-1] - y[idx]
  a = xR[idx-1]- xL[idx-1]
  b = xR[idx]  - xL[idx]

  # area of the slice and add to the total
  areaVase += (a+b)/2 * h
  areaLeft += (xL[idx]+xL[idx-1] + 2)/2 * h
  areaRigt += (2 - xR[idx]-xR[idx-1])/2 * h

print(f'Total area of the vase       : {areaVase:.3f}')
print(f'Total area of the left face  : {areaLeft:.3f}')
print(f'Total area of the right face : {areaRigt:.3f}')
print(f'Sum of three areas           : {areaLeft+areaRigt+areaVase:.3f}')

# In [ ]

