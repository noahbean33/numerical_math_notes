# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_applications_optimization.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Applications
# ### LECTURE: CodeChallenge: farmers and Qberts
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# better image resolution
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# In [ ]

# %% [markdown]
# # Exercise 1: The farmer's fence

# In [ ]
# create symbolic variables
x = sym.symbols('x')

# y in terms of x
y = 400/x

# define cost function
C = 2*x + 3*y/2
C

# In [ ]
# quick plot of cost as function of x
sym.plot(C,(x,3,70),ylim=[50,150]);

# In [ ]
# derivative of cost function
dC = sym.diff(C)

# solve x in dA=0
solX = sym.solve(dC,x)
print(solX)

# but we only need the positive solution
solX = solX[1]
solX

# In [ ]
# then solve for y
solY = y.subs(x,solX)
solY

# In [ ]
# lambda functions to plot the data in matplotlib
xx = np.linspace(3,70,2001)

C_lambda  = sym.lambdify(x,C)
dC_lambda = sym.lambdify(x,dC)

# plot the function
_,axs = plt.subplots(2,1,figsize=(8,5))
axs[0].plot(xx,C_lambda(xx),linewidth=2)
axs[0].plot(solX,C_lambda(solX),'ro')
axs[0].set_xlim(xx[[0,-1]])
axs[0].set_ylim([50,150])
axs[0].set_xlabel('Length of side x')
axs[0].set_ylabel('Cost')
axs[0].set_title('Cost as a function of side length')

# plot the derivative
axs[1].plot(xx,dC_lambda(xx),linewidth=2)
axs[1].plot(xx[[0,-1]],[0,0],'--',color=[.6,.6,.6])
axs[1].plot(solX,dC_lambda(solX),'ro')
axs[1].set_xlim(xx[[0,-1]])
axs[1].set_ylim([-10,5])
axs[1].set_xlabel('Length of side x')
axs[1].set_ylabel('dC/dx')

plt.tight_layout()
plt.show()

# In [ ]
# (out of curiosity) confirm area and calculate perimeter and cost
print(f'Confirm area = {solX*solY} m^2')
print(f'Total fence  = {sym.N(2*solX + 2*solY):.2f} meters')
print(f"Fred's cost  = {sym.N(2*solX + 3*solY/2)*100:.2f} euros")
print(f"Fran's cost  = {sym.N(solY/2)*100:.2f} euros")

# In [ ]

# %% [markdown]
# # Exercise 2: Qbert's cost-saving

# In [ ]
# need to redefine y as its own variable
y = sym.symbols('y')

# volume function
V = x**2 * y - 200

# solve for y
yInTermsOfX = sym.solve(V,y)[0]

# surface area
S = x**2 + 4*x*yInTermsOfX
S

# In [ ]
# derivative of S and solve for x
dS = sym.diff(S)
solX = sym.solve(dS,x)
solX

# In [ ]
# then solve for y
solY = yInTermsOfX.subs(x,solX[0])
solY

# In [ ]
# surface areas of the two sides sizes
print(f'Surface area of the bottom: {sym.N(solX[0]**2):.3f} cm^2')
print(f'Surface area of one side: {sym.N(solX[0]*solY):.3f} cm^2')
print(f'Surface area of all sides: {4*sym.N(solX[0]*solY):.3f} cm^2')

# In [ ]
# lambda functions to plot the data in matplotlib
xx = np.linspace(2,40,2001)

A_lambda  = sym.lambdify(x,S)
dA_lambda = sym.lambdify(x,sym.diff(S))

# plot the function
_,axs = plt.subplots(2,1,figsize=(8,5))
axs[0].plot(xx,A_lambda(xx),linewidth=2)
axs[0].plot(solX[0],A_lambda(solX[0]),'ro')
axs[0].set_xlim(xx[[0,-1]])
axs[0].set_ylim([0,1500])
axs[0].set_xlabel('Length of side x')
axs[0].set_ylabel('Area')
axs[0].set_title('Surface area as a function of side length')

# plot the derivative
axs[1].plot(xx,dA_lambda(xx),linewidth=2)
axs[1].plot(xx[[0,-1]],[0,0],'--',color=[.6,.6,.6])
axs[1].plot(solX[0],dA_lambda(solX[0]),'ro')
axs[1].set_xlim(xx[[0,-1]])
axs[1].set_ylim([-100,100])
axs[1].set_xlabel('Length of side x')
axs[1].set_ylabel('dV/dx')

plt.tight_layout()
plt.show()

# In [ ]

