# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_stats_CCmakePdfs.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in statistics
# ### LECTURE: Probability density functions (pdfs)
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import Math
from scipy import stats
import sympy.stats
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
# # Exercise 1: Create a logistic pdf

# In [ ]
# parameters
xx = np.linspace(-5,20,5001)
dx = xx[1]-xx[0]
mu = 0
s = 2

# create the pdf
pdf_num = np.exp(-(xx-mu)/s)
pdf_den = s * (1 + np.exp(-(xx-mu)/s))**2

logistic_pdf_np = pdf_num / pdf_den * dx


# and visualize it
plt.figure(figsize=(8,4))
plt.plot(xx,logistic_pdf_np)

plt.gca().set(xlabel='x',ylabel='Prob. density',xlim=[xx[0]-.5,xx[-1]+.5],
              title=f'Logistic pdf ($\\mu$ = {mu}, $\\sigma$ = {s})\nSum = {np.sum(logistic_pdf_np):.4f}')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Compare it to scipy's pdf

# In [ ]
logistic_pdf_sp = stats.logistic.pdf(xx,loc=mu,scale=s)
logistic_pdf_sp *= dx

# and visualize it
plt.figure(figsize=(8,4))
plt.plot(xx,logistic_pdf_sp)

plt.gca().set(xlabel='x',ylabel='Prob. density',xlim=[xx[0]-.5,xx[-1]+.5],
              title=f'Logistic pdf ($\\mu$ = {mu}, $\\sigma$ = {s})\nSum = {np.sum(logistic_pdf_sp):.4f}')
plt.show()

# In [ ]
# visualize both
plt.figure(figsize=(8,4))

plt.plot(xx[::50],logistic_pdf_np[::50],'o',label='numpy')
plt.plot(xx,logistic_pdf_sp,label='scipy')

plt.legend()
plt.gca().set(xlabel='x',ylabel='pdf',xlim=[xx[0]-.5,xx[-1]+.5],
              title=f'Logistic pdf ($\\mu$ = {mu}, $\\sigma$ = {s})\nSum$_{{np}}$ = {np.sum(logistic_pdf_np):.3f}, Sum$_{{sp}}$ = {np.sum(logistic_pdf_sp):.3f}')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Exact definite integral

# In [ ]
# define some variables and parameters
x = sym.symbols('x')

# a sympy expression for the distribution
L = sym.stats.Logistic('L',sym.sympify(mu),sym.sympify(s))
logistic_pdf_sym = sym.stats.density(L)(x)

# and print it
display(Math('p(x) = %s' %sym.latex(logistic_pdf_sym)))

# In [ ]
# exact definite integral
a = 0
b = sym.pi

exactInt = sym.integrate(logistic_pdf_sym,(x,a,b))

display(Math('%s \;=\; %s' %(sym.latex(sym.Integral(logistic_pdf_sym,(x,a,b))),sym.latex(exactInt))))

# In [ ]
# and draw

# lambdify first
logistic_pdf_l = sym.lambdify(x,logistic_pdf_sym)

# numerically evaluate the pdf
pdf_sym = logistic_pdf_l(xx) * dx

# and plot
plt.figure(figsize=(8,4))
plt.plot(xx,pdf_sym,'r',label='Logistic pdf')
x4defint = (xx>=a) & (xx<=b)
plt.fill_between(xx[x4defint],pdf_sym[x4defint],color='r',edgecolor='none',alpha=.5,label=f'Area = {float(exactInt):.3f}')

plt.legend()
plt.gca().set(xlabel='x',ylabel='pdf',xlim=[xx[0]-.5,xx[-1]+.5])

plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Approximate definite integral

# In [ ]
approxInt = spi.simpson(logistic_pdf_sp[x4defint],dx=dx)

print(f'Area from sympy = {float(exactInt):.8f}')
print(f'Area from scipy = {approxInt:.8f}')

# In [ ]

