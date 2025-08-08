# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_stats_pdfs.ipynb' on 2025-08-08T15:22:58
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

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # pdfs using numpy

# In [ ]
# setting up
x = np.linspace(-4,4,1001)
dx = x[1]-x[0]
sigma = 1.4

# the pdf
pdf = 1/np.sqrt(2*np.pi*sigma**2) * np.exp( -x**2/(2*sigma**2) )
pdf *= dx

# confirm: should = 1
sumPdf = np.sum(pdf)


# show the plot
plt.figure(figsize=(8,4))
plt.plot(x,pdf)

plt.gca().set(xlabel='x',ylabel='Probability',title='Gaussian (sum = %.2f)' %sumPdf)
plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # pdfs using sympy

# In [ ]
# define some variables and parameters
x = sym.symbols('x')
dfs = [5,30] # degrees of freedom (df)
fvals = np.linspace(.01,5,500)
dx = fvals[1]-fvals[0]

# a sympy expression for the distribution
F = sym.stats.FDistribution('F',dfs[0],dfs[1])
Fpdf_expr = sym.stats.density(F)(x)

# lambdify that
Fpdf_l = sym.lambdify(x,Fpdf_expr)

# numerically evaluate the pdf
pdf = Fpdf_l(fvals) * dx
sumPdf = np.sum(pdf)

# and plot
plt.figure(figsize=(8,4))
plt.plot(fvals,pdf)

# finalize the plot
plt.gca().set(xlim=fvals[[0,-1]],xlabel='F',ylabel='Probability',
              title=r'F-distribution ($d_1,d_2$ = %g,%s, sum = %.2f)' %(dfs[0],dfs[1],sumPdf))
plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # pdfs using scipy

# In [ ]
# parameters
mu = 1
sigma = .3

# get the pdf values
xx = np.linspace(-5,5,401)
pdf = stats.logistic.pdf(xx,loc=mu,scale=sigma) * (xx[1]-xx[0])

# and plot
plt.figure(figsize=(8,3))
plt.plot(xx,pdf)
plt.axvline(mu,linewidth=1,color='gray',linestyle='--')

# finalize
plt.gca().set(xlim=xx[[0,-1]],xlabel='x',ylabel='probability',title=r'Logistic pdf ($\mu = %g, \sigma = %g$)' %(mu,sigma))
plt.tight_layout()
plt.show()

# In [ ]

