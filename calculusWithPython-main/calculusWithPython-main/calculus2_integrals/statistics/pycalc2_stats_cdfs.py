# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_stats_cdfs.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in statistics
# ### LECTURE: Cumulative distribution functions (cdfs)
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
# # Creating a cdf in sympy

# In [ ]
# define some variables and parameters
x = sym.symbols('x')
m = 1.5
alpha = 2
xx = np.linspace(.01,5,500)

# a sympy expression for the distribution
P = sym.stats.Pareto('P',m,alpha)
Pcdf_expr = sym.stats.cdf(P)(x)

# lambdify that
Pcdf_l = sym.lambdify(x,Pcdf_expr)

# numerically evaluate the cdf
cdf = Pcdf_l(xx)

# and plot
plt.figure(figsize=(8,4))
plt.plot(xx,cdf)

# finalize the plot
plt.gca().set(xlim=xx[[0,-1]],xlabel='x',ylabel='Probability',
              title=f'Pareto cdf (m = {m}, $\\alpha$ = {alpha})')
plt.tight_layout()
plt.show()

# In [ ]
# let's look at some values
display(Math('c(x) \;=\; %s' %sym.latex(Pcdf_expr))), print('')
display(Math('c(%g) \;=\; %g' %(xx[-1],cdf[-1])))
display(Math('c(%s) \;=\; %g' %(sym.latex(sym.oo),Pcdf_l(sym.oo))))

# In [ ]

# %% [markdown]
# # Using scipy

# In [ ]
# parameters
mu = 1
sigma = .3

# get the pdf values
xx = np.linspace(-5,5,401)
cdf = stats.logistic.cdf(xx,loc=mu,scale=sigma)

# and plot
plt.figure(figsize=(8,3))
plt.plot(xx,cdf)
plt.axvline(mu,linewidth=1,color='gray',linestyle='--')

# finalize
plt.gca().set(xlim=xx[[0,-1]],xlabel='x',ylabel='Cumulative prob.',title=r'Logistic cdf ($\mu = %g, \sigma = %g$)' %(mu,sigma))
plt.tight_layout()
plt.show()

# In [ ]

