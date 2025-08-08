# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_stats_CCpdfsAndCdfs.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in statistics
# ### LECTURE: CodeChallenge: cdfs and pdfs
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
# # Exercise 1: cdf's from pdf

# In [ ]
x = np.linspace(-5,5,31)
dx = x[1] - x[0]

# the pdf
pdf = stats.logistic.pdf(x)

# cdf version 1: in scipy
cdf_sp = stats.logistic.cdf(x)

# cdf version 2: approximate integral of cdf in numpy
cdf_np = np.cumsum(pdf) * dx

# cdf version 3: approximate integral via simpson's method
cdf_simp = spi.cumulative_simpson(pdf,x=x,initial=0)

# create the plot
_,axs = plt.subplots(1,2,figsize=(12,4))

# pdf on top
axs[0].plot(x,pdf)
axs[0].set(xlabel='x',ylabel='pdf',xlim=x[[0,-1]],title=f'Logistic pdf (N = {len(x)})')

# cdfs
axs[1].plot(x,cdf_sp,label='cdf via sp.stats')
axs[1].plot(x,cdf_np,label='cdf via numpy')
axs[1].plot(x,cdf_simp,label='cdf via simpson')
axs[1].set(xlabel='x',ylabel='cdf',xlim=x[[0,-1]],title=f'Logistic cdf (N = {len(x)})')
axs[1].legend()

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Area from pdf and cdf

# In [ ]
x = np.linspace(-2,2,301)
dx = x[1] - x[0]

# pdf parameters
mu = np.sqrt(2)/2
sigma = 1/np.pi

# upper bound of definite integral
intVal = sigma/mu
intidx = np.argmin(abs(x-intVal))

# get the pdf and cdf
pdf = stats.norm.pdf(x,loc=mu,scale=sigma) * dx
cdf = stats.norm.cdf(x,loc=mu,scale=sigma)

# plot the prob functions
_,axs = plt.subplots(2,1,figsize=(8,8))
axs[0].plot(x,pdf,color='m',linewidth=2)
axs[0].plot([intVal,intVal],[0,pdf[intidx]],'b--')
axs[0].fill_between(x[:intidx],pdf[:intidx],color='b',alpha=.2)
axs[0].annotate(r'$\frac{2}{\sqrt{2}\pi}$',xy=(intVal,pdf[intidx]/2),xytext=((intVal+x[0])/2,pdf[intidx]/2),
                arrowprops={'facecolor':'k'},verticalalignment='center',fontsize=25)
axs[0].set(xlim=x[[0,-1]],xlabel='x',ylim=[-.00004,np.max(pdf)*1.05],
           ylabel='Probability density',title=r'Normal pdf $\left(\mu = \sqrt{2}/2, \sigma = \pi^{-1}, N = %g \right)$' %len(x))

axs[1].plot(x,cdf,color='m',linewidth=2)
axs[1].plot([intVal,intVal],[0,cdf[intidx]],'b--')
axs[1].plot(x[[0,intidx]],[cdf[intidx],cdf[intidx]],'b--')
axs[1].set(xlim=x[[0,-1]],xlabel='x',ylim=[-.015,1.02],
           ylabel='Cumulative probability',title='Normal cdf')


plt.tight_layout()
plt.show()

# In [ ]
area_from_pdf = spi.simpson(pdf[:intidx],dx=dx/dx)
area_from_cdf = cdf[intidx]

print(f'Area from pdf: {area_from_pdf:.4f}')
print(f'Area from cdf: {area_from_cdf:.4f}')

# In [ ]

# %% [markdown]
# # Exercise 3: Exact result in sympy

# In [ ]
# get the exact result in sympy
u = sym.symbols('u')

# parameters using symbols
mu_s = sym.sqrt(2)/2
sigma_s = 1/sym.pi
intVal_s = sigma_s/mu_s

# the function and its integral
N = sym.stats.Normal('N',mu_s,sigma_s)
cdf_expr = sym.stats.cdf(N)(u)

exact_defint = cdf_expr.subs(u,intVal_s)

# and show the function and result
display(Math('c(u) \;=\; %s' %sym.latex(cdf_expr))), print('')
display(Math('c\\left(%s\\right) \;=\; %s \;\\approx\; %g' %(sym.latex(intVal_s),sym.latex(exact_defint),exact_defint.evalf())))

# In [ ]

# %% [markdown]
# # Exercise 4: accuracy of pdf and cdf approximations

# In [ ]
# initialize
Ns = np.logspace(np.log10(51),np.log10(10001),25).astype(int)

areaPdf = np.zeros(len(Ns))
areaCdf = np.zeros(len(Ns))

for idx,n in enumerate(Ns):

  x = np.linspace(-2,2,n)
  dx = x[1] - x[0]
  pdf = stats.norm.pdf(x,loc=mu,scale=sigma) * dx
  cdf = stats.norm.cdf(x,loc=mu,scale=sigma)
  intidx = np.argmin(abs(x-intVal))

  areaPdf[idx] = spi.simpson(pdf[:intidx],dx=dx/dx)
  areaCdf[idx] = cdf[intidx]


# and plot
plt.figure(figsize=(10,4))
plt.axhline(exact_defint.evalf(),color='k',linestyle='--',label='Exact (sympy)')
plt.plot(Ns,areaPdf,'s-',label='From pdfs')
plt.plot(Ns,areaCdf,'s-',label='From cdfs')

plt.gca().set(xlabel='Number of points',ylabel='Area',xscale='log')
plt.legend()
plt.show()

# In [ ]

