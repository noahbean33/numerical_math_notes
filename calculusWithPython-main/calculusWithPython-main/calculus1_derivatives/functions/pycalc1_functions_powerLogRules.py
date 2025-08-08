# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_functions_powerLogRules.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Functions
# ### LECTURE: CodeChallenge: Power and log rules
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

# nicer latex displays
from IPython.display import display,Math

# In [ ]

# %% [markdown]
# # Exercise 1: Adding powers

# In [ ]
# create a few symbolic variables
from sympy.abc import x,a,b

# the terms individually
xa = x**a
xb = x**b

# and their product
product = xa*xb

# show the results
display( product )
display( sym.simplify(product) )

# In [ ]
# substituting for specific numbers
n1 = 3.4
n2 = 7.3

display(Math( 'x^{%s} \\times x^{%s} = %s' %(n1,n2,sym.latex(xa.subs(a,n1) * xb.subs(b,n2))) ))
print('')
display(Math( 'x^{%s+%s} = %s' %(n1,n2,sym.latex(x**(n1+n2))) ))

# In [ ]

# %% [markdown]
# # Exercise 2: Subtracting powers

# In [ ]
# and their quotient
quotient = xa / xb

# show the results
display( quotient )
display( sym.simplify(quotient) )

# In [ ]
# substituting for specific integers
n = 4

display(Math( '\\frac{x^{%s}}{x^{%s}} = %s' %(n,n,sym.latex((x**n)/(x**n)) )))
print('')

display(Math( 'x^{%s-%s} = %s' %(n,n,sym.latex(x**(n-n)) )))

# In [ ]
x**0

# In [ ]

# %% [markdown]
# # Exercise 3: Powers upon powers

# In [ ]
# the terms individually
xa = x**a
xab = xa**b

# show the results
display( xa**b )
display( xab ) # also try sym.simplify(xab) and sym.powsimp(xab)

# In [ ]
# substituting for specific integers
n1 = 3
n2 = 7

display(Math( '(x^{%s})^{%s} = %s' %(n1,n2,sym.latex((x**n1)**n2)) ))
print('')

display(Math( 'x^{(%s\\times %s)} = %s' %(n1,n2,sym.latex(x**(n1*n2)) )))

# In [ ]
# substituting for specific non-integers
y  = -2
n1 = 4.1
n2 = -.3

display(Math('(%x^{%s})^{%s} = %s' %(y,n1,n2,(y**n1)**n2) ))

print(' ')
display(Math('%x^{(%s\\times %s)} = %s' %(y,n1,n2,y**(n1*n2)) ))

# In [ ]
# sympy will simplify only if the statement is true in general.
y,c,d = sym.symbols('y,c,d',positive=True)

# the terms individually
yc = y**c
ycd = yc**d

# show the results
display( yc**d )
display( sym.powsimp(ycd) )

# In [ ]

# %% [markdown]
# # Exercise 4: the power of log

# In [ ]
a = np.random.rand()
b = np.random.rand()

np.log(a*b) - (np.log(a)+np.log(b))

# In [ ]
a = np.random.rand(100)
b = np.random.rand(100)

ans = np.log(a*b) - (np.log(a)+np.log(b))

plt.plot(ans,'s')
plt.title('$\\ln(ab) - [\\ln(a)+\\ln(b)]$')
plt.show()

# In [ ]
a = np.random.rand(100)
b = np.random.rand(100)

ans = np.log(a**b) - b*np.log(a)

plt.plot(ans,'s')
plt.title('$\\ln(a^b) - b\\ln(a)$')
plt.show()

# In [ ]
a = np.random.rand(100)
b = np.random.rand(100)

ans = np.log(a/b) - (np.log(a)-np.log(b)) # try reversing

plt.plot(ans,'s')
plt.title('$\\ln\\left(\\frac{a}{b}\\right) - [\\ln(a)-\\ln(b)]$')
plt.show()

# In [ ]

