# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_techniques_CCpartialfractions.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integration techniques
# ### LECTURE: CodeChallenge: Partial fractions algorithm
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import display,Math

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# In [ ]
x,A,B = sym.symbols('x,A,B')

# function
fx = (5*x+3) / (2*x**2 - 4*x - 6)

# quickie-plot
sym.plot(fx,(x,-5,5),ylim=[-20,20])
plt.show()

display(Math('f(x) = %s' %sym.latex(fx)))
print('')
display(Math('\int f(x) \,dx = ?'))

# In [ ]
# Step 1: separate numerator and denominator
numerator, denominator = fx.as_numer_denom()

# and print
display(Math('\\text{The numerator is } %s' %sym.latex(numerator)))
display(Math('\\text{The denominator is } %s' %sym.latex(denominator)))

# In [ ]
# Step 2: factor the denominator
den_factors = sym.factor(denominator)

# print them out
for i,fact in enumerate(den_factors.args):
  display(Math('\\text{Simple factors } %g: \; %s' %(i+1,sym.latex(fact))))
  print('')

# In [ ]
# Step 3: create simple fractions
simple_fract_1 = A / (den_factors.args[0]*den_factors.args[1])
simple_fract_2 = B / den_factors.args[2]

display(Math('\\text{Factor 1:} \; %s' %sym.latex(simple_fract_1)))
print('')
display(Math('\\text{Factor 2:} \; %s' %sym.latex(simple_fract_2)))

# In [ ]
# Step 4: solve for A and B
expression = sym.Eq(numerator , simple_fract_1*sym.prod(den_factors.args) + simple_fract_2*sym.prod(den_factors.args) )
solutionsAB = sym.solve(expression,(A,B))

solutionsAB

# In [ ]
# Step 5: integrate separately
defint1 = sym.integrate( simple_fract_1.subs(A,solutionsAB[A]) )
defint2 = sym.integrate( simple_fract_2.subs(B,solutionsAB[B]) )

display(Math('\int %s \,dx = %s+C' %(sym.latex(simple_fract_1.subs(A,solutionsAB[A])),sym.latex(defint1))))
print('')
display(Math('\int %s \,dx = %s+C' %(sym.latex(simple_fract_2.subs(B,solutionsAB[B])),sym.latex(defint2))))

# In [ ]
# Step 6: sum the parts
mysolution = defint1 + defint2
mysolution

display(Math('\\int %s \, dx = %s+C' %(sym.latex(fx),sym.latex(mysolution))))

# In [ ]
# Step 7: compare against sympy integration from the original function
symsolution = sym.integrate(fx)
symsolution

display(Math('\\int %s \, dx = %s+C' %(sym.latex(fx),sym.latex(symsolution))))

# In [ ]

