# -*- coding: utf-8 -*-
# Auto-generated from 'stats_whatRdata_types.ipynb' on 2025-08-08T15:22:56
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master statistics and machine learning: intuition, math, code
# ##### COURSE URL: udemy.com/course/statsml_x/?couponCode=202304 
# ## SECTION: What are (is?) data?
# ### VIDEO: Code: Representing types of data on computers
# #### TEACHER: Mike X Cohen, sincxpress.com

# In [ ]
## create variables of different types (classes)

# data numerical (here as a list)
numdata = [ 1, 7, 17, 1717 ]

# character / string
chardata = 'xyz'

# double-quotes also fine
strdata = "x"

# boolean (aka logical)
logitdata = True # notice capitalization!

# a list can be used like a MATLAB cell
listdata = [ [3, 4, 34] , 'hello' , 4 ]

# dict (kindof similar to MATLAB structure)
dictdata = dict()
dictdata['name'] = 'Mike'
dictdata['age'] = 25
dictdata['occupation'] = 'Nerdoscientist'

# In [ ]
# let's see what the workspace looks like
%whos

# In [ ]
# clear the Python workspace
%reset -sf

# In [ ]

