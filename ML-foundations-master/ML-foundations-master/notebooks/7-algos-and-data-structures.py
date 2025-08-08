# -*- coding: utf-8 -*-
# Auto-generated from '7-algos-and-data-structures.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/7-algos-and-data-structures.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Algorithms & Data Structures

# %% [markdown]
# This class, *Algorithms & Data Structures*, introduces the most important computer science topics for machine learning, enabling you to design and deploy computationally efficient data models.
# 
# Through the measured exposition of theory paired with interactive examples, you’ll develop a working understanding of all of the essential data structures across the list, dictionary, tree, and graph families. You’ll also learn the key algorithms for working with these structures, including those for searching, sorting, hashing, and traversing data.
# 
# The content covered in this class is itself foundational for the *Optimization* class of the *Machine Learning Foundations* series.

# %% [markdown]
# Over the course of studying this topic, you'll:
# 
# * Use “Big O” notation to characterize the time efficiency and space efficiency of a given algorithm, enabling you to select or devise the most sensible approach for tackling a particular machine learning problem with the hardware resources available to you.
# * Get acquainted with the entire range of the most widely-used Python data structures, including list-, dictionary-, tree-, and graph-based structures.
# * Develop an understanding of all of the essential algorithms for working with data, including those for searching, sorting, hashing, and traversing.

# %% [markdown]
# **Note that this Jupyter notebook is not intended to stand alone. It is the companion code to a lecture or to videos from Jon Krohn's [Machine Learning Foundations](https://github.com/jonkrohn/ML-foundations) series, which offer detail on the following:**
# 
# *Segment 1: Introduction to Data Structures and Algorithms*
# * A Brief History of Data
# * A Brief History of Algorithms
# * “Big O” Notation for Time and Space Complexity
# 
# *Segment 2: Lists and Dictionaries*
# * List-Based Data Structures: Arrays, Linked Lists, Stacks, Queues, and Deques
# * Searching and Sorting: Binary, Bubble, Merge, and Quick
# * Set-Based Data Structures: Maps and Dictionaries
# * Hashing: Hash Tables, Load Factors, and Hash Maps
# 
# *Segment 3: Trees and Graphs*
# * Trees: Decision Trees, Random Forests, and Gradient-Boosting (XGBoost)
# * Graphs: Terminology, Directed Acyclic Graphs (DAGs)
# * Resources for Further Study of Data Structures & Algorithms

# %% [markdown]
# # Segment 1: Intro to Data Structures & Algorithms

# %% [markdown]
# ### "Big O" Notation

# In [1]
import numpy as np
import tensorflow as tf
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time # for timing processes
import random # for generating random values

# %% [markdown]
# #### Constant Time

# In [2]
def take_first(my_list):
    return my_list[0]

# In [3]
short_list = [13, 25, 42]

# In [4]
tic = time.process_time() # if using a version of Python older than 3.3, you can use time.clock() here
first = take_first(short_list)
toc = time.process_time()

# In [5]
first

# In [6]
toc-tic

# In [7]
long_list = [42] * 10**8 # one hundred million items

# In [8]
len(long_list)

# In [9]
tic = time.process_time()
first = take_first(long_list)
toc = time.process_time()

# In [10]
toc-tic

# In [11]
list_lengths = [10**l for l in range(1, 8)]
list_lengths

# In [12]
constant_times = []

for l in list_lengths:
    lst = [42]*l

    tic = time.process_time()
    x = take_first(lst)
    toc = time.process_time()

    constant_times.append(toc-tic)

# In [13]
constant_df = pd.DataFrame(list(zip(list_lengths, constant_times)), columns=['n', 'time'])
constant_df

# %% [markdown]
# #### Linear Time

# In [14]
def find_max(my_list):
    max_value = my_list[0]
    for i in range(len(my_list)):
        if my_list[i] > max_value:
            max_value = my_list[i]
    return max_value

# In [15]
tic = time.process_time()
largest = find_max(short_list)
toc = time.process_time()

# In [16]
toc-tic

# In [17]
largest

# In [18]
tic = time.process_time()
largest = find_max(long_list)
toc = time.process_time()

# In [19]
toc-tic

# In [20]
largest

# In [21]
linear_times = []

for l in list_lengths:
    lst = [42]*l

    tic = time.process_time()
    x = find_max(lst)
    toc = time.process_time()

    linear_times.append(toc-tic)

# In [22]
linear_df = pd.DataFrame(list(zip(list_lengths, linear_times)), columns=['n', 'time'])
linear_df

# In [23]
_ = sns.lmplot(x='n', y='time', data=linear_df, ci=None) # linear model (regression) plot

# %% [markdown]
# #### Polynomial Time

# In [24]
def element_multiplier(my_list):
    for i in range(len(my_list)):
        for j in range(len(my_list)):
            x = my_list[i] * my_list[j]

# In [25]
list_lengths

# In [26]
list_lengths[:4] # compute time gets annoyingly long from list_lengths[5] onward

# In [27]
granular_list_lengths = list_lengths[:4] + [50, 500, 5000]
granular_list_lengths.sort()
granular_list_lengths

# In [28]
poly_times = []

for l in granular_list_lengths:
    lst = [42]*l

    tic = time.process_time()
    x = element_multiplier(lst)
    toc = time.process_time()

    poly_times.append(toc-tic)

# In [29]
poly_df = pd.DataFrame(list(zip(granular_list_lengths, poly_times)), columns=['n', 'time'])
poly_df

# In [30]
_ = sns.lmplot(x='n', y='time', data=poly_df, ci=None, order=2)

# %% [markdown]
# **In Big O notation**:
# 
# * Constant: O(1)
# * Linear: O($n$)
# * Polynomial: O($n^2$)

# %% [markdown]
# **Other common runtimes**:
# 
# * Logarithmic: O(log $n$)
# * Exponential: O($2^n$)
# * O($n$ log $n$)
# * O($n!$)

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ## Segment 2: Lists and Dictionaries

# %% [markdown]
# ### List-Based Data Structures

# %% [markdown]
# #### Lists

# In [31]
t = [25, 2, 5]
t

# %% [markdown]
# Lists in Python are **extensible** by default:

# In [32]
t.append(26)
t

# In [33]
t[2]

# %% [markdown]
# Cannot apply mathematical operators, e.g., try uncommenting:

# In [34]
# t/2

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# #### Arrays

# In [35]
x = np.array([25, 2, 5])
x

# In [36]
x[2]

# In [37]
x/2.

# In [38]
y = torch.tensor([[25, 2, 5], [26, 1, 4]])
y

# In [39]
y[0, 2]

# In [40]
y/2.

# In [41]
z = tf.Variable([[[0, 1], [2, 3]], [[25, 26], [7, 9]]], dtype=tf.float16)
z

# In [42]
z[1, 0, 0]

# In [43]
z/2.

# %% [markdown]
# Arrays in Python are also typically extensible by default.
# 
# Note that it's very efficient (O(1)) to append items to end of list...

# In [44]
np.append(x, 26)

# %% [markdown]
# ...but time inefficient to add at start or middle of long list as later items will need to be shifted to later in list. Worst case is insert at beginning of list, which has O($n$) complexity.
# 
# Note that as well as appending, accessing an array element by index or finding the array's length are O(1).

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# #### Stacks

# %% [markdown]
# Can be implemented in Python with lists:

# In [45]
s = []
s

# In [46]
s.append('five_of_diamonds') # push
s

# In [47]
s.append('queen_of_hearts')
s

# In [48]
s.append('ace_of_spades')
s

# In [49]
s.pop()

# In [50]
s

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ### Searching and Sorting

# %% [markdown]
# Topics in this section:
# * Binary search
# * Bubble sort
# * Merge sort
# * Quick sort

# %% [markdown]
# #### Binary Search

# %% [markdown]
# Let's say we have an array, `b`:

# In [51]
b = np.array([25, 2, 5, 14, 22, 11, 96, 1, 101])
b

# %% [markdown]
# Under normal circumstances, the only way to search for a value (e.g., `25`) is to search element by element. In the worse-case, this has O($n$) time complexity (e.g., if we were to search `b` for `25` from its end).
# 
# If `b` already happens to be sorted, however...

# In [52]
b.sort() # FYI: uses quick sort
b

# %% [markdown]
# ...we can use the **binary search** algorithm, which has O(log $n$) time complexity:
# 
# * Start at the midpoint (`b` is nine elements long, so midpoint is 5th: `14`).
# * Since `25` is larger than `14`, we need only search the top half of `b`...
# * So we find the midpoint between the 5th element and the 9th, which is the 7th, and which happens to be `25`.
# * Done! And in only two steps instead of seven (if we were to search elementwise from first onward).
# * If we were searching for `22`, then our next search would be halfway between 5th and 7th elements, which is 6th element and voilà!
# * With our nine-element array, worst case is four steps (for `1` or `101`) to find any value (or determine value *isn't* in array).
# 
# (Note: called *binary* because at each step we either search upward or downward.)
# 
# Doubling $n$ results in only one extra step in worst case:
# * One step with array of length 1($= 2^0$)
# * Two steps with length 2($= 2^1$) up to length 3
# * Three steps with $n = 4 (= 2^2$) up to $n = 7$
# * Four steps with $n = 8 (= 2^3$) up to $n = 15$
# * Five steps with $n = 16 (= 2^4$) up to $n = 31$
# 
# Hence O(log $n$) where, as is typical in CS, we use $\text{log}_2$ (because of binary bits and frequently doubling/halving quantities). E.g.:

# In [53]
np.log2(4) + 1

# In [54]
np.log2(8) + 1

# In [55]
np.log2(16) + 1

# %% [markdown]
# Recall that we retain only the dominant term, so O(log $n$ + 1) becomes O(log $n$).

# %% [markdown]
# Here's an implementation of binary search, for your reference, that *iterates* over binary search steps with a `while` loop:

# In [56]
def binary_search(array, value):

    low = 0
    high = len(array)-1

    while low <= high:
        midpoint = (low + high) // 2 # rounds down to nearest integer after division
        if value < array[midpoint]:
            high = midpoint-1
        elif value > array[midpoint]:
            low = midpoint+1
        else:
            return midpoint
    return -1

# In [57]
binary_search(b, 25)

# In [58]
binary_search(b, 255)

# %% [markdown]
# Alternatively, you could implement binary search with *recursion* instead of iteration:

# In [59]
def recursive_binary_search(array, value, low, high):

    if low <= high:

        midpoint = (low + high) // 2

        if array[midpoint] == value:
            return midpoint
        elif array[midpoint] > value:
            return recursive_binary_search(array, value, low, midpoint-1)
        else:
            return recursive_binary_search(array, value, midpoint+1, high)
    else:
        return -1

# In [60]
recursive_binary_search(b, 25, 0, len(b)-1)

# In [61]
recursive_binary_search(b, 255, 0, len(b)-1)

# %% [markdown]
# #### Bubble Sort

# %% [markdown]
# Binary search required a sorted list. To sort a list (e.g., from smallest to largest value), the most naïve (and computationally complex) approach would be to compare every given element with all other elements.
# 
# A common naïve implementation is the **bubble sort**, which allows the largest values to gradually "bubble up" toward the "top" (typically the end) of the array:

# In [62]
def bubble_sort(array):

    n = len(array)

    for i in range(n-1):
        for j in range(0, n-1): # could be n-i-1 as, e.g., top value is guaranteed to have bubbled up in 1st iteration
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]

    return array

# In [63]
b = np.array([25, 2, 5, 14, 22, 11, 96, 1, 101])
b

# In [64]
bubble_sort(b)

# %% [markdown]
# * During each iteration, we make $n-1$ comparisons
# * A total of $n-1$ iterations need to be made
# 
# $$ (n-1)(n-1) = n^2 -2n + 1 $$
# 
# Since we drop all terms but the dominant one, this leaves us with polynomial O($n^2$) time complexity for the worst case and the average case. (For more clever implementations, best case is O($n$) because array would already be sorted.)
# 
# In contrast, as is typical in algos, there is a time- vs memory-complexity trade-off: Memory complexity is constant, O(1).

# In [65]
granular_list_lengths

# In [66]
max(granular_list_lengths)

# In [67]
random.sample(range(0, max(granular_list_lengths)), 5) # samples 5 integers w/o replacement from uniform distribution

# In [68]
bubble_times = []

for l in granular_list_lengths[0:7]:
    lst = random.sample(range(0, max(granular_list_lengths)), l)

    tic = time.process_time()
    x = bubble_sort(lst)
    toc = time.process_time()

    bubble_times.append(toc-tic)

# In [69]
bubble_df = pd.DataFrame(list(zip(granular_list_lengths, bubble_times)), columns=['n', 'time'])
bubble_df

# In [70]
_ = sns.lmplot(x='n', y='time', data=bubble_df, ci=None, order=2)

# %% [markdown]
# #### Merge Sort

# %% [markdown]
# * General idea is to "divide and conquer"; specifically:
#     1. Halve the array into smaller arrays
#     2. Sort the smaller arrays
#     3. Merge them back into full array
# * The above steps are carried out recursively so ultimately sort arrays of max length 2, then merge back up to full length.

# In [71]
def merge_sort(my_list): # using list instead of array so we can .pop() (np arrays don't pop with built-in method)

    if len(my_list) > 1: # if length is 1, no need to sort (at deepest recursion, some will have len 1, others len 2)

        # 1. Halve:
        midpoint = len(my_list) // 2 # Note: This is ~O(log n) behavior
        left_half = my_list[:midpoint]
        right_half = my_list[midpoint:]

        # 2. Sort (recursively):
        left_half = merge_sort(left_half)
        right_half = merge_sort(right_half)

        my_list = []

        # 3. Merge:
        while len(left_half)>0 and len(right_half)>0: # Note: This inner loop exhibits ~O(n) behavior

            if left_half[0] < right_half[0]:
                my_list.append(left_half.pop(0)) # pop first element

            else: # 1st element of right half < 1st element of left
                my_list.append(right_half.pop(0))

        # If any elements remain from either half, they must be the largest value:
        for i in left_half:
            my_list.append(i)
        for i in right_half:
            my_list.append(i)

    return my_list

# In [72]
m = [25, 2, 5, 14, 22, 11, 96, 1, 101]

# In [73]
merge_sort(m)

# %% [markdown]
# Because of the O($n$) loop performing comparisons inside of the function halving list sizes (which we'll need to call O(log $n$) times), merge sort time complexity is the product, i.e., O($n$ log $n$).

# In [74]
granular_list_lengths

# In [75]
ext_granular_lengths = granular_list_lengths + [50000, 100000] # extended
ext_granular_lengths

# In [76]
merge_times = []

for l in ext_granular_lengths:
    lst = random.sample(range(0, max(ext_granular_lengths)), l)

    tic = time.process_time()
    x = merge_sort(lst)
    toc = time.process_time()

    merge_times.append(toc-tic)

# In [77]
merge_df = pd.DataFrame(list(zip(ext_granular_lengths, merge_times)), columns=['n', 'time'])
merge_df

# %% [markdown]
# On local laptop, $n=10000$ took ~0.05s with merge sort compared to ~10.5s with bubble sort: a ~200x speedup.

# In [78]
_ = sns.lmplot(x='n', y='time', data=merge_df, ci=None, order=2)

# %% [markdown]
# In contrast, while memory complexity for bubble sort was O(1) because we were sorting "in place" (not storing any values in memory), memory complexity is O($n$) for merge sort.
# 
# (We need at least two array copies (2$n$) for the most space-efficient merge sort implementations, but we drop constants so this becomes O($n$). Our recursive merge sort implementation would require more as it stores several copies in memory; one more copy of size $n$ would be used for every time the array length doubles (log $n$) resulting in O($n$ log $n$) memory complexity.)

# %% [markdown]
# #### Quick Sort

# %% [markdown]
# Under many common conditions, the aptly-named **quick sort** is at least as time-efficient as merge sort while also being more space-efficient.
# 
# Like merge sort, quick sort uses the "divide and conquer" principle:
# 1. Pick a *pivot* point (often the final element, though it could be any element, e.g., first, random, median)
# 2. Move pivot within array such that all elements above pivot are larger (though randomly ordered) and all elements below pivot are smaller (though likewise randomly ordered)
#     * Movement occurs element-by-element (and "in place", thus retaining O(1) memory complexity)
# 3. Recursively, perform quick sort both below and above pivot

# In [79]
def quick_sort(array):
    _quick_sort(array, 0, len(array)-1) # to start, low is index of first element in array; high is final element
    return array

# In [80]
def _quick_sort(array, low, high):

    if low < high: # when low reaches high, we know all elements in array must be in order

        # Result of 2.: Element at "partitioning index" has been pivoted to its correct position
        p_index = partition(array, low, high)

        # 3. Recursively sort the elements below the partitioning index...
        _quick_sort(array, low, p_index-1)
        # ...and, separately, above the partitioning index:
        _quick_sort(array, p_index+1, high)

    return array

# In [81]
def partition(array, low, high):

    # 1. Use the final element as the pivot:
    pivot = array[high]

    i = (low-1) # index of element on small side
    for j in range(low, high): # like a bubble swap, we compare pivot to all elements between low and high
        if array[j] < pivot: # if value < pivot, it's correctly on left so increment the smaller element's index...
            i = i+1
            array[i], array[j] = array[j], array[i]
    array[i+1], array[high] = array[high], array[i+1] # ...otherwise, swap greater value to right of pivot (now what we want)
    return(i+1) # return index of correctly positioned pivot

# In [82]
b = np.array([25, 2, 5, 14, 22, 11, 96, 1, 101])

# In [83]
quick_sort(b)

# In [84]
list_lengths

# In [85]
list_lengths[:6]

# In [86]
quick_times = []

for l in list_lengths[:6]:
    lst = random.sample(range(0, max(list_lengths[:6])), l)

    tic = time.process_time()
    x = quick_sort(lst)
    toc = time.process_time()

    quick_times.append(toc-tic)

# In [87]
quick_df = pd.DataFrame(list(zip(list_lengths[:6], quick_times)), columns=['n', 'time'])
quick_df

# %% [markdown]
# At $n=$ 10k, about a 2x speed-up relative to merge sort (so ~400 relative to bubble sort).
# 
# At $n=$ 100k, about a 5x speed-up relative to merge sort.

# In [88]
_ = sns.lmplot(x='n', y='time', data=quick_df, ci=None, order=2)

# %% [markdown]
# **Average-case time**: The efficiency of quick sort comes from moving lower-valued pivots toward the front (ideally to the midpoint) of the array then "dividing and conquering" on either side of the pivot. In which case (the average case), quick sort behaves like a merge sort (thereby it has O($n$ log $n$) time complexity), but in a manner that requires only a single copy of the array in memory.
# 
# **Worst-case time**: With the pivot starting at the final element, if the array is already sorted, this is the worst case. There will be as many calls to `quick_sort()` as there are elements in the array (i.e., $n$ calls) and quick sort will perform a full round of bubble sorting (as per the `for j in range` loop) during each of those (i.e., $n-1$ comparisons); the product of these resulting in O($n^2$) time complexity. So, don't use quick sort if the array is nearly in order!
# 
# **Best-case time**: If all values in the array are equal, quick sort will scan over them all once to confirm this, in which case it has O($n$) time complexity. The more values that are equal in the array, the closer to O($n$) performance will be.

# %% [markdown]
# Those are the most frequently covered sorting algos. They're relatively straightforward to implement and illustrate time/space complexity trade-offs clearly. There are, however, many more ways to sort, with pros and cons related to time/space complexity as well as implementation difficulty. See the [Big O Cheat Sheet here](https://www.bigocheatsheet.com).

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ### Set-Based Data Structures: Maps/Dictionaries

# In [89]
inventory = {}
inventory

# In [90]
inventory['hammer'] = [17]
inventory

# In [91]
inventory['nail'] = [2552]
inventory['unicorn'] = [2]
inventory['dolphin'] = ['none']

# In [92]
inventory

# In [93]
inventory['dolphin'] = [0]
inventory

# In [94]
inventory['unicorn'].append(3)
inventory

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ### Hashing

# %% [markdown]
# Let's say we have some value:

# In [95]
value = 5551234567 # a phone number from any American '90s sitcom

# %% [markdown]
# A common hash function approach is to use the modulo operator on the last few digits of the value...

# In [96]
split_value = [digit for digit in str(value)]
split_value

# In [97]
final_digits = int(''.join(split_value[-2:])) # final digits typically used b/c they tend to vary more than first ones
final_digits

# In [98]
hash_value = final_digits % 10 # 10 is arbitrary, but would be used consistently across values to be hashed
hash_value

# In [99]
def simple_hash(v):
    split_v = [digit for digit in str(v)]
    final_2 = int(''.join(split_v[-2:]))
    return final_2 % 10

# In [100]
simple_hash(value)

# In [101]
simple_hash(5557654321)

# %% [markdown]
# These hash values (`7` and `1`) could be used in a sequential, small-integer index, i.e., a *hash table*.

# %% [markdown]
# #### Collisions

# %% [markdown]
# Major problem with the `simple_hash()` function:
# * The hash table has at most ten indices
# * Ergo, many input values will result in **collisions**, e.g.:

# In [102]
simple_hash(555)

# In [103]
simple_hash(125)

# %% [markdown]
# Three common ways to resolve collisions:
# 1. Change the modulus denominator (e.g., `10` --> `11`); this adds procedural (and thus time) complexity to hash algo
# 2. Change the hash function entirely; ditto w.r.t. procedural complexity
# 3. Store a list (or similar) at the index, e.g.:

# In [104]
hash_table = {}

# In [105]
hash_table[simple_hash(555)] = [555]
hash_table

# In [106]
hash_table[simple_hash(125)].append(125)
hash_table

# %% [markdown]
# Such a list is called a **bucket**.
# 
# Worst case:
# * All of the values hash to the same hash value (e.g., `5`)
# * Thus, all of the values are stored in a single bucket
# * Searching through the bucket has linear O($n$) time complexity
# 
# Alternatively, we can increase memory complexity instead of time complexity:
# * Use very large modulus denominator
# * Reduces probability of collisions
# * If use denominator of `1e9`, we have a hash table with a billion buckets
# 
# Could also have a second hash function *inside* of the bucket (e.g., if we know we'll have a few very large buckets).
# 
# There is no "perfect hash". It depends on the values you're working with. There are many options to consider with various trade-offs.

# %% [markdown]
# **Load Factor**

# %% [markdown]
# Metric that guides hashing decisions:
# $$ \text{load factor} = \frac{n_\text{values}}{n_\text{buckets}} $$

# In [107]
10/1e9

# %% [markdown]
# If we have ten values to store, but a billion buckets...
# $$ \text{load factor} = \frac{10}{10^9} = 10^{-8}$$

# %% [markdown]
# ...we are probably using much more memory than we need to. This is the case whenever load factor $\approx 0$.

# %% [markdown]
# As a general rule of thumb, the "Goldilocks" sweet spot would be to maintain a load factor between 0.6 and 0.75, providing a balance between memory and time complexity.
# 
# Below 0.6, you're potentially wasting memory with too many empty buckets.
# 
# Above 0.75, you risk having too many collisions, which degrades performance.

# %% [markdown]
# #### Hash Maps

# %% [markdown]
# In all of the above examples, we were hashing "values", but these "values" could in fact be the *keys* of a key-value pair, allowing us to have a **hash map**.

# %% [markdown]
# Let's say `Jane Dough` has receipt number `5551234567`, where we're using the receipt numbers as keys to look up customers.
# 
# We can add this as an entry in a hash table for quick lookup later (once we have many more receipts...):

# In [108]
hash_map = {}

# In [109]
hash_map[simple_hash(5551234567)] = [(5551234567, 'Jane Dough')]
hash_map

# In [110]
hash_map[simple_hash(5551234568)] = [(5551234568, 'Johnny Dochevski')]
hash_map

# In [111]
hash_map[simple_hash(5551234578)].append((5551234578, 'Jon Krohn'))
hash_map

# %% [markdown]
# **FYI**: In Python, dictionaries are hash maps.

# %% [markdown]
# #### String Keys

# %% [markdown]
# If our keys are character strings, we can still make use of hashing by converting the character string into an integer.
# 
# For example, we could use the [ASCII table](http://www.asciitable.com) to convert `Jon` to `112157156`.

# %% [markdown]
# **Exercises**
# 
# 1. Use the `simple_hash()` function to add a customer named Jean d'Eau, who has receipt number `5551234569`, to `hash_map`.
# 
# 1. You have a hash table with a million buckets and two million values to store in the table. What is your load factor? Are collisions likely? If so, how can the probability of collisions be reduced?
# 
# 1. Use the octal standard of the ASCII table to convert the string `Llama` into an integer representation.

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ## Segment 3: Trees and Graphs

# %% [markdown]
# ### Decision Trees

# In [112]
titanic = sns.load_dataset('titanic')

# In [113]
titanic

# In [114]
np.unique(titanic['survived'], return_counts=True)

# In [115]
np.unique(titanic['sex'], return_counts=True)

# In [116]
np.unique(titanic['class'], return_counts=True)

# In [117]
gender = pd.get_dummies(titanic['sex'])
gender

# In [118]
clas = pd.get_dummies(titanic['class'])
clas

# In [119]
y = titanic.survived

# In [120]
X = pd.concat([clas.First, clas.Second, gender.female], axis=1)
X

# In [121]
from sklearn.model_selection import train_test_split

# In [122]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# In [123]
X_train.shape

# In [124]
y_train.shape

# In [125]
X_test.shape

# In [126]
y_test.shape

# In [127]
from sklearn.tree import DecisionTreeClassifier, plot_tree

# In [128]
dt_model = DecisionTreeClassifier()

# In [129]
dt_model.fit(X_train, y_train)

# In [130]
plot_tree(dt_model) # can read about Gini here: en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

# In [131]
rose = np.array([[1, 0, 1]]) # must be matrix format
jack = np.array([[0, 0, 0]])

# In [132]
dt_model.predict(rose)

# In [133]
dt_model.predict_proba(rose)

# In [134]
dt_model.predict(jack)

# In [135]
dt_model.predict_proba(jack)

# In [136]
dt_yhat = dt_model.predict(X_test) # decision trees are prone to overfitting training data

# In [137]
dt_yhat[0:6]

# In [138]
y_test[0:6]

# In [139]
from sklearn.metrics import accuracy_score, roc_auc_score

# In [140]
accuracy_score(y_test, dt_yhat)

# In [141]
roc_auc_score(y_test, dt_yhat)

# %% [markdown]
# **Exercises**:
# 
# 1. Create more features ([here are some ideas for creating them](https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8)) with an eye to improving model fit on the test data.
# 
# 2. Re-train the decision tree on the training data and evaluate its accuracy and ROC AUC on the test data. How does the tree compare with an ordinary-least-squares regression model (from [*Intro to Stats*](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/6-statistics.ipynb)) trained on the same data?

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ### Random Forests

# In [142]
iris = sns.load_dataset('iris')
iris

# In [143]
_ = sns.scatterplot(x='sepal_width', y='petal_length', hue='species', data=iris)

# In [144]
X = iris[['sepal_width', 'petal_length']]

# In [145]
y = iris.species

# In [146]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# In [147]
iris_dt = DecisionTreeClassifier().fit(X_train, y_train)

# In [148]
iris_dt_yhat = iris_dt.predict(X_test)

# In [149]
accuracy_score(iris_dt_yhat, y_test) # ROC AUC is for binary classifier

# In [150]
from sklearn.ensemble import RandomForestClassifier

# In [151]
rf_model = RandomForestClassifier(n_estimators=100) # n decision trees in forest

# In [152]
rf_model.fit(X_train, y_train)

# In [153]
rf_yhat = rf_model.predict(X_test)

# In [154]
accuracy_score(rf_yhat, y_test)

# %% [markdown]
# **Return to slides.**

# %% [markdown]
# ### Gradient-Boosted Trees

# In [155]
import xgboost as xgb

# %% [markdown]
# XGBoost `DMatrix()` method requires numeric inputs, not strings:

# In [156]
y_train[0:6]

# In [157]
mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
y_train_int = y_train.map(mapping)
y_test_int = y_test.map(mapping)

# In [158]
y_train_int[0:6]

# In [159]
D_train = xgb.DMatrix(X_train, label=y_train_int)
D_test = xgb.DMatrix(X_test, label=y_test_int)

# In [160]
param = {
    'eta': 0.1, # learning rate (0.1 to 0.3 are common)
    'max_depth': 2, # I mean, we only have two features...
    'objective': 'multi:softprob',
    'num_class': 3
}
steps = 10

# In [161]
xg_model = xgb.train(param, D_train, steps)

# In [162]
xg_yhats = xg_model.predict(D_test)

# In [163]
xg_yhats[0] # supports >2 classes

# In [164]
xg_yhat = np.asarray([np.argmax(line) for line in xg_yhats])

# In [165]
xg_yhat[0]

# In [166]
accuracy_score(xg_yhat, y_test_int)

# %% [markdown]
# **Exercise**:
# 
# 1. Try training random forest and XGBoost models with additional iris features. Can you attain 100% accuracy on the test data set?
# 2. Explore [CatBoost](https://catboost.ai/en/docs/concepts/python-quickstart) for situations where you have categorical *features* in the model (CatBoost works for both classification and regression problems).

# %% [markdown]
# **Return to slides.**

