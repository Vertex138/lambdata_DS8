"""
lambdata - a collection of datascience helper functions
"""

import pandas as pd
import numpy as np

# Sample Datasets
ONES = pd.DataFrame(np.ones(10))
ZEROS = pd.DataFrame(np.zeros(50))

# Sample functions
def increment(x):
    return(x+1)
