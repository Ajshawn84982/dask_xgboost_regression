# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:32:30 2024

@author: localadmin
"""

import numpy as np
import pandas as pd
num_columns=10

# Create a sample NumPy array
al=np.zeros([1000000, num_columns+2])
array = np.random.rand(1000000, num_columns)  # 100 rows, 5 columns
result=np.sum(array,axis=1)+0.5*np.sum(array*array,axis=1)
result1=2*np.sum(array,axis=1)+1.5*np.sum(array*array,axis=1)
al[:,num_columns]=result
al[:,num_columns+1]=result1

# Convert the NumPy array to a Pandas DataFrame
df = pd.DataFrame(array, columns = [f'col{i}' for i in range(1, num_columns + 1)])

# Save the DataFrame to a CSV file
df.to_csv('array.csv', index=False)


print("NumPy array saved to array.csv using Pandas")