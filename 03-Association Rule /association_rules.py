import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import apyori

# data = [ 
#     ["Milk", "Egg", "Bread"],
#     ["Milk, Bread"],
#     ["Milk", "Eggs"],
#     ["Eggs", "Banana"],
# ]

# import data
store_data = pd.read_csv("store_data.csv", headre=None)

# Preprocessing
records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i,j]) for j in range(0,20)])


