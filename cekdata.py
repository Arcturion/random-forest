import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('siappagi.csv', sep=',')
data_array = data.values

plt.scatter(data_array[:,0], data_array[:,4], s=data_array[:,6])
plt.show()
