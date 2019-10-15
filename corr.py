import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datapagi.csv', sep=',')
df2 = pd.read_csv('datamalamtanpa2019.csv', sep=',')

data_array = df.values
X = data_array[:, 1:40]

corr = df2.corr()
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(corr, ax=ax)
plt.show()
corr.to_csv('korelasi.csv', sep=',', index=False)
