import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datapagitanpa2019.csv', sep=',')
#df = df.drop(columns='tanggal')

df = df.loc[:,['press', 'press1', 'press2', 'press3', 'press4', 'press5', 'press6', 'ww']]

#df = pd.melt(df, value_vars = ['suhu', 'suhu1', 'suhu2', 'suhu3', 'suhu4', 'suhu5', 'suhu6'], id_vars='ww')
#df = pd.melt(df, value_vars = ['rh', 'rh1', 'rh2', 'rh3', 'rh4', 'rh5', 'rh6'], id_vars='ww')
#df = pd.melt(df, value_vars = ['dd', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'], id_vars='ww')
df = pd.melt(df, value_vars = ['press', 'press1', 'press2', 'press3', 'press4', 'press5', 'press6'], id_vars='ww')

df = df.rename(columns = {'value' : 'Tekanan (hPa)'})
df = df.rename(columns = {'variable' : '-'})
sns.violinplot(x='-', y='Tekanan (hPa)', hue='ww', split='True', data=df)
plt.title('TEKANAN')
plt.show()
