import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from subprocess import call
#from sklearn import tree

df1 = pd.read_csv('datamalamedittanpa2019.csv', sep=',')
df2 = pd.read_csv('datapagiedittanpa2019.csv', sep=',')

data_array1 = df1.values
data_array2 = df2.values

panjang = len(data_array1)+len(data_array2)
ujan = data_array1[:,26].sum()+data_array2[:,25].sum()

print(panjang)
print(ujan)

labels = 'Hujan', 'Tidak Hujan'
sizes = [ujan, panjang]
explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
