import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('tespairplot.csv', sep=',')
df2 = pd.read_csv('datamalamtanpa2019.csv', sep=',')

#g = sns.pairplot(iris, vars=["sepal_width", "sepal_length"])

sns.pairplot(df,hue='ww', markers = [".", "."], height=1)
plt.show()
