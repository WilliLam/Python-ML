import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./winequality-red.csv', sep = ';')
plt.scatter(df['alcohol'],df['quality'])
plt.xlabel('alcohol content')
plt.ylabel('score')
plt.show()
print(df.describe())
