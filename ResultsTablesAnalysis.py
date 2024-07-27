import pandas as pd
import numpy as np

df = pd.read_csv('DataAllMethods.csv', index_col=0)

dfmgType = df[df['microGridType']=='mg1']
boxplot = dfmgType.boxplot(column='ObjFunc', by='OptMethod')
print(type(boxplot))