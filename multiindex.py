import numpy as np
import pandas as pd

mult_ind = pd.MultiIndex(levels=[['A', 'B', 'C'], ['left', 'right']],
                         codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
df = pd.DataFrame(np.random.randn(6, 3), mult_ind, ['X', 'Y', 'Z'])
print(df)
print("_______________________")
# Выберите значение C-left из столбца X
print(df.loc["C"].loc["left"]["X"])
print("_______________________")
# Создайте названия. Для внешних индексов - 'Points', для внутренних - 'Sides'
names = df.index.names = ["Points", "Sides"]
print(df)
print("_______________________")
# Выберите данные всех правых сторон столбца Y
data = df.xs("right", level="Sides")["Y"]
print(data)
