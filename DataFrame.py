import numpy as np
import pandas as pd

# Создайте DataFrame, состоящую из 5 столбцов и 7 строк, в которой будут целые значения возраста от 1 до 100
# включительно. Для столбцов используйте названия стран, для индексов строк - буквы латинского алфавита от a до g
my_df = pd.DataFrame(np.random.randint(1, 101, 35).reshape(7, 5), ["a", "b", "c", "d", "e", "f", "g"],
                     ["France", "Spain", "Italy", "Germany", "Great Britain"])
print(my_df)