import numpy as np
import pandas as pd

# Создайте объект Series с данными из списка, состоящего из названий семи цветов радуги
colors = ["red", "orange", "green", "blue", "indigo", "violet"]
seven_colors_rainbow = pd.Series(data=colors)
print(seven_colors_rainbow)
# Создайте объект Series с данными из списка, состоящего из 6 случайных целых чисел от 3 до 7 включительно  и
# индексами из списка, состоящего из букв a, b, c, d, e, f
np_arr = np.random.randint(3, 8, 6)
indices = ["a", "b", "c", "d", "e", "f"]
num_random = pd.Series(np_arr, indices)
print(num_random)