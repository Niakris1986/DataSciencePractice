import numpy as np

# Создайте массив , состоящий из 10 случайных целых чисел из диапазона от 1 до 20, затем создайте массив состоящий
# из 10 элементов, начиная с 2. Выведите значения этих массивов, а также их сумму, при помощи функции print()
arr1 = np.random.randint(1, 21, 10)
print(arr1)
arr2 = np.arange(2, 12)
print(arr2)
print(arr1 + arr2)
print("---------------------------")
# Создайте массив , состоящий из 7 случайных целых чисел из диапазона от 0 до 9, затем создайте новый массив состоящий
# из 7 элементов первого массива, увеличенных в 3 раза. Выведите значения этих массивов при помощи функции print()
arr1 = np.random.randint(0, 10, 7)
arr2 = arr1 * 3
print(arr1)
print(arr2)
print("---------------------------")
# Создайте массив состоящий из квадратных корней 20 элементов по порядку, начиная с 7. Выведите значение массива
# при помощи функции print()
arr1 = np.arange(7, 27)
arr1 = np.sqrt(arr1)
print(arr1)