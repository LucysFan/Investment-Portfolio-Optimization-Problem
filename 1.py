import pandas as pd
import numpy as np
import scipy.sparse as sp


data = pd.read_csv('preprocessed.csv')
sts = np.array(data.columns.values)
data = np.array(data.iloc[:])

ti = data[0] == data[0].max()
if data[1, ti] < 0.2:
    print(f'Вкладываем всё в {sts[ti]}')
    print(f'Ожидаемая доходность: {data[0, ti]}')
    print(f'Уровень риска: {data[1, ti]}')
else:
    m = int(input('Введите степень дискретизации (целое больше 0)'))
    diags = []
    for i in range(data.shape[1] - 2):
        for j in range(m + 1):
            diags.append((data[0, i] - data[0, -1] - (data[0, -2] - data[0, -1]) * (data[1, i] ** 2 - data[1, -1] ** 2)
                          / (data[1, -2] ** 2 - data[1, -1] ** 2)) / 2 ** j)
    qubo = sp.dia_matrix((diags, 0), shape=[(data.shape[1] - 2) * (m + 1)] * 2)
    print(qubo)
