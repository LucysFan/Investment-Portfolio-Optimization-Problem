import pandas as pd
import numpy as np
import scipy.sparse as sp
import pyqiopt as pq


def restrain(a, b):
    diags = []
    for i in range(len(a)):
        for j in range(m):
            diags.append(((a[i] ** 2 - 2 * a[i] * b) / 2 ** j))
    q0 = sp.dia_matrix((diags, 0), shape=[len(a) * m] * 2)
    q1 = np.zeros((len(a) * m, len(a) * m))
    for i in range(len(a) * m):
        for j in range(len(a) * m):
            q1[i, j] = a[i // m] * a[j // m] / 2 ** (i % m) / 2 ** (j % m)
    return q0.tocoo() + sp.coo_matrix(q1)


m = 6
data = pd.read_csv('preprocessed.csv')
sts = np.array(data.columns.values)
data = np.array(data.iloc[:])

ti = data[0] == data[0].max()
if data[1, ti] < 0.2:
    print(f'Вкладываем всё в {sts[ti]}')
    print(f'Ожидаемая доходность: {data[0, ti]}')
    print(f'Уровень риска: {data[1, ti]}')
else:

    diags = [data[0, i] for i in range(data.shape[1])]
    for i in range(data.shape[1]):
        for j in range(m):
            diags.append((data[0, i] / 2 ** j))

    qubo = -sp.dia_matrix((diags, 0), shape=[data.shape[1] * m] * 2).tocoo() + restrain(data[0], 1) + restrain(data[1] ** 2, 0.04)
    vec = pq.solve(qubo.tocoo(), no).vector
    for i in range(data.shape[1]):
        a = 0
        for j in range(m):
            a += vec[i * m + j] / 2 ** j
        if a > 0:
            print(f'Покупаем {a * 100}% акций {sts[i]}. Прибыль {data[0, i] * a * 100}%, риск {data[1, i] * a}')
