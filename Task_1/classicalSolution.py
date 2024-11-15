import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# Установка параметров
num_stocks = 10  # Количество акций
num_days = 100   # Количество дней
mean_return = 0.01  # Средняя доходность
std_dev = 0.1  # Стандартное отклонение (риск)

# Генерация случайных доходностей для акций
np.random.seed(42)  # Для воспроизводимости
returns = np.random.normal(loc=mean_return, scale=std_dev, size=(num_days, num_stocks))

# Преобразование доходностей в цены
# Начальная цена для каждой акции
initial_prices = np.random.rand(num_stocks) * 100  # Начальные цены от 0 до 100
prices = np.zeros((num_days, num_stocks))

# Установка начальных цен
prices[0] = initial_prices

# Генерация цен на основе доходностей
for i in range(1, num_days):
    prices[i] = prices[i - 1] * (1 + returns[i])

# Создание DataFrame
price_df = pd.DataFrame(prices, columns=[f'Stock_{i+1}' for i in range(num_stocks)])

# Расчет доходностей
returns_df = price_df.pct_change().dropna()

# Расчет средней доходности и ковариационной матрицы
mean_returns = returns_df.mean()
cov_matrix = returns_df.cov()

# Генерация всех возможных сочетаний акций
all_portfolios = []

for r in range(1, num_stocks + 1):  # от 1 до 10 акций
    for combo in combinations(range(num_stocks), r):
        weights = np.random.random(len(combo))
        weights /= np.sum(weights)  # Нормализация весов
        
        # Создание полного вектора весов для всех акций
        full_weights = np.zeros(num_stocks)
        full_weights[list(combo)] = weights
        
        portfolio_return = np.dot(full_weights, mean_returns)
        portfolio_std_dev = np.sqrt(np.dot(full_weights.T, np.dot(cov_matrix, full_weights)))
        
        all_portfolios.append((portfolio_return, portfolio_std_dev, full_weights))

# Преобразование в DataFrame для удобства
portfolio_df = pd.DataFrame(all_portfolios, columns=['Return', 'Risk', 'Weights'])

# Фильтрация портфелей с риском < 0.2
filtered_portfolios = portfolio_df[portfolio_df['Risk'] < 0.2]

# Проверка на наличие подходящих портфелей
if filtered_portfolios.empty:
    print("Нет портфелей с риском меньше 0.2.")
else:
    # Выбор портфеля с максимальной доходностью среди отфильтрованных
    best_portfolio = filtered_portfolios.loc[filtered_portfolios['Return'].idxmax()]

    # Вывод результатов
    print("Наилучший портфель:")
    print(f"Доходность: {best_portfolio['Return']:.4f}")
    print(f"Риск: {best_portfolio['Risk']:.4f}")
    print(f"Веса акций: {best_portfolio['Weights']}")

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_df['Risk'], portfolio_df['Return'], alpha=0.5)
    plt.scatter(best_portfolio['Risk'], best_portfolio['Return'], color='red', marker='*', s=200)  # Наилучший портфель
    plt.title('Оптимизация портфеля')
    plt.xlabel('Риск (Стандартное отклонение)')
    plt.ylabel('Доходность')
    plt.show()
