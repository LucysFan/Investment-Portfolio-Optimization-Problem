import tensorcircuit as tc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorcircuit.templates.ansatz import QAOA_ansatz_for_Ising
from tensorcircuit.templates.conversions import QUBO_to_Ising
from tensorcircuit.applications.optimization import QUBO_QAOA, QAOA_loss
from tensorcircuit.applications.finance.portfolio import StockData, QUBO_from_portfolio

K = tc.set_backend("tensorflow")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

path = 'utils/data/stocks_1.csv'

data = pd.read_csv(path)
# _values = np.array(data.columns.values)
data = np.array(data.iloc[:])
# n = data.shape[0]
# tdata = data.transpose()
# returns = ((np.roll(tdata, -1) - tdata) / tdata)[:, :-1]
# returns_median = returns.sum(1) / (n - 1)
# returns = returns[returns_median > 0]
# _values = _values[returns_median > 0]
# returns_median = returns_median[returns_median > 0]
# returns_sigma = np.array([np.sqrt((n - 1) / (n - 2) * np.sum(np.square(returns[i] - returns_median[i]))) for i in range(returns_median.shape[0])])
# df = pd.DataFrame(np.array((returns_median, returns_sigma)), columns=_values)
# df.to_csv('preprocessed.csv', index=None)

stocks = StockData(data)
mu = stocks.get_return()
sigma = np.cov(data)
print(mu)
print(sigma)

n_stocks = stocks.n_stocks
n_days = stocks.n_days
daily_change = stocks.daily_change

q = 0.2 # risk
budget = 1_000_0000
penalty = 1.2

Q = -QUBO_from_portfolio(sigma, mu, q, budget, penalty)
portfolio_pauli_terms, portfolio_weights, portfolio_offset = QUBO_to_Ising(Q)

iterations = 1000
nlayers = 12
loss_list = []


# define a callback function to recode the loss
def record_loss(loss, params):
    loss_list.append(loss)


# apply QAOA on this portfolio optimization problem
final_params = QUBO_QAOA(Q, nlayers, iterations, callback=record_loss)

p = plt.plot(loss_list)
def print_result_prob(c, wrap=False, reverse=False):
    states = []
    n_qubits = c._nqubits
    for i in range(2**n_qubits):
        a = f"{bin(i)[2:]:0>{n_qubits}}"
        states.append(a)
        # Generate all possible binary states for the given number of qubits

    probs = K.numpy(c.probability()).round(decimals=4)
    # Calculate the probabilities of each state using the circuit's probability method

    sorted_indices = np.argsort(probs)[::-1]
    if reverse == True:
        sorted_indices = sorted_indices[::-1]
    state_sorted = np.array(states)[sorted_indices]
    prob_sorted = np.array(probs)[sorted_indices]
    # Sort the states and probabilities in descending order based on the probabilities

    print("\n-------------------------------------")
    print("    selection\t  |\tprobability")
    print("-------------------------------------")
    if wrap == False:
        for i in range(len(states)):
            print("%10s\t  |\t  %.4f" % (state_sorted[i], prob_sorted[i]))
            # Print the sorted states and their corresponding probabilities
    elif wrap == True:
        for i in range(4):
            print("%10s\t  |\t  %.4f" % (state_sorted[i], prob_sorted[i]))
        print("               ... ...")
        for i in range(-5, -1):
            print("%10s\t  |\t  %.4f" % (state_sorted[i], prob_sorted[i]))
    print("-------------------------------------")
c_final = QAOA_ansatz_for_Ising(
    final_params, nlayers, portfolio_pauli_terms, portfolio_weights
)
print_result_prob(c_final, wrap=False)
