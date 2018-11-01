from rw_fuzzy_params import run

list_fuctions = [
    {'function': 'fuzzy', 'name': 'alpha', 'params': [6]},
    {'function': 'triangular', 'name': 'beta','params': [4, 6, 8]},
    {'function': 'trapezoidal', 'name': 'beta', 'params': [2, 4, 6]},
    {'function': 'gaussian', 'name': 'alpha', 'params': [5, 10, 15]},
    {'function': 'bell', 'name': 'beta', 'params': [4, 6, 8]},
    {'function': 'mult_gaussian', 'name': 'alpha', 'params': [1, 2, 3]},
    {'function': 'mult_fuzzy', 'name': 'alpha', 'params': [0.1, 1, 2]},
]

for experim in list_fuctions:
    print(experim)
    run(experim['function'], experim)
