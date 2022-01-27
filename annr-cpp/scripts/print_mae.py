import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('table_path', help='a path to MAE table')
args = parser.parse_args()

all_data = pd.read_csv(args.table_path, header=None, sep=' ')

methods = ['annr', 'defer', 'uniform']
labels = ['ANNR', 'DEFER', 'nANNR']

for test in ['uniform', 'grid']:
    plots = []
    xs = None
    for i, train in enumerate(methods):
        data = all_data
        data = data[(data[3] == test) & (data[2] == train)]

        # print(data.shape, train)
        if data.shape[0] == 0:
            continue

        data = data[[1, 4]].to_numpy()

        xs = np.unique(data[:, 0])
        means = np.zeros_like(xs)
        stds = np.zeros_like(xs)
        for j, x in enumerate(xs):
            means[j] = np.mean(data[data[:, 0] == x, 1])
            stds[j] = np.std(data[data[:, 0] == x, 1])

        if xs.size == 1:
            print(f'{labels[i]}: ${means[0]:.4f}$ \\\\ $\\pm {stds[0]:.4f}$')
        else:
            print(xs, means)
