import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('table_path', help='a path to MAE table', default='data/afi/ball6.txt')
args = parser.parse_args()

all_data = pd.read_csv(args.table_path, header=None, sep=' ')


methods = ['annr', 'defer', 'uniform']
labels = ['ANNR', 'DEFER', 'nANNR']
colors = ['C0', 'C1', 'C3']
line_style = ['-', '--', '-.']

plt.figure(figsize=(6, 4))

test = 'uniform'
plots = []
for i, train in enumerate(methods):
    data = all_data[(all_data[4] == test) & (all_data[3] == train)]
    data = data[[2, 5]].to_numpy()
    xs = np.unique(data[:, 0])
    means = np.zeros_like(xs)
    stds = np.zeros_like(xs)
    for j, x in enumerate(xs):
        means[j] = np.mean(data[data[:, 0] == x, 1])
        stds[j] = np.std(data[data[:, 0] == x, 1])
    print(stds)

    p = plt.plot(xs, means, c=colors[i], ls=line_style[i])[0]
    plt.fill_between(xs, means - stds, means + stds, color=colors[i], alpha=.3)
    plots.append(p)

plt.legend(plots, labels, fontsize='large')
plt.xlabel('$\\Vert p\\Vert, p \\in P_{\\mathrm{test}}$', fontsize='large')
plt.ylabel('MAE', fontsize='large')
plt.tight_layout()
plt.savefig('ball6_mae.png')
# plt.show()

