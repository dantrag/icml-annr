import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='data path')
args = parser.parse_args()

import utils
cfg = utils.load_config(f'{args.data_path}/config.py')

methods = ['annr', 'defer', 'uniform']
labels = ['ANNR', 'DEFER', 'nANNR']
colors = ['C0', 'C1', 'C3']
line_style = ['-', '--', '-.']

plt.figure(figsize=(6, 4))

plots = []

for i, train in enumerate(methods):
    if train in ['annr', 'uniform']:
        data = np.load(f'{args.data_path}/{train}_data.npy')
    elif train in ['defer']:
        from defer.helpers import Variables, DensityFunctionApproximation, construct
        from defer.variables import Variable
        x = Variable(
            lower=cfg.function.domain.lower_limit_vector,
            upper=cfg.function.domain.upper_limit_vector,
            name="x"
        )
        variables = Variables([x])
        approx: DensityFunctionApproximation = construct(
            fn=cfg.function,
            is_log_fn=False,
            variables=variables,
            num_fn_calls=1,
            callback=lambda i, density:
            print("#Evals: %s. Log Z: %.2f" %
                  (density.num_partitions, np.log(density.z))),
            callback_freq_fn_calls=1000,
            is_vectorized_fn=False
        )
        approx.load(f'{args.data_path}/defer.pkl')
        data = np.array([.5 * (p.domain.lower_limit_vector + p.domain.upper_limit_vector) for p in approx.all_partitions()]).astype(np.float128)
        # values = np.array([p.f for p in approx.all_partitions()]).astype(np.float128)
    else:
        raise Exception()

    data = np.linalg.norm(data, axis=1)
    print(f'{train} size: {data.shape[0]} ')
    values, bins = np.histogram(data, bins=np.linspace(0.0, 5.0, 51))
    print(bins.shape, values.shape)
    xs = 0.5 * (bins[1:] + bins[:-1])
    # data = data[[2, 5]].to_numpy()
    # xs = np.unique(data[:, 0])
    # means = np.zeros_like(xs)
    # stds = np.zeros_like(xs)
    # for j, x in enumerate(xs):
    #     means[j] = np.mean(data[data[:, 0] == x, 1])
    #     stds[j] = np.std(data[data[:, 0] == x, 1])
    # print(stds)

    p = plt.plot(xs, values, c=colors[i], ls=line_style[i])[0]
    plots.append(p)

plt.legend(plots, labels, fontsize='large')
plt.xlabel('$\\Vert p\\Vert, p \\in P_{N}$', fontsize='large')
plt.ylabel('# of points', fontsize='large')
plt.tight_layout()
plt.savefig('ball6_queries.png')
# plt.show()

