from json import load
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

font_size = plt.rcParams['font.size']
use_tex = plt.rcParams['text.usetex']

def load_plot_style(tex=False):
    font_size = plt.rcParams['font.size']
    use_tex = plt.rcParams['text.usetex']
    plt.rcParams['font.size'] = 16
    plt.rcParams['text.usetex'] = tex

def reset_plot_style():
    plt.rcParams['font.size'] = font_size
    plt.rcParams['text.usetex'] = use_tex

def save_score_plot(score_curves, metric_names, filename, title="", skip_first=0, tex=False):
    load_plot_style(tex)

    figure, axes = plt.subplots()
    color_list = list(mcolors.TABLEAU_COLORS)
    style_list = ['-', '--', '-.', ':']

    scores_by_name = dict()
    for score_curve, name in score_curves:
        score = np.asarray(score_curve)
        if not name in scores_by_name:
            scores_by_name[name] = []
        scores_by_name[name].append(score)

    index = 0    
    for name in scores_by_name:
        scores = np.asarray(scores_by_name[name])

        for i in range(len(scores[0][0][1])):
            metric_values = []
            for j in range(len(scores)):
                metric_values.append([scores[j][k][1][i] for k in range(len(scores[j]))])
            metric_values = np.asarray(metric_values)
            average = sum(metric_values) / len(metric_values)
            minimum = np.minimum(*metric_values) if len(metric_values) > 1 else metric_values[0]
            maximum = np.maximum(*metric_values) if len(metric_values) > 1 else metric_values[0]

            indices = scores[0][:, 0][skip_first:]
            average = average[skip_first:]
            minimum = minimum[skip_first:]
            maximum = maximum[skip_first:]

            axes.plot(indices, average,
                      style_list[i % len(style_list)],
                      label=f'{name}, {metric_names[i]}',
                      c=color_list[index],
                      linewidth=1)
            axes.fill_between(indices.astype(int), minimum.astype(float), maximum.astype(float), alpha=0.25, edgecolor=color_list[index], facecolor=color_list[index])
        index += 1
    
    axes.legend()
    axes.set_title(title)

    figure.savefig(filename)

    reset_plot_style()

#scores = \
#[([[0, [1, 11]], [10, [2, 12]], [20, [1.5, 11.5]]], 'ANNR'),
# ([[0, [3, 13]], [10, [3, 13]], [20, [2.5, 12.5]]], 'ANNR'),
# ([[0, [5, 8]], [15, [7, 6]], [22, [11, 10]]], 'DEFER')]
#save_score_plot(scores, ['MAE', 'MSE'], 'qq.png')