import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

font_size = plt.rcParams['font.size']
use_tex = plt.rcParams['text.usetex']

def adaptive_marker_size(count):
    return min(20, 200 // (count ** 0.5))

def load_plot_style(tex=False):
    font_size = plt.rcParams['font.size']
    use_tex = plt.rcParams['text.usetex']
    plt.rcParams['font.size'] = 16
    plt.rcParams['text.usetex'] = tex

def reset_plot_style():
    plt.rcParams['font.size'] = font_size
    plt.rcParams['text.usetex'] = use_tex

def save_score_plot(score_curves, metric_names, filename, title="", skip_first=0, tex=False, log_scale=False, x_suffix=''):
    load_plot_style(tex)

    figure, axes = plt.subplots(dpi=600)
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
            minimum = np.minimum.reduce(metric_values) if len(metric_values) > 1 else metric_values[0]
            maximum = np.maximum.reduce(metric_values) if len(metric_values) > 1 else metric_values[0]
            std = (sum(abs(metric_values - average) ** 2) / len(metric_values)) ** 0.5

            indices = scores[0][:, 0][skip_first:]
            average = average[skip_first:]
            minimum = minimum[skip_first:]
            maximum = maximum[skip_first:]
            std = std[skip_first:]

            axes.plot(indices, average,
                      style_list[i % len(style_list)],
                      label=f'{name}' + (f', {metric_names[i]}' if metric_names[i] else ''),
                      c=color_list[index],
                      linewidth=1)
            axes.fill_between(indices.astype(int), (average - std).astype(float), (average + std).astype(float), alpha=0.25, edgecolor=color_list[index], facecolor=color_list[index])
        index += 1
    
    axes.legend()
    axes.set_title(title)
    if log_scale:
        axes.set_yscale('log')
    if x_suffix:
        from matplotlib.ticker import FormatStrFormatter
        figure.gca().xaxis.set_major_formatter(FormatStrFormatter(rf'%d{x_suffix}'))

    plt.tight_layout()
    figure.savefig(filename)

    reset_plot_style()

# Example of a scores structure, with three interpolators and two metrics
# scores = \
#     [([[0, [1, 11]], [10, [2, 12]], [20, [1.5, 11.5]]], 'ANNR'),
#     ([[0, [3, 13]], [10, [3, 13]], [20, [2.5, 12.5]]], 'ANNR'),
#     ([[0, [5, 8]], [15, [7, 6]], [22, [11, 10]]], 'DEFER')]
# save_score_plot(scores, ['MAE', 'MSE'], 'scores.png')