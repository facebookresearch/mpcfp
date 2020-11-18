#!/usr/bin/env python2

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn

# constants:
NAN = float('nan')
# From https://blog.graphiq.com/
# finding-the-right-color-palettes-for-data-visualizations-fcd4e707a283
BAR_COLORS_PURPLES = [
    (0.9020, 0.6196, 0.6157),
    (0.7765, 0.3412, 0.5294),
    (0.4471, 0.1922, 0.5647),
    (0.2549, 0.1098, 0.3804),
]
BAR_COLORS_GRAY_PURPLES = [
    (.7, .7, .7),
    (0.9020, 0.6196, 0.6157),
    (0.7765, 0.3412, 0.5294),
    (0.4471, 0.1922, 0.5647),
    (0.2549, 0.1098, 0.3804),
]
BAR_COLORS_DETECTION = [
    (.8, .8, .8),
    (.4, .4, .4),
    (0.9020, 0.6196, 0.6157),
    (0.7765, 0.3412, 0.5294),
    (0.4471, 0.1922, 0.5647),
    (0.2549, 0.1098, 0.3804),
]
LINE_COLORS = seaborn.cubehelix_palette(
    4, start=2, rot=0, dark=0.15, light=0.75, reverse=False, as_cmap=False)
BAR_COLORS = BAR_COLORS_GRAY_PURPLES
FS = 18
color_counter = [0]

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"


def set_style():
    params = {
        "legend.fontsize": FS - 4,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS - 4,
        "ytick.labelsize": FS - 4,
    }
    matplotlib.rcParams.update(params)
    fig = plt.gcf()
    for ax in fig.axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)


# make generic line plot:
def make_line_plot(Y, x=None, title='',
                   xlabel='', ylabel='', xlog=False, ylog=False,
                   xmin=None, xmax=None, ymin=None, ymax=None,
                   legend=[], legend_title=None, show_legend=True,
                   text_labels=None, colors=[], linestyle=[], markerstyle=[],
                   append=False, filename=None, linewidth=2., legloc=None,
                   errors=None, xticks=None, yticks=None):

    # assertions and defaults:
    x = np.linspace(0, Y.shape[1]) if x is None else x
    ymin = Y.min() if ymin is None else ymin
    ymax = Y.max() if ymax is None else ymax
    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax
    if len(legend) > 0:
        assert len(legend) == Y.shape[0]
    if len(colors) == 0:
        colors = LINE_COLORS
    if isinstance(linestyle, str):
        linestyle = [linestyle] * Y.shape[0]
    if len(linestyle) == 0:
        linestyle = ['-'] * Y.shape[0]
    if isinstance(markerstyle, str):
        markerstyle = [markerstyle] * Y.shape[0]
    if len(markerstyle) == 0:
        markerstyle = [''] * Y.shape[0]

    # make plot:
    if not append:
        plt.clf()
    for n in range(Y.shape[0]):
        linecolor = colors[color_counter[0] % len(colors)]
        color_counter[0] += 1
        plt.plot(x, Y[n, :],
                 label=legend[n] if len(legend) > 0 else None,
                 linewidth=linewidth, linestyle=linestyle[n],
                 marker=markerstyle[n], markersize=linewidth * 1.5,
                 color=linecolor)
        if errors is not None:
            plt.fill_between(
                x, Y[n, :] - errors[n, :], Y[n, :] + errors[n, :],
                alpha=0.2, color=linecolor)

    plt.xlabel(xlabel, fontweight='bold', fontsize=FS)
    plt.ylabel(ylabel, fontweight='bold', fontsize=FS)
    if show_legend:
        plt.legend(fontsize=FS - 4, loc=0 if legloc is None else legloc,
                   title=legend_title)

    # add text labels:
    if text_labels is not None:
        assert isinstance(text_labels, list)
        for text_label in text_labels:
            assert isinstance(text_label, list) \
                or isinstance(text_label, tuple)
            assert len(text_label) == 3
            plt.text(*text_label)

    # makes axes look pretty:
    axes = plt.gca()

    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
    if xlog:
        axes.semilogx(10.)
    if ylog:
        axes.semilogy(10.)
    if xticks is not None:
        axes.set_xticks(xticks)
    if yticks is not None:
        axes.set_yticks(yticks)
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(FS - 4)
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(FS - 4)
    if title != '':
        axes.set_title(title, fontweight='bold', fontsize=FS)
    if show_legend and legend_title is not None:
        legend_title = axes.get_legend().get_title().properties()[
            'fontproperties']
        legend_title.set_weight('bold')

    # remove legend border:
    legend = axes.get_legend()
    if legend is not None:
        legend.get_frame().set_linewidth(0.0)

    # export plot:
    set_style()
    if filename is not None:
        plt.savefig(filename, format='pdf', transparent=True,
                    bbox_inches='tight')


def read_log(logfile, timings=False, test=False):
    x = []
    y = []
    yy = []
    z = []
    with open(os.path.join("results/", logfile), 'r') as fid:
        for line in fid:
            if test and "Test Set" in line:
                fields = line.strip().split()
                if len(fields) > 4:
                    test_loss = float(fields[3][:-1])
                    test_accuracy = float(fields[5])
                else:
                    test_loss = float(fields[3])
                    test_accuracy = 0
            if "Iter" not in line:
                continue
            fields = line.strip().split()
            it = int(fields[1][:-1])
            loss = float(fields[3][:-1])
            if len(fields) > 6:
                accuracy = float(fields[5][:-1])
                runtime = float(fields[7])
                yy.append(accuracy)
            else:
                runtime = float(fields[5])
            x.append(it)
            y.append(loss)
            z.append(runtime)
    if test:
        return test_loss, test_accuracy
    return np.array(x), np.array(y), np.array(yy), np.array(z)


def read_log_synth(logfile):
    x = []
    with open(os.path.join("results/", logfile), 'r') as fid:
        for line in fid:
            if "normalizing both weights and iweights" not in line:
                continue
            fields = line.strip().split()
            diff = float(fields[7])
            x.append(diff)
    return np.array(x)


def mnist_width_train(filename):
    global color_counter
    color_counter = [0]
    xlabel = '\\textbf{Width (}$\mathbf{\gamma}$\\textbf{)}'
    ylabel = '\\textbf{Train Loss}'

    widths = ['1e3', '1e4', '1e5', '1e6', '2e6', '5e6']
    Ys = []
    links = ["Identity", "Logit", "Probit"]
    for link in links:
        files = ['mnist_width%d_link_%s.txt' % (int(float(w)), link.lower())
                 for w in widths]
        Y = []
        for logfile in files:
            it, loss, _, _ = read_log(logfile, test=False)
            Y.append(loss[-1])
        Ys.append(Y)
    Y = np.stack(Ys)

    x = np.array([float(w) for w in widths])

    # produce plots:
    make_line_plot(Y, x=x, xlabel=xlabel, ylabel=ylabel, legend=links,
                   colors=['k', 'k', 'k'],
                   linestyle=['-', '--', ':'],
                   xlog=True, ylog=False, markerstyle='s',
                   ymin=0.0, ymax=0.2,
                   xmin=9e2, xmax=6e6,
                   filename=filename, linewidth=2.,
                   legloc='upper left')


def mnist_width_test(filename):
    global color_counter
    color_counter = [0]
    xlabel = '\\textbf{Width (}$\mathbf{\gamma}$\\textbf{)}'
    ylabel = '\\textbf{Test Loss}'

    widths = ['1e3', '1e4', '1e5', '1e6', '2e6', '5e6']

    Ys = []
    links = ["Identity", "Logit", "Probit"]
    for link in links:
        files = ['mnist_width%d_link_%s.txt' % (int(float(w)), link.lower())
                 for w in widths]
        Y = []
        for logfile in files:
            loss, _ = read_log(logfile, test=True)
            Y.append(loss)
        Ys.append(Y)
    Y = np.stack(Ys)

    x = np.array([float(w) for w in widths])

    # produce plots:
    make_line_plot(Y, x=x, xlabel=xlabel, ylabel=ylabel, legend=links,
                   colors=['k', 'k', 'k'],
                   linestyle=['-', '--', ':'],
                   xlog=True, ylog=False, markerstyle='s',
                   ymin=0.0, ymax=0.2,
                   xmin=9e2, xmax=6e6,
                   filename=filename, linewidth=2.,
                   legloc='upper left')


def covtype_width_train(filename):
    global color_counter
    color_counter = [0]
    xlabel = '\\textbf{Width (}$\mathbf{\gamma}$\\textbf{)}'
    ylabel = '\\textbf{Train Loss}'

    widths = ['1e3', '1e4', '1e5', '1e6', '2e6', '5e6', '1e7']
    Ys = []
    links = ["Identity", "Logit", "Probit"]
    for link in links:
        files = ['covtype_width%d_link_%s.txt' % (int(float(w)), link.lower())
                 for w in widths]
        Y = []
        for logfile in files:
            it, loss, _, _ = read_log(logfile, test=False)
            Y.append(loss[-1])
        Ys.append(Y)
    Y = np.stack(Ys)

    x = np.array([float(w) for w in widths])

    # produce plots:
    make_line_plot(Y, x=x, xlabel=xlabel, ylabel=ylabel, legend=links,
                   colors=['k', 'k', 'k'],
                   linestyle=['-', '--', ':'],
                   xlog=True, ylog=False, markerstyle='s',
                   ymin=0.0, ymax=1.3,
                   xmin=9e2, xmax=1.2e7,
                   filename=filename, linewidth=2.,
                   legloc='upper left')


def covtype_width_test(filename):
    global color_counter
    color_counter = [0]
    xlabel = '\\textbf{Width (}$\mathbf{\gamma}$\\textbf{)}'
    ylabel = '\\textbf{Test Loss}'

    widths = ['1e3', '1e4', '1e5', '1e6', '2e6', '5e6', '1e7']

    Ys = []
    links = ["Identity", "Logit", "Probit"]
    for link in links:
        files = ['covtype_width%d_link_%s.txt' % (int(float(w)), link.lower())
                 for w in widths]
        Y = []
        for logfile in files:
            loss, _ = read_log(logfile, test=True)
            Y.append(loss)
        Ys.append(Y)
    Y = np.stack(Ys)

    x = np.array([float(w) for w in widths])

    # produce plots:
    make_line_plot(Y, x=x, xlabel=xlabel, ylabel=ylabel, legend=links,
                   colors=['k', 'k', 'k'],
                   linestyle=['-', '--', ':'],
                   xlog=True, ylog=False, markerstyle='s',
                   ymin=0.0, ymax=1.3,
                   xmin=9e2, xmax=1.2e7,
                   filename=filename, linewidth=2.,
                   legloc='upper left')


def synth_width(filename):
    global color_counter
    color_counter = [0]
    xlabel = '\\textbf{Width (}$\mathbf{\gamma}$\\textbf{)}'
    ylabel = '$\mathbf{\|\\frac{x}{\|x\|} - \\frac{w}{\|w\|}\|}$'

    widths = ['1e1', '1e2', '1e3', '1e4', '1e5', '1e6', '5e6']

    Ys = []
    links = ["Identity", "Logit", "Probit"]
    for link in ['identity', 'logit', 'probit']:
        files = ['synth_width%d_link_%s.txt' % (int(float(w)), link)
                 for w in widths]
        Y = []
        for logfile in files:
            normdiff = read_log_synth(logfile)
            Y.append(normdiff[-1])
        Ys.append(Y)
    Y = np.stack(Ys)

    x = np.array([float(w) for w in widths])

    # produce plots:
    make_line_plot(Y, x=x, xlabel=xlabel, ylabel=ylabel, legend=links,
                   colors=['k', 'k', 'k'],
                   linestyle=['-', '--', ':'],
                   xlog=True, ylog=False, markerstyle='s',
                   ymin=0.0, ymax=0.02,
                   xmin=8, xmax=6e6,
                   filename=filename, linewidth=2.,
                   legloc='upper left')


def synth_terms(filename):
    global color_counter
    color_counter = [0]
    xlabel = '\\textbf{Terms}'
    ylabel = '$\mathbf{\|\\frac{x}{\|x\|} - \\frac{w}{\|w\|}\|}$'

    terms = list(range(6, 42, 2))

    Ys = []
    links = ["Logit", "Probit"]
    for link in links:
        files = ['synth_terms%d_link_%s.txt' % (t, link.lower())
                 for t in terms]
        Y = []
        for logfile in files:
            normdiff = read_log_synth(logfile)
            Y.append(normdiff[-1])
        Ys.append(Y)
    Y = np.stack(Ys)

    x = np.array(terms)

    # produce plots:
    make_line_plot(Y, x=x, xlabel=xlabel, ylabel=ylabel, legend=links,
                   colors=['k', 'k'],
                   linestyle=['-', '--'],
                   xlog=False, ylog=False, markerstyle='s',
                   ymin=0.0, ymax=0.025, xticks=list(range(6, 42, 4)),
                   xmin=5, xmax=42,
                   filename=filename, linewidth=2.,
                   legloc='upper right')


def mnist_multi(filename):
    xlabel = '\\textbf{Width (}$\mathbf{\gamma}$\\textbf{)}'
    ylabel = '\\textbf{Accuracy}'

    widths = ['1e3', '1e4', '1e5', '1e6']
    files = ['mnist_width%d_multi.txt' % (int(float(w))) for w in widths]
    Y = []
    Y_train = []
    for logfile in files:
        _, acc = read_log(logfile, test=True)
        _, _, train_acc, _ = read_log(logfile)
        Y.append(acc)
        Y_train.append(train_acc[-1])
    Y = np.stack([Y_train, Y])

    x = np.array([float(w) for w in widths])

    # produce plots:
    make_line_plot(Y, x=x, xlabel=xlabel, ylabel=ylabel,
                   legend=['Train', 'Test'],
                   colors=['k', 'k'],
                   linestyle=['-', '--'],
                   xlog=True, ylog=False, markerstyle='s',
                   ymin=0.7, ymax=1,
                   xmin=9e2, xmax=1.2e6,
                   filename=filename, linewidth=2.,
                   legloc='upper left')


def covtype_multi(filename):
    xlabel = '\\textbf{Width (}$\mathbf{\gamma}$\\textbf{)}'
    ylabel = '\\textbf{Accuracy}'

    widths = ['1e3', '1e4', '1e5', '1e6']
    files = ['covtype_width%d_multi.txt' % (int(float(w))) for w in widths]
    Y = []
    Y_train = []
    for logfile in files:
        _, acc = read_log(logfile, test=True)
        _, _, train_acc, _ = read_log(logfile)
        Y.append(acc)
        Y_train.append(train_acc[-1])
    Y = np.stack([Y_train, Y])

    x = np.array([float(w) for w in widths])

    # produce plots:
    make_line_plot(Y, x=x, xlabel=xlabel, ylabel=ylabel,
                   legend=['Train', 'Test'],
                   colors=['k', 'k'],
                   linestyle=['-', '--'],
                   xlog=True, ylog=False, markerstyle='s',
                   ymin=0.5, ymax=0.8,
                   xmin=9e2, xmax=1.2e6,
                   filename=filename, linewidth=2.,
                   legloc='upper left')


# make all the plots:
def main():

    # get destination folder:
    parser = argparse.ArgumentParser(
        description='Make plots for floating point MPC')
    parser.add_argument('--destination', default='./results/', type=str,
                        help='folder in which to dump figures')
    args = parser.parse_args()

    # create plots:
    mnist_width_train(os.path.join(args.destination,
                                   'mnist_widths_train_loss.pdf'))
    mnist_width_test(os.path.join(args.destination,
                                  'mnist_widths_test_loss.pdf'))
    covtype_width_train(os.path.join(args.destination,
                                     'covtype_widths_train_loss.pdf'))
    covtype_width_test(os.path.join(args.destination,
                                    'covtype_widths_test_loss.pdf'))
    synth_width(os.path.join(args.destination, 'synth_widths_weightdiffs.pdf'))
    synth_terms(os.path.join(args.destination, 'synth_terms_weightdiffs.pdf'))
    mnist_multi(os.path.join(args.destination,
                'mnist_multiclass_accuracy.pdf'))
    covtype_multi(os.path.join(args.destination,
                  'covtype_multiclass_accuracy.pdf'))


# run all the things:
if __name__ == '__main__':
    main()
