"""
plot.py
-------

  Modules to provide some useful ploters.

  Created by Liu Guoqing; Last modification: 2024/12/6
"""


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import utils

__all__ = ['plt', 'sns', 'func_plot', 'plot', 'plot_frame', 'corr']


plt.rcParams['font.sans-serif'] = ['SimSun']    # SimHei
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

sns.set_context(context='talk', font_scale=1)


def calc_lambda(func, x):
    try:
        y = func(x)
    except:
        x = list(x)
        y = []
        i = 0
        while i < len(x):
            try:
                y.append(func(x[i]))
                i += 1
            except Exception as e:
                print(repr(e) + ' (skip xi = %.2e)' % x.pop(i))
                y.append(None)
    return y


def func_plot(function, x0=0, x1=10, n=100, title=None, x_label='x', y_label='y', marked_x=None):
    """绘制函数图像，可指定标注点"""
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #fig.tight_layout()
    ax.grid()
    x = np.linspace(x0, x1, n)
    y = calc_lambda(function, x)
    ax.plot(x, y, 'k-')
    if hasattr(marked_x, '__getitem__'):
        for xi, yi in zip(marked_x, calc_lambda(function, marked_x)):
            ax.scatter((xi, ), (yi, ), s=20, c='red')
            ax.text(xi, yi, f'({xi:.3g}, {yi:.3g})', fontdict={'size': 14, 'color': 'red'})
    fig.show()
    return fig, ax


def plot(x, *y, title=None, x_label='x', y_label='y', legend=None, save_path=None):
    """在同一图窗上绘制多条曲线"""
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(title)
    for yi in y:
        ax.plot(x, yi, '.-', markersize=2, lw=1)
    ax.set_xlabel(x_label, fontsize=24, weight='bold')
    ax.set_ylabel(y_label, fontsize=24, weight='bold')
    margin = 0.1 * (x[-1] - x[0])
    ax.set_xticks(np.linspace(x[0] - margin, x[-1] + margin, 13))
    if legend:
        ax.legend(legend)
    ax.grid()
    plt.autoscale()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        fig.show()
    return fig, ax


def plot_frame(frame, column_group, column_x, column_y,
               title=None):
    """分组绘制二维表格（DataFrame）中的特定数据"""
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(title)
    for value in frame[column_group].unique():
        sub_frame = frame[frame[column_group] == value]
        ax.plot(sub_frame[column_x], sub_frame[column_y],
                '.-', markersize=4, lw=1,
                label=utils.num2str(value))
    ax.set_xlabel(column_x, fontsize=14)
    ax.set_ylabel(column_y, fontsize=14)
    ax.legend()
    ax.grid()
    fig.show()
    return fig, ax


def corr(frame, indep_vars=None, tag=None):
    """绘制二维表格（DataFrame）各列的相关系数，可指定自变量"""
    # Pearson积差相关系数局限：仅线性关系、离群值敏感、变量接近联合正态分布
    # Spearman秩相关系数：利用两变量的秩次大小作线性相关分析，对原始变量的分布不做要求
    corr_matrix = frame.corr(method='spearman', numeric_only=True)
    if indep_vars:
        dep_vars = list(set(frame.columns).difference(indep_vars))
        indep_ind = np.where(np.in1d(frame.columns, indep_vars))[0]
        dep_ind = np.where(np.in1d(frame.columns, dep_vars))[0]
    else:
        indep_ind = dep_ind = range(len(corr_matrix.columns))
    if tag:
        corr_matrix.columns = tag
        corr_matrix.index = tag
    sns.heatmap(corr_matrix.iloc[indep_ind, dep_ind],
                annot=True,
                cmap='coolwarm',
                linewidths=1)


