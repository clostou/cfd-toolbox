"""
post.py
-------

  Modules containing some mathematical tools for analysis or postprocessing.

  Created by Liu Guoqing; Last modification: 2024/10/22
"""


import numpy as np
import pandas as pd

__all__ = ['fft', 'fft_filter', 'spl', 'corr']


def fft(data: np.ndarray, duration: float = 2.0, factor: int = 1):
    """
    一维快速傅里叶变换

    :param data: 时域采样数据（按列排列）
    :param duration: 时域数据的总采样时间（假定为等间隔采样）
    :param factor: 平滑因子，仅能为正整数，实际上为原数据的非重叠分组数
    :return: 频域数据（第一列为频率）
    """
    y = np.array(data).T
    m, n = y.shape
    k = int(n / factor) // 2    # 频域数据点数
    sub_n = 2 * k    # 时域子块数据点数
    sub_t = duration * (sub_n - 1) / (n - 1)    # 时域子块的时长
    df = 1. / sub_t    # 频域频率间隔
    fft_y = np.zeros((m, sub_n), dtype=np.complex64)
    for i in range(factor):
        fft_y += np.fft.fft(y[:, i * sub_n: (i + 1) * sub_n])
    abs_y = np.abs(fft_y) / (sub_n * factor)
    angle_y = np.angle(fft_y)
    freq = np.arange((k) * df, step=df)
    return np.hstack([freq.reshape((-1, 1)),
                      2 * abs_y.T[: k]])


def fft_filter(data, duration, upper_freq=1000):
    """
    低频滤波器

    :param data: 一维原始数据
    :param duration: 数据的总采样时间（假定为等间隔采样）
    :param upper_freq: 截止频率，原数据大于该频率的高频部分将被舍弃
    :return: 滤波后的数据
    """
    y = np.array(data).flatten()
    fft_y = np.fft.fft(y)
    ind = min(int(upper_freq * duration), y.shape[-1])
    fft_y[ind: ] = .0
    y_flit = np.fft.ifft(fft_y)
    return np.real(y_flit)


def spl(data: np.ndarray, p0: float = 2e-5):
    """计算声压级"""
    return 20 * np.log10(data / p0)


def corr(data):
    """计算Spearman相关系数"""
    return data.corr(method='spearman', numeric_only=True)


