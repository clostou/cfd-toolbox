"""
Cpp Implementation
++++++++++++++++++

  c++实现的部分数值算法，用于填补python第三方库的空白

  已有函数：
    - PyCubicSpline: 三次样条
      - cubic_spline: 执行单次三次样条插值
      - cp_plot: 基于cubic_spline的函数拟合，用于测试

  作者：
    仰望 (Github@clostou)
"""

from typing import Union, Iterable, Optional, Callable, Tuple
import numpy as np
from .Interplotation import PyCubicSpline
from matplotlib import pyplot as plt

__all__ = ['PyCubicSpline', 'cubic_spline', 'cp_plot']


def cubic_spline(x_simple: Union[np.ndarray, Iterable],
                 y_simple: Union[np.ndarray, Iterable],
                 x: Union[np.ndarray, Iterable],
                 type: str = 'natural',
                 v0: float = .0, vn: float = .0):
    """ 分段三次样条插值
    
    给定采样点的x、y坐标，在指定位置计算拟合函数值
    根据边界条件的不同，可分为以下六类：
      1. natural:
        自然三次样条，需要指定端点处的曲率（默认为零）
      2. clamped:
        钳制三次样条，需要指定端点处的斜率（默认为零）
      3. quard:
        抛物线端点三次样条，即样条的起始段S1和结束段Sn至多为2阶多项式
      4. not-a-dot:
        非扭结三次样条，即S1和S2、Sn-1和Sn分别为相同的三阶多项式
      5. periodic:
        周期三次样条，即端点处的函数值和1阶、2阶导数相等
      6. half-clamped:
        半钳制三次样条，需要指定左端点处的斜率和右端点处的曲率（默认为零）
    """
    _x = np.array(x_simple, dtype=np.float64)
    _y = np.array(y_simple, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.zeros(x.shape, dtype=np.float64)
    cp = PyCubicSpline(_x, _y, type, v0=v0, vn=vn)
    cp.fillY(x, y)
    return y


def cp_test():
    x = np.array([-1, 0.5, 2, 3])
    y = np.array([3, -2, 1, 3])
    x_interp = np.linspace(x.min() - 1, x.max() + 1, 100)
    y_interp = cubic_spline(x, y, x_interp, type='natural')
    fig, ax = plt.subplots()
    ax.plot(x_interp, y_interp, 'k-')
    ax.plot(x, y, 'r.', markersize=10)
    ax.grid()
    fig.show()


def cp_plot(func: Callable,
            type: str = 'natural',
            fit_n: int = 4,
            section: Tuple[float] = (-10, 10)):
    """ 使用三次样条对函数进行插值拟合，计算均方误差并绘图
    
    详见cubic_spline函数
    """
    x_simple = np.linspace(*section, fit_n)
    y_simple = func(x_simple)
    margin = 0.1 * (section[1] - section[0]) / fit_n
    x_test = np.linspace(section[0] - margin,
                         section[1] + margin, fit_n * 100)
    y_actual = func(x_test)
    y_pred = cubic_spline(x_simple, y_simple, x_test, type)
    print("RMSE: %.2e" % np.sqrt(np.power(y_pred - y_actual, 2).mean()))
    fig, ax = plt.subplots()
    ax.plot(x_test, y_actual, 'y-.')
    ax.plot(x_test, y_pred, 'b-')
    ax.plot(x_simple, y_simple, 'r.', markersize=10)
    ax.grid()
    fig.show()


if __name__ == '__main__':
    #cp_test()
    f_cos = lambda x: np.cos(0.5 * x)
    f_poly = lambda x: (0.01 * x**2 - 1) * (0.003 * x**3 - 0.2 * x + 1)
    f_exp = lambda x: (0.01 * x**2 - 1) * 0.04 * x * np.exp(- 0.3 * x)
    cp_plot(f_poly, type='half-clamped', fit_n=5)
    
    input()


