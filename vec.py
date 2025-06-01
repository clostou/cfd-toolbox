"""
vec.py
------

  Some useful functions of vector and geometric operation.

  Created by Liu Guoqing; Last modification: 2025/3/28
"""


import numpy as np
import quaternion as qua
from scipy.optimize import fsolve, root_scalar
import scipy.interpolate as interpolate

from .cpp import PyCubicSpline

__all__ = ['three_point_to_normal', 'vector_to_quaternion', 'offset_space', 'offset_space_nd',
           'func_space_2d', 'func_space_3d', 'ThirdPoly', 'QuasiBezier', 'BSpline', 'CubicSpline']


def _get_ind(x, x_list):
    """计算给定值的所在的区间编号，要求划分区间升序排列。
    基于分治法因此该函数会递归调用"""
    if len(x_list) < 3:
        return 0
    i = int((len(x_list) - 1) / 2)
    if x == x_list[i]:
        return i
    elif x < x_list[i]:
        return _get_ind(x, x_list[: i + 1])
    else:
        return i + _get_ind(x, x_list[i:])


def three_point_to_normal(*points):
    """通过空间三点计算面的法向量"""
    p1, p2, p3 = points
    tang_1 = np.array(p2) - np.array(p1)
    tang_2 = np.array(p3) - np.array(p1)
    normal = np.cross(tang_1, tang_2)
    return normal / np.linalg.norm(normal)


def vector_to_quaternion(vector, spin=0):
    """通过方向矢量和自转角度计算四元数(w, x, y, z)。
    注意cfdpost相机旋转四元数为(x, y, z, w)，初始全局z轴指向屏幕"""
    axis_z = np.array(vector, dtype=np.float64)
    axis_z /= np.linalg.norm(axis_z)
    # 相机坐标系的x轴：x` = cross(y`, z`) = cross(y, z`)
    axis_x = np.cross(np.array((0., 1., 0.)), axis_z)
    if np.linalg.norm(axis_x) == 0:
        axis_y = np.cross(np.array((1., 0., 0.)), axis_z)
        axis_y /= np.linalg.norm(axis_y)
        axis_x = np.cross(axis_y, axis_z)
    else:
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(axis_z, axis_x)
    # 旋转相机xOy平面
    spin = np.pi * spin / 180
    axis_x_new = np.cos(spin) * axis_x + np.sin(spin) * axis_y
    axis_y_new = np.cross(axis_z, axis_x_new)
    rotate_matrix = np.array([axis_x_new, axis_y_new, axis_z])
    # 由旋转矩阵计算四元数：将相机坐标系复位至标准位置
    q = qua.from_rotation_matrix(rotate_matrix)
    return q


def offset_space(start, end, size_or_n, factor=1.0):
    """通过数值范围、q1（或n），以及qn/q1（即factor）来指定等比数列"""
    if factor == 1:
        if isinstance(size_or_n, int):
            n = size_or_n
        else:
            _n = abs(end - start) / size_or_n
            n = int(_n) + 1
        return np.linspace(start, end, n)
    else:
        if isinstance(size_or_n, int):
            n = size_or_n
            q = np.power(factor, 1 / (n - 1))
            size = abs(end - start) * (1 - q) / (1 - np.power(q, n))
        else:
            _n = abs(end - start) / size_or_n
            n = int(np.log(factor) / np.log((_n - 1) / (_n - factor)) + 1)  # n为正整数
            q = np.power(factor, 1 / (n - 1))
            k = _n * (1 - q) / (1 - np.power(q, n))  # 修正起始尺寸
            size = k * size_or_n
        offset = np.concatenate([[0], size * np.power(q, np.arange(n))]).cumsum()
        if start > end:
            return start - offset
        else:
            return start + offset


def offset_space_nd(p_start, p_end, size_or_n, factor=1.0):
    """由两端点对直线进行划分, 可指定划分偏置"""
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    if isinstance(size_or_n, float):
        size_or_n /= np.linalg.norm(p_end - p_start)
    divide = offset_space(0, 1, size_or_n, factor)
    return divide.reshape((-1, 1)) * (p_end - p_start) + p_start


class CurveSpace:
    """
    基于曲率的自适应曲线函数划分，要求给定的曲线方程为参数形式
    """

    def __init__(self, func, t0, t1, size, factor=1., exp=0., beta=0., dim=2):
        if isinstance(func(t0), np.ndarray):
            self.func = func
        else:
            self.func = lambda t: np.array(func(t))
        self.t0 = t0
        self.t1 = t1
        self.size = size

        if factor == 1:    # 尺寸偏置的公比 (factor > 0)
            self.scale_offset = lambda t: 1.
        else:
            self.scale_offset = lambda t: factor ** ((t - t0) / (t1 - t0))

        if exp == 0:    # 曲率映射的指数，即曲率自适应参数 (exp ≥ 0)
            self.scale_curv = lambda c: 1.
        else:
            self.scale_curv = lambda c: (1 - 0.5 * size * c) ** exp

        self.c_limit = 0.05    # 曲率截断，即曲线的曲率过大时当作直线考虑
        self.beta = beta    # 曲率平滑的衰减因子
        self._scale_curv = 1.

        if dim == 2:
            self.spatial = False
        elif dim == 3:
            self.spatial = True
        else:
            raise ValueError("Only division of plannar and spatial curves are supported (dim=2 or dim=3)")

        self.p_start = self.func(t0)
        self.p_end = self.func(t1)
        self.t = []
        self.points = []

    def calc_size(self):
        if len(self.t) < 3:
            self._scale_curv = 1.
        else:
            scale = self.scale_curv(self.curvature())
            self._scale_curv = self.beta * self._scale_curv + (1 - self.beta) * scale  # 移动平均
        return self.size * self.scale_offset(self.t[-1]) * self._scale_curv

    def next_p(self, size):
        f = lambda t: np.linalg.norm(self.func(t) - self.points[-1]) - size
        if f(self.t[-1]) * f(self.t1) >= 0:
            raise ValueError("Cannot find next point because given size is too large.")
        t = root_scalar(f, bracket=[self.t[-1], self.t1]).root
        self.t.append(t)
        self.points.append(self.func(t))

    def curvature(self):
        if len(self.points) < 3:
            raise ValueError("At least three points are needed to calculate the curvature")
        else:
            p1, p2, p3 = self.points[-3: ]
        side_1 = p3 - p2
        side_2 = p2 - p1
        normal = np.cross(side_1, side_2)
        area = np.linalg.norm(normal)
        area_min = self.c_limit * np.linalg.norm(side_1) * np.linalg.norm(side_2)
        if area < area_min:
            return 0.
        if self.spatial:
            normal /= area
            def circle(p0):
                v1 = np.linalg.norm(p1 - p0)
                v2 = np.linalg.norm(p2 - p0)
                v3 = np.linalg.norm(p3 - p0)
                return v1 - v2, v2 - v3, np.dot(normal, p0 - p2)
        else:
            def circle(p0):
                v1 = np.linalg.norm(p1 - p0)
                v2 = np.linalg.norm(p2 - p0)
                v3 = np.linalg.norm(p3 - p0)
                return v1 - v2, v2 - v3
        p0 = fsolve(circle, p2)    # 计算圆心的坐标
        return 1.0 / np.linalg.norm(p0 - p2)

    def divide(self, combine_result=False):
        self.t.append(self.t0)
        self.points.append(self.p_start)
        self.next_p(self.calc_size())
        self.next_p(self.calc_size())
        while True:
            size = self.calc_size()
            if np.linalg.norm(self.p_end - self.points[-1]) > 1.5 * size:
                self.next_p(size)
            else:
                self.t.append(self.t1)
                self.points.append(self.p_end)
                break
        if combine_result:
            return np.vstack(self.points)
        else:
            return self.points


def func_space_2d(func, t0, t1, size, factor=1.):
    """由曲线方程对平面曲线进行划分, 可指定尺寸偏置"""
    curve = CurveSpace(func, t0, t1, size, factor=factor, exp=4, beta=0.6, dim=2)
    return curve.divide(True)


def func_space_3d(func, t0, t1, size, factor=1.):
    """由曲线方程对空间曲线进行划分, 可指定尺寸偏置
    注：部分点的曲率计算可能会不收敛（待解决）"""
    curve = CurveSpace(func, t0, t1, size, factor=factor, exp=4, beta=0.8, dim=3)
    return curve.divide(True)


class ThirdPoly:
    """
    分段三次多项式插值
    
    根据给定的函数值和一阶导数值进行插值，要求各点按x的升序排列
    """

    def __init__(self, x, y, y_prime):
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.y_prime = np.array(y_prime, dtype=np.float64)
        self.m = len(x)

        self._poly_param = []
        if self.m < 2:
            raise ValueError("Piecewise cubic polynomial interpolation needs at least 2 points.")
        i = 0
        while i < self.m - 1:
            self._poly_param.append(self._calc_param(i))
            i += 1

    def _calc_param(self, i):
        x11 = self.x[i]
        x12 = x11 * self.x[i]
        x13 = x12 * self.x[i]
        x21 = self.x[i + 1]
        x22 = x21 * self.x[i + 1]
        x23 = x22 * self.x[i + 1]
        A = np.array([[x13, x12, x11, 1.],
                      [x23, x22, x21, 1.],
                      [3. * x12, 2 * x11, 1., 0.],
                      [3. * x22, 2 * x21, 1., 0.]])
        B = np.array([[self.y[i]],
                      [self.y[i + 1]],
                      [self.y_prime[i]],
                      [self.y_prime[i + 1]]])
        return np.linalg.solve(A, B).flatten().tolist()

    def __len__(self):
        return self.m

    def __call__(self, x):
        i = _get_ind(x, self.x)
        poly = self._poly_param[i]
        return x * (x * (x * poly[0] + poly[1]) + poly[2]) + poly[3]

    def sample(self, x):
        """单点或批量采样，给定x坐标返回对应点处的y坐标"""
        if np.ndim(x) == 1:
            return np.array(list(map(self.__call__, x)))
        elif np.ndim(x) == 0:
            return self.__call__(x)
        else:
            raise ValueError("Input should be a single number or 1-d array, but given x has %d dim." % np.ndim(x))


class QuasiBezier:
    """
    基于分段四点三阶贝塞尔曲线的插值

    其中每一点的斜率由与前后两点的连线决定，参数s∈[0,1]控制斜率的影响范围
    """
    
    def __init__(self, x, y, s, theta_l=None, theta_r=None):
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.s = np.array(s, dtype=np.float64)
        self.points = np.vstack([self.x, self.y]).T
        self.m = len(self.points)
        self.t = np.arange(self.m)
        self.alpha = 0.5  # 斜率计算的比例因子

        diff_vec = np.vstack([np.convolve(self.x, np.array([1, -1]), mode='valid'),
                              np.convolve(self.y, np.array([1, -1]), mode='valid')]).T
        div_len = np.linalg.norm(diff_vec, axis=1, keepdims=True)
        diff_vec /= div_len
        div_len = np.vstack([.0, div_len, .0])
        diff_vec = np.vstack([diff_vec[0], diff_vec, diff_vec[-1]])
        if theta_l is not None:
            diff_vec[0] = [np.cos(theta_l), np.sin(theta_l)]
        if theta_r is not None:
            diff_vec[-1] = [np.cos(theta_r), np.sin(theta_r)]
        # 计算每个采样点上的偏置向量，用于计算其余控制点
        self.offset = []
        i = 0
        while i < self.m:
            self.offset.append(self.s[i] * ((1 - self.alpha) * diff_vec[i] * div_len[i + 1] + \
                                            self.alpha * diff_vec[i + 1] * div_len[i]))
            i += 1

    def __len__(self):
        return self.m

    def __call__(self, t):
        i = _get_ind(t, self.t)
        p1 = self.points[i]
        p2 = self.points[i] + self.offset[i]
        p3 = self.points[i + 1] - self.offset[i + 1]
        p4 = self.points[i + 1]
        t -= i
        w1 = (1 - t) ** 3
        w2 = 3 * (1 - t) ** 2 * t
        w3 = 3 * (1 - t) * t ** 2
        w4 = t ** 3
        return w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4

    def sample(self, t):
        """单点或批量采样，给定参数t返回各点的x、y坐标"""
        if np.ndim(x) == 1:
            return np.array(list(map(self.__call__, t)))
        elif np.ndim(x) == 0:
            return self.__call__(t)
        else:
            raise ValueError("Input should be a single number or 1-d array, but given x has %d dim." % np.ndim(x))


class BSpline:
    """
    使用scipy库实现的三阶B样条插值

    边界处的曲率固定为零，但通过引入样本点权重和平滑性度量来平衡拟合的贴合度和光滑度
    """
    
    def __init__(self, x, y, smooth=False, weight=None):
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()
        self.m = len(self.x)
        self.w = weight
        if smooth:
            s = self.m - np.sqrt(2 * self.m)    # recommended value of s
        else:
            s = 0
        self.t, self.c, self.k = interpolate.splrep(self.x, self.y, w=self.w, s=s, k=3)
        self.spline = interpolate.BSpline(self.t, self.c, self.k, extrapolate=False)

    def __len__(self):
        return self.m
    
    def __call__(self, x):
        return self.spline(x).item()

    def sample(self, x):
        """单点或批量采样，给定x坐标返回对应点处的y坐标"""
        if np.ndim(x) == 1:
            return self.spline(x)
        elif np.ndim(x) == 0:
            return self.spline(x).item()
        else:
            raise ValueError("Input should be a single number or 1-d array, but given x has %d dim." % np.ndim(x))


class CubicSpline:
    """
    分段三次样条插值

    相比分段三次多项式插值曲线更加平滑（C2连续），需要指定端点条件：
      1. natural: 自然三次样条，需要指定端点处的曲率（默认为零）
      2. clamped: 钳制三次样条，需要指定端点处的斜率（默认为零）
      3. quard: 抛物线端点三次样条，即样条的起始段S1和结束段Sn至多为2阶多项式
      4. not-a-dot: 非扭结三次样条，即S1和S2、Sn-1和Sn分别为相同的三阶多项式
      5. periodic: 周期三次样条，即端点处的函数值和1阶、2阶导数相等
      6. half-clamped: 半钳制三次样条，需要指定左端点处的斜率和右端点处的曲率（默认为零）
    """

    def __init__(self, x, y, type='natural', v0=0., vn=0.):
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.spline = PyCubicSpline(self.x, self.y, type, v0, vn)

    def __len__(self):
        return self.spline.p_n

    def __call__(self, x):
        return self.spline.getY(float(x))

    def sample(self, x):
        """单点或批量采样，给定x坐标返回对应点处的y坐标"""
        if np.ndim(x) == 1:
            x = np.array(x, dtype=np.float64)
            y = np.zeros(x.shape, dtype=np.float64)
            self.spline.fillY(x, y)
            return y
        elif np.ndim(x) == 0:
            return self.spline.getY(float(x))
        else:
            raise ValueError("Input should be a single number or 1-d array, but given x has %d dim." % np.ndim(x))


if __name__ == '__main__':
    print("======== Test of quaternion ========")
    vec_1 = np.array([0, 0, 1])
    vec_2 = three_point_to_normal([-3914e-3, 984e-3, 3097.01e-3],
                                  [-3914e-3, -663.58e-3, 3097e-3],
                                  [-4109e-3, 1500.07e-3, 2963.68e-3])
    vec_3 = three_point_to_normal([-3914e-3, 984e-3, 3097.01e-3],
                                  [-3914e-3, -663.58e-3, 3097e-3],
                                  [-3676.52e-3, -874.82e-3, 2972.95e-3])
    print(vec_2, vec_3)
    print(vector_to_quaternion(vec_1))
    print(vector_to_quaternion(-vec_2))
    print(vector_to_quaternion(vec_3))
    print("======== Test of offset-space ========")
    print(offset_space(0, 5, 10, factor=2))
    print("======== Test of function-space ========")
    from cfd_toolbox.plot import plot, func_plot, plt
    poly = ThirdPoly(np.arange(-2, 4, 1), np.random.random(6), np.random.random(6))
    #func_plot(lambda x: poly(x), -2, 3, n=100, marked_x=np.arange(-2, 4, 1))
    p = func_space_2d(lambda t: (t, poly(t)), t0=2, t1=-1, size=0.02, factor=4)
    plot(p[:, 0], p[:, 1])
    '''import matplotlib.pyplot as plt
        p = func_space_3d(lambda t: (t, t**2, t**3), t0=-1, t1=2, size=0.1, factor=4)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(p[:, 0], p[:, 1], p[:, 2], '.-')
        fig.show()'''
    print("======== Test of Interpolation ========")
    bez = QuasiBezier(np.arange(6), [9, 8, 6, 3, 1, 0], [1]*6, theta_l=np.tan(np.pi*-71/180))
    func_plot(lambda x: bez(x)[1], 0, 5, n=100, marked_x=np.arange(6))
    x = np.array([0., 1.2, 1.9, 3.2, 4., 6.5])
    y = np.array([0., 2.3, 3., 4.3, 2.9, 3.1])
    xx = np.linspace(x.min()-0.5, x.max()+0.5, 100)
    bsp = BSpline(x, y, weight=[1,1,0.1,1,1,1], smooth=True)
    sp = CubicSpline(x, y, type='not-a-dot')
    fig, ax = plot(xx, bsp.sample(xx), sp.sample(xx), legend=['BSpline', 'Spline', 'Original points'])
    ax.plot(x, y, 'bo')
    print('''\tt: {}\n\tc: {}\n\tk: {}'''.format(bsp.t, bsp.c, bsp.k))


