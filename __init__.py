"""
CFD Toolbox
+++++++++++

  基于python的代码工具箱，主要包括CFD领域自己常用的一些代码

  关键模组：
    - submit.py: fluent/cfdpost任务提交与管理
    - post.py: 后处理
    - plot.py: 图像绘制（数据可视化）
    - fileio.py: 文件读写操作
    - vec.py: 向量相关操作
    - utils.py: 便利函数
    - gasdy.py: 气体动力学公式

  作者：
    仰望 (Github@clostou)

  第三方依赖库：
    quaternion
    seaborn

"""

from . import submit, plot, fileio, post, vec, gasdy


