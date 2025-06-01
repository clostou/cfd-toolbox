"""
fileio.py
---------

  Modules to read or write file of specific format.

  Created by Liu Guoqing; Last modification: 2024/9/2
"""


import numpy as np
import pandas as pd
import re

__all__ = ['read_fluent_result', 'read_fluent_profile', 'read_fluent_xy']


def read_fluent_result(path):
    """读取FluentQuest生成的报告文件，并返回为DataFrame数据表"""
    _data = []
    with open(path, 'r', encoding='utf-8') as fr:
        header = fr.readline().strip().split(',')
        column_n = len(header)
        while True:
            _line = fr.readline()
            if not _line:
                break
            line = _line[:-1].split()
            if len(line) == column_n:
                _data.append(line)
    frame = pd.DataFrame(np.array(_data, dtype=np.float32), columns=header)
    return frame


def read_excel(path):
    pd.read_excel(path)


def read_fluent_profile(path):
    """读取Fluent生成的prifile文件，并返回为DataFrame数据表"""
    with open(path, 'r', encoding='utf-8') as fr:
        info = fr.readline()[2: -2].split()
        n = int(info[2])
        data = {}
        while True:
            label = fr.readline()
            if not label:
                break
            if label.startswith('('):
                column = []
                i = 0
                while i < n:
                    column.append(float(fr.readline()[: -1]))
                    i += 1
                data[label[1: -1]] = column
    print("Read profile on %s (%i points) of:" % (info[0], n))
    for label in data.keys():
        print("    %s" % label)
    return pd.DataFrame(data)


def read_fluent_xy(path):
    """读取Fluent生成的xy文件，并返回为DataFrame数据表"""
    with open(path, 'r', encoding='utf-8') as fr:
        title = re.search('''(?<=")\w[^"]*(?=")''', fr.readline()).group()
        label = re.findall('''(?<=")\w[^"]*(?=")''', fr.readline())
        data = []
        for line in fr:
            cuts = line.split()
            try:
                data.append([float(cuts[0]), float(cuts[1])])
            except:
                pass
    print("Read XY file (%i points) of:\n    %s" % (len(data), title))
    return pd.DataFrame(data, columns=label)


if __name__ == '__main__':
    data = read_fluent_result(r'F:\Nozzle\plug\CFD\allParams\1.txt')
    pass


