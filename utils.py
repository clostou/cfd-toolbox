"""
utils.py
--------

  Common tool functions used by other modules.

  Created by Liu Guoqing; Last modification: 2024/8/21
"""


import os
import math
import json

__all__ = ['sec2str', 'num2str', 'dict_disp', 'copy_file']


def sec2str(t: int) -> str:
    """将秒数转换为含分、时、天的字符串"""
    day = int(t // 86400)
    t = t % 86400
    hour = int(t // 3600)
    t = t % 3600
    minute = int(t // 60)
    t = t % 60
    second = int(t)
    if day != 0:
        return f"{day}d {hour}h {minute}m"
    elif hour != 0:
        return f"{hour}h {minute}m"
    else:
        return f"{minute}m {second}s"


def num2str(x: float, ) -> str:
    """将浮点数转换为带计数单位的字符串"""
    # K (kilo,1e3)      m (1e-3)
    # M (mega,1e6)      μ (1e-6)
    # G (giga,1e9)      n (1e-9)
    # T (tera,1e12)     p (1e-12)
    # P (peta,1e15)     a (1e-15)
    # E (exa,1e18)
    if x == 0:
        return '0'
    else:
        exp = math.floor(math.log10(abs(x)))
        ind = exp // 3
        if -6 < ind < 7:
            return format(x * 1e3**-ind, '.%df' % (2 - exp % 3)) + \
                ['', 'K', 'M', 'G', 'T', 'P', 'E', 'a', 'p', 'n', 'μ', 'm'][ind]
        else:
            return '{:.2e}'.format(x)


def dict_disp(d: dict) -> str:
    """使用json序列化将字典转换为易阅读的格式"""
    return json.dumps(d, ensure_ascii=True, separators=(',', ': '), indent=4)


def copy_file(file1: str, file2: str, start:int=0, ending:int=-1, append:bool=False, buffer:int=1024) -> None:
    """文件（部分）复制与追加操作"""
    if ending >= 0 and ending <= start:
        raise ValueError("invalid start-ending of read")
    mode = 'wb'
    if os.path.exists(file1):
        if append:
            mode = 'ab'
        else:
            os.remove(file1)
    if ending < 0:
        length = os.path.getsize(file2) - start
    else:
        length = min(os.path.getsize(file2), ending) - start
    with open(file2, 'rb') as f:
        f.seek(start)
        with open(file1, mode) as _f:
            while True:
                _f.write(f.read(buffer))
                length -= buffer
                if length <= 0:
                    break


