"""
submit.py
---------

  Automatically submission of scripts of Fluent and CFD-POST.

  Created by Liu Guoqing; Last modification: 2024/8/21
"""

# 待实行：
# 1. fluent收敛自动检测
# 2. cfdpost参数自动解析


import os
import queue
import sys
import subprocess
import threading
import time
import re

from pathlib import Path
from typing import Union, Optional, List
from itertools import product
from queue import PriorityQueue

from cfd_toolbox import utils

__all__ = ['FluentQuest', 'CFDPostQuest', 'QuestManager']

# sys.stderr = sys.stdout


class BaseScript:
    """
    参数化脚本基类
    """

    _type = None
    _raw = ''

    def __init__(self, script_type: str) -> None:
        self._type = str.lower(script_type)

    def _from_file(self, path: Path) -> None:
        """
        从文件读取脚本，并自动识别脚本参数

        :param path: 本地脚本路径
        """
        if os.path.basename(path).split('.')[-1] != self._type:
            raise TypeError("Incompatible file type (accepted file suffix: .%s)" % self._type)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                self._raw = f.read()
            params = re.findall('''(?<={)[a-zA-Z][a-zA-Z0-9_]*(?=})''', self._raw)
            self.__dict__.update(dict.fromkeys(params))
            print(self.__repr__())

    def _to_file(self, path: Path) -> str:
        """
        生成特定参数下的脚本并导出为文件

        :param path: 脚本导出目录
        :return: 生成脚本的本地路径
        """
        file_path = path / f'{self.__class__.__name__}.{self._type}'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.__call__())
        return file_path

    def _get_params(self) -> dict:
        """
        获取脚本的全部参数以及参数的当前值

        :return: 参数字典
        """
        return {k: self.__dict__[k] for k in self.__dict__.keys() if not k.startswith('_')}

    def _update_params(self, params_dict: dict) -> None:
        """
        设置脚本参数的当前值

        :param params_dict: 参数更新字典
        """
        keys = list(self._get_params().keys())
        for k in params_dict.keys():
            if k not in keys:
                raise KeyError("unknown parameter '%s'" % k)
        self.__dict__.update(params_dict)

    def __call__(self, params_dict: Optional[dict] = None) -> str:
        """
        返回已填入参数的脚本

        :param params_dict: 用于更新脚本参数的字典，可选参数
        :return: 脚本字符串
        """
        params = self._get_params()
        if params_dict:
            params.update(params_dict)
        content = self._raw[:]
        for k in set(re.findall('''(?<={)[a-zA-Z][a-zA-Z0-9_]*(?=})''', self._raw)):
            content = content.replace('{%s}' % k, str(params[k]))
        return content

    def __add__(self, other):
        if isinstance(other, BaseScript):
            self._raw += other._raw
            self.__dict__.update(other._get_params())
        elif isinstance(other, str):
            self._raw += other
        else:
            raise TypeError("'+' operator only supported with instances of 'script' and 'str'")
        return self

    def __repr__(self):
        params = self._get_params()
        return f"Script type: {self._type} | Length: {len(self._raw)} | Parameter Count: {len(params)}" \
               f"\n({', '.join(['%s=%s' % item for item in params.items()])})" if len(params) > 0 else ""

    def __str__(self):
        return self._raw


class FluentScript(BaseScript):
    """
    读取Fluent脚本文件（.jou）并解析参数
    """
    # 任务组根路径（case文件所在路径）
    ROOTPATH = ''

    def __init__(self, script_path):
        super(FluentScript, self).__init__('jou')
        self._from_file(Path(script_path))


class FluentScriptTemplate(BaseScript):

    def __init__(self, transient=False):
        super(FluentScriptTemplate, self).__init__('jou')
        self._raw = '''\
rc {case_file}.cas.h5
/define/boundary-conditions/set/pressure-inlet/inlet () p0 n {inlet_p} \
supersonic-or-initial-gauge-pressure n {inlet_p_init} t0 n {inlet_t} q
/solve/initialize/hyb-initialization
/solve/set/ri 1
/file/auto-save/root-name {root_path}
/file/auto-save/data-frequency {save_freq}\n''' + '''\
/solve/set/time-step {time_step}
/solve/dual {iter} {iter_inner}\n''' if transient else '''\
/solve/iterate {iter}\n'''+ '''\
/file/write-case-data/{case_file_end}.cas.h5
exit
yes'''
        self.root_path = ''
        self.case_file = ''
        self.case_file_end = ''
        self.save_freq = 0
        self.inlet_p = 0
        self.inlet_p_init = 0
        self.inlet_t = 300
        self.time_step = 1e-6
        self.iter = 1000
        self.iter_inner = 30


class CFDPostScript(BaseScript):
    """
    读取CFD Post脚本文件（.cse）并解析参数
    """
    # 子任务根路径（data文件所在位置）
    ROOTPATH = ''
    # data文件名
    DATAFILE = ''

    def __init__(self, script_path):
        super(CFDPostScript, self).__init__('cse')
        self._from_file(Path(script_path))


class BaseQuest:
    """
    任务组基类，每次迭代应当返回一条子任务
    """

    def __init__(self,  exe_path: Path, work_path: Path, thread_n: int, priority: int):
        self.exe_path = exe_path
        self.work_path = work_path
        self.thread_n = thread_n
        self.priority = priority
        self.id = None

    def __lt__(self, other):
        if not isinstance(other, BaseQuest):
            raise TypeError("'<' operator only supported with instances of 'quest'")
        return self.priority < other.priority

    def __iter__(self):
        self._ind = 0
        return self

    def __next__(self):
        raise StopIteration

    def __len__(self):
        return 0


class FluentQuest(BaseQuest):
    """
    Fluent参数化任务组，通过含参数的Fluent脚本文件定义任务的具体操作

    通过迭代创建子任务目录并生成所需文件，迭代完成后可调用收集计算结果
    """

    def __init__(self, fluent_path, script_path, planar_geom=True, thread_n=16, priority=100):
        super(FluentQuest, self).__init__(Path(fluent_path),
                                          Path(os.path.dirname(script_path)),
                                          thread_n,
                                          priority)
        # fluent脚本模板所在路径（所有子任务的根路径）
        self.base_path = self.work_path
        # fluent脚本模板
        self.script = FluentScript(script_path)
        # 参数列表（参数表的列标签）
        self.params = list(self.script._get_params().keys())
        # Fluent脚本保留参数（单独指定）：任务根路径
        self.params.remove('ROOTPATH')
        self.script._update_params({'ROOTPATH': self.base_path})
        # 任务列表（参数表的行）
        self.job_list = set()
        self.solver = '2ddp' if planar_geom else '3ddp'

    def add_params(self, **kwargs):
        """定义全部参数的候选值，并将参数的全部可能组合添加到任务列表中"""
        params = []
        for param in self.params:
            value = kwargs.get(param)
            if value == None:
                raise KeyError("unspecified parameter '%s'" % param)
            params.append(list(value))
        self.job_list.update(product(*params))

    def clear_params(self):
        """清空任务列表"""
        self.job_list.clear()

    def get_result(self, report_file, *pic_file):
        """指定文件名，可将fluent报告文件和图片文件收集至任务根目录"""
        fw = open(self.base_path / 'fluent_result.txt', 'w', encoding='utf-8')
        pic_dir = self.base_path / 'fluent_pictures'
        os.makedirs(pic_dir, exist_ok=True)
        for i, job_dir in enumerate(self._job_dir_list):
            has_result = False
            for file in os.listdir(job_dir):
                path = job_dir / file
                if os.path.isdir(path):
                    continue
                if file == report_file:
                    has_result = True
                    with open(path, 'r', encoding='utf-8') as fr:
                        if fw.tell() == 0:
                            for _ in range(3):
                                line = fr.readline()
                            column_tag = ['index'] + self.params + re.findall('''(?<=")[A-Za-z0-9_-]+(?=")''', line)
                            fw.write(','.join(column_tag) + '\n')
                        while True:
                            _line = fr.readline()
                            if not _line:
                                break
                            line = _line
                        fw.write('%d %s %s' % (i + 1,
                                ' '.join(map(str, self._job_list[i])),
                                line))
                if file in pic_file:
                    cuts = file.split('.')
                    new_file = ''.join(cuts[: -1]) + f'-{job_dir}.' + cuts[-1]
                    utils.copy_file(pic_dir / new_file, path)
            if not has_result:
                if fw.tell() == 0:
                    column_tag = ['index'] + self.params
                    fw.write(','.join(column_tag) + '\n')
                fw.write('%d %s\n' % (i + 1, ' '.join(map(str, self._job_list[i]))))

    def __iter__(self):
        # 计数器归零
        self._ind = 0
        # 每次迭代过程中确保子任务列表不变
        self._job_list = list(self.job_list)
        self._job_list.sort()
        self._job_dir_list = []
        return self

    def __next__(self):
        if self._ind >= len(self._job_list):
            raise StopIteration
        params_value = self._job_list[self._ind]
        self.work_path = self.base_path / '_'.join(map(utils.num2str, params_value))
        os.makedirs(self.work_path, exist_ok=True)
        self.script._update_params(dict(zip(self.params, params_value)))
        script_path = self.script._to_file(self.work_path)
        output_path = self.work_path / 'fluent.out'
        self._job_dir_list.append(self.work_path)
        ret = self._ind, \
            f'"{self.exe_path}" {self.solver} -g -t{self.thread_n} -pinfiniband -mpi=intel -ssh' + \
            f' < {script_path} > {output_path} 2>&1'
        self._ind += 1
        return ret

    def __len__(self):
        return len(self.job_list)

    def __repr__(self):
        info = f"Total {len(self.job_list)} Fluent Jobs ({self.thread_n} threads, {self.solver} solver)\n" + \
               "parameter group:\n\t" + '\t'.join(self.params)
        for job in self.job_list:
            info += ('\n\t' + '\t'.join(map(str, job)))
        return info


class CFDPostQuest(BaseQuest):

    def __init__(self, cfdpost_path, script_path, priority=100):
        super(CFDPostQuest, self).__init__(Path(cfdpost_path),
                                            Path(os.path.dirname(script_path)),
                                            1, priority)
        # cfdpost脚本模板所在路径（所有子任务的根路径）
        self.base_path = self.work_path
        # cfdpost脚本模板
        self.script = CFDPostScript(script_path)
        # 任务列表（data文件列表）
        self.job_list = []
        # 参数列表（每个data文件的文件夹名，即可能参数）
        self.params = []

    def set_datafile(self, prefix='-end.dat.h5'):
        self.job_list.clear()
        self.params.clear()
        for root, dirs, files in os.walk(self.base_path):
            files.sort(reverse=True)
            for f in files:
                if f.endswith(prefix):
                    self.job_list.append(os.path.join(root, f))
                    mid_dir = root.replace('\\', '/').replace(str(self.base_path).replace('\\', '/'), '')[1: ]
                    self.params.append(mid_dir.replace('/', '_'))
                    break

    def get_result(self, res_file, *pic_file):
        """指定文件名，可将cfdpost结果文件和图片文件收集至任务根目录"""
        fw = open(self.base_path / 'cfdpost_result.txt', 'w', encoding='utf-8')
        pic_dir = self.base_path / 'cfdpost_pictures'
        os.makedirs(pic_dir, exist_ok=True)
        for i, data_file_path in enumerate(self._job_list):
            job_dir = Path(os.path.dirname(data_file_path))
            has_result = False
            for file in os.listdir(job_dir):
                file_path = job_dir / file
                if os.path.isdir(file_path):
                    continue
                if file == res_file:
                    has_result = True
                    with open(file_path, 'r', encoding='utf-8') as fr:
                        while True:
                            _line = fr.readline()
                            if not _line:
                                break
                            line = _line
                        fw.write('%d %s %s' % (i + 1, self._params[i], line))
                if file in pic_file:
                    cuts = file.split('.')
                    new_file = ''.join(cuts[: -1]) + f'-{self._params[i]}.' + cuts[-1]
                    utils.copy_file(pic_dir / new_file, file_path)
            if not has_result:
                fw.write('%d %s\n' % (i + 1, self._params[i]))

    def __iter__(self):
        # 计数器归零
        self._ind = 0
        # 每次迭代过程中确保子任务列表不变
        self._job_list = self.job_list.copy()
        self._params = self.params.copy()
        return self

    def __next__(self):
        if self._ind >= len(self._job_list):
            raise StopIteration
        data_file_path = self._job_list[self._ind]
        self.work_path = Path(os.path.dirname(data_file_path))
        self.script._update_params({'ROOTPATH': self.work_path,
                                    'DATAFILE': os.path.basename(data_file_path)})
        script_path = self.script._to_file(self.work_path)
        output_path = self.work_path / 'cfdpost.out'
        ret = self._ind, \
            f'"{self.exe_path}" -batch {script_path} > {output_path} 2>&1'
        self._ind += 1
        return ret

    def __len__(self):
        return len(self.job_list)

    def __repr__(self):
        info = f"Total {len(self.job_list)} CFD-Post Jobs\nfluent data file:"
        for i in range(len(self.job_list)):
            info += '\n\t%s' % self.job_list[i]
        return info


class QuestManager(threading.Thread):
    """
    任务组调度器，用于任务组的提交与管理
    """

    def __init__(self, parallel_n=64):
        super(QuestManager, self).__init__(name='cfd quest manager')
        # 任务组队列（动态排列）：[(instance of quest class, quest index in queue), ...]
        self.quest_list = PriorityQueue()
        # **任务组元信息列表**（静态排列，在列表中的索引与任务组id一致）：
        self.quest_info = []
        # 需删除的任务组id集合
        self.discard_quest = set()
        self.parallel_n = parallel_n
        self.running = False

    def run(self):
        self.running = True
        # 运行列表：[(instance of quest class, instance of Popen), ...]
        running_quest = []
        worker_counter = {}
        free_threads = self.parallel_n
        while self.running:
            try:
                quest, ind = self.quest_list.get(block=False)
            except queue.Empty:
                pass
            else:
                quest_info = self.quest_info[quest.id]
                thread_n = quest_info['thread'] * quest_info['worker']
                if thread_n  > free_threads:
                    self.quest_list.put((quest, ind + self.quest_list.qsize()))
                else:
                    quest_info['start time'] = time.ctime()
                    quest_info['duration'] = -time.time()
                    quest_iter = iter(quest)
                    for _ in range(quest_info['worker']):
                        running_quest.append([quest_iter, None])
                    worker_counter[quest.id] = quest_info['worker']
                    free_threads -= thread_n
            # 中止并移除需删除的任务组
            for ind, (quest, job) in enumerate(running_quest):
                if quest.id in self.discard_quest:
                    if job != None:
                        job.kill()
                    running_quest[ind] = None
                    worker_counter[quest.id] -= 1
                    if worker_counter[quest.id] == 0:
                        quest_info = self.quest_info[quest.id]
                        quest_info['end time'] = time.ctime()
                        quest_info['duration'] += time.time()
                        quest_info['state'] = 'D'
            self.discard_quest.clear()
            running_quest = [item for item in running_quest if item is not None]
            # 检测子任务状态，并进行任务提交等操作
            for ind, (quest, job) in enumerate(running_quest):
                if job == None or job.poll() != None:
                    # 对于初次提交或者停止运行的任务组，提交运行下一条子任务
                    quest_info = self.quest_info[quest.id]
                    try:
                        job_i, cmd = next(quest)
                        quest_info['state'] = 'R:%s/%s' % (job_i + 1, len(quest))
                        #print(shlex.split(cmd.replace('\\', '/')))
                        p = subprocess.Popen(cmd.replace('\\', '/'),
                                             shell=True, cwd=quest.work_path,
                                             universal_newlines=None, stderr=subprocess.STDOUT)
                        running_quest[ind][1] = p
                    except StopIteration:
                        # 对于运行完成的子任务，从运行列表中删除并释放进程
                        running_quest[ind] = None
                        worker_counter[quest.id] -= 1
                        free_threads += quest.thread_n
                        if worker_counter[quest.id] == 0:
                            # 若全部子任务都已执行完毕，结束计时并将任务组标记为完成
                            quest_info['end time'] = time.ctime()
                            quest_info['duration'] += time.time()
                            quest_info['state'] = 'E'
            running_quest = [item for item in running_quest if item is not None]
            time.sleep(10)
        for quest, job in running_quest:
            if job != None:
                job.kill()
            running_quest[ind] = None
            worker_counter[quest.id] -= 1
            if worker_counter[quest.id] == 0:
                quest_info = self.quest_info[quest.id]
                quest_info['end time'] = time.ctime()
                quest_info['duration'] += time.time()
                quest_info['state'] = 'D'
        running_quest = [item for item in running_quest if item is not None]

    def stop(self):
        """关闭任务调度器"""
        self.running = False

    def submit(self, *quests, worker_n=1):
        """批量提交任务，可指定用于执行任务组的进程数"""
        for quest in quests:
            if not isinstance(quest, BaseQuest):
                print("Unknown quest type: %s" % type(q))
                continue
            quest.id = len(self.quest_info)
            quest_meta = {'quest': os.path.basename(quest.exe_path),
                          'path': str(quest.work_path),
                          'thread': quest.thread_n,
                          'priority': quest.priority,
                          'worker': worker_n,
                          'detail': repr(quest),
                          'jobs': [],
                          'start time': '',
                          'end time': '',
                          'duration': 0,
                          'state': 'Q'
                          }
            # 实际开始计算前进行一轮迭代，用于生成子任务文件并收集信息
            for _, cmd in quest:
                quest_meta['jobs'].append(cmd)
            self.quest_info.append(quest_meta)
            self.quest_list.put((quest, quest.id))

    def delete(self, quest_id):
        """删除特定任务组"""
        quest_id -= 1
        if quest_id >= len(self.quest_info) or self.quest_info[quest_id]['state'] == 'E':
            print("Invalid quest id")
        else:
            self.discard_quest.add(quest_id)

    def state(self):
        """统计队列状态，并列出所有任务组的详细信息"""
        print("{:>4s} {:>16s} {:>8s} {:>10s} {:>16s} {:>12s}".format(
            "#", "Quest", "Jobs", "Threads", "Duration ", "State "))
        for ind, info in enumerate(self.quest_info, start=1):
            if info['worker'] > 1:
                quest = info['quest'] + ' x%d' % info['worker']
            else:
                quest = info['quest']
            if info['duration'] < 0:
                duration = utils.sec2str(time.time() + info['duration'])
            elif info['duration'] > 0:
                duration = utils.sec2str(info['duration'])
            else:
                duration = '-'
            print(f"{ind:>4d} {quest:>16s} {len(info['jobs']):>8d} {info['thread']:>10d}"
                  f"{duration:>16s} {info['state']:>12s}")

    def info(self, quest_id):
        """列出单个任务的详细信息"""
        print(utils.dict_disp(self.quest_info[quest_id - 1]))


if __name__ == '__main__':

    # 启动任务调度队列，并设置总进程数
    q = QuestManager(parallel_n=24)
    q.start()

    # 根据参数化脚本创建fluent任务组
    task_1 = FluentQuest('/public/software/ansys_inc211/v211/fluent/bin/fluent',
                         '/public/home/liugq/Fluent/plug/allParam/origin/ejector.jou',
                         planar_geom=True, thread_n=16)
    # 通过输入参数的笛卡尔积创建子任务
    task_1.add_params(inlet_p=[3e6, 5e6, 8e6, 12e6, 15e6],
                      atmo_p=[101325, 89875.3, 67805.7, 54019.4, 26497.5,
                              2549.13, 79.7782, 2.06784, 0.0320116],
                      inlet_t=[3500])
    # 任务组提交
    q.submit(task_1)
    # 自动从子任务的报告文件收集计算结果
    task.get_result('report-def-0-rfile.out')

    # 根据参数化脚本创建cfdpost任务组
    task_2 = CFDPostQuest('/public/software/ansys_inc211/v211/CFD-Post/bin/cfdpost',
                          '/public/home/liugq/Fluent/plug/allParam/plug.cse')
    # 通过搜索目录下的data文件来创建子任务
    task_2.set_datafile(prefix='-end.dat.h5')
    # 由于cfdpost为单进程，可通过分配多条队列进程来实现并行
    q.submit(task_2, worker_n=16)
    # 自动从收集整理子任务的计算结果和云图等
    task.get_result('result.txt', 'machNumber.png')

    # 查看队列状态
    q.state()
    # 查看队列中特定任务的详细信息
    q.info(0)


