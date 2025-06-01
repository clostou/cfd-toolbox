import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfd_toolbox.submit import *


if __name__ == '__main__':
    q = QuestManager(parallel_n=16)
    q.start()

    '''
    task_1 = FluentQuest('/public/software/ansys_inc211/v211/fluent/bin/fluent',
                         '/public/home/liugq/Fluent/experiment/new/4_50/ejector.jou',
                         planar_geom=False)
    task_1.add_params(inlet_p=[3e6, 5e6, 8e6],
                      inlet_t=[3500],
                      outlet_p=[0],
                      outlet_t=[300])
                         
    task_2 = CFDPostQuest('/public/software/ansys_inc211/v211/CFD-Post/bin/cfdpost',
                          '/public/home/liugq/Fluent/plug/allParam/cutoff-1_3/plug.cse')
    task_2.set_datafile()'''

    #q.submit(task_1, task_2, task_3, worker_n=8)


