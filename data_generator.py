from rlbench import ObservationConfig
from rlbench.action_modes import ActionMode
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

import numpy as np 
import argparse
from matplotlib.image import imread
import pickle




class DataGenerator(object):
    '''
    此类用来生成网络可用的数据，一次task为一个输出集包括：
    1.按帧排列的rgb图像数组，帧为不定长
    2.手臂运动中间过程的state信息
    3.每一帧图像对应的action
    '''
    def __init__(self,args):
        self.mode = args.mode



    def load_data(self,args,itr):
        '''
        加载视觉数据文件并处理成一个向量（T，-1），每一个itr生成一个task的量
        '''

        dir = args.data_root+'/episode'+str(itr)
        with open(dir+"/low_dim_obs.pkl",'rb') as fo:
            low_dim_obs = pickle.load(fo,encoding='bytes')
        batch_images = [obs.front_rgb for obs in low_dim_obs]
        task_length = len(batch_images) # 获取任务长度
        images = []
        for i in range(task_length):
            image = imread(dir+'/front_rgb/'+str(i)+'.png')
            images.append(image)

        #获得的actions和state都是list类型，里面包含task_length长的数组
        ground_truth_actions = [obs.joint_velocities for obs in low_dim_obs]
        states = [obs.joint_positions for obs in low_dim_obs]

        return images,states,ground_truth_actions
       
        




    
