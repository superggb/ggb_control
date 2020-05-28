
import os
import pdb
import copy
import random
import argparse
import tensorflow as tf
import numpy as np
from utils import Timer

from data_generator import DataGenerator
from vnet import Vnet
from rlbench_test import act



FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode', choices=['train', 'test','act'])
# Hyper-parameters 参数名前加--表示可选参数

FLAGS.add_argument('--im_width',type=int,default=128, help="image width")
FLAGS.add_argument('--im_height',type=int,default=128,  help="image height")
FLAGS.add_argument('--im_channels',type=int,default=3, help="image channels")
FLAGS.add_argument('--data_root',type=str,default='~/RLBench-master/tools/data/reach_target/variation0/episodes', help="root of data stored")
FLAGS.add_argument('--random_seed',type=int,default=0, help="random seed")
FLAGS.add_argument('--conv_layers_num',type=int,default=3, help="number of conv layers")
FLAGS.add_argument('--total_itr',type=int, help="train itr")
FLAGS.add_argument('--state_indexs',type=int,default=7, help="状态参数的长度")
FLAGS.add_argument('--keep_prob',type=float, help="keep probility when drop")
FLAGS.add_argument("--norm_type",type=str, help="batch_norm或者layer_norm")
FLAGS.add_argument("--conv_weights_init",type=str,default='random',help="卷积层权值初始化器")
FLAGS.add_argument("--conv_filters_num",default="[30, 30, 30]")     # 此条待考虑 # 每个卷积层使用的卷积核数目
FLAGS.add_argument("--conv_filters_size",default="[3, 3, 3]")            # 每个卷积层使用的卷积核尺寸
FLAGS.add_argument("--conv_strides",default="[[1, 2, 2, 1]-[1, 2, 2, 1]-[1, 2, 2, 1]]")  # 每个卷积层使用的步长strides   需要使用stringtolist进行转化
FLAGS.add_argument("--fc_layers_num",type=int,default=2)                 # 全连接层数
FLAGS.add_argument("--fc_layers_size",default="[200, 200]")   # 隐藏层尺寸
FLAGS.add_argument("--fc_bt_dim",type=int,default=10)         #fc层的偏置项维度
FLAGS.add_argument("--output_dim",type=int ,default=7)
FLAGS.add_argument("--loss_multiplier",type=int)        #与损失相乘的数 100 reach 50 push
FLAGS.add_argument("--train_lr",type=float,default=0.001)

#由于arg无法解析bool参数，所以全由default控制变量
FLAGS.add_argument("--enable_1th_fc_bt",type=bool,default=False)            #是否增加参数偏置项
FLAGS.add_argument("--enable_spatial_softmax",type=bool,default=False)      #在原作中对push任务有空间softmax操作
FLAGS.add_argument("--enable_dropout",type=bool,default=True)        
FLAGS.add_argument("--enable_l1_loss",type=bool,default=False)
FLAGS.add_argument("--enable_1th_conv_bt",type=bool,default=False)
FLAGS.add_argument("--enable_all_fc_bt",type=bool,default=False)
FLAGS.add_argument("--enable_resume",type=bool,default=True)

FLAGS.add_argument("--save_itr",type=int,default=50)
FLAGS.add_argument("--save_dir",default="checkpoint")
FLAGS.add_argument("--restore_itr",type=int,default=0)


def train(graph,model,saver,sess,data_generator,args):
    total_itr = args.total_itr
    total_loss = []
    save_itr = args.save_itr
    save_dir = args.save_dir+'/model'
    restore_itr = args.restore_itr

    if restore_itr==0:
        trainning_range = range(total_itr)
    else:
        trainning_range = range(restore_itr+1,total_itr)
        print('restore model from ----- itr%d'%restore_itr)

    for itr in trainning_range:
        vision,state,action = data_generator.load_data(args,itr)
        feed_dict = {
            model.vision:vision,
            model.state:state,
            model.action:action
        }
        run_ops = [model.train_op]
        run_ops.extend([model.train_summary_op,model.total_loss,model.predict_out])

        with graph.as_default():
            # res的结果为run_ops对应的，所以res[-1]为action,res[-2]为loss
            res = sess.run(run_ops,feed_dict=feed_dict)     
        total_loss.append(res[-2])        

        print("iteration --------%d---loss is---%.3f"%(itr,np.mean(total_loss)))

        if itr != 0 and (itr % save_itr == 0):
            with graph.as_default():
                save_path = save_dir + '_%d' % itr
                saver.save(sess,save_path)
                print("Saved model to "+save_path+".")


def test(graph,model,sess,data_generator,args):
    total_itr = args.total_itr
    total_loss = []
    for itr in range(total_itr):
        vision,state,action = data_generator.load_data(args,itr)
        feed_dict = {
            model.vision:vision,
            model.state:state,
            model.action:action
        }
        run_ops = [model.total_loss]

        with graph.as_default():
            res = sess.run(run_ops,feed_dict=feed_dict)
        
        total_loss.append(res)
        print("iteration ------ %d------loss is: %.3f, average loss is: %.3f"%(itr,res[0],np.mean(total_loss)))





def string2list(str, recursive_num=1):
    # print(str)
    if recursive_num == 1:
        return list(map(int, str[1:-1].split(",")))
    elif recursive_num == 2:
        return list(map(string2list, str[1:-1].split("-")))

def main():

    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    # 对一些str参数转list
    args.conv_filters_num = string2list(args.conv_filters_num)
    args.conv_filters_size = string2list(args.conv_filters_size)
    args.conv_strides = string2list(args.conv_strides,2)
    args.fc_layers_size = string2list(args.fc_layers_size)


    # 设置随机数种子 待进入网络测试时使用
    # tf.set_random_seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # random.seed(args.random_seed)

    
    # 构建Sess
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    data_generator = DataGenerator(args)
    # images,states,actions = data_generator.load_data(args,0) # 返回三个list对象，长度相同且为动作长度
    model = Vnet(args)
    model.init_network(graph,mode=args.mode)
    with graph.as_default():
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    if args.enable_resume:
        with Timer("load latest model from file..."):
            model_file = tf.train.latest_checkpoint(args.save_dir)
            if args.restore_itr > 0:
                model_file = model_file[:model_file.index('model')]+'model_'+str(args.restore_itr)
            if model_file:
                with graph.as_default():
                    print(model_file)
                    saver.restore(sess,model_file)

    if args.mode == 'train':
        train(graph,model,saver,sess,data_generator,args)
    elif args.mode == 'test':
        test(graph,model,sess,data_generator,args)
    else:
        act(graph,model,sess)



if __name__=='__main__':
    main()