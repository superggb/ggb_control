import numpy as np 
import tensorflow as tf
from utils import Timer
from tf_utils import *

from natsort import natsorted


class Vnet(object):

    def __init__(self,args):
        self.im_width = args.im_width
        self.im_height =args.im_height
        self.im_channels = args.im_channels
        self.state_indexs = range(args.state_indexs)  #state的长度，range(0,7)
        self.vision = None
        self.state = None
        self.action = None
        self.keep_prob = args.keep_prob
        self.activation_fn = tf.nn.relu  # by default, we use relu
        self.norm_type = args.norm_type  # 使用batch_norm or layer_norm
        self.conv_layers_num = args.conv_layers_num  # 卷积层数目
        self.conv_weights_init = args.conv_weights_init  # 卷积层权值初始化器
        self.conv_filters_num = args.conv_filters_num  # 每个卷积层使用的卷积核数目
        self.conv_filters_size = args.conv_filters_size  # 每个卷积层使用的卷积核尺寸
        self.conv_strides = args.conv_strides  # 每个卷积层使用的步长strides
        self.fc_layers_num = args.fc_layers_num # 全连接层数
        self.fc_layers_size = args.fc_layers_size
        self.fc_bt_dim = args.fc_bt_dim

        self.enable_1th_fc_bt = args.enable_1th_fc_bt  #是否增加参数偏置项
        self.enable_spatial_softmax = args.enable_spatial_softmax  #在原作中对push任务有空间softmax操作
        self.enable_dropout = args.enable_dropout
        self.enable_l1_loss = args.enable_l1_loss
        self.enable_1th_conv_bt = args.enable_1th_conv_bt
        self.enable_all_fc_bt = args.enable_all_fc_bt

        self.weights = None
        self.sorted_weight_keys = None
        self.conv_out_size = 0   # conv层输出的特征维度
        self.conv_out_size_final = 0  # conv层最终输出的特征维度
        self.output_dim =args.output_dim

        self.loss_multiplier = args.loss_multiplier
        self.train_lr = args.train_lr
        self.total_loss = []
        self.predict_out = None
    

    def init_network(self,graph,input_tensors=None,mode="train"):
        """
        初始化网络计算图，并定义train_op
        """
        with graph.as_default():
            with Timer("Build TF network("+mode+")"):
                if mode == 'train':
                    res = self.build_model(mode="train")
                else:
                    res = self.build_model(mode="test")

            predict_out,predict_loss = res

            if mode == 'train':
                self.total_loss = tf.reduce_sum(predict_loss)
                self.predict_out = predict_out
                self.train_op = tf.train.AdamOptimizer(self.train_lr).minimize(self.total_loss)
                summ = []
                summ.append(tf.summary.scalar("Loss of th update in mode"+mode,self.total_loss))
                self.train_summary_op = tf.summary.merge(summ)
            else:
                self.total_loss = tf.reduce_sum(predict_loss)
                self.predict_out = predict_out
                
    # 这个代码用于替换掉原init_network函数，以实现分iteration处理数据
    # 传入此函数的loss会从 tensor 变成 list 所以暂时放弃这个写法
    # def train(self,graph,total_loss,iteration):
    #     """
    #     此部分定义train_op，
    #     """
    #     with graph.as_default():
    #         print(type(self.total_loss))
    #         input('chekkkk')
    #         train_op = self.train_op = tf.train.AdamOptimizer(self.train_lr).minimize(self.total_loss)
    #     return train_op


    def build_model(self,input_tensors=None,mode="train"):
        '''
        return: predict_out,predict_loss
        '''

        # if input_tensors is None:
        #     pass  #此处用于实际环境测试用
        # else:
        #     self.vision =  vision = tf.stack(input_tensors['vision'])
        #     self.state = state = tf.stack(input_tensors['states'])
        #     self.action = actions = tf.stack(input_tensors['actions'])

        self.vision = vision = tf.placeholder(tf.float32,name='vision')
        self.state = state = tf.placeholder(tf.float32,name='state')
        self.action = action = tf.placeholder(tf.float32,name='action')

        # input_all = tf.concat(axis=2,values=[state,vision]) # 此处是把输入全部铺平，待考虑

        with tf.variable_scope("model",reuse=None) as train_scope:

            if self.weights is None:
                # 一次性创建整个网络参数
                self.weights = weights = self.build_weights() 
                self.sorted_weight_keys = natsorted(self.weights.keys()) #对key进行自然排序
            else:
                train_scope.reuse_variables()
                weights = self.weights

            loss_multiplier = self.loss_multiplier
            enable_l1_loss = self.enable_l1_loss

            predict_out = self.conv_forward(vision,state,weights,mode=mode)
            predict_loss = euclidean_loss_layer(predict_out,action,multiplier=loss_multiplier,
                                                use_l1=enable_l1_loss)
            # predict_loss = tf.reduce_mean(predict_loss,name='eucilidean_loss')
            
            return [predict_out,predict_loss]

    def conv_forward(self,vision_input,state_input,weights,mode="train"):
        """
        卷积操作
        : param vision_input: shape=(?,h,w,c)
        : param state_input: shape=(?,7)
        """

        # 这里有个增加参数偏置项的操作，即加入一个和weight同大小的0矩阵
        if self.enable_1th_fc_bt:
            flatten_image = tf.reshape(vision_input, [-1, self.im_height*self.im_width*self.im_channels])
            fc_context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), list(range(self.fc_bt_dim))))
            fc_context += weights["fc_bt_1"]

        
        decay = 0.9
        norm_type = self.norm_type
        conv_layers_num = self.conv_layers_num
        conv_strides = self.conv_strides
        enable_dropout = self.enable_dropout
        keep_prob = self.keep_prob

        is_training = mode == "train"

        conv_layer = vision_input

        if self.enable_1th_conv_bt:
            img_context = tf.zeros_like(conv_layer)  # 只是为了保证shape?
            img_context += weights["conv_bt_1"]
            conv_layer = tf.concat(axis=3, values=[conv_layer, img_context])


        for i in range(conv_layers_num):
            conv_layer = norm(conv2d(img=conv_layer, w=weights["conv_weights_%d" % (i + 1)],
                                     b=weights["conv_bias_%d" % (i+1)],strides=conv_strides[i]),
                                    norm_type=norm_type,decay = decay,id = i,
                                    is_training=is_training,activation_fn=self.activation_fn)
            
            if enable_dropout:
                dropout(conv_layer,keep_prob=keep_prob,is_training=is_training,name="conv_dropout_%d"%(i+1))

        if self.enable_spatial_softmax:
            pass #空间softmax先不加入
        else:
            conv_flatten_out = tf.reshape(conv_layer,[-1,self.conv_out_size])
        
        fc_input = tf.add(conv_flatten_out,0) # 此步作用未知
        fc_input = tf.to_float(fc_input)
        state_input = tf.to_float(state_input)

        return self.fc_forward(fc_input,weights,state_input = state_input,mode=mode)


    def fc_forward(self,fc_input,weights,state_input=None,mode='train'):
        is_training = mode == "train"
        fc_output = tf.add(fc_input,0) # 此步作用未知

        if state_input is not None:
            fc_output = tf.concat(axis=1,values=[fc_output,state_input]) # 连接卷积输出和状态输入
        

        for i in range(self.fc_layers_num+1):
            if i > 0 and self.enable_all_fc_bt:
                context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(fc_output)), list(range(self.fc_bt_dim))))
                context += weights['fc_bt_%d' % (i+1)]
                fc_output = tf.concat(axis=1, values=[fc_output, context])

            # 全连接计算
            fc_output = tf.matmul(fc_output, weights["fc_weights_%d" % (i+1)]) + weights["fc_bias_%d" % (i+1)]
            if i!= self.fc_layers_num:
                fc_output = self.activation_fn(fc_output)
                if self.enable_dropout:
                    fc_output = dropout(fc_output,keep_prob=self.keep_prob,is_training=is_training,name="fc_dropout_%d"%(i+1))
        
        return fc_output
                    

    # 用于每一层卷积计算后的标准化，包括激活
    def norm(self,layer, norm_type='batch_norm', decay=0.9, id=0, is_training=True, activation_fn=tf.nn.relu, prefix='conv'):
        if norm_type != 'batch_norm' and norm_type != 'layer_norm':
            return tf.nn.relu(layer)
        with tf.variable_scope('norm_layer_%s_%d' % (prefix, id)) as vs:
            if norm_type == 'batch_norm':
                if is_training:
                    try:
                        layer = tf.contrib.layers.batch_norm(layer, is_training=True, center=True,
                            scale=False, decay=decay, activation_fn=activation_fn, updates_collections=None, scope=vs)
                    except ValueError:
                        layer = tf.contrib.layers.batch_norm(layer, is_training=True, center=True,
                            scale=False, decay=decay, activation_fn=activation_fn, updates_collections=None, scope=vs, reuse=True)
                else:
                    layer = tf.contrib.layers.batch_norm(layer, is_training=False, center=True,
                        scale=False, decay=decay, activation_fn=activation_fn, updates_collections=None, scope=vs, reuse=True)
            elif norm_type == 'layer_norm':
                try:
                    layer = activation_fn(tf.contrib.layers.layer_norm(layer, center=True,
                                                                    scale=False, scope=vs))  # updates_collections=None
                except ValueError:
                    layer = activation_fn(tf.contrib.layers.layer_norm(layer, center=True,
                                                                    scale=False, scope=vs, reuse=True))
            else:
                raise NotImplementedError('Other types of norm not implemented.')
            return layer

    # 用于每一层后的dropout
    def dropout(self,layer, keep_prob=0.9, is_training=True, name=None):
        if is_training:
            return tf.nn.dropout(layer, keep_prob=keep_prob, name=name)
        else:
            return tf.add(layer, 0, name=name)

                
    def build_weights(self):

        weights = {}
        conv_layers_num = self.conv_layers_num  # 卷积层数目, 3 for reach
        conv_filters_size = self.conv_filters_size  # 每个卷积层使用的卷积核尺寸, [3, 3, 3] for reach
        conv_filters_num = self.conv_filters_num  # 每个卷积层使用的卷积核数目, [64, 64, 64] for reach
        conv_weights_init = self.conv_weights_init  # 卷积层权值初始化器, xavier for reach
        im_width = self.im_width  # 图片宽度
        im_height = self.im_height  # 图片高度
        im_channels = self.im_channels  # 图片通道数目
        enable_spatial_softmax = self.enable_spatial_softmax  # 是否在最后一个卷积层使用spatial_softmax

        if enable_spatial_softmax:
            self.conv_out_size = int(conv_filters_num[-1] * 2)  # 空间softmax
        else:
            # 默认使用SAME填充方式, 则最终图像大小只与stride有关
            down_sample_factor = 1
            for stride in self.conv_strides:
                down_sample_factor *= stride[1]
            self.conv_out_size = int(np.ceil(im_width/(down_sample_factor))) * \
                                    int(np.ceil(im_height/(down_sample_factor))) * \
                                    conv_filters_num[-1]    # np.ceil向上取整  此部分计算了卷积的输出大小
            # print(self.conv_out_size)   # 在enable参数均为false时， 卷积输出size为 7680


        # conv bt weights
        fan_in = im_channels  # 每一层输出的通道维度, 会影响后面filter的通道维度
        # 第一个卷积层+bt，就是加一个需要学习的同输入图片大小一样的参数输入
        if self.enable_1th_conv_bt:
            fan_in += im_channels
            conv_bt_name = "conv_bt_1"
            weights[conv_bt_name] = safe_get(conv_bt_name, initializer=tf.zeros([im_height, im_width, im_channels],
                                                                                dtype=tf.float32))
            weights[conv_bt_name] = tf.clip_by_value(weights[conv_bt_name], 0., 1.)  # clip_by_value使数值大小控制在min和max之间，此处控制在0～1之间

        # conv weights
        for i in range(conv_layers_num):
            # filter_size * filter_size * channel' * filter_num, 这里channel'不一定等于channel   3*3*3*30
            weights_shape = [conv_filters_size[i], conv_filters_size[i], fan_in, conv_filters_num[i]]  
            conv_weight_name = "conv_weights_%d" % (i + 1)
            if conv_weights_init == "xavier":
                weights[conv_weight_name] = init_conv_weights_xavier(weights_shape, name=conv_weight_name)
            elif conv_weights_init == "random":
                weights[conv_weight_name] = init_weights(weights_shape, name=conv_weight_name)
            else:
                raise NotImplementedError("Need choose a conv_weights_init param!")
            conv_bias_name = "conv_bias_%d" % (i+1)
            weights[conv_bias_name] = init_bias([conv_filters_num[i]], name=conv_bias_name)
            fan_in = conv_filters_num[i]

        # fc bt weights
        in_shape = self.conv_out_size + len(self.state_indexs)
        if self.enable_1th_fc_bt:
            in_shape += self.fc_bt_dim  # 此处长度加上了卷积的输出和状态的输入维度
            fc_bt_name = 'fc_bt_1'
            weights[fc_bt_name] = safe_get(fc_bt_name, initializer=tf.zeros([self.fc_bt_dim], dtype=tf.float32))
        self.conv_out_size_final = in_shape  # 最终conv输出的特征维度
        # fc weights
        fc_weights = self.build_fc_weights(in_shape)
        weights.update(fc_weights)  # 将dc_weights添加到之前的dict中
        return weights

    def build_fc_weights(self, in_dim):
        fc_layers_num = self.fc_layers_num + 1  # number of fc layers, add output layer
        fc_layers_size = self.fc_layers_size  # size of each fc layers
        if len(fc_layers_size)+1 != fc_layers_num:
            raise ValueError("The length of fc_layers_num param must equal to fc_layers_num param!")
        fc_layers_size.append(self.output_dim)  # 添加输出层
        weights = {}
        in_shape = in_dim
        for i in range(fc_layers_num):
            if i > 0 and self.enable_all_fc_bt:
                in_shape += self.fc_bt_dim
                weight_name = "fc_bt_%d" % (i+1)
                weights[weight_name] = init_bias([self.fc_bt_dim], name=weight_name)
            fc_weights_name = "fc_weights_%d" % (i+1)
            fc_bias_name = "fc_bias_%d" % (i+1)
            weights[fc_weights_name] = init_weights([in_shape, fc_layers_size[i]], name=fc_weights_name)
            weights[fc_bias_name] = init_bias([fc_layers_size[i]], name=fc_bias_name)
            in_shape = fc_layers_size[i]
        return weights
