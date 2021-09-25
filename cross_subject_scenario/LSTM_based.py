#coding=utf-8

import tensorflow as tf
import numpy as np
import pickle
import normalization
import cv2
import math
import os
import random
import time
import pylab


class Seg_network():
    def __init__(
            self,
            train_data = None,
            test_data=None,
            test_no_trainSet_data=None,
            activity_num=11,
            interval_len=20,
            batch_size=1,
            learning_rate = 0.000005,#pre-training stage: 0.00005, fine-tuning stage:0.000005
            weight_decay=0.00001,
            con_params=0.0,
            training_epochs = 200,
            is_train=False,
            show_picture=False
                 ):

        self.train = train_data
        self.test_data=test_data
        self.test_no_trainSet_data=test_no_trainSet_data
        self.activity_num = activity_num
        self.interval_len=interval_len
        self.batch_size=batch_size
        self.lr=learning_rate
        self.weight_decay=weight_decay
        self.con_params = con_params
        self.is_train = is_train
        self.training_epochs = training_epochs
        self.show_picture = show_picture

        self.build()
        print ("Neural networks build!")
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)

        if is_train is False:
            self.saver.restore(self.sess, "./Params_cross_subject_scenario/envir1/LSTM_based/params_300/train.ckpt")
            # self.saver.restore(self.sess, "./Params_cross_subject_scenario/envir2/LSTM_based/params_300/train.ckpt")

            print('load params sucess')
            print('0.5:')
            self.test_Seg(self.test_no_trainSet_data, 'person not in trainset test', 0.50)
            print('0.75:')
            self.test_Seg(self.test_no_trainSet_data, 'person not in trainset test', 0.75)


    def build(self):

            self.input = tf.placeholder(tf.float32, shape = [None, 90, 20, 2], name='csi_input')
            self.tag = tf.placeholder(tf.float32, shape = [None,self.activity_num], name ='activity_flag')
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
            self.dimention=tf.placeholder(tf.int32,name='dimention')
            self.AllWeight=[]

            self.resnet_output=self.resnet(self.input)
            self.resnet_output = tf.reshape(self.resnet_output, [-1, 12*3*128])
            self.state_vector=self.active_founction(self.resnet_output,int(self.resnet_output.get_shape()[-1]))

            self.features = tf.reshape(self.state_vector, [-1,self.dimention, 2048])
            print(self.features.shape)
            self.channel_list=[1024,512,256,128,64,self.activity_num]

            self.pred, self.final_states = self.LSTM_network(self.features, self.channel_list)
            self.output = tf.reshape(self.pred, [1, -1, self.activity_num])
            self.pro_output = tf.nn.softmax(tf.squeeze(self.output))
            self.tempTarget = tf.reshape(self.tag, [-1, self.dimention, self.activity_num])

            self.cons_loss = self.punish_overSeg(self.pro_output, self.tag)

            with tf.variable_scope('loss'):
                """
                soft_max cross entropy
                """

                samples = tf.cast(self.dimention, tf.float32)
                self.L2_Regul = tf.reduce_sum([tf.nn.l2_loss(x) * self.weight_decay for x in self.AllWeight]) / samples

                """ follow loss is uses in the pretraining stage"""
                # self.loss = tf.reduce_sum(
                #     tf.nn.softmax_cross_entropy_with_logits(labels=self.tempTarget, logits=self.output)) + self.L2_Regul

                """ follow loss is uses in the fine-tuning stage"""
                self.loss=self.cons_loss+self.L2_Regul

            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def resnet(self,input):
        layer_1=self.make_layer_one(input,2,2,3,1,1)
        layer_2=self.make_layer_two(layer_1,2,2,3,1,2)
        renet_1=self.resnet_same_connection(input,layer_2)
        print(renet_1.shape)

        layer_3 = self.make_layer_one(renet_1, 2, 8, 3, 2, 3)
        layer_4 = self.make_layer_two(layer_3, 8, 8, 3, 1, 4)
        renet_2 = self.resnet_conv_connection(renet_1, layer_4,2,8,3,2,2)
        print(renet_2.shape)


        layer_5 = self.make_layer_one(renet_2, 8, 8, 3, 1, 5)
        layer_6 = self.make_layer_two(layer_5, 8, 8, 3, 1, 6)
        renet_3 = self.resnet_same_connection(renet_2, layer_6)
        print(renet_3.shape)

        layer_7 = self.make_layer_one(renet_3, 8, 32, 3, 2, 7)
        layer_8 = self.make_layer_two(layer_7, 32, 32, 3, 1, 8)
        renet_4 = self.resnet_conv_connection(renet_3, layer_8, 8, 32, 3, 2, 4)
        print(renet_4.shape)



        layer_9 = self.make_layer_one(renet_4, 32, 32, 3, 1, 9)
        layer_10 = self.make_layer_two(layer_9, 32, 32, 3, 1, 10)
        renet_5 = self.resnet_same_connection(renet_4, layer_10)
        print(renet_5.shape)

        layer_11 = self.make_layer_one(renet_5, 32, 128, 3, 2, 11)
        layer_12 = self.make_layer_two(layer_11, 128, 128, 3, 1, 12)
        renet_6 = self.resnet_conv_connection(renet_5, layer_12, 32, 128, 3, 2, 6)
        print(renet_6.shape)

        return renet_6

    def active_founction(self,input,fin):
        with tf.variable_scope('activation'):
            w_initializer = tf.random_normal_initializer(0., 1./tf.sqrt(fin/2.))
            b_initializer = tf.constant_initializer(0.0)
            W_e_conv = tf.get_variable('w', [12*3*128,2048], initializer=w_initializer)
            b_e_conv = tf.get_variable('b', [2048, ], initializer=b_initializer)
            state = tf.nn.relu(tf.matmul(input, W_e_conv) + b_e_conv)
            state = tf.layers.batch_normalization(state)
            state = tf.nn.relu(state)
            self.AllWeight.append(W_e_conv)
        return state


    def make_layer_one(self, input_x, input_channel, output_channel, kenel, stride, layer_index):
        block_name = 'layer' + str(layer_index)
        # print(input_x.get_shape(),11111111111)
        fin=int(input_x.get_shape()[1])*int(input_x.get_shape()[2])*int(input_x.get_shape()[3])
        with tf.variable_scope(block_name):
            w_initializer = tf.random_normal_initializer(0., 1./tf.sqrt(fin/2.))
            b_initializer = tf.constant_initializer(0.0)
            W_e_conv = tf.get_variable('w', [kenel, kenel, input_channel, output_channel], initializer=w_initializer)
            b_e_conv = tf.get_variable('b', [output_channel, ], initializer=b_initializer)
            conv_temp =  tf.add(tf.nn.conv2d(input_x, W_e_conv,
                                     strides=[1, stride, stride, 1], padding='SAME'),b_e_conv)
            conv_temp = tf.layers.batch_normalization(conv_temp)
            conv_output = tf.nn.relu(conv_temp)
            self.AllWeight.append(W_e_conv)
        return conv_output

    def make_layer_two(self, input_x, input_channel, output_channel, kenel, stride, layer_index):
        block_name = 'layer' + str(layer_index)
        fin=int(input_x.get_shape()[1])*int(input_x.get_shape()[2])*int(input_x.get_shape()[3])
        with tf.variable_scope(block_name):
            w_initializer = tf.random_normal_initializer(0., 1./tf.sqrt(fin/2.))
            b_initializer = tf.constant_initializer(0.0)
            W_e_conv = tf.get_variable('w', [kenel, kenel, input_channel, output_channel], initializer=w_initializer)
            b_e_conv = tf.get_variable('b', [output_channel, ], initializer=b_initializer)
            conv_output = tf.add(tf.nn.conv2d(input_x, W_e_conv,
                                            strides=[1, stride, stride, 1], padding='SAME'), b_e_conv)
            self.AllWeight.append(W_e_conv)
        return conv_output

    def resnet_same_connection(self,before_input,after_input):
        same_add = tf.add(before_input, after_input)
        same_result_temp=tf.layers.batch_normalization(same_add )
        same_result = tf.nn.relu(same_result_temp)
        return same_result

    def resnet_conv_connection(self,before_input,after_input,input_channel,output_channel, kenel, stride, layer_index):
        block_name='add_layer'+str(layer_index)
        fin=int(before_input.get_shape()[1])*int(before_input.get_shape()[2])*int(before_input.get_shape()[3])
        with tf.variable_scope(block_name):
            w_initializer = tf.random_normal_initializer(0., 1./tf.sqrt(fin/2.))
            W_e_conv = tf.get_variable('w', [kenel, kenel, input_channel, output_channel], initializer=w_initializer)
            x_shortcut = tf.nn.conv2d(before_input, W_e_conv,
                                      strides=[1, stride, stride, 1], padding='SAME')
            self.AllWeight.append(W_e_conv)
        conv_add = tf.add(x_shortcut, after_input)
        conv_result_temp=tf.layers.batch_normalization(conv_add)
        conv_result = tf.nn.relu(conv_result_temp)
        return conv_result


    def LSTM_network(self,input, channel_list):
        batch_size = tf.shape(input)[0]
        time_step = tf.shape(input)[1]
        rnn_unit=channel_list
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit[0])
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input, initial_state=init_state,dtype=tf.float32)
        # output_rnn是记录lstm每个输出节点的结果，[batch_size,max_lenght,cell.outputsize],final_states是最后一个cell的结果[2,batch_size,cell.outputsize]



        output = tf.reshape(output_rnn, [-1, rnn_unit[0]])  # 作为输出层的输入
        print(output.shape)

        with tf.variable_scope('lstm_output'):
            fin=rnn_unit[0]
            w_initializer = tf.random_normal_initializer(0., 1./tf.sqrt(fin/2.))
            b_initializer = tf.constant_initializer(0.0)
            W_e_conv = tf.get_variable('w', [1024, self.activity_num], initializer=w_initializer)
            b_e_conv = tf.get_variable('b', [ self.activity_num, ], initializer=b_initializer)
            pred = tf.matmul(output, W_e_conv) + b_e_conv
        print(pred.shape)
        return pred, final_states


    def punish_overSeg(self,output,target):

        self.out_win = tf.map_fn(lambda i: output[i] - output[i - 1],
                                 tf.range(1, self.dimention, dtype=tf.int32), dtype=tf.float32)
        self.tar_win = tf.map_fn(lambda i: target[i] - target[i - 1],
                                 tf.range(1, self.dimention, dtype=tf.int32), dtype=tf.float32)

        constraint_loss = tf.abs(tf.reduce_sum(tf.abs(self.out_win)) - tf.reduce_sum(tf.abs(self.tar_win)))

        return constraint_loss













    def CSi_split(self,data):
        CSI_input, ac_tar = None, None
        CSI_data = data[0]
        CSI_index = data[1]
        CSI_flag = data[2]
        split_num = len(data[0][0]) // self.interval_len
        # print(CSI_data.shape,CSI_index,split_num)

        current_index = CSI_index[0]
        for i in range(split_num):
            current_index = current_index + self.interval_len
            # print(current_index)
            flag = 0
            for j in range(len(CSI_index)):
                if current_index <= CSI_index[j]:
                    flag = j - 1
                    break
            activity_flag = CSI_flag[flag]
            # print(activity_flag)
            onehot_vector = np.zeros([self.activity_num])
            onehot_vector[activity_flag ] = 1
            csi_image = CSI_data[:, i * self.interval_len:(i + 1) * self.interval_len, :]
            CSI_input = np.array([csi_image]) if CSI_input is None else np.append(CSI_input, [csi_image], axis=0)
            ac_tar = np.array([onehot_vector]) if ac_tar is None else np.append(ac_tar, [onehot_vector], axis=0)
        # activity_tar = np.argmax(ac_tar, axis=1)
        # print('activity_tar',activity_tar)
        return CSI_input, ac_tar



    def test_ac(self,input_data):
        # print(len(input_data))
        all_CSI_input_test, all_target_test = [], []
        for one_data in input_data:
            CSI_input, ac_tar = self.CSi_split(one_data)
            all_CSI_input_test.append(CSI_input)
            all_target_test.append(ac_tar)
        # print(len(all_CSI_input_test), len(all_target_test))
        # print(all_CSI_input[0].shape, all_target[0].shape)
        correct_count = 0
        test_sum = 0
        for num in range(len(all_CSI_input_test)):
            output = self.sess.run(self.output,feed_dict={self.input: all_CSI_input_test[num],self.keep_prob: 1.0,
                                                          self.dimention: all_target_test[num].shape[0]})
            output=np.squeeze(output)
            # print(output.shape,'output')
            activity_pre = np.argmax(output, axis=1)
            activity_target = np.argmax(all_target_test[num], axis=1)
            for i in range(len(activity_target)):
                if activity_pre[i] == activity_target[i]:
                    correct_count += 1
                test_sum += 1
        correct_pro = correct_count / test_sum
        # print(correct_pro)
        return correct_pro

    def test_trainData_ac(self):
        all_CSI_input_test, all_target_test = [], []
        trainData_toTest=self.train[0:len(self.test_no_trainSet_data)]
        # print(len(trainData_toTest))
        for one_data in trainData_toTest:
            CSI_input, ac_tar = self.CSi_split(one_data)
            all_CSI_input_test.append(CSI_input)
            all_target_test.append(ac_tar)
        # print(len(all_CSI_input), len(all_target))
        # print(all_CSI_input[0].shape, all_target[0].shape)
        correct_count = 0
        test_sum = 0
        for num in range(len(all_CSI_input_test)):
            output = self.sess.run(self.output,feed_dict={self.input: all_CSI_input_test[num],self.keep_prob: 1.0,
                                                          self.dimention: all_target_test[num].shape[0]})
            output=np.squeeze(output)
            # print(output.shape,'output')
            activity_pre = np.argmax(output, axis=1)
            activity_target = np.argmax(all_target_test[num], axis=1)
            for i in range(len(activity_target)):
                if activity_pre[i] == activity_target[i]:
                    correct_count += 1
                test_sum += 1
        correct_pro = correct_count / test_sum
        # print(correct_pro)
        return correct_pro


    def test_Seg(self,input_data,str1,f1_param):
        # time_1 = time.time()
        correct_count = 0
        test_sum = 0
        all_TP,all_FP,all_FN=0,0,0
        flag=0
        all_cons_loss = 0
        pre_data_save = []
        for one_data in input_data:
            csi_data = one_data[0]
            csi_index = one_data[1]
            csi_flag = one_data[2]
            # print(csi_index,csi_flag,'truth')

            flag+=1
            CSI_input, ac_tar = self.CSi_split(one_data)
            output,cons_loss = self.sess.run([self.output,self.cons_loss], feed_dict={self.input: CSI_input,
                                                           self.keep_prob: 1.0,self.tag: ac_tar,
                                                           self.dimention: ac_tar.shape[0]})

            all_cons_loss += cons_loss
            output = np.squeeze(output)
            activity_pre = np.argmax(output, axis=1)
            activity_tar=np.argmax(ac_tar,axis=1)
            # print('true list',activity_tar)
            # print('pre list',activity_pre)
            pre_data_save.append([csi_index, csi_flag, activity_pre])

            pre_index = []
            pre_flag = []
            for i in range(1, len(activity_pre)):
                if activity_pre[i - 1] != activity_pre[i]:
                    pre_index.append(i*20)
                    pre_flag.append(activity_pre[i - 1])
            pre_index.append(len(activity_pre)*20)
            pre_flag.append(activity_pre[-1])
            # print(pre_index,pre_flag,'pre')
            # pre_index,pre_flag=self.slide_modify(pre_index,pre_flag)
        # print('testing time is ', (time.time() - time_1) / len(input_data) * 2)

            if self.show_picture is True:
                'draw a picture'
                csi_index_convt = []
                pre_index_convt = []
                for i in range(1, len(csi_index) - 1, 1):
                    csi_index_convt.append(csi_index[i] - csi_index[0])
                # print(csi_index_convt, csi_flag, 'truth')
                for i in range(len(pre_index) - 1):
                    pre_index_convt.append(pre_index[i])
                # print(pre_index_convt, pre_flag, 'pre')

                csi_index_draw, pre_index_draw = [], []
                for i in range(1, len(csi_index)):
                    if i == 1:
                        csi_index_draw.append((csi_index[i] - csi_index[0]) // 2)
                    else:
                        csi_index_draw.append((csi_index[i] + csi_index[i - 1] - 2 * csi_index[0]) // 2)

                for i in range(len(pre_index)):
                    if i == 0:
                        pre_index_draw.append(pre_index[i] // 2)
                    else:
                        pre_index_draw.append((pre_index[i] + pre_index[i - 1]) // 2)

                # print(csi_index_draw, csi_flag, 111111)
                # print(pre_index_draw, pre_flag, 222222)

                """
                0-static, 1-walking, 2-wave hand, 3-box, 4-lift leg, 5-sit down, 
                6-stand up, 7-jump, 8-draw circle, 9-pick up, 10-raise hand
                """

                pylab.figure(figsize=(9, 7))
                pylab.subplot(2, 1, 1)
                pylab.plot(csi_data[0, :, 1], linewidth=1)
                for i in range(len(csi_index_convt)):
                    pylab.axvline(csi_index_convt[i], color='red')
                for i in range(len(csi_index_draw)):
                    pylab.text(csi_index_draw[i], 0.1, csi_flag[i])
                pylab.ylim(-0.5, 1.5)
                pylab.xlabel("Ground Truth")

                pylab.subplot(2, 1, 2)
                pylab.plot(csi_data[0, :, 1], linewidth=1)
                for i in range(len(pre_index_convt)):
                    pylab.axvline(pre_index_convt[i], color='blue')
                for i in range(len(pre_index_draw)):
                    pylab.text(pre_index_draw[i], 0.1, pre_flag[i])
                pylab.ylim(-0.5, 1.5)
                pylab.xlabel("Prediction Result")
                pylab.show()


            creat_pre_flag = np.zeros((pre_index[-1],))
            start_index=0
            for i in range(len(pre_flag)):
                creat_pre_flag[start_index:pre_index[i]]=pre_flag[i]
                # print(pre_flag[i])
                start_index=pre_index[i]
            # print(creat_pre_flag.shape,creat_pre_flag,'creat_pre_flag')

            creat_csi_flag=np.zeros((csi_index[-1]-csi_index[0],))
            start_index=0
            for i in range(len(csi_flag)):
                creat_csi_flag[start_index:(start_index+csi_index[i+1]-csi_index[i])]=csi_flag[i]
                start_index=start_index+csi_index[i+1]-csi_index[i]
            # print(creat_csi_flag.shape,'creat_csi_flag')

            leaf_len=creat_csi_flag.shape[0]-creat_pre_flag.shape[0]
            # print(leaf_len,'leaf_len')
            add_csi_flag=np.ones((creat_csi_flag.shape[0]-creat_pre_flag.shape[0],))*20
            creat_pre_flag=np.concatenate((creat_pre_flag,add_csi_flag),axis=0)
            # print(creat_csi_flag)
            consult=(creat_pre_flag==creat_csi_flag)
            # print(consult)
            correct_count+=np.sum(consult)
            test_sum+=consult.shape[0]
            if leaf_len!=0:
                yi = creat_csi_flag[0:-leaf_len]
                pi = creat_pre_flag[0:-leaf_len]
            else:
                yi = creat_csi_flag
                pi = creat_pre_flag
            # print(yi.shape, pi.shape,111111111)
            TP,FP,FN=self.overlap_f(pi, yi, self.activity_num, f1_param)
            all_TP+=TP
            all_FP+=FP
            all_FN+=FN

        # with open('result/envir2/LSTM_cons_1/' +str1 + '.kpl','wb') as handle:
        #     pickle.dump(pre_data_save, handle, -1)
        # print(str1+' result data stored')

        print(str1,'package-wise accuracy is :',correct_count/test_sum)
        # print('all_TP,all_FP,all_FN ',all_TP,all_FP,all_FN)
        precision = all_TP / (all_TP + all_FP)
        recall = all_TP / (all_TP + all_FN)
        F1 = 2 * (precision * recall) / (precision + recall)
        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)
        print('all F1@'+str(f1_param)+' and cons_loss are ', F1, all_cons_loss)

        if self.is_train is True:
            return F1,all_cons_loss








    def segment_labels(self,Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        # print(idxs)
        Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
        # print(Yi_split)
        return Yi_split

    def segment_intervals(self, Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        # print(intervals)
        return intervals

    def overlap_f(self,p, y,n_classes, overlap):
        true_intervals = np.array(self.segment_intervals(y))
        true_labels = self.segment_labels(y).astype(int)
        pred_intervals = np.array(self.segment_intervals(p))
        pred_labels = self.segment_labels(p).astype(int)
        # print('true',true_intervals, true_labels)
        # print('pre',pred_intervals, pred_labels)


        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]
        # print(n_true, n_pred)

        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j, 1], true_intervals[:, 1]) - np.maximum(pred_intervals[j, 0],
                                                                                               true_intervals[:, 0])
            union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) - np.minimum(pred_intervals[j, 0],
                                                                                        true_intervals[:, 0])
            IoU = (intersection / union) * (pred_labels[j] == true_labels)
            # print(IoU)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1
        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()


        return TP,FP,FN























def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        # print path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print path + ' 目录已存在'
        return False

def MAXMINnormalization(CSIimage):
    CSIimage=np.transpose(CSIimage,[2,0,1]) #2*90*len
    for num1 in range(len(CSIimage)):
        for num2 in range(len(CSIimage[0])):
            CSIimage[num1][num2]=normalization.MINMAXNormalization(CSIimage[num1][num2])
    CSIimage = np.transpose(CSIimage, [1, 2, 0])#90*len*2



if __name__ =="__main__":

    np.set_printoptions(threshold=np.inf)
    train_data = []
    test_data = []
    no_trainSet_data = []
    # for person_index in range(1, 5, 1):
    #     for list_index in range(1, 3, 1):
    #         path = 'TestData/envir_1/person_' + str(person_index) + '/list_' + str(list_index) + '/'
    #         temp_paths = os.listdir(path, )
    #         print(len(temp_paths))
    #
    #         for i in range(len(temp_paths)):
    #             with open(path + temp_paths[i], 'rb') as handle:
    #                 data_temp = pickle.load(handle)
    #             MAXMINnormalization(data_temp[0])
    #             # print(data_temp[0].shape)
    #             test_data.append(data_temp)

    for person_index in range(5, 6, 1):
        for list_index in range(1, 3, 1):
            path = 'TestData_cross_subject_scenario/envir_1/person_' + str(person_index) + '/list_' + str(list_index) + '/'
            temp_paths = os.listdir(path, )
            print(len(temp_paths))
            for i in range(len(temp_paths)):
                with open(path + temp_paths[i], 'rb') as handle:
                    data_temp = pickle.load(handle)
                MAXMINnormalization(data_temp[0])
                no_trainSet_data.append(data_temp)

    # print('testData_len is:', len(test_data))
    print('test_no_trainSet_data len is', len(no_trainSet_data))
    Seg_network(train_data=train_data, test_data=test_data, test_no_trainSet_data=no_trainSet_data)








