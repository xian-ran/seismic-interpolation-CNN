# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/20 16:22

import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
from cnn_sr import *


if __name__ == '__main__':
    learning_rate = 1e-3
    batch_size = 32
    step_num = 10000
    g = ResNet([106, 106, 1], learning_rate)
    data = sio.loadmat('TEST_exp')['TEST_exp']
    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)
    seismic_l = np.zeros((53, 53, data.shape[2]))
    seismic_h = np.zeros(data.shape)

    if tf.train.get_checkpoint_state('./backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, './backup/')
        print('********Restore the latest trained parameters.********')
        for i in range(data.shape[2]):
            xs = data[:, :, i]
            xs = np.expand_dims(xs, axis=0)
            xs = np.expand_dims(xs, axis=-1)
            zs, gs = sess.run([g.z, g.g], feed_dict={g.x: xs})
            seismic_l[:,:,i] = np.squeeze(zs)
            seismic_h[:,:,i] = np.squeeze(gs)

        sio.savemat('cnn_sr.mat', {'seismic_cnn_l': seismic_l, 'seismic_cnn_h': seismic_h})

    else:
        print('Trained model is not existed!')