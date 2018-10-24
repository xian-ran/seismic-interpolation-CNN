# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/19 20:17


import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio


def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def relu(x):
    return tf.nn.relu(x)


def elu(x):
    return tf.nn.elu(x)


def xavier_init(size):
    input_dim = size[0]
    stddev = 1. / tf.sqrt(input_dim / 2.)
    return tf.random_normal(shape=size, stddev=stddev)


def he_init(size, stride):
    input_dim = size[2]
    output_dim = size[3]
    filter_size = size[0]

    fan_in = input_dim * filter_size ** 2
    fan_out = output_dim * filter_size ** 2 / (stride ** 2)
    stddev = tf.sqrt(4. / (fan_in + fan_out))
    minval = -stddev * np.sqrt(3)
    maxval = stddev * np.sqrt(3)
    return tf.random_uniform(shape=size, minval=minval, maxval=maxval)


class SR(object):
    def __init__(self, img_shape, batch_size, learning_rate):
        self.height = img_shape[0]
        self.width = img_shape[1]
        self.batch_size = batch_size
        # self.learning_rate = learning_rate
        self.channel_num = img_shape[2]
        # self.vgg = VGG19(None, None, None)
        self.layer_num = 0

        self.x = tf.placeholder(
            tf.float32,
            [None, self.height, self.width, self.channel_num],
            name='x'
        )
        self.z = self.downscale(self.x, 2)
        self.g = self.generator(self.z)
        self.loss = self.reconstruction_loss(self.x, self.g)

        self.opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.loss)

    def generator(self, z):
        # Network.deconv2d(input, input_shape, output_dim, filter_size, stride)
        h = tf.nn.relu(self.deconv2d(z, 64, 3, 1))
        bypass = h

        h = self.residual_block(h, 64, 3, 2)

        h = self.deconv2d(h, 64, 3, 1)
        h = self.batch_norm(h)
        h = tf.add(h, bypass)

        h = self.deconv2d(h, 256, 3, 1)
        h = self.pixel_shuffle(h, 2, 64)
        h = tf.nn.relu(h)

        # h = self.deconv2d(h, 64, 3, 1)
        # h = self.pixel_shuffle(h, 2, 16)
        # h = tf.nn.relu(h)

        h = self.deconv2d(h, self.channel_num, 3, 1)

        return h

    def downscale(self, x, K):
        mat = np.zeros([K, K, self.channel_num, self.channel_num])
        for i in range(self.channel_num):
            mat[:, :, i, i] = 1.0 / K ** 2
        filter = tf.constant(mat, dtype=tf.float32)
        return tf.nn.conv2d(x, filter, strides=[1, K, K, 1], padding='SAME')

    def vgg19_loss(self, x, g):
        _, real_phi = self.vgg.build_model(x, tf.constant(False), False)
        _, fake_phi = self.vgg.build_model(g, tf.constant(False), True)

        loss = None
        for i in range(len(real_phi)):
            l2_loss = tf.nn.l2_loss(real_phi[i] - fake_phi[i])
            if loss is None:
                loss = l2_loss
            else:
                loss += l2_loss

        return tf.reduce_mean(loss)

    @staticmethod
    def reconstruction_loss(x, g):
        return tf.reduce_sum(tf.square(x - g))

    def conv2d(self, input, input_dim, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('conv' + str(self.layer_num)):
            init_w = he_init([filter_size, filter_size, input_dim, output_dim], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d(
                input,
                weight,
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)

            self.layer_num += 1

        return output

    def deconv2d(self, input, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('deconv' + str(self.layer_num)):
            input_shape = input.get_shape().as_list()
            init_w = he_init([filter_size, filter_size, output_dim, input_shape[3]], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d_transpose(
                value=input,
                filter=weight,
                output_shape=[
                    tf.shape(input)[0],
                    input_shape[1] * stride,
                    input_shape[2] * stride,
                    output_dim
                ],
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)
            output = tf.reshape(output,
                                [tf.shape(input)[0], input_shape[1] * stride, input_shape[2] * stride, output_dim])

            self.layer_num += 1

        return output

    def batch_norm(self, input, scale=False):
        ''' batch normalization
        ArXiv 1502.03167v3 '''
        with tf.variable_scope('batch_norm' + str(self.layer_num)):
            output = tf.contrib.layers.batch_norm(input, scale=scale)
            self.layer_num += 1

        return output

    def dense(self, input, output_dim):
        with tf.variable_scope('dense' + str(self.layer_num)):
            input_dim = input.get_shape().as_list()[1]

            init_w = xavier_init([input_dim, output_dim])
            weight = tf.get_variable('weight', initializer=init_w)

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable('bias', initializer=init_b)

            output = tf.add(tf.matmul(input, weight), bias)

            self.layer_num += 1

        return output

    def residual_block(self, input, output_dim, filter_size, n_blocks=5):
        output = input
        with tf.variable_scope('residual_block'):
            for i in range(n_blocks):
                bypass = output
                output = self.deconv2d(output, output_dim, filter_size, 1)
                output = self.batch_norm(output)
                output = tf.nn.relu(output)

                output = self.deconv2d(output, output_dim, filter_size, 1)
                output = self.batch_norm(output)
                output = tf.add(output, bypass)

        return output

    def pixel_shuffle(self, x, r, n_split):
        def PS(x, r):
            bs, a, b, c = x.get_shape().as_list()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, (bs, a, b, r, r))
            x = tf.transpose(x, (0, 1, 2, 4, 3))
            x = tf.split(x, a, 1)
            x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)
            x = tf.split(x, b, 1)
            x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)
            return tf.reshape(x, (bs, a * r, b * r, 1))

        xc = tf.split(x, n_split, 3)
        xc = tf.concat([PS(x_, r) for x_ in xc], 3)
        return xc

def show_result(xs, zs, gs, step):
    zs = np.squeeze(zs)
    xs = np.squeeze(xs)
    gs = np.squeeze(gs)
    fig = plt.figure(figsize=(5, 15))

    #graph = gridspec.GridSpec(1, 3)
    #graph.update(wspace=0.5, hspace=0.5)

    ax = fig.add_subplot(131)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(zs, cmap='Greys_r')

    ax = fig.add_subplot(132)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(gs, cmap='Greys_r')

    ax = fig.add_subplot(133)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(xs, cmap='Greys_r')

    plt.savefig('out/{}.png'.format(str(step).zfill(6)), bbox_inches='tight')
    plt.close(fig)


def sample_data(batch_size, sample_shape, data):
    data_shape = data.shape
    samples = np.zeros((batch_size, sample_shape[0], sample_shape[1]))
    for i in range(batch_size):
        row = np.random.randint(0, (data_shape[0]-sample_shape[0]+1))
        col = np.random.randint(0, (data_shape[1]-sample_shape[1]+1))
        sli = np.random.randint(0, data_shape[2])
        samples[i,:,:] = data[row:(row+sample_shape[0]), col:(col+sample_shape[1]), sli]

    return samples


if __name__ == '__main__':
    learning_rate = 1e-3
    batch_size = 32
    step_num = 10000

    data = sio.loadmat('volume.mat')['volume']

    g = SR([106, 106, 1], batch_size, learning_rate)

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    if not os.path.exists('./backup/'):
        os.mkdir('./backup/')
    if not os.path.exists('./out/'):
        os.mkdir('./out/')
    saver = tf.train.Saver()

    for step in tqdm(range(step_num), total=step_num, ncols=70, leave=False, unit='b'):
        xs = sample_data(batch_size, (106, 106), data)
        xs = np.expand_dims(xs, axis=-1)
        _, l = sess.run([g.opt, g.loss], feed_dict={g.x: xs})

        if step % 100 == 0:
            print('step: {}, loss: {}'.format(step, l))
            # zs, gs = sess.run([g.z, g.g], feed_dict={g.x: xs})
            # show_result(xs[0], zs[0], gs[0], step)
        if step % 1000 == 0:
            saver.save(sess, './backup/', write_meta_graph=False)


