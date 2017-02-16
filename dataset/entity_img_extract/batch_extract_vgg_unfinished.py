import cPickle as pk
import glob

import numpy as np
import tensorflow as tf

import utils
import vgg16

file_list = tuple(glob.glob('./mirflickr/*'))
pic_dict = {}
training_tuple = []
training_size = 128
training_pic_name = []


def batch_vgg(training_tuple):
    batch = np.concatenate(tuple(training_tuple), 0)
    tf.reset_default_graph()
    with tf.Session() as sess:
        images = tf.placeholder("float", [training_size, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)

    print 'assign prob'
    for x, y in zip(training_pic_name, prob):
        pic_dict[x] = y


for fid, f in enumerate(file_list):
    print 'reshaping', fid, f
    training_pic_name.append(f)
    cur_img = utils.load_image(f).reshape((1, 224, 224, 3))
    training_tuple.append(cur_img)

    prob = []
    if len(training_tuple) == training_size:
        print 'training batch'
        batch = np.concatenate(tuple(training_tuple), 0)
        tf.reset_default_graph()
        with tf.Session() as sess:
            images = tf.placeholder("float", [training_size, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16()
            vgg.build(images)

            prob = sess.run(vgg.prob, feed_dict=feed_dict)

        print 'assign prob'
        for x, y in zip(training_pic_name, prob):
            pic_dict[x] = y

        training_pic_name = []
        training_tuple = []

if training_tuple:
    batch = np.concatenate(tuple(training_tuple), 0)
    tf.reset_default_graph()
    with tf.Session() as sess:
        images = tf.placeholder("float", [len(training_tuple), 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)

    print 'assign prob'
    for x, y in zip(training_pic_name, prob):
        pic_dict[x] = y

pk.dump(pic_dict, open('pic_dict.pk', 'wb'))
