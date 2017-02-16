import glob
import os, sys
import re

import cPickle as pk
import numpy as np
import tensorflow as tf
import entity_img_extract.utils as utils
import entity_img_extract.vgg19 as vgg19

file = open("storyline_train")
base_path = ''

# read storyline entity database
storyline = {}
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        id_story = line.split(':')
        sid = id_story[0]
        story = id_story[1].strip()
        entity = re.split('\||,', story)

        if sid in storyline:
            storyline[sid].extend([_.lower() for _ in entity])
        else:
            storyline[sid] = [_.lower() for _ in entity]

# read avatar file and computer VGG19 feat
img_dic = {}
for dir in glob.glob('entity_img' + os.sep + '*'):
    print dir
    entity_id = dir.split(os.sep)[-1]
    for f in glob.glob(dir + os.sep + '*'):
        img_dic[entity_id.lower()] = f

img_dic = {}
for dir in glob.glob('entity_img' + os.sep + '*'):
    print dir
    entity_id = dir.split(os.sep)[-1]
    for f in glob.glob(dir + os.sep + '*'):
        img_dic[entity_id.lower()] = f

img_set = [np.asarray(utils.load_image(x)) for x in img_dic.values()]
name_set = img_dic.keys()

img_set_len = len(img_set)
segment_len = 48

feats = []
with tf.Session() as sess:
    tmp_len = img_set_len
    segment_iter_len = [0]
    while tmp_len > 0:
        if tmp_len > segment_len:
            tmp_len -= segment_len
            segment_iter_len.append(segment_len + segment_iter_len[-1])
        else:
            segment_iter_len.append(img_set_len)
            tmp_len = 0

    for x in xrange(len(segment_iter_len) - 1):
        img_subset = img_set[segment_iter_len[x]:segment_iter_len[x + 1]]
        name_subset = name_set[segment_iter_len[x]:segment_iter_len[x + 1]]

        images = tf.placeholder("float", [len(img_subset), 224, 224, 3])
        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        feed_dict = {images: img_subset}
        feats = sess.run(vgg.fc7, feed_dict=feed_dict)

        for name, feat in zip(name_subset, feats):
            img_dic[name] = feat

# re-organize in event category
pic_avatar_dict = {}
for sid, entities in storyline.iteritems():
    pic_avatar_dict[sid] = {}
    for en in entities:
        if en in img_dic:
            pic_avatar_dict[sid][en] = img_dic[en]

pk.dump(pic_avatar_dict, open('pic_avatar_dict.pk', 'wb'))
