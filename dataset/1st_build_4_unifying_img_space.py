import cPickle as pk
from nltk.stem.wordnet import WordNetLemmatizer
import glob
import re
import tensorflow as tf
import entity_img_extract.utils as utils
import entity_img_extract.vgg19 as vgg19
import sys
import os
import os.path
import numpy as np
from gensim.models import Word2Vec
import urllib

from stop_words import get_stop_words

en_stop_words = [x.encode('utf-8') for x in get_stop_words('en')]
# customized_stopwords = ''
# en_stop_words.append(customized_stopwords)
lmtzr = WordNetLemmatizer()
sent_list = []
img_id_cnt = 0

sent_img_corpus = {}
sent_img_corpus_file = 'sent_img_corpus.pk'

if not os.path.isfile(sent_img_corpus_file):
    for root_dir in ['./homicide', './protest', './test']:
        for dir in glob.glob(root_dir + os.sep + '*'):
            print dir
            if not os.path.isdir(dir):
                continue
            story_id = dir.split(os.sep)[-1].split('_')[0]
            sent_img_corpus[story_id] = {}
            for file in glob.glob(dir + os.sep + '*'):
                sent_list = []
                file_cont = open(file, 'r').readlines()

                # stopwords
                sents = [filter(lambda x: x not in en_stop_words, x.split()) for x in
                         re.split('\. ', ' '.join([_.strip().lower() for _ in file_cont[:-1]]))]
                # stemming
                sents = [[lmtzr.lemmatize(x.decode('utf-8')) for x in y] for y in sents]
                for s in sents:
                    if len(s) > 5:
                        sent_list.append([x.replace(',', '') for x in s])

                new_img_addr = 'news_img' + os.sep + str(img_id_cnt) + '.jpg'

                # download image
                try:
                    print('Downloading images ' + str(img_id_cnt) + ' at:' + file_cont[-1])
                    from subprocess import call

                    call(["wget", "--tries=2", "-O", new_img_addr, file_cont[-1]])
                    t_shape = np.asarray(utils.load_image(new_img_addr)).shape
                    if t_shape != (224, 224, 3):
                        continue
                    img_id_cnt += 1

                    sent_img_corpus[story_id][new_img_addr] = sent_list
                    print('Caught^_^')
                except:
                    print('Failed-_-!')
                    continue

    print('Finish downloading...')
    pk.dump(sent_img_corpus, open(sent_img_corpus_file, 'wb'))
else:
    sent_img_corpus = pk.load(open(sent_img_corpus_file, 'rb'))

print('Calculating images feats...')
img_set = glob.glob('news_img/*')
img_set_len = len(img_set)
segment_len = 64

img_dict = {}
feats = []

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
    name_subset = img_set[segment_iter_len[x]:segment_iter_len[x + 1]]
    img_subset = [np.asarray(utils.load_image(_)) for _ in name_subset]

    tf.reset_default_graph()
    feats = []
    with tf.Session() as sess:
        images = tf.placeholder("float", [len(img_subset), 224, 224, 3])
        # vgg = vgg16.Vgg16()
        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        feed_dict = {images: img_subset}
        feats = sess.run(vgg.fc7, feed_dict=feed_dict)

    for name, feat in zip(name_subset, feats):
        img_dict[name] = feat

img_postfix = '_train_ims.npy'
txt_postfix = '_train_caps.txt'
prefix = 'unifying_model_4_img'

# output training file
for story_id, img_sents in sent_img_corpus.iteritems():
    img_train = story_id + '_train_ims.npy'
    txt_train = story_id + '_train_caps.txt'
    img_arr = []
    train_caps_file = open(prefix + os.sep + txt_train, 'wb')
    for img, sents in img_sents.iteritems():
        for sent in sents:
            img_arr.append(img_dict[img])
            print >> train_caps_file, u' '.join(sent).encode('utf-8').strip()
    np.save(prefix + os.sep + img_train, np.array(img_arr))
