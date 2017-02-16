# import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np

# sess = tf.Session()
# saver = tf.train.Saver()
# saver.restore(sess, 'my-model')

# recover
# xx = generator.generate(sess)
storyline_trans = np.load('./trans_file.npy')[:3]

# load w2v model
w2v_vocab = Word2Vec.load('../dataset/w2v/mexico43_w2v.model')

# test entries
start_entries = ['abarca', 'pineda']

for s in start_entries:
    base = w2v_vocab[s]
    for trans in storyline_trans:
        print('\n===========')
        delta = trans[0] - base
        for en in trans:
            toplist = w2v_vocab.similar_by_vector(en - delta)
            print toplist
        print('\n===========')


# # load w2v model
# w2v_vocab = Word2Vec.load('../dataset/w2v/mexico43_w2v.model')
#
# # test entries
# start_entries = ['abarca', 'pineda']
#
# for s in start_entries:
#     base = w2v_vocab[s]
#     for trans in storyline_trans:
#         print('\n===========')
#         delta = trans[0] - base
#         for en in trans:
#             toplist = w2v_vocab.similar_by_vector(en - delta)
#             print toplist
#         print('\n===========')

