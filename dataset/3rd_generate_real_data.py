from __future__ import division
import cPickle as pk
import os
import numpy as np
from gensim.models import Word2Vec
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer


storylines = pk.load(open('storyline.pk', 'rb'))
protest_tag = ['baltimore', 'wall', 'martin']
min_len = 3
protest_data = []
homicide_data = []
en_stop_words = [x.encode('utf-8') for x in get_stop_words('en')]
lmtzr = WordNetLemmatizer()

def mean(a):
    return sum(a) / len(a)


def get_w2v(w2v_vocab, word_list):
    ret = []
    for wd in word_list:
        if len(wd.split()) > 1:
            wd = filter(lambda x: x not in en_stop_words, wd.split())
            wd = [lmtzr.lemmatize(x.decode('utf-8')).lower() for x in wd]
            missed = False
            for w in wd:
                if w not in w2v_vocab:
                    print 'Missed Multiple Entites:', wd
                    missed = True
            if missed:
                continue
            vec = [w2v_vocab[_] for _ in wd if _ in w2v_vocab]
            mean_vec = map(mean, zip(*vec))
            ret.append(np.array(mean_vec))
        elif wd.lower() in w2v_vocab:
            ret.append(w2v_vocab[wd.lower()])
        else:
            print 'Missed Single Entity:', wd
    return ret


if __name__ == '__main__':
    for tag, story in storylines.iteritems():
        print '-- Processing', tag
        w2v_model = Word2Vec.load('./w2v' + os.sep + tag + '_w2v.model')
        for s in story:
            if len(s) < min_len:
                continue
            else:
                for i in xrange(len(s) - min_len + 1):
                    slice = s[i:i + min_len]
                    slice_vec = get_w2v(w2v_model, slice)

                    if tag in protest_tag:
                        protest_data.append(slice)
                    else:
                        homicide_data.append(slice)

np.save(open('protest_in_num.npy', 'wb'), protest_data)
np.save(open('homicide_in_num.npy', 'wb'), homicide_data)
