import cPickle as pk
import os
from gensim.models import Word2Vec

storylines = pk.load(open('storyline.pk', 'rb'))
protest_tag = ['baltimore', 'wall', 'martin']

min_len = 3

protest_data = []
homicide_data = []


def get_w2v(w2v_vocab, word_list):
    return [w2v_vocab[_.lower()] for _ in word_list if _.lower() in w2v_vocab]


if __name__ == '__main__':
    for tag, story in storylines.iteritems():
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
