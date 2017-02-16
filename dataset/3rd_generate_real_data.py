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
img_feat_dic = pk.load(open('./unifying_pic_avatar_dict.pk', 'rb'))
tmp = {}
for story, spic in img_feat_dic.iteritems():
    for name, p in spic.iteritems():
        tmp[name] = p
img_feat_dic = tmp
missed_cnt = 0

feat_style = 0  # 0-text, 1-img, 2-txt+img, 3-txt+img+mm


def mean(a):
    return sum(a) / len(a)


def get_w2v_and_img_and_mm(w2v_vocab, word_list):
    ret = []
    for wd in word_list:
        if wd.lower() not in img_feat_dic:
            break

        if len(wd.split()) > 1:
            wd = filter(lambda x: x not in en_stop_words, wd.split())
            wd = [lmtzr.lemmatize(x.decode('utf-8')).lower() for x in wd]
            missed = False
            for w in wd:
                if w not in w2v_vocab:
                    print 'Missed Entity in multimple words:', wd
                    missed = True
            # if missed:
            #     continue
            vec = [w2v_vocab[_] for _ in wd if _ in w2v_vocab]
            mean_vec = map(mean, zip(*vec))
            ret.append(np.concatenate((np.array(mean_vec), img_feat_dic[wd.lower()],
                                       np.subtract(np.array(mean_vec), img_feat_dic[wd.lower()]))))
        elif wd.lower() in w2v_vocab:
            ret.append(np.concatenate((w2v_vocab[wd.lower()], img_feat_dic[wd.lower()],
                                       np.subtract(w2v_vocab[wd.lower()], img_feat_dic[wd.lower()]))))
        else:
            print 'Missed Single Entity:', wd
    return ret


def get_w2v_and_img_feat(w2v_vocab, word_list):
    ret = []
    for wd in word_list:
        if wd.lower() not in img_feat_dic:
            break

        if len(wd.split()) > 1:
            wd = filter(lambda x: x not in en_stop_words, wd.split())
            #wd = [lmtzr.lemmatize(x.decode('utf-8')).lower() for x in wd]
            missed = False
            for w in wd:
                if w not in w2v_vocab:
                    print 'Missed Entity in multimple words:', wd
                    missed = True
            # if missed:
            #     continue
            vec = [w2v_vocab[_] for _ in wd if _ in w2v_vocab]
            mean_vec = map(mean, zip(*vec))
            ret.append(np.concatenate((np.array(mean_vec), img_feat_dic[' '.join(wd)])))
        elif wd.lower() in w2v_vocab:
            ret.append(np.concatenate((w2v_vocab[wd.lower()], img_feat_dic[wd])))
        else:
            print 'Missed Single Entity:', wd
    return ret


def get_img_feat(w2v_vocab, word_list):
    ret = []
    for wd in word_list:
        if wd.lower() not in img_feat_dic:
            break

        if len(wd.split()) > 1:
            wd = filter(lambda x: x not in en_stop_words, wd.split())
            # wd = [lmtzr.lemmatize(x.decode('utf-8')).lower() for x in wd]

            ret.append(img_feat_dic[' '.join(wd)])
        elif wd.lower() in w2v_vocab:
            ret.append(img_feat_dic[wd.lower()])
        else:
            print 'Missed Single Entity:', wd
    return ret


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
            # if missed:
            #     continue
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
                # slicing window for data augmentation
                for i in xrange(len(s) - min_len + 1):
                    one_transfer_vec = []
                    slice = s[i:i + min_len]
                    # 0-text, 1-img, 2-txt+img, 3-txt+img+mm
                    if feat_style == 0:
                        slice_vec = get_w2v(w2v_model, slice)
                    elif feat_style == 1:
                        slice_vec = get_img_feat(w2v_model, slice)
                    elif feat_style == 2:
                        slice_vec = get_w2v_and_img_feat(w2v_model, slice)
                    elif feat_style == 3:
                        slice_vec = get_w2v_and_img_and_mm(w2v_model, slice)

                    if len(slice_vec) != len(slice):
                        print '-- This slice don\'t have enough data', s
                        break

                    # normalize vector, substracting the first one
                    for i in xrange(min_len):
                        diff = np.subtract(slice_vec[i], slice_vec[0])
                        one_transfer_vec.append(diff)

                    if tag in protest_tag:
                        protest_data.append(one_transfer_vec)
                    else:
                        homicide_data.append(one_transfer_vec)

    print 'protest number', len(protest_data)
    print 'homicide number', len(homicide_data)

    if feat_style == 0:
        np.save(open('feat_pool/protest_in_num_w2v.npy', 'wb'), protest_data)
        np.save(open('feat_pool/homicide_in_num_w2v.npy', 'wb'), homicide_data)
    elif feat_style == 1:
        np.save(open('feat_pool/protest_in_num_img.npy', 'wb'), protest_data)
        np.save(open('feat_pool/homicide_in_num_img.npy', 'wb'), homicide_data)
    elif feat_style == 2:
        np.save(open('feat_pool/protest_in_num_w2v_img.npy', 'wb'), protest_data)
        np.save(open('feat_pool/homicide_in_num_w2v_img.npy', 'wb'), homicide_data)
    elif feat_style == 3:
        np.save(open('feat_pool/protest_in_num_w2v_img_mm.npy', 'wb'), protest_data)
        np.save(open('feat_pool/homicide_in_num_w2v_img_mm.npy', 'wb'), homicide_data)
