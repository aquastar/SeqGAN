from gensim.models import Word2Vec
import os
import numpy as np
from scipy import spatial
import math

w2v_path = '/Users/danny/PycharmProjects/SeqGAN/dataset/w2v'

model_a = Word2Vec.load(w2v_path + os.sep + '911' + '_w2v.model')
# model_b = Word2Vec.load(w2v_path + os.sep + 'charlie' + '_w2v.model')
# model_c = Word2Vec.load(w2v_path + os.sep + 'vtshooting' + '_w2v.model')
model_b = model_a
model_c = model_a

# iraq|israel|saudi (cause)
# -> osama|laden|al-qaeda (attacker)
# -> united states|us|u.s.|world trade center|pentagon (victim)
# -> pentagon|flight 93|world trade center site memorial (aftermath)
# a = ['israel', 'laden', 'world trade center', 'memorial']
# a = ['israel iraq israel', 'osama laden al-qaeda', 'world trade center pentagon', 'site memorial']

# satirical|muhammad (cause)
# -> al-qaeda|aqap|said|cherif|kouachi (attacker)
# -> charlie|hebdo|cabu|elsa|cayat|charb|wolinski|tignous (victim)
# -> vigipirate|jesuischarlie|#jesuischarlie (aftermath)
# b = ['muhammad', 'kouachi', 'charlie', '#jesuischarlie']

# anxiety|disorder|mental|ill|therapy|autism,seung-hui|seung-hui cho
# c = ['anxiety disorder mental', 'seung-hui cho', 'jamie bishop jocelyne couture-nowak liviu librescu',
#      'hokie spirit memorial fund']


a = ['iraq', 'laden', 'hani', 'world trade center']
# a = ['israel iraq israel', 'osama laden al-qaeda', 'world trade center pentagon', 'site memorial']

# satirical|muhammad (cause)
# -> al-qaeda|aqap|said|cherif|kouachi (attacker)
# -> charlie|hebdo|cabu|elsa|cayat|charb|wolinski|tignous (victim)
# -> vigipirate|jesuischarlie|#jesuischarlie (aftermath)
# b = ['muhammad', 'kouachi', 'charlie', '#jesuischarlie']

# anxiety|disorder|mental|ill|therapy|autism,seung-hui|seung-hui cho
b = ['saudi', 'al-qaeda', 'mohammed', 'new york']

c = ['al-hazmi', 'flight', 'nawaf', 'september 11']

model_set = [model_a, model_b, model_c]
story_set = [a, b, c]


def get_KNN(model, word, sim_type='hyb'):
    K_thres = 10
    sim_thres = 0.5
    candidates = []

    if sim_type == 'add':
        candidates = model.wv.most_similar(positive=[word], topn=K_thres)
    elif sim_type == 'mul':
        candidates = model.wv.most_similar_cosmul(positive=[word], topn=K_thres)
    elif sim_type == 'hyb':
        candidates_add = model.wv.most_similar(positive=[word], topn=K_thres)
        candidates_mul = model.wv.most_similar_cosmul(positive=[word], topn=K_thres)
        candidates = candidates_add + candidates_mul

    candidates = [_[0] for _ in candidates if _[1] > sim_thres]
    candidates = sum([model.wv[_] for _ in candidates] + [model.wv[word]]) / (len(candidates) + 1)
    return candidates


def get_w2v(model, word):
    words = word.split()
    if len(words) > 1:
        return sum([get_KNN(model, x) for x in words]) / len(words)
    else:
        return get_KNN(model, word)


def get_degree(x):
    return round(math.degrees(math.acos(x)), 2)


def relative_len(prev, next):
    prev_len = np.linalg.norm(prev)
    next_len = np.linalg.norm(next)
    return prev_len, next_len, next_len / prev_len, next_len - prev_len


def relative_angle(prev, next):
    tmp = 1 - spatial.distance.cosine(prev, next)
    return get_degree(tmp), tmp


if __name__ == '__main__':
    rel_len_list = []
    rel_ang_list = []
    for m, s in zip(model_set, story_set):
        print 'Story', s
        for _ in xrange(len(s) - 2):
            prev_link_vec = get_w2v(m, s[_ + 1]) - get_w2v(m, s[_])
            next_link_vec = get_w2v(m, s[_ + 2]) - get_w2v(m, s[_ + 1])

            rel_len = relative_len(prev_link_vec, next_link_vec)
            rel_ang = relative_angle(prev_link_vec, next_link_vec)

            rel_ang_list.append(rel_ang)
            rel_len_list.append(rel_len)
            # print 'Link', _, 'to', _ + 1
            # print '------------------------------------'
            # print 'ang', rel_ang
            # print 'len', rel_len
            # print '===================================='
            # print ''
    print '========'
    print 'Summary'
    print 'Rel Ang(degree)(Div)', [rel_ang_list[_][0] / rel_ang_list[int(_ / 3) * 3][0] for _ in
                                   xrange(len(rel_ang_list))]
    print 'Rel Ang(degree)(Sub)', [rel_ang_list[_][0] - rel_ang_list[int(_ / 3) * 3][0] for _ in
                                   xrange(len(rel_ang_list))]
    print 'Abs Ang(degree)', [_[0] for _ in rel_ang_list]
    print ' -- Abs Ang Avg Err', (abs(rel_ang_list[2][0]-rel_ang_list[0][0])+abs(rel_ang_list[3][0]-rel_ang_list[1][0]))/2, (abs(rel_ang_list[4][0]-rel_ang_list[0][0])+abs(rel_ang_list[5][0]-rel_ang_list[1][0]))/2

    print '----'
    print 'Rel Ang(value)(Div)', [rel_ang_list[_][1] / rel_ang_list[int(_ / 3) * 3][1] for _ in
                                  xrange(len(rel_ang_list))]
    print 'Rel Ang(value)(Sub)', [rel_ang_list[_][1] - rel_ang_list[int(_ / 3) * 3][1] for _ in
                                  xrange(len(rel_ang_list))]
    print 'Abs Ang(value)', [_[1] for _ in rel_ang_list]
    print '----'
    print 'Rel Len(Div)', [rel_len_list[_][-2] / rel_len_list[int(_ / 3) * 3][-2] for _ in xrange(len(rel_len_list))]
    print 'Rel Len(Sub)', [rel_len_list[_][-1] - rel_len_list[int(_ / 3) * 3][-1] for _ in xrange(len(rel_len_list))]
    print 'Abs Lng', [_ for _ in rel_len_list]
    print str("%.2f" % rel_len_list[0][0]) + '-' + str("%.2f" % rel_len_list[0][1]) + '-' + str(
        "%.2f" % rel_len_list[1][1]), str(
        "%.2f" % rel_len_list[2][0]) + '-' + str("%.2f" % rel_len_list[2][1]) + '-' + str(
        "%.2f" % rel_len_list[3][1]), str(
        "%.2f" % rel_len_list[4][0]) + '-' + str("%.2f" % rel_len_list[4][1]) + '-' + str("%.2f" % rel_len_list[5][1])
    print ' -- Abs Len Avg Err', "%.4f" % ((abs(rel_len_list[2][0] - rel_len_list[0][0]) + abs(
        rel_len_list[3][0] - rel_len_list[1][0]) + abs(rel_len_list[3][1] - rel_len_list[1][1])) / 3), '/', "%.4f" % (
    (abs(
        rel_len_list[4][0] - rel_len_list[0][0]) + abs(rel_len_list[5][0] - rel_len_list[1][0]) + abs(
        rel_len_list[5][1] - rel_len_list[1][1])) / 3)

