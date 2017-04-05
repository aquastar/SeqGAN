from gensim.models import Word2Vec
import os
import numpy as np
from scipy import spatial
import math

basepath = '/Users/danny/Downloads/KB2E/'
# read data as dict
transx = 'TransR'
trans_method = 'bern'

# load entity/relation index
f_ent = open('%sdata/entity2id.txt' % (basepath), 'rb')
f_rel = open('%sdata/entity2id.txt' % (basepath), 'rb')

dict_ent = {}
with open('%sdata/entity2id.txt' % (basepath), 'rb') as f:
    for line in f:
        ent = line.strip().split()
        dict_ent[ent[0]] = ent[1]

dict_rel = {}
with open('%sdata/relation2id.txt' % (basepath), 'rb') as f:
    for line in f:
        ent = line.strip().split()
        dict_rel[ent[0]] = ent[1]

entity2vec = '%s%s/entity2vec.%s' % (basepath, transx, trans_method)
relation2vec = '%s%s/relation2vec.%s' % (basepath, transx, trans_method)
entid2vec = {}
relid2vec = {}
with open(entity2vec, 'rb') as f:
    for lid, line in enumerate(f):
        vec = [float(_) for _ in line.strip().split()]
        entid2vec[lid] = vec
with open(relation2vec, 'rb') as f:
    for lid, line in enumerate(f):
        vec = [float(_) for _ in line.strip().split()]
        relid2vec[lid] = vec

for _id, _ in dict_ent.iteritems():
    dict_ent[_id] = entid2vec[int(_)]
for _id, _ in dict_rel.iteritems():
    dict_rel[_id] = relid2vec[int(_)]

# reload entity/relation vectors


# iraq|israel|saudi (cause)
# -> osama|laden|al-qaeda (attacker)
# -> united states|us|u.s.|world trade center|pentagon (victim)
# -> pentagon|flight 93|world trade center site memorial (aftermath)
a = ['iraq', 'laden', 'hani', 'world_trade_center']
# a = ['israel iraq israel', 'osama laden al-qaeda', 'world trade center pentagon', 'site memorial']

# satirical|muhammad (cause)
# -> al-qaeda|aqap|said|cherif|kouachi (attacker)
# -> charlie|hebdo|cabu|elsa|cayat|charb|wolinski|tignous (victim)
# -> vigipirate|jesuischarlie|#jesuischarlie (aftermath)
# b = ['muhammad', 'kouachi', 'charlie', '#jesuischarlie']

# anxiety|disorder|mental|ill|therapy|autism,seung-hui|seung-hui cho
b = ['saudi', 'al-qaeda', 'mohammed', 'new_york']

c = ['al-hazmi', 'flight', 'nawaf', 'september_11']

story_set = [a, b, c]


def get_w2v(word):
    return np.array(dict_ent[word])


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

    for s in story_set:
        print 'Story', s
        for _ in xrange(len(s) - 2):
            prev_link_vec = get_w2v(s[_ + 1]) - get_w2v(s[_])
            next_link_vec = get_w2v(s[_ + 2]) - get_w2v(s[_ + 1])

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
    print ' -- Abs Ang Avg Err', (abs(rel_ang_list[2][0] - rel_ang_list[0][0]) + abs(
        rel_ang_list[3][0] - rel_ang_list[1][0])) / 2, '/', (abs(rel_ang_list[4][0] - rel_ang_list[0][0]) + abs(
        rel_ang_list[5][0] - rel_ang_list[1][0])) / 2
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
