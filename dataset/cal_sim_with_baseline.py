from config import positive_file
import numpy as np
from scipy import linalg, mat, dot


# read input file
input = np.load(positive_file)

# MLE
mle = np.load('../MLE_SeqGAN/trans_file_mle.npy')

# seqgan
seqgan = np.load('../MLE_SeqGAN/trans_file_seqgan.npy')

# pg
pg = np.load('../pg_bleu/trans_file_pg.npy')

# ss
ss = np.load('../schedule_sampling/trans_file_ss.npy')

# random
ran = np.random.rand(*mle.shape)
total_num = input.shape[0] * mle.shape[0]


for base in [mle, seqgan, pg, ss, ran]:
    summ = 0.0
    for _ in base:
        summ += sum([np.sum(dot(mat(_), mat(__).T)) / linalg.norm(mat(__)) / linalg.norm(mat(_)) for __ in input])
    print summ

# sum(
#    [np.sum(dot(mat(poem), mat(_).T)) / linalg.norm(mat(poem)) / linalg.norm(mat(_)) for _ in self.references])
