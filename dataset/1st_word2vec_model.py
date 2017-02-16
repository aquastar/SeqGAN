import cPickle as pk
from nltk.stem.wordnet import WordNetLemmatizer
import glob
import re
import sys
import os
from gensim.models import Word2Vec

from stop_words import get_stop_words

en_stop_words = [x.encode('utf-8') for x in get_stop_words('en')]
# customized_stopwords = ''
# en_stop_words.append(customized_stopwords)
lmtzr = WordNetLemmatizer()
sent_list = []

for root_dir in ['./test']:

    for dir in glob.glob(root_dir + os.sep + '*'):
        print dir
        if not os.path.isdir(dir):
            continue
        story_id = dir.split(os.sep)[-1].split('_')[0]
        sent_list = []
        for file in glob.glob(dir + os.sep + '*'):

            # stopwords
            sents = [filter(lambda x: x not in en_stop_words, x.split()) for x in
                     re.split('\. ', ' '.join([_.strip().lower() for _ in open(file, 'r').readlines()[:-1]]))]
            # stemming
            sents = [[lmtzr.lemmatize(x.decode('utf-8')) for x in y] for y in sents]
            for s in sents:
                if len(s) > 5:
                    sent_list.append([x.replace(',', '') for x in s])

        model = Word2Vec(sent_list, size=300, window=5, min_count=5, workers=4, iter=100)
        model.save('w2v' + os.sep + story_id + '_w2v.model')

        # test purpose only
        if 'protest' in model:
            print 'protest', model.most_similar(positive='protest')
        if 'attack' in model:
            print 'attack', model.most_similar(positive='attack')
