#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import string
import unicodedata
from goose import Goose
import re
import shutil
import sys

pattern = re.compile(
    r'(?:(((Jan(uary)?|Ma(r(ch)?|y)|Jul(y)?|Aug(ust)?|Oct(ober)?|Dec(ember)?)\ 31)|((Jan(uary)?|Ma(r(ch)?|y)|Apr(il)?|Ju((ly?)|(ne?))|Aug(ust)?|Oct(ober)?|(Sept|Nov|Dec)(ember)?)\ (0?[1-9]|([12]\d)|30))|(Feb(ruary)?\ (0?[1-9]|1\d|2[0-8]|(29(?=,\ ((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00)))))))\,\ ((1[6-9]|[2-9]\d)\d{2}))',
    re.IGNORECASE)
pattern_spanish = re.compile(r'\d{1,2} de [a-z]+ (de|del) (2|1)\d{3}', re.IGNORECASE)


def noramlize_spanish(str):
    return unicodedata.normalize('NFKD', str).encode('ASCII', 'ignore')


def catchpg(x, dir_to_write):
    g = Goose()
    print '=== start ==='
    print x
    try:
        a = g.extract(url=x)

        to_write = a.cleaned_text.replace('\r', '').replace('\n', ' ').strip() + '\n'
        to_write += a.title + '\n'
        to_write += a.top_image.src

        # translate(string.maketrans(string.punctuation, ' ' * len(string.punctuation)))

        if len(to_write.strip()) > 0:
            output = open(dir_to_write + os.sep + a.title, 'wb')
            output.write(to_write.encode('utf-8'))
        else:
            print 'caught ^_^', x
    except Exception as e:
        print e
        print 'miss a link --!', x

    print '=== end ==\n'


if __name__ == '__main__':
    corpus_name = sys.argv[1]
    corpus_file = open(corpus_name)
    folder_name = corpus_name + '_dir'
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        print 'Found and deleted'
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)
    urls = []
    while 1:
        lines = corpus_file.readlines(100000)
        if not lines:
            break
        for line in lines:
            urls.append(line.strip())

    for x in xrange(len(urls)):
        catchpg(urls[x], folder_name)
