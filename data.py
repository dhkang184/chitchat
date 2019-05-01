# -*- coding: utf-8 -*-
EN_WHITELIST = "0123456789abcdefghijklmnopqrstuvwxyz?.!~,' " # space is included in whitelist
EN_BLACKLIST = '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~\''

FILENAME = 'ko0529.txt'
limit = {
        'maxq' : 30,
        'minq' : 1,
        'maxa' : 30,
        'mina' : 2
        }

token_limit = {
                'maxq' : 30,
                'minq' : 1,
                'maxa' : 30,
                'mina' : 1}

UNK = '<unk>'
VOCAB_SIZE = 20000

import random
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import csv
import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle

if __name__=="__main__":
    print "__file__" + __file__

def ddefault():
    return 1

'''
 read lines from file
     return [list of lines]

'''
def read_lines(filename):
    return open(filename).read().split('\n')[:-1]

def csv_read_lines(filename):
    csv_list = []
    csv_f = open(filename, 'r')
    csvReader = csv.reader(csv_f)
    for x in csvReader:
        csv_list.append(x[1])
        csv_list.append(x[2])
    return csv_list

'''
 split sentences in one line
  into multiple lines
    return [list of lines]

'''
def split_line(line):
    return line.split('.')


'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)

'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2
    if len(sequences) %2 == 1:
        sequences = sequences[:-1]
    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i+1])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a





'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, token_limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, token_limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, token_limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, token_limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


def process_data():

    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)
    #lines = csv_read_lines(filename= csv_filename)
    # change to lower case (just for en)
    lines = [ line.lower() for line in lines ]

    print('\n:: Sample from read(p) lines')
    print(lines[121:125])

    # filter out unnecessary characters
    print('\n>> Filter lines')
    t_lines=[]
    """
    for x in lines :
        x = x.replace('?',' ?')
        x = x.replace('!',' !')
        x = x.replace('.',' .')
        x = x.replace(',',' ,')
        x = x.replace('  ', ' ')
        t_lines.append(x)
    """
    for x in lines :
        x = x.replace('?','')
        x = x.replace('!','')
        x = x.replace('.','')
        x = x.replace(',','')
        x = x.replace('-','')
        x = x.replace('  ', ' ')
        t_lines.append(x)        
    #lines = [ filter_line(line, EN_WHITELIST) for line in t_lines ]
    lines = t_lines
    print lines[121:125]
    #for i, line in enumerate(lines):
    #    lines[i] = ' '.join(line)
        

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(lines)
    print('\nq : {0} ; a : {1}'.format(qlines[60], alines[60]))
    print('\nq : {0} ; a : {1}'.format(qlines[61], alines[61]))


    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [ wordlist.split(' ') for wordlist in qlines ]
    atokenized = [ wordlist.split(' ') for wordlist in alines ]
    #qtokenized = [ ' '.join(wordlist).split(' ') for wordlist in qlines ]
    #atokenized = [ ' '.join(wordlist).split(' ') for wordlist in alines ]
    print('\n:: Sample from segmented list of words')
    print('\nq : {0} ; a : {1}'.format(qtokenized[60], atokenized[60]))
    print('\nq : {0} ; a : {1}'.format(qtokenized[61], atokenized[61]))
    
    


    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)
    """
    for x in range(len(tidx2w)):
        try:
            a = w2idx[tidx2w[x]]
        except KeyError:
            idx2w.append(tidx2w[x])
        if len(idx2w) == 20000:
            break
    """
    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n >> Save numpy arrays to disk')
    # save them
    
    np.save('test_q.npy', idx_q)
    np.save('test_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('test.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata_test.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_ta = np.load(PATH + 'idx_test.npy')
    idx_en = np.load(PATH + 'idx_test.npy')
    return metadata, idx_ta, idx_en


if __name__ == '__main__':
    process_data()

