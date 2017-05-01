import time

from rnnlm import *
from utils import TextIterator,load_model


import sys
import os
import numpy
numpy.set_printoptions(threshold=numpy.nan)

lr=0.01
n_batch=20
NEPOCH=200

n_input=256
maxlen=50
train_datafile='../data/wikitext-103/idx_wiki.train.tokens'
filepath='../data/wikitext-103/frequenties.pkl'
n_words_source=-1
vocabulary_size=267736
brown_or_huffman='huffman'
matrix_or_vector='matrix'


n_words_source=-1


def train(lr):
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,filepath,
        n_words_source=n_words_source,
        n_batch=n_batch,
        brown_or_huffman=brown_or_huffman,
        mode=matrix_or_vector,
        maxlen=maxlen)
        
    print 'building model...'
    model=RNNLM(n_input,vocabulary_size,mode=matrix_or_vector)
    print 'training start...'
    for epoch in xrange(NEPOCH):
        gain=0
        for x,x_mask,(y_node,y_choice,y_bit_mask),y_mask in train_data:
            cur_time=time.time()
            model.train(x,y_node,y_choice,y_bit_mask,y_mask,lr)
            #model.forward(x,y_node,y_choice,y_bit_mask,y_mask)
            gain=time.time()-cur_time
            print gain
        time_list.append(gain)



if __name__ == '__main__':
    train(lr=lr)
