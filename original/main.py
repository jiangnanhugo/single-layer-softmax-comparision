import time
import os
from rnnlm import *
from utils import TextIterator
import sys

lr=0.001
NEPOCH=20

n_input=256
n_hidden=256


lr=0.01
n_batch=20
NEPOCH=20

n_input=256
maxlen=50
optimizer='sgd'
train_datafile='../data/wikitext-103/idx_wiki.train.tokens'
n_words_source=-1
vocabulary_size=267736


def train():
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_words_source=n_words_source,n_batch=n_batch,maxlen=maxlen)
    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size)
    print 'training start...'
    time_list=[]
    for epoch in xrange(NEPOCH):
        gain=0
        for x,x_mask,y,y_mask in train_data:
            cur_time=time.time()
            model.train(x,y,y_mask,lr)
            #model.forward(x,y,y_mask)
            gain=time.time()-cur_time
            print gain
        time_list.append(gain)

    print time_list


if __name__ == '__main__':
    train()
