import time

from rnnlm import *
from utils import TextIterator,save_model

lr=0.01
p=0.5
n_batch=50
NEPOCH=100

n_input=100
n_hidden=250
maxlen=100
optimizer='sgd'
train_datafile='../ptb/idx_ptb.train.txt'
valid_datafile='../ptb/idx_ptb.valid.txt'
test_datafile='../ptb/idx_ptb.test.txt'
n_words_source=-1
vocabulary_size=10001


def train():
    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_words_source=n_words_source,n_batch=n_batch,maxlen=maxlen)
    test_data=TextIterator(test_datafile,n_words_source=n_words_source,n_batch=n_batch,maxlen=maxlen)

    print 'building model...'
    model=RNNLM(n_input,vocabulary_size,optimizer)
    print 'training start...'
    start=time.time()
    for epoch in xrange(NEPOCH):
        for x,x_mask,y,y_mask in train_data:
            model.train(x,y,y_mask,lr)

    print "Finished. Time = ",(time.time()-start)/NEPOCH


if __name__ == '__main__':
    train()
