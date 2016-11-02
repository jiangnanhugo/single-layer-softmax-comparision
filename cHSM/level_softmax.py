import numpy as np
import theano
import theano.tensor as T
import logging
from theano.tensor.nnet import h_softmax

class level_softmax:
    def __init__(self,n_input,n_output,x,y):
        self.n_input=n_input
        self.n_output=n_output

        self.level1_size=np.ceil(np.sqrt(n_output)).astype('int32')
        self.level2_size=np.ceil(n_output/(self.level1_size-1)).astype('int32')
        print n_input
        print "level1_size=%d level2_size=%d",self.level1_size,self.level2_size
        assert self.level1_size*self.level2_size>=n_output

        self.logitx_shape=x.shape
        self.logity_shape=y.shape
        self.x=x.reshape([self.logitx_shape[0]*self.logitx_shape[1],self.logitx_shape[2]])
        self.y=y.reshape([self.logity_shape[0],self.logity_shape[1]])

        init_W1=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input,self.level1_size)),dtype=theano.config.floatX)
        init_b1=np.zeros((self.level1_size),dtype=theano.config.floatX)

        init_W2=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(self.level1_size,n_input,self.level2_size)),dtype=theano.config.floatX)
        init_b2=np.zeros((self.level1_size,self.level2_size),dtype=theano.config.floatX)

        self.W1=theano.shared(value=init_W1,name='output_W1')
        self.b1=theano.shared(value=init_b1,name='output_b1')
        self.W2=theano.shared(value=init_W2,name='output_W2')
        self.b2=theano.shared(value=init_b2,name='output_b2')

        self.params=[self.W1,self.b1,self.W2,self.b2]

        self.build()

    def build(self):
        '''
        the input is always 3-dimensional:
            the first dimension is the sequences,
            the second dimension are n_batch,
            the third dimension is n_hidden.
        :return:
        '''
        self.activation = h_softmax(self.x,
                                    self.x.shape[0], self.n_output,
                                    self.level1_size,self.level2_size,
                                    self.W1, self.b1, self.W2, self.b2,
                                    self.y)

        predicted=h_softmax(self.x, self.x.shape[0], self.x.shape[1], self.level1_size,
                         self.level2_size, self.W1, self.b1, self.W2, self.b2)
        self.prediction=T.argmax(predicted,axis=1)


def test_h_softmax():
    input_size = 4
    batch_size = 2
    h_softmax_level1_size = 5
    h_softmax_level2_size = 3
    output_size = h_softmax_level1_size * h_softmax_level2_size

    #############
    # Initialize shared variables
    #############

    floatX = theano.config.floatX
    shared = theano.shared

    # First level of h_softmax
    W1 = np.asarray(np.random.normal(
        size=(input_size, h_softmax_level1_size)), dtype=floatX)
    W1 = shared(W1)
    b1 = shared(np.asarray(np.zeros((h_softmax_level1_size,)),
                              dtype=floatX))

    # Second level of h_softmax
    W2 = np.asarray(np.random.normal(
        size=(h_softmax_level1_size, input_size, h_softmax_level2_size)),
        dtype=floatX)
    W2 = shared(W2)
    b2 = shared(
        np.asarray(np.zeros((h_softmax_level1_size,
                                   h_softmax_level2_size)), dtype=floatX))

    #############
    # Build graph
    #############
    x = T.matrix('x')
    y = T.ivector('y')

    # This only computes the output corresponding to the target
    y_hat_tg = h_softmax(x, batch_size, output_size, h_softmax_level1_size,
                         h_softmax_level2_size, W1, b1, W2, b2, y)

    # This computes all the outputs
    y_hat_all= h_softmax(x, batch_size, output_size, h_softmax_level1_size,
                         h_softmax_level2_size, W1, b1, W2, b2)

    #############
    # Compile functions
    #############
    fun_output_tg = theano.function([x, y], y_hat_tg)
    fun_output = theano.function([x], y_hat_all)

    #############
    # Test
    #############
    x_mat = np.random.normal(size=(batch_size, input_size)).astype(floatX)
    y_mat = np.random.randint(0, output_size, batch_size).astype('int32')
    tg_output = fun_output_tg(x_mat, y_mat)
    all_outputs = fun_output(x_mat)

    assert(tg_output.shape == (batch_size,))
    assert(all_outputs.shape == (batch_size, output_size))

    # Verifies that the outputs computed by fun_output_tg are the same as those
    # computed by fun_output.
    #utt.assert_allclose(all_outputs[np.arange(0, batch_size), y_mat], tg_output)
        

        
        
