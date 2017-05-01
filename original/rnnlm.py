import theano
if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from softmax import softmax
from updates import *


class RNNLM(object):
    def __init__(self, n_input, n_hidden, n_output):
        self.x = T.imatrix('batched_sequence_x')  # n_batch, maxlen
        self.y = T.imatrix('batched_sequence_y')
        self.y_mask = T.matrix('y_mask')

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        init_Embd = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_output),
                                                 high=np.sqrt(1. / n_output),
                                                 size=(n_output, n_input)),
                               dtype=theano.config.floatX)
        self.E = theano.shared(value=init_Embd, name='word_embedding',borrow=True)




        self.rng = RandomStreams(1234)
        self.build()

    def build(self):
        output_layer = softmax(self.n_input, self.n_output, self.E[self.x,:])
        cost = self.categorical_crossentropy(output_layer.activation, self.y)
        self.params = [self.E, ]
        self.params += output_layer.params

        lr = T.scalar("lr")
        gparams = [T.clip(T.grad(cost, p), -10, 10) for p in self.params]
        updates = sgd(self.params, gparams, lr)

        self.train = theano.function(inputs=[self.x, self.y, self.y_mask, lr],
                                     outputs=cost,
                                     updates=updates)

        self.forward=theano.function(inputs=[self.x,self.y,self.y_mask],
                                     outputs=cost)

    def categorical_crossentropy(self, y_pred, y_true):
        y_true = y_true.flatten()
        nll = T.nnet.categorical_crossentropy(y_pred, y_true)
        return T.sum(nll * self.y_mask.flatten()) / T.sum(self.y_mask)
