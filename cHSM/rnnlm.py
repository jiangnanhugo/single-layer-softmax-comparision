from theano.tensor.shared_randomstreams import RandomStreams



from level_softmax import level_softmax
from updates import *

class RNNLM:
    def __init__(self,n_input,n_output,optimizer='sgd'):
        self.x=T.imatrix('batched_sequence_x')  # n_batch, maxlen
        self.y=T.imatrix('batched_sequence_y')
        self.y_mask=T.matrix('y_mask')
        
        self.n_input=n_input
        self.n_output=n_output
        init_Embd=np.asarray(np.random.uniform(low=-np.sqrt(1./n_output),
                                               high=np.sqrt(1./n_output),
                                               size=(n_output,n_input)),
                           dtype=theano.config.floatX)
        self.E=theano.shared(value=init_Embd,name='word_embedding')

        self.optimizer=optimizer
        self.n_batch=T.iscalar('n_batch')

        self.epsilon=1.0e-15
        self.rng=RandomStreams(1234)
        self.build()

    def build(self):

        output_layer=level_softmax(self.n_input,self.n_output,self.E[self.x,:],self.y)
        self.params=[self.E,]
        self.params+=output_layer.params

        cost=self.categorical_crossentropy(output_layer.activation)
        lr=T.scalar("lr")
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]
        updates=sgd(self.params,gparams,lr)



        self.train=theano.function(inputs=[self.x,self.y,self.y_mask,lr],
                                   outputs=cost,
                                   updates=updates)


    def categorical_crossentropy(self,y_pred):
        return T.sum(y_pred*self.y_mask.flatten())/T.sum(self.y_mask)
    
