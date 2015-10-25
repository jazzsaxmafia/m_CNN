import theano
import theano.tensor as T

import numpy as np

from keras import initializations
from theano.sandbox.cuda import dnn


class m_CNN():
    def __init__(self, n_words, dim_emb, dim_img):
        self.n_words = n_words
        self.dim_emb = dim_emb
        self.dim_img = dim_img

        self.emb_W = initializations.uniform((n_words, dim_emb))

        self.cnn_word_W1 = initializations.uniform((dim_emb*3 + dim_img, 200))
        self.cnn_word_W2 = initializations.uniform((200*3, 300))
        self.cnn_word_W3 = initializations.uniform((300*3, 300))

        self.cnn_phs_W1 = initializations.uniform((dim_emb*3, 200))
        self.cnn_phs_W2 = initializations.uniform((200*3 + dim_img, 300))
        self.cnn_phs_W3 = initializations.uniform((300*3, 300))

        self.cnn_phl_W1 = initializations.uniform((dim_emb*3, 200))
        self.cnn_phl_W2 = initializations.uniform((200*3, 300))
        self.cnn_phl_W3 = initializations.uniform((300*3 + dim_img, 300))

        self.cnn_st_W1 = initializations.uniform((dim_emb*3, 200))
        self.cnn_st_W2 = initializations.uniform((200*3, 300))
        self.cnn_st_W3 = initializations.uniform((300*3, 300))

    def convolving(self, inp, weight, n_samples, input_dim):
        inp_shuffle = inp.dimshuffle(1,0,2)
        n_timestep = inp_shuffle.shape[1]

        inp_shuffle = inp_shuffle.reshape((n_samples, -1))

        output, _ = theano.scan(
                fn=lambda timestep: T.dot(inp_shuffle[:,(timestep-1)*input_dim:(timestep+2)*input_dim], weight),
                sequences=T.arange(1, n_timestep-1)
                )

        return output

    def multi_convolving(self, inp1, inp2, weight, n_samples, input_dim):
        inp_shuffle = inp1.dimshuffle(1,0,2)
        n_timestep = inp_shuffle.shape[1]

        inp_shuffle = inp_shuffle.reshape((n_samples, -1))

        output, _ = theano.scan(
                fn=lambda timestep: T.dot(T.concatenate([inp_shuffle[:,(timestep-1)*input_dim:(timestep+2)*input_dim], inp2], axis=1), weight),
                sequences=T.arange(1, n_timestep-1)
                )

        return output

    def pooling(self, inp, input_dim):
        inp_shuffle = inp.dimshuffle(1,0,2)
        n_timestep = inp_shuffle.shape[1]

        output, _ = theano.scan(
                fn=lambda timestep: T.max(inp_shuffle[:,timestep:timestep+1,:], axis=1),
                sequences=T.arange(0, n_timestep/2)*2
                )
        return output

    def build_model(self):
        image = T.matrix('image')
        sentence = T.imatrix('sentence')

        n_samples, n_timestep = sentence.shape
        emb = self.emb_W[sentence.flatten()]
        emb = emb.reshape((n_samples, n_timestep, -1))
        emb = emb.dimshuffle(1,0,2)

        cnn_word_output1 = self.multi_convolving(emb, image, self.cnn_word_W1, n_samples, self.dim_emb)
        cnn_word_output1 = self.pooling(cnn_word_output1, 200)
        cnn_word_output2 = self.convolving(cnn_word_output1, self.cnn_word_W2, n_samples, 200)
        cnn_word_output2 = self.pooling(cnn_word_output2, 300)
        cnn_word_output3 = self.convolving(cnn_word_output2, self.cnn_word_W3, n_samples, 300)
        cnn_word_output3 = self.pooling(cnn_word_output3, 300)
        cnn_word_output3 = cnn_word_output3.dimshuffle(1,0,2).reshape((n_samples, -1))

        cnn_phs_output1 = self.convolving(emb, self.cnn_phs_W1, n_samples, self.dim_emb)
        cnn_phs_output1 = self.pooling(cnn_phs_output1, 200)
        cnn_phs_output2 = self.multi_convolving(cnn_phs_output1, image, self.cnn_phs_W2, n_samples, 200)
        cnn_phs_output2 = self.pooling(cnn_phs_output2, 300)
        cnn_phs_output3 = self.convolving(cnn_phs_output2, self.cnn_phs_W3, n_samples, 300)
        cnn_phs_output3 = self.pooling(cnn_phs_output2, 300)
        cnn_phs_output3 = cnn_phs_output3.dimshuffle(1,0,2).reshape((n_samples, -1))

        cnn_phl_output1 = self.convolving(emb, self.cnn_phl_W1, n_samples, self.dim_emb)
        cnn_phl_output1 = self.pooling(cnn_phl_output1, 200)
        cnn_phl_output2 = self.convolving(cnn_phl_output1, self.cnn_phl_W2, n_samples, 200)
        cnn_phl_output2 = self.pooling(cnn_phl_output2, 300)
        cnn_phl_output3 = self.multi_convolving(cnn_phl_output2, image, self.cnn_phl_W3, n_samples, 300)
        cnn_phl_output3 = self.pooling(cnn_phl_output3, 300)
        cnn_phl_output3 = cnn_phl_output3.dimshuffle(1,0,2).reshape((n_samples, -1))

        cnn_st_output1 = self.convolving(emb, self.cnn_st_W1, n_samples, self.dim_emb)
        cnn_st_output1 = self.pooling(cnn_st_output1, 200)
        cnn_st_output2 = self.convolving(cnn_st_output1, self.cnn_st_W2, n_samples, 200)
        cnn_st_output2 = self.pooling(cnn_st_output2, 300)
        cnn_st_output3 = self.convolving(cnn_st_output2, self.cnn_st_W3, n_samples, 300)
        cnn_st_output3 = self.pooling(cnn_st_output3, 300)
        cnn_st_output3 = T.concatenate([cnn_st_output3.dimshuffle(1,0,2).reshape((n_samples, -1)), image], axis=1)

        ff = theano.function(
                inputs=[image, sentence],
                #outputs=cnn_phs_output3,
                outputs=[cnn_word_output3, cnn_phs_output3, cnn_phl_output3, cnn_st_output3],
                allow_input_downcast=True)

        return ff

m_cnn = m_CNN(n_words=5000, dim_emb=512, dim_img=4096)

dummy_image = np.random.randn(10, 4096)
dummy_sentence = np.random.randint(5000, size=[10, 50])

ff = m_cnn.build_model()
