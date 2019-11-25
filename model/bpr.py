import sys
import math
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, optimizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda, Add
from tensorflow.python.keras.models import Model
from tensorflow.keras import regularizers

from utils import *

def line_loss(y_true, y_pred):

    r1 = y_true * y_pred
    r2 = K.sigmoid(r1)
    r3 = K.log(r2)
    result = - K.mean(r3)
    return result


def create_model(numNodes, embedding_size, lamb_V):
    
    u     = Input(shape=(1,))
    pos   = Input(shape=(1,))
    neg   = Input(shape=(1,))
    
    vertex_emb  = Embedding(numNodes, embedding_size, 
                            name='vertex_emb')
    
    user     = vertex_emb(u)
    pos_item = vertex_emb(pos)
    neg_item = vertex_emb(neg)

    # pos
    pos_mal = Lambda(lambda x: x[0]*x[1], name='pos')([user, pos_item])
    
    # neg
    neg_mal = Lambda(lambda x: x[0]*x[1], name='neg')([user, neg_item])
    
    # subtract + sum_dim
    score = Lambda(lambda x: tf.reduce_sum( x[0] - x[1], axis=-1, keep_dims=False), name='score')([pos_mal, neg_mal])

    model = Model(inputs=[u, pos, neg], outputs=score)

    return model, vertex_emb

    
class BPR:
    def __init__(self, graph, data_name, embedding_size=100, negative_ratio=5, alpha=0.1,
                 lamb=0.05, lamb_V=0.025, k=2, save_epoch=5):

        
        self.lr         = alpha
        self.lamb       = lamb  # for NS loss
        self.lamb_V     = lamb_V  # L2_reg
        self.k          = k
        self.save_epoch = save_epoch
        self.graph      = graph
        self.data_name  = data_name
        
        '''
        # idx2node:LIST ; node2idx:DIST
        
        ##########################################
        
        ADD code : return Embedding for User, Item seperately.
        
        ##########################################
        '''
        self.idx2node, self.node2idx, self.user_idx, self.item_idx = preprocess_bigraph(graph)
        
        self.emb_size = embedding_size
        self._embeddings = {}

        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        # 3 : k_step_NS_h, k_step_NS_t, DS
        self.samples_per_epoch = self.edge_size * negative_ratio
        self.negative_ratio = negative_ratio

        self.reset_model()

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = (
            (self.samples_per_epoch - 1) // self.batch_size + 1)*times
        
        print('> steps_per_epoch: ', self.steps_per_epoch)
        print('> edge_size: ', self.edge_size)
        print('> samples/epoch: ', self.samples_per_epoch)
        print('> batch_size: ', self.batch_size)
        

    def reset_model(self,opt='sgd'):

        self.model, self.embedding_dict = create_model(
            self.node_size, self.emb_size, self.lamb_V)
        adam = tf.keras.optimizers.Adam(lr=0.0005, clipvalue=0.20)
        self.model.compile(adam, line_loss)
        
        print(self.model.summary())
       
        self.batch_it = self.batch_iter(self.node2idx)
       
    def batch_iter(self, node2idx):
        
        # construct edges [(idx, idx) ... ]
        edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph.edges()]
        data_size = self.graph.number_of_edges()
        shuffle_edge_ids = np.random.permutation(np.arange(data_size))
        
        # positive or negative mod => 0:POS, 1~ratio:NEG
        mod = 'neg_sampling'
        neg_size = self.negative_ratio
        user = []
        pos  = []
        neg  = []
        sign = 0
        count = 0
        start_index = 0
        step = 0
        num_walk = 0
        neg_counts = [0,0,0]
        step_incre = (1*(1+self.negative_ratio)+2*(self.k+self.negative_ratio))
        end_index = min(start_index + self.batch_size, data_size)
        
        print('data_Size: ', data_size)
        
        while True:
            
            '''
            ##################    Sampling    #####################
            '''
            
            # Positive Samping (mod:0)
            if mod == 'neg_sampling':
                
                neg = []
                
                if neg_counts[0] == 0:
                    user = []
                    pos  = []

                    for i in range(start_index, end_index):

                        cur_u = edges[shuffle_edge_ids[i]][0]
                        cur_p = edges[shuffle_edge_ids[i]][1]
                        user.append(cur_u)
                        pos.append(cur_p)
            
                # Neg Sampling
                for u in user:
                    
                    cur_neg = self.idx2node[random.choice(self.item_idx)]
                    neg_check = cur_neg not in list(self.graph.neighbors(self.idx2node[u]))
                    while neg_check is not True:
                        cur_neg = self.idx2node[random.choice(self.item_idx)]
                        neg_check = cur_neg not in list(self.graph.neighbors(self.idx2node[u]))
                    
                    # append the idx
                    neg.append(self.node2idx[cur_neg])
                    
                lamb = np.ones(len(user))
                neg_counts[0] += 1
                
                if neg_counts[0] >= self.negative_ratio:
                    mod = 'next'
                    
                yield ([np.array(user), np.array(pos), np.array(neg)], lamb)

            '''
            ##################    Sampling    #####################
            '''
                
            if mod == 'next':

                neg_counts = [0,0,0]
                
                mod = 'neg_sampling'
                
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index >= data_size:
                count += 1
                if count % self.save_epoch == 0:
                    
#                     print_info('Count: {} ({}/{})'.format(count,start_index,data_size), ['yellow', 'bold'])
                    
                    self._embeddings = {}
                    # Get trained_Emb from Keras.Layers.Embedding            
                    embeddings = self.embedding_dict.get_weights()[0]
                    idx2node = self.idx2node
                    for i, embedding in enumerate(embeddings):
                        self._embeddings[idx2node[i]] = embedding

                    np.savez('./saved_embeddings/BPR_{}_{}'.format(self.data_name, count), self._embeddings)
                
                step = 0
                user = []
                pos  = []
                neg  = []
                shuffle_edge_ids = np.random.permutation(np.arange(data_size))
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)

         
                
    def get_embeddings(self,):
        
        self._embeddings = {}
        
        # Get trained_Emb from Keras.Layers.Embedding            
        embeddings = self.embedding_dict.get_weights()[0]
        
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding
        
        return self._embeddings

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        self.reset_training_config(batch_size, times)
        
        hist = self.model.fit_generator(self.batch_it, epochs=epochs, initial_epoch=initial_epoch, steps_per_epoch=self.steps_per_epoch,
                                        verbose=verbose)

        return hist
    
