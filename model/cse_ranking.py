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
    '''
    y_true[0]: -1 or +1 (indicating pos/neg samples)
    y_true[1]: lamb (lamb * NS_loss)
    '''

    r1 = y_true[0][0] * y_pred
    r2 = K.sigmoid(r1)
    r3 = K.log(r2)
    result = y_true[0][1] * - K.mean(r3)

    return result


def create_model(numNodes, embedding_size, lamb_V):
    
    u     = Input(shape=(1,))
    pos   = Input(shape=(1,))
    neg   = Input(shape=(1,))
    train_type = Input(shape=(1,))

    # No reg
    vertex_emb  = Embedding(numNodes, embedding_size, 
                            name='vertex_emb')
    context_emb = Embedding(numNodes, embedding_size, 
                            name='context_emb')
    
    
    u_emb   = vertex_emb(u)
    pos_emb = vertex_emb(pos)
    neg_emb = vertex_emb(neg)
    
    pos_ctx = context_emb(pos)
    neg_ctx = context_emb(neg)
    
    # DS pair score
    DS_score = Lambda(lambda x: x[0]*x[1]-x[0]*x[2], name='DS_SCORE')([u_emb, pos_emb, neg_emb])
    
    # NS pair
    NS_score = Lambda(lambda x: x[0]*x[1]-x[0]*x[2], name='NS_SCORE')([u_emb, pos_ctx, neg_ctx])
    
    score = Lambda(lambda x: K.switch(K.equal(x[2], 1), 
                                     tf.reduce_sum(x[0], axis=-1, keep_dims=False), 
                                     tf.reduce_sum(x[1], axis=-1, keep_dims=False)),
                       name='switch')([DS_score, NS_score, train_type])

    model = Model(inputs=[u, pos, neg, train_type], outputs=score)

    return model, vertex_emb

    
class CSE:
    def __init__(self, graph, data_name, embedding_size=100, negative_ratio=5, alpha=0.1,
                 lamb=0.05, lamb_V=0.025, k=2, save_epoch=5):

        
        self.lr         = alpha
        self.lamb       = lamb  # for NS loss
        self.lamb_V     = lamb_V  # L2_reg
        self.k          = k
        self.save_epoch = save_epoch
        self.graph      = graph
        self.data_name  = data_name

        # idx2node:LIST ; node2idx:DIST

        self.idx2node, self.node2idx, self.user_idx, self.item_idx = preprocess_bigraph(graph)

        self.emb_size = embedding_size
        self._embeddings = {}
        self.negative_ratio = negative_ratio

        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        # 3 : k_step_NS_h, k_step_NS_t, DS
        self.samples_per_epoch = self.edge_size * (1*(negative_ratio)+2*(self.k*negative_ratio))
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
        mod = 'DS'
        neg_size = self.negative_ratio
        h = []
        t = []
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

            if mod == 'DS':

                if neg_counts[0] == 0:
                    h = []
                    t = []

                    for i in range(start_index, end_index):

                        cur_h = edges[shuffle_edge_ids[i]][0]
                        cur_t = edges[shuffle_edge_ids[i]][1]
                        h.append(cur_h)
                        t.append(cur_t)

            # Negative Sampling
                neg = np.random.randint(0, len(self.idx2node), size=len(h))
                neg_counts[0] += 1
            
                if neg_counts[0] >= self.negative_ratio:
                    mod = 'NS_h'

                lamb = np.ones(len(h))
                sign = np.ones(len(h))
                train_type = np.ones(len(h), dtype=int)
    
                yield ([np.array(h), np.array(t), neg, train_type], np.vstack([sign, lamb]).T)

            elif mod == 'NS_h':

                # k-step RandomWalk
                if num_walk == 0 and neg_counts[1] == 0:
                    ns_list = [[] for _ in range(self.k)]
                    sources = h

                    for src in sources:
                        neighbor = random.choice(list(self.graph.neighbors(self.idx2node[src])))
                        ns_list[num_walk].append(self.node2idx[neighbor])

                neg = np.random.randint(0, len(self.idx2node), size=len(h))
                neg_counts[1] += 1                    


                lamb = np.ones(len(h)) * self.lamb
                sign = np.ones(len(h))
                train_type = np.ones(len(h), dtype=int)*2
                
                
                if num_walk == self.k-1 and neg_counts[1] >= self.negative_ratio:
                    mod = 'NS_t'
                    num_walk = 0
                    yield ([np.array(h), np.array(ns_list[-1]), neg, train_type], np.vstack([sign, lamb]).T)
    
                elif neg_counts[1] < self.negative_ratio:
                    yield ([np.array(h), np.array(ns_list[num_walk]), neg, train_type], np.vstack([sign, lamb]).T)
                
                else:
                    sources = ns_list[num_walk]
                    num_walk += 1
                    neg_counts[1] = 0
                    for src in sources:
                        neighbor = random.choice(list(self.graph.neighbors(self.idx2node[src])))
                        ns_list[num_walk].append(self.node2idx[neighbor])
                    
                    yield ([np.array(h), np.array(ns_list[num_walk-1]), neg, train_type], np.vstack([sign, lamb]).T)
            
        

            elif mod == 'NS_t':
                if num_walk == 0 and neg_counts[2] == 0:
                    ns_list = [[] for _ in range(self.k)]
                    sources = t

                    for src in sources:
                        neighbor = random.choice(list(self.graph.neighbors(self.idx2node[src])))
                        ns_list[num_walk].append(self.node2idx[neighbor])

                neg = np.random.randint(0, len(self.idx2node), size=len(h))
                neg_counts[2] += 1                    


                lamb = np.ones(len(h)) * self.lamb
                sign = np.ones(len(h))
                train_type = np.ones(len(h), dtype=int)*2
                
                
                if num_walk == self.k-1 and neg_counts[2] >= self.negative_ratio:
                    mod = 'next'
                    num_walk = 0
                    yield ([np.array(t), np.array(ns_list[-1]), neg, train_type], np.vstack([sign, lamb]).T)
    
                elif neg_counts[2] < self.negative_ratio:
                    yield ([np.array(t), np.array(ns_list[num_walk]), neg, train_type], np.vstack([sign, lamb]).T)
                
                else:
                    sources = ns_list[num_walk]
                    num_walk += 1
                    neg_counts[2] = 0
                    for src in sources:
                        neighbor = random.choice(list(self.graph.neighbors(self.idx2node[src])))
                        ns_list[num_walk].append(self.node2idx[neighbor])

                    yield ([np.array(t), np.array(ns_list[num_walk-1]), neg, train_type], np.vstack([sign, lamb]).T)

                
            '''
            ##################    Sampling    #####################
            '''
                
            if mod == 'next':
                
                neg_counts = [0,0,0]
                
                mod = 'DS'
                
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

                    np.savez('./saved_embeddings/CSE_Rank_{}_{}'.format(self.data_name, count), self._embeddings)
                
                step = 0
                h = []
                t = []
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