import sys
import math
import random
import time

import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow.python.keras import layers, optimizers, activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda, Add, Activation
from tensorflow.python.keras.models import Model
from tensorflow.keras import regularizers

from utils import *


def line_loss(y_true, y_pred):
    '''
    y_true = np.vstack([k_weight, odw]).T
    '''

    r1 = layers.multiply([y_true[:,1], y_pred])
    r2 = K.sigmoid(r1)
    r3 = K.log(r2)
    r4 = layers.multiply([y_true[:,0], r3])
    result = - K.mean(r4)
    
    return result


def create_model(numNodes, embedding_size, lamb_V, nbs):
    
    u     = Input(shape=(1,))
    pos   = Input(shape=(1,))
    neg   = Input(shape=(1,))
    # neighbors to spread and Sum
    pos_s = Input(shape=(nbs,)) 
    neg_s = Input(shape=(nbs,))
    # transmisison factor
    trans_f2 = Input(shape=(1,))
    trans_f3 = Input(shape=(1,))
    train_type = Input(shape=(1,))

    # No reg
    vertex_emb  = Embedding(numNodes, embedding_size, 
                            name='vertex_emb')
    
    u_emb   = vertex_emb(u)
    pos_emb = vertex_emb(pos)
    neg_emb = vertex_emb(neg)
    pos_s_emb = vertex_emb(pos_s) # size=(3,bs,100)
    neg_s_emb = vertex_emb(neg_s) # size=(3,bs,100)
    
    trans_f1_emb = vertex_emb(u)
    trans_f2_emb = vertex_emb(trans_f2)
    trans_f3_emb = vertex_emb(trans_f3)
    
    
    # DS pair score
    BPR_score = Lambda(lambda x: x[0]*x[1]-x[0]*x[2], 
                       name='BPR_diff')([u_emb, pos_emb, neg_emb])
    
    # Sum high-order neighbors
    pn_s_diff   = Lambda(lambda x: tf.reduce_sum(x[0]-x[1], axis=1, keep_dims=True), 
                         name='PN_s_diff')([pos_s_emb, neg_s_emb])
    BPR_s_score = Lambda(lambda x: (1/nbs) * x[0] * x[1], 
                         name='BPR_s_diff')([u_emb, pn_s_diff])
    
    # Transmission Attention 
    Atten_mult    = Lambda(lambda x: K.switch(K.equal(x[-1], 1),
                                             x[0] * x[1],
                                             x[0] * x[1] * x[2]),
                           name='Atten_switch')([trans_f1_emb, trans_f2_emb, trans_f3_emb, train_type])
    
    Atten_softmax = Activation(activations.softmax, name='Atten_softmax')(Atten_mult)
    
    '''
    Output: 1 of 3 
        - Direct     * 1/Emb_size (type=0)
        - 1-hop      * Atten      (type=1)
        - 2-hop(sum) * Atten      (type=2)
    '''
    Atten_Score = Lambda(lambda x: K.switch(K.equal(x[-1], 1), 
                                     tf.reduce_sum(x[0]*x[2], axis=-1, keep_dims=False), 
                                     tf.reduce_sum(x[1]*x[2], axis=-1, keep_dims=False)),
                       name='Sum_switch')([BPR_score, BPR_s_score, Atten_softmax, train_type])
    
    score       = Lambda(lambda x: K.switch(K.equal(x[-1], 0), 
                                     tf.reduce_sum(x[0], axis=-1, keep_dims=False), 
                                     x[1]*embedding_size),
                       name='Final_switch')([BPR_score, Atten_Score, train_type])
    
    model = Model(inputs=[u, pos, neg, pos_s, neg_s, 
                          trans_f2, trans_f3,
                          train_type], 
                  outputs=score)

    return model, vertex_emb

    
class HNB:
    def __init__(self, graph, data_name, embedding_size=100, negative_ratio=5, alpha=0.1,
                 lamb=0.05, lamb_V=0.025, k=2, nbs=3,save_epoch=5):

        
        self.lr         = alpha
        self.lamb       = lamb  # for NS loss
        self.lamb_V     = lamb_V  # L2_reg
        self.k          = k-1
        self.nbs        = nbs # neighbors to spread
        self.save_epoch = save_epoch
        self.graph      = graph
        self.data_name  = data_name

        # idx2node:LIST ; node2idx:DIST

        self.idx2node, self.node2idx, self.user_idx, self.item_idx, self.degree_w = preprocess_bigraph_degree(graph)

        self.emb_size = embedding_size
        self._embeddings = {}
        self.negative_ratio = negative_ratio
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()

        self.samples_per_epoch = self.edge_size * (1*self.negative_ratio + 2*self.nbs*self.negative_ratio)
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
            self.node_size, self.emb_size, self.lamb_V, self.nbs)
    
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
        all_nodes = set(self.graph.nodes())

        neg_size = self.negative_ratio
        h = []
        t = []
        sign = 0
        count = 0
        start_index = 0
        start_time = time.time()
        step = 0
        num_walk = 0
        num_k = 0
        neg_counts = [0,0,0]
        step_incre = (1*(self.negative_ratio)+2*(self.nbs*self.negative_ratio))
        end_index = min(start_index + self.batch_size, data_size)
        
        print('data_Size: ', data_size)

        while True:
            
            '''
            ##################    Sampling    #####################
            
            inputs=[u, pos, neg, pos_s, neg_s, trans_f2, trans_f3, train_type]
            '''

            if mod == 'DS':
                
                if neg_counts[0] == 0:
                    h = []
                    t = []
                    odw_h = []
                    odw_t = []
                    for i in range(start_index, end_index):
                        cur_h = edges[shuffle_edge_ids[i]][0]
                        cur_t = edges[shuffle_edge_ids[i]][1]
                        odw_h.append(self.degree_w[cur_h])
                        odw_t.append(self.degree_w[cur_t])
                        h.append(cur_h)
                        t.append(cur_t)
                    # OWD 
                    odw_h, odw_t = np.array(odw_h), np.array(odw_t)
                
                # Neg
                neg = np.random.randint(0, len(self.idx2node), size=len(h))
                odw_neg = np.array([self.degree_w[idx] for idx in neg])
                odw = np.power(odw_h*odw_t*odw_neg, 1/3)
                odw = (odw-np.min(odw))/(np.max(odw)-np.min(odw))

                neg_counts[0] += 1
                
                if neg_counts[0] >= self.negative_ratio:
                    mod = 'h_info_flow'
                
                # dummy
                k_weight           = np.ones(len(h))
                pos_s, neg_s       = np.ones((len(h), self.nbs)), np.ones((len(h), self.nbs)) #size=(1024,3)
                trans_f2, trans_f3 = np.ones(len(h)), np.ones(len(h))
                
                train_type = np.zeros(len(h), dtype=int) # type=0
                # inputs=[u, pos, neg, pos_s, neg_s, trans_f2, trans_f3, train_type]
                yield ([np.array(h), np.array(t), np.array(neg), pos_s, neg_s, trans_f2, trans_f3, train_type],
                       np.vstack([k_weight, odw]).T)
                
                
            
            elif mod == 'h_info_flow':

                if neg_counts[1] == self.negative_ratio:
                    neg_counts[1] = 0
                    if num_walk == self.nbs-1 and num_k == self.k-1:
                        mod = 't_info_flow'
                        num_k = 0
                        num_walk = 0
                    elif num_k == self.k-1:
                        num_walk += 1
                        num_k    =  0
                    else:
                        num_k += 1
                    
                # k-step, nbs-walk RandomWalk
                if num_k == 0 and neg_counts[1] == 0:

                    ns_list = [[] for _ in range(self.nbs)]
                    sources = t
                    for src in sources:
                        neighbor = random.choice(list(self.graph.neighbors(self.idx2node[src])))
                        ns_list[num_walk].append(self.node2idx[neighbor])                
                    odw_pos = np.array([self.degree_w[idx] for idx in ns_list[num_walk]])
                
                # -> if depth is not enough : 2-hop (type=1)
                if num_k == self.k-1 and neg_counts[1] < self.negative_ratio:
            
                    # Neg
                    neg = np.random.randint(0, len(self.idx2node), size=len(h))
                    odw_neg = np.array([self.degree_w[idx] for idx in neg])
                    odw = np.power(odw_h*odw_pos*odw_neg, 1/3)
                    odw = (odw-np.min(odw))/(np.max(odw)-np.min(odw))
                    
                    neg_counts[1] += 1
                    
                    # variables
                    k_weight           = np.ones(len(h)) * np.power(1/2, 3)
                    pos_s, neg_s       = np.ones((len(h), self.nbs)), np.ones((len(h), self.nbs)) #size=(1024,3)
                    trans_f2, trans_f3 = np.array(t), np.ones(len(h))
                    train_type         = np.ones(len(h), dtype=int) #type=1

                    yield ([np.array(h), np.array(ns_list[num_walk]), np.array(neg), pos_s, neg_s, trans_f2, trans_f3, train_type], 
                           np.vstack([k_weight, odw]).T)
                
            elif mod == 't_info_flow':

                if neg_counts[2] == self.negative_ratio:
                    neg_counts[2] = 0
                    if num_walk == self.nbs-1 and num_k == self.k-1:
                        mod = 'next'
                        num_k = 0
                        num_walk = 0
                    elif num_k == self.k-1:
                        num_walk += 1
                        num_k    =  0
                    else:
                        num_k += 1
                    
                # k-step, nbs-walk RandomWalk
                if num_k == 0 and neg_counts[2] == 0:

                    ns_list = [[] for _ in range(self.nbs)] 
                    sources = h
                    for src in sources:
                        neighbor = random.choice(list(self.graph.neighbors(self.idx2node[src])))
                        ns_list[num_walk].append(self.node2idx[neighbor])
                    odw_pos = np.array([self.degree_w[idx] for idx in ns_list[num_walk]])

                
                # -> if depth is not enough : 2-hop (type=1)
                if num_k == self.k-1 and neg_counts[2] < self.negative_ratio:
            
                    # Neg
                    neg = np.random.randint(0, len(self.idx2node), size=len(h))
                    odw_neg = np.array([self.degree_w[idx] for idx in neg])
                    odw = np.power(odw_t*odw_pos*odw_neg, 1/3)
                    odw = (odw-np.min(odw))/(np.max(odw)-np.min(odw))
                    neg_counts[2] += 1
                    
                    # variables
                    k_weight           = np.ones(len(h)) * np.power(1/2, 3)
                    pos_s, neg_s       = np.ones((len(h), self.nbs)), np.ones((len(h), self.nbs)) #size=(1024,3)
                    trans_f2, trans_f3 = np.array(h), np.ones(len(h))
                    train_type         = np.ones(len(h), dtype=int) #type=1

                    yield ([np.array(t), np.array(ns_list[num_walk]), np.array(neg), pos_s, neg_s, trans_f2, trans_f3, train_type], 
                           np.vstack([k_weight, odw]).T)
                
            '''
            ##################    Sampling    #####################
            '''
                
            if mod == 'next':
                
                neg_counts = [0,0,0]
                mod = 'DS'
                
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

                start_time = time.time()

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

                    np.savez('./saved_embeddings/HNB_2hop_{}_{}'.format(self.data_name, count), self._embeddings)
                
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