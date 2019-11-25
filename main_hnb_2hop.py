# ignore all future warnings from sklearn
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import argparse, os

import numpy as np
from model.hnb_2h import *

# Data to BiGraph
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from utils.utils import *

parser = argparse.ArgumentParser(description='Training_Config')

parser.add_argument('--batch_size', default=4096, type=int, help='Batchsize of each mini-batch.')
parser.add_argument('--epoch', default=150, type=int, help='Total # epochs to train.')
parser.add_argument('--dataset', default=amazon-book, type=str, help='Dataset.')
parser.add_argument('--eta', default=3, type=int, help='Number of high-order neighbors to sample.')
parser.add_argument('--emb_size', default=100, type=int, help='Embedding size.')
parser.add_argument('--save_epoch', default=5, type=int, help='Save embedding per # epochs.')
parser.add_argument('--negative_ratio', default=5, type=int, help='Negative examples for each positive example.')

args = parser.parse_args()


'''
Example of edge dataset style:
    Node Node
    ...
    u0   i13
    u199 i48343

'''

def constr_bigraph(data_path):
    # read data in Bipartite Graph
    
    _data = np.load(data_path)
    users = _data['users']
    items = _data['items']
    edges = _data['edges']
    train_edgelist = _data['train_edgelist']
#     test_edgelist = _data['test_edgelist']
    
    
    G = nx.Graph()
    G.add_nodes_from(users, bipartite=0)
    G.add_nodes_from(items, bipartite=1)
    G.add_edges_from(train_edgelist)
    
    print_info(nx.info(G), ['yellow', 'bold'])
    print_info('\nUsers: {}\nItems: {}\nWhole Dataset:{}\nTraining Data:{}'.format(len(users), len(items), len(edges), len(train_edgelist)), 
               ['white', 'bold'])

    return G
    
    
if __name__ == "__main__":
    
    dataset_dict = {
        'citeulike': './data/citeulike-a_edgelist.npz',
        'gowalla'    : './data/gowalla.npz',
        'amzbook'    : './data/amazon_book.npz',
        'ML-1M'      : './data/ml-1m.npz'
    }
    
    data_name = args.dataset
    data_path = dataset_dict[data_name]
    G = constr_bigraph(data_path)

    # init model
    model = HNB(G, data_name, embedding_size=args.emb_size, negative_ratio=args.negative_ratio,
                alpha=0.1, lamb=0.5, lamb_V=0.025, k=2, nbs=args.nbs, save_epoch=args.save_epoch) 
    
    # train model
    model.train(batch_size=args.batch_size, epochs=args.epoch, verbose=2)
    
    # get embedding vectors
    embeddings = model.get_embeddings()
    
    print_info('\n>>>> Saving Embeddings.....', ['white', 'bold'])
    np.savez('./saved_embeddings/HNB_2hop_{}_Final'.format(data_name), embeddings)
    print_info('>>>> Successfully Saved Embeddings!', ['yellow', 'bold'])
    
    