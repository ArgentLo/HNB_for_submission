import sys

import numpy as np
from utils import *
import argparse, os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

start_t = time.time()

parser = argparse.ArgumentParser(description='Eval_Multiprocseeing')
parser.add_argument('--saved_emb', default=None, type=str, help='Path to the saved embedding.')
parser.add_argument('--data_path', default='./data/ml-1m.npz',
                    type=str, help='Path to data containing Train and Test edges.')
parser.add_argument('--top_k', default=20, type=int, help='Top-K recommendation.')
parser.add_argument('--partition', default=10, type=int, help='Item_Matrix Partition.')
parser.add_argument('--emb_size', default=100, type=int, help='Embedding size.')
args = parser.parse_args()

data_path = args.data_path
saved_emb = args.saved_emb

trained_emb = np.load(saved_emb, allow_pickle=True)['arr_0']
trained_emb = trained_emb.tolist()

_data       = np.load(data_path, 
                      allow_pickle=True)
test_set    = _data['test_edgelist']
train_set   = _data['train_edgelist']
items       = _data['items']
users       = _data['users']

dataset_name = args.data_path.split('/')[-1]
print_info('\nDataset: {}\nUsers: {}\nItems: {}'.format(dataset_name, len(users), len(items)), ['yellow', 'bold'])

test_ui  = dict()
train_ui = dict()

for row in test_set:
    if row[0] not in test_ui.keys():
        test_ui[row[0]] = [row[1]]
    else:
        test_ui[row[0]].append(row[1])
        
for row in train_set:
    if row[0] not in train_ui.keys():
        train_ui[row[0]] = [row[1]]
    else:
        train_ui[row[0]].append(row[1])

top_k = args.top_k
emb_size = args.emb_size
idx2name_user = {}
name2idx_item = {}
idx2name_item = {}
u_mtx = np.zeros((len(users), emb_size), dtype=np.float32)
i_mtx = np.zeros((len(items), emb_size), dtype=np.float32)
train_mask = np.ones((len(users), len(items)), dtype=np.float32)

for i_idx in range(len(items)):
    idx2name_item[i_idx] = items[i_idx]
    name2idx_item[items[i_idx]] = i_idx
    i_mtx[i_idx] = trained_emb[items[i_idx]]
for u_idx in range(len(users)):
    idx2name_user[u_idx] = users[u_idx]
    u_mtx[u_idx] = trained_emb[users[u_idx]]
    for i_name in train_ui[users[u_idx]]:
        train_mask[u_idx][name2idx_item[i_name]] = 0.

print_info('Start computing similarity...', ['blue', 'bold'])

parts = len(u_mtx)//args.partition

for p in tqdm(range(args.partition + 1)):

    # RUN on CPU
    cur_u_mtx = torch.from_numpy(u_mtx[p*parts:min((p+1)*parts, len(u_mtx))])
    cur_i_mtx = torch.from_numpy(i_mtx)
    cur_mask  = torch.from_numpy(train_mask[p*parts:min((p+1)*parts, len(u_mtx)), :])
    
    cur_dist_mtx = - torch.mm(cur_u_mtx, cur_i_mtx.t())
    cur_dist_mtx = cur_dist_mtx  * cur_mask
    cur_dist_mtx = cur_dist_mtx
    cur_dist_mtx = torch.argsort(cur_dist_mtx, dim=1)[:, :top_k]
    cur_dist_mtx = cur_dist_mtx.numpy()
    
    if p == 0:
        dist_mtx = cur_dist_mtx
    else:
        dist_mtx = np.concatenate((dist_mtx, cur_dist_mtx), axis=0)


print_info('Completed computing similarity.', ['blue', 'bold'])

print_info('Evaluating.....', ['blue', 'bold'])
count = 0
prec_cnt = 0
ndcg  = 0

for u_idx in range(len(users)):
    
    u_name    = idx2name_user[u_idx]
    rec_items = [idx2name_item[i_idx] for i_idx in dist_mtx[u_idx]]
    denom     = len(test_ui[u_name])
    prec_denom= top_k
    
    cur_count, rel_list = count_repeated(rec_items, test_ui[u_name])
    count    += cur_count / denom
    prec_cnt += cur_count / prec_denom
    ndcg     += ndcg_at_k(rel_list)
    
avg_recall = count/len(users)
avg_prec   = prec_cnt/len(users)
avg_ndcg   = ndcg/len(users)

print_info('Total Eval Time : {:.1f}s.'.format(time.time() - start_t), ['white', 'bold'])
print_info('Top_{} Results of Emb_Name: {} \n - Recall: {:.4f}  Prec: {:.4f}  NDCG: {:.4f}'.format(top_k, args.saved_emb, avg_recall, avg_prec, avg_ndcg), ['yellow', 'bold'])
