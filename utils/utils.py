from termcolor import cprint
import numpy as np


def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info,str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info,list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)


def preprocess_bigraph_degree(graph):
    node2idx = {}
    idx2node = []
    user_idx = []
    item_idx = []
    degree_w = []  # degree[idx]: weight
    
    node_idx = 0
    for node in graph.nodes():
        node2idx[node] = node_idx
        idx2node.append(node)
        degree_w.append(graph.degree(node)+2) # Degree+2

#         print('NODE: ', graph.node[node]['bipartite'])
        
        if graph.node[node]['bipartite'] == 0:
            user_idx.append(node_idx)
        else:
            item_idx.append(node_idx)

        node_idx += 1
#     # normalize the degree_weight
#     log_dw   = 1 / np.log(degree_w)
#     norm_dw  = (log_dw-min(log_dw))/(max(log_dw)-min(log_dw))
#     degree_w = norm_dw + 0.7
    degree_w = 1 / np.log(degree_w)
    
    return idx2node, node2idx, user_idx, item_idx, degree_w


def preprocess_bigraph(graph):
    node2idx = {}
    idx2node = []
    user_idx = []
    item_idx = []
    
    node_idx = 0
    for node in graph.nodes():
        node2idx[node] = node_idx
        idx2node.append(node)
        
#         print('NODE: ', graph.node[node]['bipartite'])
        
        if graph.node[node]['bipartite'] == 0:
            user_idx.append(node_idx)
        else:
            item_idx.append(node_idx)

        node_idx += 1
    return idx2node, node2idx, user_idx, item_idx


def top_k_rec(i, rec_list, u_name, top_k, train_ui):

    while len(rec_list[u_name]) < top_k:
        if i not in train_ui[u_name]:
            rec_list[u_name].append(i)

def count_repeated(rec_it, test_it):
    # relevant list for NDCG
    rel_list = []
    count = 0.
    
    for i in rec_it:
        if i in test_it:
            rel_list.append(1.)
            count += 1.
        else:
            rel_list.append(0.)
    return count, rel_list

def dcg_at_k(r):
    r = np.asfarray(r)
    if r.size:
#         if method == 0:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
#         elif method == 1:
#         return np.sum(r / np.log2(np.arange(2, r.size + 2)))
#     return 0.


def ndcg_at_k(r):
    dcg_max = dcg_at_k(sorted(r, reverse=True))
    if not dcg_max:
        return 0.
    return dcg_at_k(r) / dcg_max


def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]

    
