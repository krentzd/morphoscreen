import numpy as np
from sklearn import metrics

def index(input_maps, input_choices):
    idx_list_ = []
    for maps, choices in zip(input_maps, input_choices):    
        idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))
    return np.logical_and.reduce(idx_list_)

def cosine_similarity(A, B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def get_acc_dict(data_dict):
    acc_dict = dict()
    for p_idx in range(6):
        for l in data_dict['cmpd_labels']:
            if l not in acc_dict.keys():
                acc_dict[l] = []
            acc = metrics.accuracy_score(data_dict[(f'P{p_idx}', l, 'preds')], data_dict[(f'P{p_idx}', l, 'labels')])
            acc_dict[l].append(acc)

    return acc_dict