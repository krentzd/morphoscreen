import numpy as np

def index(input_maps, input_choices):
    idx_list_ = []
    for maps, choices in zip(input_maps, input_choices):    
        idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))
    return np.logical_and.reduce(idx_list_)

def cosine_similarity(A, B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))