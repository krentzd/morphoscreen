import numpy as np
import matplotlib.pyplot as plt
from utils import index, cosine_similarity, get_acc_dict

def plot_classification_accuracies(data_dict):
    acc_dict = get_acc_dict(data_dict)
    
    bar_labels = []
    bar_vals = []

    color_dict = data_dict['color_dict']

    bar_data = []
    for key in acc_dict.keys():
        bar_data.append((key, np.median(acc_dict[key])))

    bar_labels = [x[0] for x in sorted(bar_data, key=lambda x: x[1])]
    
    for key in bar_labels: 
        bar_vals.append(np.median(acc_dict[key]))
    
    plt.figure(figsize=(3.5, 7))
    bar = plt.barh([i for i in range(len(bar_labels))], bar_vals, alpha=0.95)
    
    for b_id, c_n in enumerate(bar_labels):
        col_n = color_dict[c_n]
        bar[b_id].set_color(col_n)
    
    for i, c_n in enumerate(bar_labels):
    
        for d, m in zip(acc_dict[c_n], ['o', 's', '^', 'D', 'P', 'p']): 
            plt.scatter([d], [i], marker=m, alpha=0.5, edgecolor='k', c=[color_dict[c_n]])
    
    plt.yticks([i for i in range(len(bar_labels))], bar_labels)
    plt.vlines(1/6, -0.5, 12.5, linestyle='dashed', color='k')
    plt.xlabel('Median classification accuracy')
    plt.title('MoA classification accuracy')
    plt.ylim([-0.5, 12.5])
    plt.show()


def make_prediction_matrix(labels, preds, classes_true, classes_pred, mode='normalize'):
    from matplotlib.ticker import MultipleLocator

    def return_counts_array(labels, preds):
        counts_array = np.zeros((int(max(labels)) + 1, int(max(preds)) + 1))
        for l in np.unique(labels):
            x = np.array(preds)[np.array(labels) == l]
            p = np.unique(x, return_counts=True)
            if p[0].size > 0:
                for p_idx, p_val in zip(p[0], p[1]):
                    counts_array[l, p_idx] = p_val
        return counts_array

    counts_array = return_counts_array(labels, preds)
    counts_array_temp = np.zeros_like(counts_array)
    for i, row in enumerate(counts_array):
        counts_array_temp[i] = row / row.sum()

    if mode == 'normalize':
        counts_array = counts_array_temp
        counts_array_ = counts_array_temp

    else:
        counts_array_ = counts_array_temp
        
    fig, ax = plt.subplots(figsize=(3.5,7))
    ax.matshow(counts_array_, cmap=plt.cm.Blues)
    plt.gca().xaxis.tick_bottom()

    ax.set_xticklabels([''] + classes_pred, rotation=90)
    ax.set_yticklabels([''] + classes_true)

    for i in range(counts_array.shape[1]):
        for j in range(counts_array.shape[0]):
            c = counts_array[j,i]
            if mode == 'normalize':
                ax.text(i, j, f'{c:.2f}', va='center', ha='center', c='black' if c < 0.5 else 'white')
            else:
                ax.text(i, j, f'{int(c):d}', va='center', ha='center', c='black' if c < 0.1 * counts_array.max() else 'white')

    plt.title('Confusion matrix')

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel('Predicted MoA label')
    plt.ylabel('True compound label')
    plt.show()

def plot_confusion_matrix(data_dict, cmpd_vs_moa=True):
    if cmpd_vs_moa:
        make_prediction_matrix(
            labels=data_dict['labels'],
            preds=data_dict['preds'],
            classes_true=data_dict['cmpd_labels'],
            classes_pred=data_dict['moa_labels'],
            mode=''
        )
    else:
        make_confusion_matrix(
            data_dict['labels_moa'], 
            data_dict['preds'], 
            data_dict['classes_moa'], 
            data_dict['classes_moa']
        )

def plot_cosine_similarity_matrix(data_dict):
    sim_matrix = np.empty((len(data_dict['srtd_cmpd_labels']), len(data_dict['srtd_cmpd_labels'])))
    
    for x, fvec1 in enumerate(data_dict['feat_vecs_med']):
        for y, fvec2 in enumerate(data_dict['feat_vecs_med']):
            sim_matrix[x,y] = cosine_similarity(fvec1, fvec2)
    
    plt.figure(figsize=(5,5))
    plt.matshow(sim_matrix, cmap='coolwarm', fignum=0)
    plt.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    plt.xticks([i for i in range(len(data_dict['srtd_cmpd_labels']))], data_dict['srtd_cmpd_labels'])
    plt.xticks(rotation=90)
    plt.yticks([i for i in range(len(data_dict['srtd_cmpd_labels']))], data_dict['srtd_cmpd_labels'])
    plt.title('Cosine similarity of median feature vectors')
    plt.show()

def get_umap(data, n_components=2, n_neighbors=200, min_dist=0.1, metric='cosine'):
    import umap
    umap_ = umap.UMAP(n_components=n_components, random_state=1, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    umap_data = umap_.fit_transform(data)
    return umap_data

def _plot_umap_antibiotics(data_dict):

    color_dict = data_dict['color_dict']
    centroid_labels_name = []
    V_centroid = np.empty((len(data_dict['labels_names']), 16), dtype=data_dict['feat_vecs'].dtype)
    for l_idx, label_ in enumerate(data_dict['labels_names']):
        idx = index([data_dict['labels_as_name']], [[label_]])
        centroid_labels_name.append(label_)
        V_centroid[l_idx] = np.median(data_dict['feat_vecs'][idx], axis=0)
    
    feat_vecs_w_centroid = np.vstack([data_dict['feat_vecs'], V_centroid])

    X_w_centroid = get_umap(feat_vecs_w_centroid) 

    X = X_w_centroid[:len(data_dict['labels_as_name'])]
    X_c = X_w_centroid[len(data_dict['labels_as_name']):]


    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7,7))
    
    for label_ in data_dict['labels_train']:
        idx = index([data_dict['labels_as_name']], [[label_]])
        X_ = X[idx]
        ax.scatter(X_[:,0], X_[:,1], alpha=0.5, color=color_dict[label_], edgecolor='grey')
    
    for label_ in data_dict['labels_test']:
        idx = index([data_dict['labels_as_name']], [[label_]])
        X_ = X[idx]
        ax.scatter(X_[:,0], X_[:,1], alpha=0.5, color=color_dict[label_], edgecolor='grey', s=100, marker='P')
    
    for labels_id_ in data_dict['labels_train']:
        idx = index([centroid_labels_name], [[labels_id_]])
        X_ = X_c[idx]
    
        ax.scatter(X_[:,0], X_[:,1], label=labels_id_, alpha=1, facecolor=color_dict[labels_id_], edgecolor='black', s=100) 
    
    for labels_id_ in data_dict['labels_test']:
        idx = index([centroid_labels_name], [[labels_id_]])
        X_ = X_c[idx]
    
        ax.scatter(X_[:,0], X_[:,1], label=labels_id_, alpha=1, facecolor=color_dict[labels_id_], edgecolor='black', s=150, marker='P') 
        
    ax.legend(frameon=True, ncol=2, fontsize=8, labelspacing=1)
    
    ax.set_xticks([])
    ax.set_xlabel('UMAP 1')
    ax.set_yticks([])
    ax.set_ylabel('UMAP 2')    
    plt.show()

def _plot_umap_mutants_and_drugs(data_dict):
    
    color_dict = data_dict['color_dict']
    
    centroid_labels = []
    centroid_labels_name = []
    
    labels_names = np.unique(data_dict['labels_as_name'])
    V_centroid = np.empty((len(labels_names), 16), dtype=data_dict['feat_vecs'].dtype)
    for l_idx, label_ in enumerate(labels_names):
        idx = index([data_dict['labels_as_name']], [[label_]])
        centroid_labels_name.append(label_)
        V_centroid[l_idx] = np.median(data_dict['feat_vecs'][idx], axis=0)
    
    feat_vecs_w_centroid = np.vstack([data_dict['feat_vecs'], V_centroid])

    X_w_centroid = get_umap(feat_vecs_w_centroid, n_neighbors=50, min_dist=0.01, metric='cosine')
    
    X = X_w_centroid[:len(data_dict['labels_as_name'])]
    X_c = X_w_centroid[len(data_dict['labels_as_name']):]
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7,7))
    
    for label_ in data_dict['mutant_labels']:
        idx = index([data_dict['labels_as_name']], [[label_]])
        X_ = X[idx]
        ax.scatter(X_[:,0], X_[:,1], alpha=0.5, color=color_dict[label_], edgecolor='grey', s=100, marker='P')
    
    for label_ in data_dict['drug_labels']:
        idx = index([data_dict['labels_as_name']], [[label_]])
        X_ = X[idx]
        ax.scatter(X_[:,0], X_[:,1], alpha=0.5, color=color_dict[label_], edgecolor='grey')
    
    for labels_id_ in data_dict['mutant_labels']:
        idx = index([centroid_labels_name], [[labels_id_]])
        X_ = X_c[idx]
    
        ax.scatter(X_[:,0], X_[:,1], label=data_dict['name_dict'][labels_id_], alpha=1, facecolor=color_dict[labels_id_], edgecolor='black', s=150, marker='P')
        
    for labels_id_ in data_dict['drug_labels']:
        idx = index([centroid_labels_name], [[labels_id_]])
        X_ = X_c[idx]

        ax.scatter(X_[:,0], X_[:,1], label=data_dict['name_dict'][labels_id_], alpha=1, facecolor=color_dict[labels_id_], edgecolor='black', s=150)
        
    ax.legend(frameon=True, ncol=2, fontsize=8, labelspacing=1)
    ax.set_xticks([])
    ax.set_xlabel('UMAP 1')
    ax.set_yticks([])
    ax.set_ylabel('UMAP 2')
    plt.show()

def _plot_umap_cellcycle(data_dict):
    
    centroid_labels = []
    centroid_labels_name = []
    
    labels_names = np.unique(data_dict['labels_as_name'])
    
    V_centroid = np.empty((len(labels_names), 16), dtype=data_dict['feat_vecs'].dtype)
    
    for l_idx, label_ in enumerate(labels_names):
        idx = index([data_dict['labels_as_name']], [[label_]])
        centroid_labels.append(l_idx)
        centroid_labels_name.append(label_)
        V_centroid[l_idx] = np.median(data_dict['feat_vecs'][idx], axis=0)
    
    feat_vecs_w_centroid = np.vstack([data_dict['feat_vecs'], V_centroid])
    
    X_w_centroid = get_umap(feat_vecs_w_centroid, n_neighbors=50, min_dist=0.1, metric='cosine')
    
    X = X_w_centroid[:len(data_dict['labels_as_name'])]
    X_c = X_w_centroid[len(data_dict['labels_as_name']):]
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7,7))
    
    for label_ in data_dict['labels']:
        idx = index([data_dict['labels_as_name']], [[label_]])
        X_ = X[idx]
        ax.scatter(X_[:,0], X_[:,1], alpha=0.5, color=data_dict['color_dict'][label_], edgecolor='grey', s=100, marker='p')
    
    for labels_id_ in data_dict['labels']:
        idx = index([centroid_labels_name], [[labels_id_]])
        X_ = X_c[idx]
    
        ax.scatter(X_[:,0], X_[:,1], label=data_dict['name_dict'][labels_id_], alpha=1, facecolor=data_dict['color_dict'][labels_id_], edgecolor='black', s=150, marker='p') #color=cmap(norm(labels_id_))
        
    ax.legend(frameon=True, ncol=2, fontsize=8, labelspacing=1)
    
    ax.set_xticks([])
    ax.set_xlabel('UMAP 1')
    ax.set_yticks([])
    ax.set_ylabel('UMAP 2')
    plt.show()

def _plot_umap_fine_tuning(data_dict, model='fine-tuned'):
    
    labels_names = np.unique(data_dict[model]['labels_as_name'])
    
    centroid_labels_name = []
    V_centroid = np.empty((len(labels_names), data_dict[model]['feat_vecs'].shape[-1]), dtype=data_dict[model]['feat_vecs'].dtype)
    for l_idx, label_ in enumerate(labels_names):
        idx = index([data_dict[model]['labels_as_name']], [[label_]])
        centroid_labels_name.append(label_)
        V_centroid[l_idx] = np.median(data_dict[model]['feat_vecs'][idx], axis=0)
    
    feat_vecs_w_centroid = np.vstack([data_dict[model]['feat_vecs'], V_centroid])

    X_w_centroid = get_umap(feat_vecs_w_centroid)

    X = X_w_centroid[:len(data_dict[model]['labels_as_name'])]
    X_c = X_w_centroid[len(data_dict[model]['labels_as_name']):]
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7,7))
    
    for label_ in labels_names:
        idx = index([data_dict[model]['labels_as_name']], [[label_]])
        X_ = X[idx]
        ax.scatter(X_[:,0], X_[:,1], alpha=0.5, color=data_dict['color_dict'][label_], edgecolor='grey')
    
    for labels_id_ in labels_names:
        idx = index([centroid_labels_name], [[labels_id_]])
        X_ = X_c[idx]
    
        ax.scatter(X_[:,0], X_[:,1], label=labels_id_, alpha=1, facecolor=data_dict['color_dict'][labels_id_], edgecolor='black', s=100) 
    
    ax.legend(frameon=True, ncol=2, fontsize=9, labelspacing=1)
    
    ax.set_xticks([])
    ax.set_xlabel('UMAP 1')
    ax.set_yticks([])
    ax.set_ylabel('UMAP 2')
    plt.show()

    
def plot_umap(data_dict, experiment='antibiotics', **kwargs):
    if experiment == 'antibiotics':
        _plot_umap_antibiotics(data_dict)
    elif experiment == 'mutants-drugs':
        _plot_umap_mutants_and_drugs(data_dict)
    elif experiment == 'cellcycle':
        _plot_umap_cellcycle(data_dict)
    elif experiment == 'fine-tuning':
        _plot_umap_fine_tuning(data_dict, **kwargs)

    
def plot_hierarchical_clustering(data_dict):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from collections import Counter
    import matplotlib
    
    Z = linkage(data_dict['feat_vecs'], method='ward')
    
    # Plot the dendrogram
    fig = plt.figure(figsize=(2.5, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    labels = [''] * len(data_dict['labels_as_name'])

    threshold = 6
    matplotlib.rcParams['lines.linewidth'] = 1
    dn = dendrogram(Z, color_threshold=threshold,
                    orientation='left',
                    ax=ax)
    leaf_labels = np.array(dn['ivl'])
    leaf_labels_moa = np.array([data_dict['labels_moa_as_name'][i] for i in dn['leaves']])
    cluster_labels = np.array(dn['leaves_color_list'])
    
    midpoint_idx_list = []
    most_common_list = []
    count_fraction_list = []
    y_cluster_rec_size = []
    for c_label in np.unique(cluster_labels):
        cluster_idx = index([cluster_labels], [[c_label]])
        leaf_labels_moa_ = leaf_labels_moa[cluster_idx]
        num_labels = len(leaf_labels_moa_)
        y_cluster_rec_size.append(num_labels)
        most_common = Counter(leaf_labels_moa_).most_common(1)[0][0]
        max_count = Counter(leaf_labels_moa_).most_common(1)[0][1]
        midpoint_idx = np.where(cluster_idx)[0].size // 2 + np.where(cluster_idx)[0][0]
        midpoint_idx_list.append(midpoint_idx)
        most_common_list.append(most_common)
        count_fraction_list.append(max_count / num_labels)
    
    midpoint_leaf_labels = [''] * len(data_dict['labels_as_name'])
    
    for idx, l, c in zip(midpoint_idx_list, most_common_list, count_fraction_list):
        midpoint_leaf_labels[idx] = f'{l} ({c:.2f})'
        
    ax.axvline(x=threshold, c='red', lw=1, linestyle='dashed', alpha=0.5, label='Threshold')
    ax.legend(fontsize=7)
    
    ax.set_yticklabels(midpoint_leaf_labels)
    ax.tick_params(axis='y', which='major', labelsize=7)
    ax.set_xticks([])
    plt.show()

    leaf_labels_cmpd = np.array([data_dict['labels_as_name'][i] for i in dn['leaves']])

    color_dict = data_dict['color_dict']

    for c_label, c_title in zip(np.unique(cluster_labels), [x for x in midpoint_leaf_labels if x not in ['']]):
        c_idx = index([cluster_labels], [[c_label]])
        c_labels_moa = leaf_labels_cmpd[c_idx]
        
        labels, sizes = np.unique(c_labels_moa, return_counts=True)
        
        colors = [color_dict[m] for m in labels]
        
        fig, ax = plt.subplots()
        patches, texts = ax.pie(sizes, 
               startangle=0,
               colors=colors,
               wedgeprops=dict(width=0.5))
        
        sort_legend = True
        if sort_legend:
            patches, lgnd_labels, dummy =  zip(*sorted(zip(patches, labels, sizes),
                                                  key=lambda x: x[2],
                                                  reverse=True))
        plt.title(c_title, fontsize=16, y=0.92)
        plt.show()

    feat_vecs_img = np.empty(data_dict['feat_vecs'].shape)
    
    y_rec_sizes = []
    for idx, l_idx in enumerate(dn['leaves'][::-1]):
        feat_vecs_img[idx,:] = data_dict['feat_vecs'][l_idx,:]
    
    fig, ax = plt.subplots(figsize=(2.5,5))
    ax.imshow(feat_vecs_img, aspect='auto', cmap='gray')
    plt.show()

def plot_clustermap(data_dict):
    from scipy import cluster
    import seaborn as sns
    
    labels_ = data_dict['mutant_labels'] + data_dict['drug_labels']

    
    feat_vecs_med = np.empty((len(labels_), 16), dtype=data_dict['feat_vecs'].dtype)
    centroid_labels_name = []
    for l_idx, label_ in enumerate(labels_):
        idx = index([data_dict['labels_as_name']], [[label_]])
        centroid_labels_name.append(label_)
        feat_vecs_med[l_idx] = np.median(data_dict['feat_vecs'][idx], axis=0)
    
    feat_vecs_med = feat_vecs_med - np.median(feat_vecs_med, axis=0)
    
    sim_matrix = np.empty((len(labels_), len(labels_)))
    
    for x, fvec1 in enumerate(feat_vecs_med):
        for y, fvec2 in enumerate(feat_vecs_med):
            sim_matrix[x,y] = cosine_similarity(fvec1, fvec2)
    
    linkage = cluster.hierarchy.linkage(feat_vecs_med, method='complete', metric='cosine', optimal_ordering=True) 
    
    sns.clustermap(data=sim_matrix, 
                   xticklabels=[data_dict['name_dict'][l] for l in labels_],
                   yticklabels=[data_dict['name_dict'][l] for l in labels_],
                   cmap='coolwarm',
                   row_linkage=linkage,
                   col_linkage=linkage,
                   figsize=(6.5,6.5),
                   tree_kws={'linewidth': 1.5},
                   dendrogram_ratio=0.15)
    plt.show()

def plot_mlp_preds(data_dict):
    from sklearn.neural_network import MLPClassifier
    from collections import Counter
    color_dict = data_dict['color_dict']
    
    for l_idx, l in enumerate(data_dict['test_labels']):
        cntr_preds_list = []
        for p_idx in [0,1,2]:
            
            train_idx = index([data_dict['labels_as_name'], data_dict['plate_id']], [data_dict['train_labels'], [p for p in [0, 1, 2] if p not in [p_idx]]])
            
            clf = MLPClassifier(hidden_layer_sizes=(6), random_state=5).fit(data_dict['embeddings'][train_idx], data_dict['labels_as_name'][train_idx])
    
            test_idx = index([data_dict['labels_as_name'], data_dict['plate_id']], [[l], [p_idx]])
                    
            pred = clf.predict(data_dict['embeddings'][test_idx])
            
            cntr_preds = Counter(pred)
            cntr_preds_list.append(cntr_preds)
    
        plt.figure(figsize=(2.5,1.5))
        bar_labels_ = [x[0] for x in (cntr_preds_list[0] + cntr_preds_list[1] + cntr_preds_list[2]).most_common(2)]
        bar_labels = [data_dict['name_dict'][x] for x in bar_labels_] + [''] * (2 - len(bar_labels_)) + ['Other']
        
        bar_vals_ = [np.mean([cntr_preds_list[i][bar_labels_[j]] for i in range(3)]) for j in range(2)]
        bar_vals = [x for x in bar_vals_] + [0] * (2 - len(bar_vals_))
        other_val = 50 - np.sum(bar_vals)
        bar_vals = bar_vals + [other_val]    
        
        plt.title(f"{data_dict['name_dict'][l]}", fontsize=14)
    
        datapoints = [[cntr_preds_list[i][bar_labels_[j]] for i in range(3)] for j in range(2)]
        datapoints += [[50 - np.array(datapoints)[:,i].sum() for i in range(3)]]
        
        bar = plt.barh([0,1,2], bar_vals, alpha=0.95)
        
        for b_id, c_n in enumerate(bar_labels_):
            col_n = color_dict[c_n]
            bar[b_id].set_color(col_n)
        
        for i, c_n in enumerate(bar_labels_ + ['tab:grey']):
            if c_n != 'tab:grey':
                c_n = color_dict[c_n]
            for d, m in zip(datapoints[i], ['o', 's', '^']):
                plt.scatter([d], [i], marker=m, c=c_n, alpha=0.5, edgecolor='k')
        
        bar[-1].set_color('tab:grey')
        plt.yticks([0,1,2], bar_labels, fontsize=12)
        plt.xlabel('Number of FOV', fontsize=12)
        plt.xticks([0,25,50], fontsize=12)
        plt.xlim([0,50])
        plt.gca().invert_yaxis()
        plt.show()

def plot_adjacency_graph(data_dict, model='cnn'):
    from sklearn.preprocessing import StandardScaler
    import networkx as nx
    import math 

    feat_vecs_med = np.empty((len(data_dict['labels']), data_dict['feat_vecs'].shape[-1]), dtype=data_dict['feat_vecs'].dtype)
    centroid_labels_name = []
    for l_idx, label_ in enumerate(data_dict['labels']):
        idx = index([data_dict['labels_as_name']], [[label_]])
        centroid_labels_name.append(label_)
        feat_vecs_med[l_idx] = np.median(data_dict['feat_vecs'][idx], axis=0)

    if model == 'cnn':
        feat_vecs_med = feat_vecs_med - feat_vecs_med.mean(axis=0)
    elif model == 'hand-crafted':
        scaler = StandardScaler()
        feat_vecs_med = scaler.fit_transform(feat_vecs_med)
    
    sim_matrix = np.empty((len(data_dict['labels']), len(data_dict['labels'])))
    
    for x, fvec1 in enumerate(feat_vecs_med):
        for y, fvec2 in enumerate(feat_vecs_med):
            sim_matrix[x,y] = cosine_similarity(fvec1, fvec2)

    colors = [data_dict['color_dict'][i] for i in data_dict['labels']]
    
    mut_labels = [data_dict['name_dict'][i].replace(' ', ' ') for i in data_dict['labels']]
    
    adj_matrix = sim_matrix.copy()
    np.fill_diagonal(adj_matrix, 0)

    if model == 'cnn':
        thresh_list = []
        for i in range(adj_matrix.shape[0]):
            thresh_list.append(np.max(np.unique(adj_matrix[i])))
        
        thresh = math.floor(min(thresh_list) * 100) / 100
    elif model == 'hand-crafted':
        thresh = 0.144
        
    mask = sim_matrix > thresh
    adjacency_matrix = sim_matrix * mask
    np.fill_diagonal(adjacency_matrix, 0)
    
    G = nx.from_numpy_array(adjacency_matrix)
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7,7))

    if model == 'cnn':
        pos = nx.spring_layout(G, k=0.4, iterations=50, weight='weight', seed=2)
    elif model == 'hand-crafted':
        pos = nx.spring_layout(G, k=0.5, iterations=25, weight='weight', seed=8)

    label_mapping = dict(zip([i for i in range(len(mut_labels))], [l for l in mut_labels]))
    nx.draw_networkx(G, with_labels=True, pos=pos, labels=label_mapping, node_color=colors, node_size=200, alpha=1, node_shape='o', font_size=8, width=2)
    edge_labels = nx.get_edge_attributes(G, "weight")
    edge_labels_ = dict()
    for key, val in zip(edge_labels.keys(), edge_labels.values()):
        edge_labels_[key] = f'{float(val):.2f}'
        
    nx.draw_networkx_edge_labels(G, pos, edge_labels_, font_size=8)
    plt.axis('equal')
    plt.title('Adjacency graph', fontsize=12)
    plt.show()

def plot_cell_densities(data_dict, t=6):
    import matplotlib.patches as mpatches

    plt.figure(figsize=(5,2))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['Replicate 1', 'Replicate 2', 'Replicate 3']
    xpos = [-0.55, 0, 0.55]
    
    for idx, rep in enumerate([1,2,3]):
        vp = plt.violinplot([data_dict['cell_density'][f'R{rep}', dose, f'{t}h'] for dose in ['0.5xMIC', '1xMIC', '5xMIC', '10xMIC', '50xMIC', '100xMIC']], positions=[i+xpos[idx] for i in range(1,12,2)])
        
        for pc in vp['bodies']:
            pc.set_facecolor(colors[idx])
    
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    plt.legend(handles=patches)
    
    plt.xticks([i for i in range(1,12,2)], ['0.5xMIC', '1xMIC', '5xMIC', '10xMIC', '50xMIC', '100xMIC'])
    plt.ylabel('Estimated cell density')
    plt.title(f'Cell density estimation from FM4 channel ({t}h)')
    plt.show()

def plot_accuracies(data_dict):
    from sklearn import metrics
    def _return_accuracy(t, m, r):
        idx = index([data_dict['time_id'], data_dict['mic_id'], data_dict['rep_id']], [[t], [m], [r]])
        return metrics.accuracy_score(data_dict['labels_moa'][idx], data_dict['preds'][idx])

    plt.figure(figsize=(5,5))

    data_6 = np.array([[_return_accuracy(0, m, r) for r in range(3)] for m in range(6)])
    data_mean_6 = np.asarray([np.mean(d) for d in data_6])
    data_std_6 = np.asarray([np.std(d) for d in data_6])
    
    data_16 = np.array([[_return_accuracy(1, m, r) for r in range(3)] for m in range(6)])
    data_mean_16 = np.asarray([np.mean(d) for d in data_16])
    data_std_16 = np.asarray([np.std(d) for d in data_16])
    
    plt.plot(data_mean_6, 'o-', alpha=0.95, c='tab:blue', label='t=6h')
    plt.fill_between([i for i in range(6)], data_mean_6 - data_std_6, data_mean_6 + data_std_6, alpha=0.2)
    
    plt.plot(data_mean_16, 'o-', alpha=0.95, c='tab:orange', label='t=16h')
    plt.fill_between([i for i in range(6)], data_mean_16 - data_std_16, data_mean_16 + data_std_16, alpha=0.2)
    
    plt.hlines(1/5, 0, 5, linestyle='dashed', color='k')
    plt.ylim([0, 1.05])
    plt.xticks([i for i in range(6)], [f'{m}xMIC' for m in [0.5, 1, 5, 10, 50, 100]])
    plt.ylabel('MoA classification accuracy')
    plt.xlabel('Concentration')
    plt.title('MoA clf. acc. by incubation time and concentration')
    
    plt.legend()
    plt.show()

def plot_preds(data_dict):
    for i in range(6):
        plt.figure(figsize=(1.5,1))
        bar_labels = data_dict['labels'][::-1]
        bar = plt.barh([0,1], data_dict[f'P{i}'][::-1], alpha=0.95)
        plt.yticks([0,1], bar_labels, fontsize=10)
        plt.xticks([0, 0.5, 1.00], fontsize=10)
        plt.xlabel('Clf. accuracy')
        for b_id, c_n in enumerate(['tab:brown', 'tab:grey'][::-1]):
            bar[b_id].set_color(c_n)
        
        plt.title(f'Clofazimine (Pl. {i + 1})', fontsize=10)
        plt.show()

def get_acc_list(data_dict, model='cnn'):
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from collections import Counter 
    
    acc_list = []    
    if model == 'cnn':
        for p_idx in range(6):
            max_labels = []
            max_preds = []
            for l in data_dict[model]['labels']:
                max_pred = Counter(data_dict[model][(f'P{p_idx}', l, 'preds')]).most_common(1)[0][0]
                max_label = data_dict[model][(f'P{p_idx}', l, 'labels')][0]
                max_labels.append(max_label)
                max_preds.append(max_pred)
                
            acc_list.append(metrics.accuracy_score(max_preds, max_labels))
            
    elif model == 'random-forest':
        for p_idx in range(6): 
    
            train_idx = index([data_dict[model]['labels'], data_dict[model]['plate_id']], [data_dict['cmpd_labels'], [p for p in [0, 1, 2, 3, 4, 5] if p not in [p_idx]]])
            test_idx = index([data_dict[model]['labels'], data_dict[model]['plate_id']], [data_dict['cmpd_labels'], [p_idx]])
        
            scaler = StandardScaler()
            feat_vecs_train = scaler.fit_transform(data_dict[model]['feat_vecs'][train_idx])
            feat_vecs_test = scaler.transform(data_dict[model]['feat_vecs'][test_idx])
            
            clf = RandomForestClassifier(n_estimators=1000, random_state=0).fit(feat_vecs_train, data_dict[model]['labels_moa'][train_idx])
        
            pred = clf.predict(feat_vecs_test)
            acc_list.append(metrics.accuracy_score(pred, data_dict[model]['labels_moa'][test_idx]))   

    return acc_list

def plot_cnn_vs_rf_benchmark(data_dict):
    from scipy import stats

    acc_seen_labels_cnn = get_acc_list(data_dict, model='cnn')
    acc_seen_labels_rf = get_acc_list(data_dict, model='random-forest')

    plt.figure(figsize=(2,4))
    bar = plt.bar([i for i in range(2)], [np.mean(acc_seen_labels_cnn), np.mean(acc_seen_labels_rf)], 
                  yerr=[np.std(acc_seen_labels_cnn), np.std(acc_seen_labels_rf)],
                  capsize=5)#, c=clr_list)

    print(f'RF accuracy: {np.mean(acc_seen_labels_rf)} +/- {np.std(acc_seen_labels_rf)}')
    print(f'CNN accuracy: {np.mean(acc_seen_labels_cnn)} +/- {np.std(acc_seen_labels_cnn)}')
    
    plt.hlines(1/6, -0.4, 1.4, linestyle='dashed', color='k')
    
    pval = stats.wilcoxon(acc_seen_labels_cnn, acc_seen_labels_rf).pvalue
    print(f'p={pval}')
    
    if pval < 0.5:
        plt.hlines(1.07, -0.001, 1.001, linestyle='solid', color='k', linewidth=0.75)
        plt.text(0.47, 1.07, '*', fontsize=14)
    
    plt.ylim([0,1.15])
    plt.xticks([0,1,], ['CNN', f'Random\nforest'], fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Well-level hold-out test accuracy')
    plt.title(f'MoA clf. accuracy on\npreviously seen drugs', fontsize=10)
    plt.show()

def _plot_rf_preds_antibiotics(data_dict):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    proba_unseen = dict()
    
    train_idx = index([data_dict['random-forest']['labels'], data_dict['random-forest']['plate_id']], [data_dict['cmpd_labels'], [p for p in [0, 1, 2, 3, 4, 5] if p not in [2]]])
    test_idx = index([data_dict['random-forest']['labels'], data_dict['random-forest']['plate_id']], [data_dict['dropped_labels'], [2]])
    
    scaler = StandardScaler()
    feat_vecs_train = scaler.fit_transform(data_dict['random-forest']['feat_vecs'][train_idx])
    feat_vecs_test = scaler.transform(data_dict['random-forest']['feat_vecs'][test_idx])
    
    clf = RandomForestClassifier(n_estimators=1000, random_state=0).fit(data_dict['random-forest']['feat_vecs'][train_idx], data_dict['random-forest']['labels_moa'][train_idx])
            
    for l in data_dict['dropped_labels']:
        test_idx_ = index([data_dict['random-forest']['labels'], data_dict['random-forest']['plate_id']], [[l], [2]])
        feat_vecs_test_ = scaler.transform(data_dict['random-forest']['feat_vecs'][test_idx_])
        proba = clf.predict_proba(feat_vecs_test_)
    
        proba_unseen[(2, l)] = proba[0]
    
    for l in data_dict['dropped_labels']:
        plt.figure(figsize=(2.5,1.5))
        bar_labels = clf.classes_
        data = np.vstack([proba_unseen[i, l] for i in [2]])
        bar_vals = data
        
        bar = plt.barh([i for i in range(6)], bar_vals[0], alpha=0.95)
    
        plt.title(f'{l}')
        plt.yticks([i for i in range(6)], bar_labels)
        plt.xlabel(f'$\mu$(predicted class probability)')
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlim([0,1])
        plt.gca().invert_yaxis()
        plt.show()

def _plot_rf_preds_mutants_and_drugs(data_dict, by_pathway=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    if by_pathway:
        train_labels = np.asarray([data_dict['pathway_dict'][x] if x in data_dict['pathway_dict'].keys() else x for x in data_dict['labels']])
    else:
        train_labels = data_dict['labels']
    
    proba_drugs = dict()
    for p_idx in [0,1,2]:
        
        train_idx = index([data_dict['labels'], data_dict['plate_id']], [data_dict['mutant_labels'], [p for p in [0, 1, 2] if p not in [p_idx]]])
        test_idx = index([data_dict['labels'], data_dict['plate_id']], [data_dict['drug_lables'], [p_idx]])
    
        scaler = StandardScaler(with_std=True)
        feat_vecs_train = scaler.fit_transform(data_dict['feat_vecs'][train_idx])
        feat_vecs_test = scaler.transform(data_dict['feat_vecs'][test_idx])
        
        clf = RandomForestClassifier(n_estimators=1000, random_state=7).fit(data_dict['feat_vecs'][train_idx], train_labels[train_idx])

        for l in data_dict['drug_lables']:
            test_idx_ = index([data_dict['labels'], data_dict['plate_id']], [[l], [p_idx]])
            feat_vecs_test_ = scaler.transform(data_dict['feat_vecs'][test_idx_])
            proba = clf.predict_proba(feat_vecs_test_)
            pred = clf.predict(feat_vecs_test_)
            proba_drugs[(p_idx, l)] = proba[0]  
    
    for l in data_dict['drug_lables']:
        
        plt.figure(figsize=(2.5,1.75))
        bar_labels = clf.classes_ if by_pathway else [data_dict['name_dict'][c] for c in clf.classes_]
        data = np.vstack([proba_drugs[i, l] for i in range(3)])
        
        bar_vals = [np.mean(data[:,i]) for i in range(len(clf.classes_))]
    
        plt.title(f"{data_dict['name_dict'][l]}")
            
        bar = plt.barh([i for i in range(len(clf.classes_))], bar_vals, alpha=0.95)
        
        for i, c_n in enumerate(bar_labels):
            for d, m in zip(data[:,i], ['o', 's', '^']):
                plt.scatter([d], [i], marker=m, c='tab:blue',alpha=0.5, edgecolor='k')
        
        plt.yticks([i for i in range(len(clf.classes_))], bar_labels)
        plt.xlabel(f'$\mu$(predicted class probability)')
        plt.xticks([0.0,0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlim([0,1])
        plt.gca().invert_yaxis()
        plt.show()
    
def plot_rf_preds(data_dict, experiment='antibiotics', **kwargs):
    if experiment == 'antibiotics':
        _plot_rf_preds_antibiotics(data_dict)
    elif experiment == 'mutant-drugs':
        _plot_rf_preds_mutants_and_drugs(data_dict, **kwargs)
        
    
def plot_umap_w_bg(data, labels, input_maps, input_choices, classes, title, labels_dict=None, cmap='tab10'):
    import matplotlib as mpl
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    cmap_a = mpl.colormaps[cmap]

    idx_list_ = []
    for maps, choices in zip(input_maps, input_choices):    
        idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))
        
    idx_list = np.logical_and.reduce(idx_list_)
    
    labels = labels[idx_list] 

    classes_ = classes
    
    data_fg = data[idx_list]
    data_bg = data[~idx_list]
    
    labels_grounded_dict = dict(zip([i for i in np.unique(labels)], [i for i in range(len(np.unique(labels)))]))
    
    classes_from_labels = [(classes[int(l)], labels_grounded_dict[l]) for l in np.unique(labels)]

    num_l = min(len(np.unique(labels)), 20)

    handles = []
    for c_l in classes_from_labels:
        c, l = c_l
        handles.append(mpatches.Patch(color=cmap_a(int(l % num_l)), label=c))
    
    labels = [labels_grounded_dict[l] for l in labels]

    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111)
    ax.scatter(data_fg[:, 0], data_fg[:, 1], s=50, c=[cmap_a(int(l % num_l)) for l in labels], edgecolor='grey', alpha=0.75)
    ax.scatter(data_bg[:, 0], data_bg[:, 1], s=50, c='gray', alpha=0.05)

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    legend = ax.legend(handles=handles, frameon=False)
    plt.show()

def plot_preds_by_concentration(data_dict):
    from collections import Counter
    from matplotlib.colors import ListedColormap
    
    cmap = ListedColormap(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'])

    moa_class_matrix = np.empty((len(data_dict['srtd_cmpd_labels']), len(data_dict['concentrations'])))    
    for c_id, cmpd in enumerate(data_dict['srtd_cmpd_labels']):
        for d_id in [0,1,2,3]:
            idx = index([data_dict['labels_as_name'], data_dict['mic_id']], [[cmpd], [d_id]])
    
            moa_class_matrix[c_id, d_id] = Counter(data_dict['preds'][idx]).most_common(1)[0][0]
    
    fig = plt.figure(figsize=(20,4))
    fig, ax = plt.subplots()
    ax.pcolormesh(moa_class_matrix, cmap=cmap, edgecolors='k', linewidth=1)
    ax.set_aspect(0.5)
    ax.invert_yaxis()
    ax.set_yticks([i + 0.5 for i in range(len(data_dict['srtd_cmpd_labels']))], data_dict['srtd_cmpd_labels'])
    ax.set_xticks([i + 0.5 for i in range(len(data_dict['concentrations']))], data_dict['concentrations'], rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
    plt.show()

def plot_umap_trajectories(data_dict):
    feat_vecs_umap = get_umap(data_dict['feat_vecs'], n_components=2, n_neighbors=int(200/4), min_dist=0.1, metric='cosine')
    
    for d_id, ds in enumerate(data_dict['concentrations']):
        plot_umap_w_bg(
            feat_vecs_umap,
            labels=data_dict['labels_moa'],
            input_maps=[data_dict['mic_id']],  
            classes=data_dict['classes_moa'],
            title=f'{ds}',
            input_choices=[[d_id]]
        )

    for moa in ['Cell wall (PBP)', 'Gyrase', 'RNA polymerase']:
        for d_id, ds in enumerate(data_dict['concentrations']):
            plot_umap_w_bg(
                feat_vecs_umap, 
                labels=data_dict['labels'],
                input_maps=[data_dict['mic_id'], data_dict['labels_moa_as_name']],  
                classes=data_dict['classes'], 
                title=f'{moa} ({ds})',
                input_choices=[[d_id], [moa]]                
            )

def plot_cnn_preds(data_dict):
    from scipy import special, stats
    import matplotlib as mpl
    
    for dropped_cmpd in data_dict['labels_test']:
        idx =  index([data_dict['labels_as_name']], [[dropped_cmpd]])
        
        proba = special.softmax(data_dict['logits'][idx], axis=1)
         
        plt.figure(figsize=(3,2))    
        bar = plt.barh(data_dict['classes_moa'][:-2][::-1], proba.mean(axis=0)[::-1])
    
        for l, i in enumerate([5,4,3,2,1,0]):
            bar[i].set_color('tab:blue')
        
        plt.title(f'{dropped_cmpd}')
        plt.xlabel(f'$\\mu$(predicted class probability)')
        plt.xlim([0,1])
        plt.show()  

def plot_outlier_score(data_dict):
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    N_ticks = len(data_dict['labels_test'])
    
    detect_val_matrix = np.zeros((1, N_ticks))

    idx_train = index([data_dict['labels_as_name']], [data_dict['labels_train']])
    idx_test = index([data_dict['labels_as_name'], data_dict['mic_id']], [data_dict['labels_test'], [0]])
    
    X_train = data_dict['feat_vecs'][idx_train]
    X_test = data_dict['feat_vecs'][idx_test]

    scaler = StandardScaler(with_std=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = LocalOutlierFactor(n_neighbors=10, metric='cosine', novelty=True, p=2) #Euclidean is more principled!
    clf.fit(X_train)
    y_pred = clf.predict(X_test)

    for l_idx, test_label in enumerate(data_dict['labels_test']):
        idx = index([data_dict['labels_as_name'][idx_test]], [[test_label]])
        X_test_ = X_test[idx]
        X_test_med = np.median(X_test_, axis=0)
        y_pred_ = y_pred[idx]

        detect_val_matrix[0, l_idx] = list(y_pred_).count(-1) / len(list(y_pred_))
    
    plt.figure(figsize=(N_ticks * 1/2,3.5))
    bar = plt.bar([i for i in range(N_ticks)], detect_val_matrix[0])
    for b_id, c_n in enumerate(data_dict['labels_test']):
        col_n = data_dict['color_dict'][c_n]
        bar[b_id].set_color(col_n)
    
    plt.xticks([i for i in range(N_ticks)], data_dict['labels_test'], rotation=90)
    plt.ylabel('Outlier score')
    plt.title(f'Outlier score')
    plt.show()

def plot_cross_validated_outlier_score(data_dict, moa):
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    
    N_ticks = len(data_dict[moa]['labels_test'])
    detect_val_matrix = np.zeros((6, N_ticks))
    
    for p_idx in range(6):
        idx_train = index([data_dict[moa][f'P{p_idx}']['labels_as_name']], [data_dict[moa]['labels_train']])
        idx_test = index([data_dict[moa][f'P{p_idx}']['labels_as_name']], [data_dict[moa]['labels_test']])

        X_train = data_dict[moa][f'P{p_idx}']['feat_vecs'][idx_train]
        X_test = data_dict[moa][f'P{p_idx}']['feat_vecs'][idx_test]
    
        scaler = StandardScaler(with_std=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = LocalOutlierFactor(n_neighbors=10, metric='cosine', novelty=True, p=2)
        clf.fit(X_train)
        y_pred = clf.predict(X_test)
    
        for l_idx, test_label in enumerate(data_dict[moa]['labels_test']):
            idx = index([data_dict[moa][f'P{p_idx}']['labels_as_name'][idx_test]], [[test_label]])
            X_test_ = X_test[idx]
            X_test_med = np.median(X_test_, axis=0)
            y_pred_ = y_pred[idx]
            detect_val_matrix[p_idx, l_idx] = list(y_pred_).count(-1) / len(list(y_pred_))
    
    
    plt.figure(figsize=(N_ticks * 1/2,3.5))
    clr_list = [data_dict['color_dict'] [l] for l in data_dict[moa]['labels_test']] 
    std = np.std(detect_val_matrix, axis=0)
    bar = plt.bar([i for i in range(N_ticks)],np.median(detect_val_matrix, axis=0))#, c=clr_list)
    
    for b_id, c_n in enumerate(data_dict[moa]['labels_test']):
        col_n = data_dict['color_dict'] [c_n]
        bar[b_id].set_color(col_n)
    
    for p, m in zip(range(6),['o', 's', '^', 'P', 'p', 'D']):
        plt.scatter([i for i in range(N_ticks)], detect_val_matrix[p], alpha=0.5, c=clr_list, marker=m, edgecolor='black')
    
    plt.xticks([i for i in range(N_ticks)], data_dict[moa]['labels_test'], rotation=90)
    plt.ylabel('Outlier score')
    plt.title(moa)
    plt.show()

def mean_pairwise_distance(moa, data_dict, moa_dict, shuffle=False, k_shuffle=3, n_iters_shuffle=10000):
    from sklearn import metrics 
    import random 
    
    centroid_labels_name = []
    labels_names = np.unique(data_dict['labels_as_name'])
    V_centroid = np.empty((len(labels_names), 16), dtype=data_dict['feat_vecs'].dtype)
    
    for l_idx, label_ in enumerate(labels_names):
        idx = index([data_dict['labels_as_name']], [[label_]])
        centroid_labels_name.append(label_)
        V_centroid[l_idx] = np.median(data_dict['feat_vecs'][idx], axis=0)
    
    pw_dist = metrics.pairwise_distances(V_centroid, metric='euclidean')

    if shuffle:
        dists = []
        for n in range(n_iters_shuffle):
            idx = index([centroid_labels_name], [random.sample(centroid_labels_name, k=k_shuffle)])
            k = idx.sum() 
            n_pairs = k * (k - 1) / 2
            dists.append(np.triu(pw_dist[idx][:,idx], k=1).sum() / n_pairs)

        return dists
        
    else:
        idx = index([centroid_labels_name], [moa_dict[moa]])
        k = idx.sum()
        n_pairs = k * (k - 1) / 2
        return np.triu(pw_dist[idx][:,idx], k=1).sum() / n_pairs

def plot_cross_validated_clustering_score(data_dict, moa='RNA polymerase'):

    labels_test = [moa, 'DNA synthesis', 'Control']
    
    N_ticks = len(labels_test)
    
    detect_val_matrix = np.zeros((6, N_ticks))
    n_iters = 10000
    for p_idx in range(6):
        for l_idx, test_label in enumerate(labels_test):
            detect_val_matrix[p_idx, l_idx] = 1 - np.sum(mean_pairwise_distance(test_label, data_dict[moa][f'P{p_idx}'], data_dict['moa_dict'], shuffle=True, k_shuffle=len(data_dict['moa_dict'][test_label]), n_iters_shuffle=n_iters) < mean_pairwise_distance(test_label, data_dict[moa][f'P{p_idx}'], data_dict['moa_dict'], shuffle=False)) / n_iters
    
    plt.figure(figsize=(N_ticks * 1/2,3.5))
    clr_list = [data_dict['color_dict'][data_dict['moa_dict'][l][0]] for l in labels_test[:-1] + ['Control']]
    
    bar = plt.bar([i for i in range(N_ticks)],np.median(detect_val_matrix, axis=0))#, c=clr_list)
    
    for b_id, c_n in enumerate(labels_test):
        col_n = data_dict['color_dict'][data_dict['moa_dict'][c_n][0]]
        bar[b_id].set_color(col_n)
    
    for p, m in zip(range(6),['o', 's', '^', 'P', 'p', 'D']):
        plt.scatter([i for i in range(N_ticks)], detect_val_matrix[p], alpha=0.5, c=clr_list, marker=m, edgecolor='black')
    print(np.median(detect_val_matrix, axis=0))
    plt.xticks([i for i in range(N_ticks)], labels_test, rotation=90)
    plt.ylabel('Clustering score')
    plt.title(moa)
    plt.hlines(0.5, -0.5, N_ticks-0.5, colors='k', linestyle='dashed')
    plt.ylim([0,1.1])
    plt.show()

def make_confusion_matrix(labels, preds, classes_true, classes_pred, mode='normalize', title='Confusion matrix'):
    from matplotlib.ticker import MultipleLocator

    def return_counts_array(labels, preds):
        counts_array = np.zeros((len(classes_pred), len(classes_true)))
        for l in np.unique(labels):
            x = np.array(preds)[np.array(labels) == l]
            p = np.unique(x, return_counts=True)
            if p[0].size > 0:
                for p_idx, p_val in zip(p[0], p[1]):
                    counts_array[int(l), int(p_idx)] = p_val
        return counts_array

    counts_array = return_counts_array(labels, preds)
    counts_array_norm = np.zeros_like(counts_array)
    for i, row in enumerate(counts_array):
        counts_array_norm[i] = row / row.sum()

    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(counts_array_norm, cmap=plt.cm.Blues)
    plt.gca().xaxis.tick_bottom()

    ax.set_xticklabels([''] + classes_true, rotation=90)
    ax.set_yticklabels([''] + classes_pred)

    for i in range(counts_array.shape[1]):
        for j in range(counts_array.shape[0]):
            c = counts_array[j,i]
            c_n = counts_array_norm[j,i]
            ax.text(i, j, f'{int(c)}', va='center', ha='center', c='black' if c_n < 0.5 else 'white')
    
    plt.title(title)

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

def plot_bf_vs_all_benchmark(data_dict):
    from matplotlib.lines import Line2D
    from scipy import stats
    
    def _return_accuracy_list(data_dict):
        from sklearn import metrics
        data = []
        for mode in ['BF', 'multi-channel']:
            acc_list = []
            for i in range(6):
                acc_list.append(metrics.accuracy_score(data_dict[mode][(f'P{i}', 'preds')], data_dict[mode][(f'P{i}', 'labels_moa')] ))
            data.append(acc_list)
        return data

    ch_name_list = ['BF', 'Hoechst_FM4_BF']
    colors = ['tab:blue', 'tab:red']

    fig = plt.figure(figsize =(1,2.5))
    ax = fig.add_axes([0, 0, 1, 1])
        
    data = _return_accuracy_list(data_dict)
    ax.set_xticklabels([ch.replace('_', '\n') for ch in ch_name_list], fontsize=9)
    bp = ax.boxplot(data, widths=[0.5] * len(ch_name_list), positions=[i + 1 for i in range(len(ch_name_list))], showfliers=False, meanline=True, showmeans=True)    
    
    for k in range(len(ch_name_list)):
        bp['means'][k].set_color(colors[k])
        bp['means'][k].set_linewidth(1)
        bp['means'][k].set_linestyle('-')
    
    l_width = 0
    for k in range(len(ch_name_list)):
        bp['boxes'][k].set_linewidth(l_width)
        bp['medians'][k].set_linewidth(l_width)
    
    for k in range(len(ch_name_list) * 2):
        bp['whiskers'][k].set_linewidth(l_width)
        bp['caps'][k].set_linewidth(l_width)
    
    for i, (vals, c) in enumerate(zip(data, [i + 1 for i in range(len(ch_name_list))])):
        b = c + 0.1
        a = c - 0.1
        for j, m in enumerate(['o', 's', '^', 'D', 'P', 'p']):
            ax.scatter([(b - a) * np.random.random_sample(1) + a], vals[j], color=colors[i], marker=m, s=50, alpha=0.5)

    print(f'BF accuracy: {np.mean(data[0])} +/- {np.std(data[0])}')
    print(f'Hoechst FM4 BF accuracy: {np.mean(data[1])} +/- {np.std(data[1])}')
    
    pval = stats.wilcoxon(data[0], data[1]).pvalue
    print(f'p={pval}')
    
    if pval <= 0.5:
        plt.hlines(1.02, 1.001, 2.001, linestyle='solid', color='k', linewidth=0.75)
        plt.text(1.45, 1.02, '*', fontsize=14)
    
    plt.xlim([0.7, 2.3])
    plt.ylim([0,1.1])
    plt.hlines(1/6, 0.7, 7.3, linestyle='dashed', color='black')
    ax.set_ylabel('Hold-out test accuracy', fontsize=9)
    ax.set_title('MoA clf. acc.', fontsize=10)    
    plt.show()