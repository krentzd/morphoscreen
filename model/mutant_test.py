import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import glob
import scikitplot as skplt
import json
import numpy as np
import torchvision
from torch import nn
from tqdm import tqdm

from load_data import load_data
from utils import plot_predictions, plot_representations, get_tsne, get_umap, get_pca, plot_image_representations
from models import ContrastiveClassifierAvgPoolCNN

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def make_prediction_matrix(labels, preds, classes_true, classes_pred, save_name, mode='normalize'):
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
    if mode == 'normalize':
        counts_array_temp = np.zeros_like(counts_array)
        for i, row in enumerate(counts_array):
            counts_array_temp[i] = row / row.sum()
        counts_array = counts_array_temp
    elif mode == 'max':
        counts_array_temp = np.zeros_like(counts_array)
        for i, row in enumerate(counts_array):
            counts_array_temp[i, np.argmax(row)] = 1
        counts_array = counts_array_temp

    fig, ax = plt.subplots(figsize=(10,20))
    ax.matshow(counts_array, cmap=plt.cm.Blues)
    plt.gca().xaxis.tick_bottom()

    ax.set_xticklabels([''] + classes_true, rotation=90)
    ax.set_yticklabels([''] + classes_pred)

    for i in range(counts_array.shape[1]):
        for j in range(counts_array.shape[0]):
            c = counts_array[j,i]
            if mode == 'normalize':
                ax.text(i, j, str(c), va='center', ha='center', c='black' if c < 0.5 else 'white')
            else:
                ax.text(i, j, str(c), va='center', ha='center', c='black' if c < 0.5 * counts_array.max() else 'white')
    if mode == 'normalize':
        plt.title('Normalized Prediction Counts')
    if mode == 'max':
        plt.title('Max Prediction Counts')
    else:
        plt.title('Prediction Counts')

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    print('Saving to', save_name)
    plt.savefig(save_name)

def test(model, test_loader, classes, classes_moa, save_dir, **kwargs):
    """
    Compute accuracy on test dataset, plot AUC, show predicted images
    """
    images = []
    labels = []
    labels_moa = []

    preds = []
    class_probs = []
    test_accuracy = 0
    feat_vecs = []
    test_outputs = []

    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            label = [x.to(device) for x in label]

            test_output, feat_vec = model(data)
            test_pred = (test_output.argmax(dim=1))
            label_ = label[1] if kwargs.get('classify_by_moa', False) else label[0]
            acc = (test_pred == label_).float().mean()
            test_accuracy += acc/len(test_loader)

            images.append(data[:,4])
            labels.append(label[0])
            labels_moa.append(label[1])
            feat_vecs.append(feat_vec)
            test_outputs.append(test_output)
            preds.append(test_pred)


    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    labels_moa = torch.cat(labels_moa, dim=0)
    preds = torch.cat(preds, dim=0)
    test_outputs = torch.cat(test_outputs, dim=0)
    feat_vecs = torch.cat(feat_vecs, dim=0)

    np.savetxt(os.path.join(save_dir, 'feat_vecs.txt'), feat_vecs.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'test_outputs.txt'), test_outputs.cpu().numpy())

    np.savetxt(os.path.join(save_dir, 'labels.txt'), labels.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'labels_moa.txt'), labels_moa.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'preds.txt'), preds.cpu().numpy())
    np.save(os.path.join(save_dir, 'images.npy'), images.cpu().numpy())

    classes_ = classes_moa if kwargs.get('classify_by_moa', False) else classes
    labels_ = labels_moa if kwargs.get('classify_by_moa', False) else labels

    plot_predictions(images, labels_, preds, 16, test_accuracy, save_dir, classes=classes_, save_name='predicted_images.png')
    plot_predictions(images, labels_, preds, 16, test_accuracy, save_dir, classes=classes_, save_name='predicted_images.svg')

    tsne_data = get_tsne(feat_vecs.cpu().numpy())

    plot_representations(tsne_data, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='tsne_plot.png', dropped_labels=list(kwargs.get('dropped_classes', None)))
    plot_representations(tsne_data, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='tsne_plot.svg', dropped_labels=list(kwargs.get('dropped_classes', None)))

    plot_representations(tsne_data, labels_moa.cpu().numpy(), classes=classes_moa, save_dir=save_dir, save_name='tsne_plot_on_moa.png', dropped_labels=list(kwargs.get('dropped_classes', None)), ref_labels=list(labels.cpu().numpy()), dropped_classes=classes)
    plot_representations(tsne_data, labels_moa.cpu().numpy(), classes=classes_moa, save_dir=save_dir, save_name='tsne_plot_on_moa.svg', dropped_labels=list(kwargs.get('dropped_classes', None)), ref_labels=list(labels.cpu().numpy()), dropped_classes=classes)

    # plot_representations(tsne_data, labels_moa.cpu().numpy(), classes=classes_moa, save_dir=save_dir, save_name='tsne_plot_on_moa.png')
    # plot_representations(tsne_data, labels_moa.cpu().numpy(), classes=classes_moa, save_dir=save_dir, save_name='tsne_plot_on_moa.svg')

    if kwargs.get('classify_by_moa', False):
        make_prediction_matrix(labels.cpu().numpy(), preds.cpu().numpy(), kwargs.get('training_classes', classes_moa), classes, save_name=os.path.join(save_dir, 'prediction_matrix.png'))
        make_prediction_matrix(labels.cpu().numpy(), preds.cpu().numpy(), kwargs.get('training_classes', classes_moa), classes, save_name=os.path.join(save_dir, 'prediction_matrix.svg'))

    ax = skplt.metrics.plot_confusion_matrix(labels_moa.cpu().numpy() if kwargs.get('classify_by_moa', False) else labels.cpu().numpy(), preds.cpu().numpy(), normalize=True, figsize=(10,10) if kwargs.get('classify_by_moa', False) else (20,20))
    ax.set_xticklabels(classes_moa if kwargs.get('classify_by_moa', False) else classes, fontsize="large", rotation=90)
    ax.set_yticklabels(classes_moa if kwargs.get('classify_by_moa', False) else classes, fontsize="large", rotation=0)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    ax = skplt.metrics.plot_confusion_matrix(labels_moa.cpu().numpy() if kwargs.get('classify_by_moa', False) else labels.cpu().numpy(), preds.cpu().numpy(), normalize=True, figsize=(10,10) if kwargs.get('classify_by_moa', False) else (20,20))
    ax.set_xticklabels(classes_moa if kwargs.get('classify_by_moa', False) else classes, fontsize="large", rotation=90)
    ax.set_yticklabels(classes_moa if kwargs.get('classify_by_moa', False) else classes, fontsize="large", rotation=0)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.svg'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--test_dir', default=None)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--ckpt', default=1000, type=int)
    parser.add_argument('--dropped_classes', nargs='+', default=[])
    parser.add_argument('--dose', default=None, type=str)

    args = parser.parse_args()

    with open(os.path.join(args.save_dir, 'commandline_args_fine_tune.txt'), 'r') as f:
        cmd_args = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print('CUDA available')

    torch.manual_seed(111)
    print(cmd_args)
    print(args.data_dir)

    if args.test_dir:
        cmd_args['test_dir'] = args.test_dir

    # args.dropped_classes shows a specific class with stars in t-SNE plot
    if len(args.dropped_classes) > 0:
        cmd_args['dropped_classes'] = [c for c in cmd_args['dropped_classes'] if c not in args.dropped_classes]#args.dropped_classes

    if isinstance(cmd_args['test_dir'], list):
        cmd_args['test_dir'] = cmd_args['test_dir'][0]

    print('Test: ', cmd_args['test_dir'])

    all_dir = os.listdir(os.path.join(args.data_dir, cmd_args['test_dir']))
    print([c for c in all_dir if c.split('_')[-1]])
    dropped_cls = [c for c in all_dir if c.split('_')[-1] in ['1', '2', '3']]
    dropped_cls = [c for c in dropped_cls if c.split('_')[0] not in ['PinoGA', 'PinoGB', 'PinoSepF', 'PinoWag31', 'steA', 'wip1', 'glp', 'glpR', 'pknA', 'pknB', 'pknG', 'pknL', 'WT']]
    print(dropped_cls)

    __, __, test_loader, classes, classes_moa, __ = load_data(root_dir=cmd_args['data_dir'] if args.data_dir is None else args.data_dir,
                                                             train_val_test_dir=[cmd_args['test_dir'], cmd_args['test_dir'], cmd_args['test_dir']],
                                                  	         dropped_classes=dropped_cls,
                                                             batch_size=cmd_args['batch_size'],
                                                             val_split=cmd_args['val_split'],
                                                             crop_size=cmd_args['crop_size'],
                                                             channels=cmd_args.get('channels', [0, 1, 2]),
                                                             data_type=cmd_args.get('data_type', 'tiff'),
                                                             loader_type='triplet',
                                                             classify_by_moa=False,
                                                             use_e_coli_moa=cmd_args['use_e_coli_moa'],
                                                             use_h_pylori_moa=cmd_args.get('use_h_pylori_moa', False),
                                                             use_cglu_moa=cmd_args.get('use_cglu_moa', False))

    if args.ckpt > 0:
        ckpt_path = glob.glob(os.path.join(args.save_dir, 'ckpts', '*_' + str(args.ckpt) + '_*.tar'))[0]

    elif args.ckpt == -1:
        ckpt_paths = glob.glob(os.path.join(args.save_dir, 'ckpts', '*.tar'))
        ckpt_path = sorted(ckpt_paths, key=lambda s: os.path.basename(s).split('_')[3])[0]

    print(ckpt_path)
    ckpt = torch.load(ckpt_path)

    num_tiles = int((1500 / cmd_args['crop_size']) ** 2)

    model = ContrastiveClassifierAvgPoolCNN(model=cmd_args.get('model', 'b0'),
                                            num_classes=len(cmd_args['classes']) if not cmd_args.get('classify_by_moa', False) else len(cmd_args['classes_moa']),
                                            num_channels=cmd_args['num_channels'],
                                            num_features=cmd_args.get('num_features', 16),
                                            pretrained=cmd_args['pretrained']).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    if args.dose:
        dose_ = args.dose

    else:
        dose_ = cmd_args['dose']

    if args.save_dir:
        save_dir_ = args.save_dir

    else:
        save_dir_ = cmd_args['test_dir']

    save_dir = os.path.join(save_dir_, f"{cmd_args['test_dir']}_mutants")

    make_dir(save_dir)

    with open(os.path.join(save_dir, 'classes.txt'), 'w') as f:
        json.dump(classes, f, indent=2)

    with open(os.path.join(save_dir, 'classes_moa.txt'), 'w') as f:
        json.dump(classes_moa, f, indent=2)

    test(model=model,
         test_loader=test_loader,
         classes=classes,
         classes_moa=classes_moa,
         training_classes=cmd_args['classes'] if not cmd_args.get('classify_by_moa', False) else cmd_args['classes_moa'],
         save_dir=save_dir,
         classify_by_moa=cmd_args.get('classify_by_moa', False),
         dropped_classes=[classes.index(l) for l in ['PinoGA_state_1', 'PinoGB_state_1', 'PinoSepF_state_1', 'PinoSepF_state_2', 'PinoSepF_state_3', 'PinoSepF_state_4', 'PinoWag31_state_1', 'PinoWag31_state_2', 'PinoWag31_state_3', 'PinoWag31_state_4', 'steA_state_1', 'wip1_state_1', 'glp_state_1', 'glpR_state_1', 'pknA_state_1', 'pknB_state_1', 'pknG_state_1', 'pknL_state_1']])
