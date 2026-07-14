import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import glob
import json
import numpy as np
import torchvision
from torch import nn
from tqdm import tqdm

from load_data import load_data
from models import ContrastiveClassifierAvgPoolCNN
from utils import make_dir
from class_params import get_cglu_classes

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

    if isinstance(cmd_args['test_dir'], list):
        cmd_args['test_dir'] = cmd_args['test_dir'][0]

    print('Test: ', cmd_args['test_dir'])

    if cmd_args['use_cglu_moa']:
        cglu_classes = get_cglu_classes()
        if args.dose:
            cmd_args['dropped_classes'] = [x for x in cmd_args['dropped_classes'] if x.split('_')[-1] != args.dose]
            dropped_doses = [d for d in ['0.5MIC', '1MIC', '5MIC', '10MIC'] if d != args.dose]
            cmd_args['dropped_classes'] += [c for c in cglu_classes if c.split('_')[-1] in dropped_doses]

    __, __, test_loader, classes, classes_moa, __ = load_data(root_dir=cmd_args['data_dir'] if args.data_dir is None else args.data_dir,
                                                             train_val_test_dir=[cmd_args['test_dir'], cmd_args['test_dir'], cmd_args['test_dir']],
                                                  	         dropped_classes=cmd_args['dropped_classes'],
                                                             batch_size=cmd_args['batch_size'],
                                                             val_split=cmd_args['val_split'],
                                                             crop_size=cmd_args['crop_size'],
                                                             channels=cmd_args.get('channels', [0, 1, 2]),
                                                             data_type=cmd_args.get('data_type', 'tiff'),
                                                             loader_type='triplet',
                                                             classify_by_moa=cmd_args.get('classify_by_moa', False),
                                                             use_e_coli_moa=cmd_args['use_e_coli_moa'],
                                                             use_h_pylori_moa=cmd_args.get('use_h_pylori_moa', False),
                                                             use_cglu_moa=cmd_args.get('use_cglu_moa', False))

    if args.ckpt > 0:
        ckpt_path = glob.glob(os.path.join(args.save_dir, 'ckpts', '*_' + str(args.ckpt) + '_*.tar'))[0]

    elif args.ckpt == -1:
        ckpt_paths = glob.glob(os.path.join(args.save_dir, 'ckpts', '*.tar'))
        ckpt_path = sorted(ckpt_paths, key=lambda s: os.path.basename(s).split('_')[3])[0]
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

    save_dir = os.path.join(save_dir_, f"{cmd_args['test_dir']}_{dose_}")

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
         dropped_classes=[classes.index(l) for l in [l_ for l_ in args.dropped_classes if l_.split('_')[-1] == cmd_args['dose']]])
