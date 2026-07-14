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
from models import ContrastiveClassifierAvgPoolCNN

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))
            
def test(model, test_loader, classes, save_dir, **kwargs):
    """
    Compute accuracy on test dataset, plot AUC, show predicted images
    """
    images = []
    labels = []

    preds = []
    class_probs = []
    test_accuracy = 0
    feat_vecs = []
    test_outputs = []

    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            label = label.to(device)

            test_output, feat_vec = model(data)
            test_pred = (test_output.argmax(dim=1))
            acc = (test_pred == label).float().mean()
            test_accuracy += acc/len(test_loader)

            images.append(data[:,4])
            labels.append(label)
            feat_vecs.append(feat_vec)
            test_outputs.append(test_output)
            preds.append(test_pred)


    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    test_outputs = torch.cat(test_outputs, dim=0)
    feat_vecs = torch.cat(feat_vecs, dim=0)

    np.savetxt(os.path.join(save_dir, 'feat_vecs.txt'), feat_vecs.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'labels.txt'), labels.cpu().numpy())
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

    with open(os.path.join(args.save_dir, 'commandline_args_pre_train.txt'), 'r') as f:
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

    if cmd_args['use_cglu_moa']:
        cglu_classes = ['Amoxicillin_0.5MIC', 'Amoxicillin_10MIC', 'Amoxicillin_1MIC', 'Amoxicillin_5MIC', 'Ampicillin_0.5MIC', 'Ampicillin_10MIC', 'Ampicillin_1MIC', 'Ampicillin_5MIC', 'BDM_0.5MIC', 'BDM_10MIC', 'BDM_1MIC', 'BDM_5MIC', 'Carbenicillin_0.5MIC', 'Carbenicillin_10MIC', 'Carbenicillin_1MIC', 'Carbenicillin_5MIC', 'Cefotaxim_0.5MIC', 'Cefotaxim_10MIC', 'Cefotaxim_1MIC', 'Cefotaxim_5MIC', 'Ciprofloxacin_0.5MIC', 'Ciprofloxacin_10MIC', 'Ciprofloxacin_1MIC', 'Ciprofloxacin_5MIC', 'Clarithromycin_0.5MIC', 'Clarithromycin_10MIC', 'Clarithromycin_1MIC', 'Clarithromycin_5MIC', 'Clofazimine_0.5MIC', 'Clofazimine_10MIC', 'Clofazimine_1MIC', 'Clofazimine_5MIC', 'DMSO', 'Doxycycline_0.5MIC', 'Doxycycline_10MIC', 'Doxycycline_1MIC', 'Doxycycline_5MIC', 'Ethambutol_0.5MIC', 'Ethambutol_10MIC', 'Ethambutol_1MIC', 'Ethambutol_5MIC', 'Gepotidacin_0.5MIC', 'Gepotidacin_10MIC', 'Gepotidacin_1MIC', 'Gepotidacin_5MIC', 'Linezolid_0.5MIC', 'Linezolid_10MIC', 'Linezolid_1MIC', 'Linezolid_5MIC', 'Moxifloxacin_0.5MIC', 'Moxifloxacin_10MIC', 'Moxifloxacin_1MIC', 'Moxifloxacin_5MIC', 'Novobiocin_0.5MIC', 'Novobiocin_10MIC', 'Novobiocin_1MIC', 'Novobiocin_5MIC', 'Rifabutin_0.5MIC', 'Rifabutin_10MIC', 'Rifabutin_1MIC', 'Rifabutin_5MIC', 'Rifampicin_0.5MIC', 'Rifampicin_10MIC', 'Rifampicin_1MIC', 'Rifampicin_5MIC', 'Sulfamethoxazole_0.5MIC', 'Sulfamethoxazole_10MIC', 'Sulfamethoxazole_1MIC', 'Sulfamethoxazole_5MIC', 'Trimethoprim_0.5MIC', 'Trimethoprim_10MIC', 'Trimethoprim_1MIC', 'Trimethoprim_5MIC', 'Water']

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
                                                              loader_type='class',
                                                              classify_by_moa=cmd_args.get('classify_by_moa', False),
                                                              use_e_coli_moa=cmd_args['use_e_coli_moa'],
                                                              use_h_pylori_moa=cmd_args.get('use_h_pylori_moa', False),
                                                              use_cglu_moa=cmd_args.get('use_cglu_moa', False))

    if args.ckpt > 0:
        ckpt_path = glob.glob(os.path.join(args.save_dir, 'ckpts_pre_train', '*_' + str(args.ckpt) + '_*.tar'))[0]

    elif args.ckpt == -1:
        ckpt_paths = glob.glob(os.path.join(args.save_dir, 'ckpts_pre_train', '*.tar'))
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

    save_dir = os.path.join(save_dir_, f"{cmd_args['test_dir']}")

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
