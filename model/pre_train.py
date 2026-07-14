import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import argparse
import json
import os
import numpy as np
import torchvision
import random
import glob

from load_data import load_data
from utils import plot_sample_batch
from models import ContrastiveClassifierAvgPoolCNN
from losses import TripletLoss, HardTripletLoss, CenterLoss

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def train(model,
          criterion,
          optimizer,
          epochs,
          train_loader,
          val_loader,
          save_dir,
          **kwargs):
    """
    Training loop
    """
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        model.train()
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output, __ = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = ((output.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc/len(train_loader)
            epoch_loss += loss/len(train_loader)

        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        train_loss_history.append(epoch_loss.cpu().detach().numpy())
        train_accuracy_history.append(epoch_accuracy.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy= 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output, __ = model(data)

                val_loss = criterion(val_output,label)
                acc = ((val_output.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc/ len(val_loader)
                epoch_val_loss += val_loss/ len(val_loader)

            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))
            val_loss_history.append(epoch_val_loss.cpu().detach().numpy())
            val_accuracy_history.append(epoch_val_accuracy.cpu().detach().numpy())

        if (epoch + 1) % 10 == 0 and epoch != 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
            }, os.path.join(save_dir, f'ckpts_pre_train/model_ckpt_{epoch + 1}_{epoch_val_loss.cpu().detach().numpy():.2f}.tar'))

        if kwargs.get('self_attention', False):
            save_attention_map_overlay(attention_map, data, epoch, save_dir)

        # Save updated training curves after each epoch
        plt.figure('Pre-training Loss')
        plt.plot(train_loss_history, label='train')
        plt.plot(val_loss_history, label='validation')
        plt.xlim([0, epochs])
        plt.legend()
        plt.title('Loss')
        plt.savefig(os.path.join(save_dir, f'pre_train_loss_curves.png'))
        plt.close()

        plt.figure('Pre-training Accuracy')
        plt.plot(train_accuracy_history, label='train')
        plt.plot(val_accuracy_history, label='validation')
        plt.xlim([0, epochs])
        plt.legend()
        plt.title('Accuracy')
        plt.savefig(os.path.join(save_dir, f'pre_train_accuracy_curves.png'))
        plt.close()

if __name__ == '__main__':
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--model', default='b0')
    parser.add_argument('--dropped_classes', nargs='+', default=[])
    parser.add_argument('--train_dir', nargs='+', default=[])
    parser.add_argument('--val_dir', nargs='+', default=[])
    parser.add_argument('--test_dir', nargs='+', default=[])
    parser.add_argument('--dose', default=None, type=str)
    parser.add_argument('--num_channels', default=3, type=int)
    parser.add_argument('--num_features', default=128, type=int)
    parser.add_argument('--channels', default=None, type=str)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--val_split', default=0.2, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--bottleneck_size', default=128)
    parser.add_argument('--n_crops', default=9, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_cent', default=0.5, type=float)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--subsampling_factor', default=1, type=float)
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--freeze_layers', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--data_type', default='tiff')
    parser.add_argument('--use_e_coli_moa', action='store_true', default=False)
    parser.add_argument('--classify_by_moa', action='store_true', default=False)
    parser.add_argument('--use_h_pylori_moa', action='store_true', default=False)
    parser.add_argument('--use_cglu_moa', action='store_true', default=False)
    parser.add_argument('--label_smoothing', default=0, type=float)

    args = parser.parse_args()

    torch.manual_seed(111)

    if not os.path.exists(args.save_dir):
        make_dir(args.save_dir)

    if not os.path.exists(os.path.join(args.save_dir, 'ckpts_pre_train')):
        os.mkdir(os.path.join(args.save_dir, 'ckpts_pre_train'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print('CUDA available')

    if args.channels:
        # Parse arguments that are compatible with sbatch wrap function
        args.channels = [int(x) for x in args.channels]
        args.num_channels = len(args.channels)


    all_dir = os.listdir(args.data_dir)

    if args.val_dir[0] == 'None':
        args.val_dir = None

    elif len(args.val_dir) == 0:
        args.val_dir = [random.choice([dir for dir in all_dir if dir not in args.test_dir])]

    if len(args.train_dir) == 0:
        if args.val_dir:
            args.train_dir = [dir for dir in all_dir if dir not in args.test_dir + args.val_dir]
        else:
            args.train_dir = [dir for dir in all_dir if dir not in args.test_dir]

    if eval(args.ckpt_path) == -1:
        try:
            ckpt_paths = glob.glob(os.path.join(args.save_dir, 'ckpts_pre_train', '*.tar'))
            args.ckpt_path = sorted(ckpt_paths, key=lambda s: os.path.basename(s).split('_')[2])[-1]
        except IndexError:
            args.ckpt_path = None

    if args.use_cglu_moa:
        cglu_classes = ['Amoxicillin_0.5MIC', 'Amoxicillin_10MIC', 'Amoxicillin_1MIC', 'Amoxicillin_5MIC', 'Ampicillin_0.5MIC', 'Ampicillin_10MIC', 'Ampicillin_1MIC', 'Ampicillin_5MIC', 'BDM_0.5MIC', 'BDM_10MIC', 'BDM_1MIC', 'BDM_5MIC', 'Carbenicillin_0.5MIC', 'Carbenicillin_10MIC', 'Carbenicillin_1MIC', 'Carbenicillin_5MIC', 'Cefotaxim_0.5MIC', 'Cefotaxim_10MIC', 'Cefotaxim_1MIC', 'Cefotaxim_5MIC', 'Ciprofloxacin_0.5MIC', 'Ciprofloxacin_10MIC', 'Ciprofloxacin_1MIC', 'Ciprofloxacin_5MIC', 'Clarithromycin_0.5MIC', 'Clarithromycin_10MIC', 'Clarithromycin_1MIC', 'Clarithromycin_5MIC', 'Clofazimine_0.5MIC', 'Clofazimine_10MIC', 'Clofazimine_1MIC', 'Clofazimine_5MIC', 'DMSO', 'Doxycycline_0.5MIC', 'Doxycycline_10MIC', 'Doxycycline_1MIC', 'Doxycycline_5MIC', 'Ethambutol_0.5MIC', 'Ethambutol_10MIC', 'Ethambutol_1MIC', 'Ethambutol_5MIC', 'Gepotidacin_0.5MIC', 'Gepotidacin_10MIC', 'Gepotidacin_1MIC', 'Gepotidacin_5MIC', 'Linezolid_0.5MIC', 'Linezolid_10MIC', 'Linezolid_1MIC', 'Linezolid_5MIC', 'Moxifloxacin_0.5MIC', 'Moxifloxacin_10MIC', 'Moxifloxacin_1MIC', 'Moxifloxacin_5MIC', 'Novobiocin_0.5MIC', 'Novobiocin_10MIC', 'Novobiocin_1MIC', 'Novobiocin_5MIC', 'Rifabutin_0.5MIC', 'Rifabutin_10MIC', 'Rifabutin_1MIC', 'Rifabutin_5MIC', 'Rifampicin_0.5MIC', 'Rifampicin_10MIC', 'Rifampicin_1MIC', 'Rifampicin_5MIC', 'Sulfamethoxazole_0.5MIC', 'Sulfamethoxazole_10MIC', 'Sulfamethoxazole_1MIC', 'Sulfamethoxazole_5MIC', 'Trimethoprim_0.5MIC', 'Trimethoprim_10MIC', 'Trimethoprim_1MIC', 'Trimethoprim_5MIC', 'Water']

        if args.dose:
            dropped_doses = [d for d in ['0.5MIC', '1MIC', '5MIC', '10MIC'] if d != args.dose]
            args.dropped_classes += [c for c in cglu_classes if c.split('_')[-1] in dropped_doses]

    train_loader, val_loader, test_loader, classes, classes_moa, class_weights = load_data(root_dir=args.data_dir,
                                                                          	               train_val_test_dir=[args.train_dir, args.val_dir, args.test_dir],
                                    				                                  	   dropped_classes=args.dropped_classes,
                                    				                                  	   batch_size=args.batch_size,
                                    				                                  	   val_split=args.val_split,
                                    				                                  	   crop_size=args.crop_size,
                                    				                                  	   channels=args.channels,
                                    				                                  	   data_type=args.data_type,
                                                                                           loader_type='class',
                                    				                                  	   subsampling_factor=args.subsampling_factor,
                                                                                           classify_by_moa=args.classify_by_moa,
                                                                                           use_e_coli_moa=args.use_e_coli_moa,
                                                                                           use_h_pylori_moa=args.use_h_pylori_moa,
                                                                                           use_cglu_moa=args.use_cglu_moa)
    args.classes = classes
    args.classes_moa = classes_moa
    args.class_weights = class_weights
    with open(os.path.join(args.save_dir, 'commandline_args_pre_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    dataiter = iter(train_loader)
    images, __ = next(dataiter)
    plot_sample_batch(torchvision.utils.make_grid(images[:4].view(-1,1,256,256)), args.save_dir)

    num_tiles = int((1500 / args.crop_size) ** 2)
    model = ContrastiveClassifierAvgPoolCNN(model=args.model,
                                            num_classes=len(classes) if not args.classify_by_moa else len(classes_moa),
                                            num_channels=args.num_channels,
                                            num_features=args.num_features,
                                            n_crops=num_tiles,
                                            pretrained=args.pretrained).to(device)

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        if args.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc_head.parameters():
                param.requires_grad = True
            for param in model.clsf_head.parameters():
                param.requires_grad = True

    model = model.to(device)
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.l2)
    class_weights = torch.Tensor(class_weights).to(device)
    classifier_loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    train(model=model,
          criterion=classifier_loss_fn,
          optimizer=optimizer,
          epochs=args.epochs,
          train_loader=train_loader,
          val_loader=val_loader,
          save_dir=args.save_dir)
