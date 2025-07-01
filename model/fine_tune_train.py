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
          triplet_loss_fn,
          center_loss_fn,
          classifier_loss_fn,
          optimizer,
          epochs,
          train_loader,
          val_loader,
          save_dir,
          alpha,
          beta,
          gamma,
          lr,
          lr_cent,
          **kwargs):
    """
    Training loop
    """

    if kwargs.get('classify_by_moa', False):
        l_idx = 1
    else:
        l_idx = 0

    train_accuracy_history = []
    train_loss_history = []
    train_triplet_loss_history = []
    train_center_loss_history = []
    train_class_loss_history = []

    val_accuracy_history = []
    val_loss_history = []
    val_triplet_loss_history = []
    val_center_loss_history = []
    val_class_loss_history = []

    start_epoch = kwargs.get('start_epoch', 0)
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_triplet_loss = 0
        epoch_center_loss = 0
        epoch_classifier_loss = 0
        model.train()
        for (a_d, a_l), (p_d, p_l), (n_d, n_l) in train_loader:
            a_d = a_d.to(device)
            a_l = [x.to(device) for x in a_l]
            p_d = p_d.to(device)
            p_l = [x.to(device) for x in p_l]
            n_d = n_d.to(device)
            n_l = [x.to(device) for x in n_l]

            a_pred, anchor = model(a_d)
            p_pred, positive = model(p_d)
            n_pred, negative = model(n_d)

            triplet_loss = triplet_loss_fn(torch.cat((anchor, positive, negative)), torch.cat((a_l[l_idx], p_l[l_idx], n_l[l_idx])))
            center_loss = alpha * center_loss_fn(torch.cat((anchor, positive, negative)), torch.cat((a_l[l_idx], p_l[l_idx], n_l[l_idx])))
            classifier_loss = classifier_loss_fn(torch.cat((a_pred, p_pred, n_pred)), torch.cat((a_l[l_idx], p_l[l_idx], n_l[l_idx])))

            loss = beta * triplet_loss + alpha * center_loss + gamma * classifier_loss

            optimizer.zero_grad()
            loss.backward()
            if alpha > 0:
                for param in center_loss_fn.parameters():
                    param.grad.data *= lr_cent / alpha * lr
            optimizer.step()

            acc = ((torch.cat((a_pred, p_pred, n_pred)).argmax(dim=1) == torch.cat((a_l[l_idx], p_l[l_idx], n_l[l_idx]))).float().mean())
            epoch_accuracy += acc/len(train_loader)

            epoch_loss += loss/len(train_loader)
            epoch_triplet_loss += triplet_loss/len(train_loader)
            epoch_center_loss += center_loss/len(train_loader)
            epoch_classifier_loss += classifier_loss/len(train_loader)

        print('Epoch : {}, train loss : {}'.format(epoch+1,epoch_loss))
        train_accuracy_history.append(epoch_accuracy.cpu().detach().numpy())
        train_loss_history.append(epoch_loss.cpu().detach().numpy())
        train_triplet_loss_history.append(epoch_triplet_loss.cpu().detach().numpy())
        train_center_loss_history.append(epoch_center_loss.cpu().detach().numpy())
        train_class_loss_history.append(epoch_classifier_loss.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_accuracy = 0
            epoch_val_triplet_loss = 0
            epoch_val_center_loss = 0
            epoch_val_classifier_loss = 0
            for (a_d, a_l), (p_d, p_l), (n_d, n_l) in val_loader:
                a_d = a_d.to(device)
                a_l = [x.to(device) for x in a_l]
                p_d = p_d.to(device)
                p_l = [x.to(device) for x in p_l]
                n_d = n_d.to(device)
                n_l = [x.to(device) for x in n_l]

                a_pred, anchor = model(a_d)
                p_pred, positive = model(p_d)
                n_pred, negative = model(n_d)

                val_triplet_loss = triplet_loss_fn(torch.cat((anchor, positive, negative)), torch.cat((a_l[l_idx], p_l[l_idx], n_l[l_idx])))
                val_center_loss = alpha * center_loss_fn(torch.cat((anchor, positive, negative)), torch.cat((a_l[l_idx], p_l[l_idx], n_l[l_idx])))
                val_classifier_loss = classifier_loss_fn(torch.cat((a_pred, p_pred, n_pred)), torch.cat((a_l[l_idx], p_l[l_idx], n_l[l_idx])))

                val_loss = beta * val_triplet_loss + alpha * val_center_loss + gamma * val_classifier_loss

                val_acc = ((torch.cat((a_pred, p_pred, n_pred)).argmax(dim=1) == torch.cat((a_l[l_idx], p_l[l_idx], n_l[l_idx]))).float().mean())
                epoch_val_accuracy += val_acc/len(val_loader)

                epoch_val_loss += val_loss/ len(val_loader)
                epoch_val_triplet_loss += val_triplet_loss/len(val_loader)
                epoch_val_center_loss += val_center_loss/len(val_loader)
                epoch_val_classifier_loss += val_classifier_loss/len(val_loader)

            print('Epoch : {}, val_loss : {}'.format(epoch+1,epoch_val_loss))
            val_accuracy_history.append(epoch_val_accuracy.cpu().detach().numpy())
            val_loss_history.append(epoch_val_loss.cpu().detach().numpy())
            val_triplet_loss_history.append(epoch_val_triplet_loss.cpu().detach().numpy())
            val_center_loss_history.append(epoch_val_center_loss.cpu().detach().numpy())
            val_class_loss_history.append(epoch_val_classifier_loss.cpu().detach().numpy())

        if (epoch + 1) % 5 == 0 and epoch != 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
            }, os.path.join(save_dir, f'ckpts/model_ckpt_{epoch + 1}_{epoch_val_loss.cpu().detach().numpy():.2f}.tar'))

        plt.figure('Loss')
        plt.plot(train_loss_history, label='train')
        plt.plot(val_loss_history, label='validation')

        plt.plot(train_triplet_loss_history, label='triplet_train')
        plt.plot(val_triplet_loss_history, label='triplet_validation')

        plt.plot(train_center_loss_history, label='center_train')
        plt.plot(val_center_loss_history, label='center_validation')

        plt.plot(train_class_loss_history, label='class_train')
        plt.plot(val_class_loss_history, label='class_validation')

        plt.xlim([0, epochs])
        plt.legend()
        plt.title('Loss')
        plt.savefig(os.path.join(save_dir, f'loss_curves_from_{start_epoch}.png'))
        plt.close()

        plt.figure('Accuracy')
        plt.plot(train_accuracy_history, label='train')
        plt.plot(val_accuracy_history, label='validation')

        plt.xlim([0, epochs])
        plt.legend()
        plt.title('Accuracy')
        plt.savefig(os.path.join(save_dir, f'accuracy_curves_from_{start_epoch}.png'))
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
    parser.add_argument('--bottleneck_size', default=None)
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
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--label_smoothing', default=0, type=float)

    args = parser.parse_args()

    torch.manual_seed(111)

    if not os.path.exists(args.save_dir):
        make_dir(args.save_dir)

    if not os.path.exists(os.path.join(args.save_dir, 'ckpts')):
        os.mkdir(os.path.join(args.save_dir, 'ckpts'))

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

    print('Train: ', args.train_dir)
    print('Val: ', args.val_dir)
    print('Test: ', args.test_dir)

    if not os.path.isdir(args.ckpt_path):
        if eval(args.ckpt_path) == -1:
            try:
                ckpt_paths = glob.glob(os.path.join(args.save_dir, 'ckpts', '*.tar'))
                args.ckpt_path = sorted(ckpt_paths, key=lambda s: os.path.basename(s).split('_')[2])[-1]
            except IndexError:
                args.ckpt_path = None

    # if args.ckpt_path:
    #     start_epoch = eval(os.path.basename(args.ckpt_path).split('_')[2])
    # else:
    start_epoch = 0

    if args.use_e_coli_moa:
        e_coli_classes = ['Avibactam_0.125xIC50', 'Avibactam_0.25xIC50', 'Avibactam_0.5xIC50', 'Avibactam_1xIC50', 'Aztreonam_0.125xIC50', 'Aztreonam_0.25xIC50', 'Aztreonam_0.5xIC50', 'Aztreonam_1xIC50', 'Cefepime_0.125xIC50', 'Cefepime_0.25xIC50', 'Cefepime_0.5xIC50', 'Cefepime_1xIC50', 'Cefsulodin_0.125xIC50', 'Cefsulodin_0.25xIC50', 'Cefsulodin_0.5xIC50', 'Cefsulodin_1xIC50', 'Ceftriaxone_0.125xIC50', 'Ceftriaxone_0.25xIC50', 'Ceftriaxone_0.5xIC50', 'Ceftriaxone_1xIC50', 'Chloramphenicol_0.125xIC50', 'Chloramphenicol_0.25xIC50', 'Chloramphenicol_0.5xIC50', 'Chloramphenicol_1xIC50', 'Ciprofloxacin_0.125xIC50', 'Ciprofloxacin_0.25xIC50', 'Ciprofloxacin_0.5xIC50', 'Ciprofloxacin_1xIC50', 'Clarithromycin_0.125xIC50', 'Clarithromycin_0.25xIC50', 'Clarithromycin_0.5xIC50', 'Clarithromycin_1xIC50', 'Clavulanate_0.125xIC50', 'Clavulanate_0.25xIC50', 'Clavulanate_0.5xIC50', 'Clavulanate_1xIC50', 'Colistin_0.125xIC50', 'Colistin_0.25xIC50', 'Colistin_0.5xIC50', 'Colistin_1xIC50', 'DMSO', 'Doxycycline_0.125xIC50', 'Doxycycline_0.25xIC50', 'Doxycycline_0.5xIC50', 'Doxycycline_1xIC50', 'Kanamycin_0.125xIC50', 'Kanamycin_0.25xIC50', 'Kanamycin_0.5xIC50', 'Kanamycin_1xIC50', 'Levofloxacin_0.125xIC50', 'Levofloxacin_0.25xIC50', 'Levofloxacin_0.5xIC50', 'Levofloxacin_1xIC50', 'Mecillinam_0.125xIC50', 'Mecillinam_0.25xIC50', 'Mecillinam_0.5xIC50', 'Mecillinam_1xIC50', 'Meropenem_0.125xIC50', 'Meropenem_0.25xIC50', 'Meropenem_0.5xIC50', 'Meropenem_1xIC50', 'Norfloxacin_0.125xIC50', 'Norfloxacin_0.25xIC50', 'Norfloxacin_0.5xIC50', 'Norfloxacin_1xIC50', 'PenicillinG_0.125xIC50', 'PenicillinG_0.25xIC50', 'PenicillinG_0.5xIC50', 'PenicillinG_1xIC50', 'PolymyxinB_0.125xIC50', 'PolymyxinB_0.25xIC50', 'PolymyxinB_0.5xIC50', 'PolymyxinB_1xIC50', 'Relebactam_0.125xIC50', 'Relebactam_0.25xIC50', 'Relebactam_0.5xIC50', 'Relebactam_1xIC50', 'Rifampicin_0.125xIC50', 'Rifampicin_0.25xIC50', 'Rifampicin_0.5xIC50', 'Rifampicin_1xIC50', 'Sulbactam_0.125xIC50', 'Sulbactam_0.25xIC50', 'Sulbactam_0.5xIC50', 'Sulbactam_1xIC50', 'Trimethoprim_0.125xIC50', 'Trimethoprim_0.25xIC50', 'Trimethoprim_0.5xIC50', 'Trimethoprim_1xIC50']

        if args.dose:
            dropped_doses = [d for d in ['0.125xIC50', '0.25xIC50', '0.5xIC50', '1xIC50'] if d != args.dose]
            args.dropped_classes += [c for c in e_coli_classes if c.split('_')[-1] in dropped_doses]

    elif args.use_h_pylori_moa:
        h_pylori_classes = ['Avibactam_0.125xIC50', 'Avibactam_0.25xIC50', 'Avibactam_0.5xIC50', 'Avibactam_1xIC50', 'Aztreonam_0.125xIC50', 'Aztreonam_0.25xIC50', 'Aztreonam_0.5xIC50', 'Aztreonam_1xIC50', 'Cefepime_0.125xIC50', 'Cefepime_0.25xIC50', 'Cefepime_0.5xIC50', 'Cefepime_1xIC50', 'Cefsulodin_0.125xIC50', 'Cefsulodin_0.25xIC50', 'Cefsulodin_0.5xIC50', 'Cefsulodin_1xIC50', 'Ceftriaxone_0.125xIC50', 'Ceftriaxone_0.25xIC50', 'Ceftriaxone_0.5xIC50', 'Ceftriaxone_1xIC50', 'Chloramphenicol_0.125xIC50', 'Chloramphenicol_0.25xIC50', 'Chloramphenicol_0.5xIC50', 'Chloramphenicol_1xIC50', 'Ciprofloxacin_0.125xIC50', 'Ciprofloxacin_0.25xIC50', 'Ciprofloxacin_0.5xIC50', 'Ciprofloxacin_1xIC50', 'Clarithromycin_0.125xIC50', 'Clarithromycin_0.25xIC50', 'Clarithromycin_0.5xIC50', 'Clarithromycin_1xIC50', 'Clavulanate_0.125xIC50', 'Clavulanate_0.25xIC50', 'Clavulanate_0.5xIC50', 'Clavulanate_1xIC50', 'Colistin_0.125xIC50', 'Colistin_0.25xIC50', 'Colistin_0.5xIC50', 'Colistin_1xIC50', 'DMSO', 'Doxycycline_0.125xIC50', 'Doxycycline_0.25xIC50', 'Doxycycline_0.5xIC50', 'Doxycycline_1xIC50', 'Kanamycin_0.125xIC50', 'Kanamycin_0.25xIC50', 'Kanamycin_0.5xIC50', 'Kanamycin_1xIC50', 'Levofloxacin_0.125xIC50', 'Levofloxacin_0.25xIC50', 'Levofloxacin_0.5xIC50', 'Levofloxacin_1xIC50', 'Mecillinam_0.125xIC50', 'Mecillinam_0.25xIC50', 'Mecillinam_0.5xIC50', 'Mecillinam_1xIC50', 'Meropenem_0.125xIC50', 'Meropenem_0.25xIC50', 'Meropenem_0.5xIC50', 'Meropenem_1xIC50', 'Norfloxacin_0.125xIC50', 'Norfloxacin_0.25xIC50', 'Norfloxacin_0.5xIC50', 'Norfloxacin_1xIC50', 'PenicillinG_0.125xIC50', 'PenicillinG_0.25xIC50', 'PenicillinG_0.5xIC50', 'PenicillinG_1xIC50', 'PolymyxinB_0.125xIC50', 'PolymyxinB_0.25xIC50', 'PolymyxinB_0.5xIC50', 'PolymyxinB_1xIC50', 'Relebactam_0.125xIC50', 'Relebactam_0.25xIC50', 'Relebactam_0.5xIC50', 'Relebactam_1xIC50', 'Rifampicin_0.125xIC50', 'Rifampicin_0.25xIC50', 'Rifampicin_0.5xIC50', 'Rifampicin_1xIC50', 'Sulbactam_0.125xIC50', 'Sulbactam_0.25xIC50', 'Sulbactam_0.5xIC50', 'Sulbactam_1xIC50', 'Temocillin_0.125xIC50', 'Temocillin_0.25xIC50', 'Temocillin_0.5xIC50', 'Temocillin_1xIC50']

        if args.dose:
            dropped_doses = [d for d in ['0.125xIC50', '0.25xIC50', '0.5xIC50', '1xIC50'] if d != args.dose]
            args.dropped_classes += [c for c in h_pylori_classes if c.split('_')[-1] in dropped_doses]

    elif args.use_cglu_moa:
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
                                                                                           loader_type='triplet',
                                    				                                  	   subsampling_factor=args.subsampling_factor,
                                                                                           classify_by_moa=args.classify_by_moa,
                                                                                           use_e_coli_moa=args.use_e_coli_moa,
                                                                                           use_h_pylori_moa=args.use_h_pylori_moa,
                                                                                           use_cglu_moa=args.use_cglu_moa)
    args.classes = classes
    args.classes_moa = classes_moa
    args.class_weights = class_weights
    with open(os.path.join(args.save_dir, 'commandline_args_fine_tune.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print('Classes', classes)
    # Plot a sample batch and save
    dataiter = iter(train_loader)
    (anchor_images, __), (positive_images, __), (negative_images, __) = next(dataiter)

    plot_sample_batch(torchvision.utils.make_grid(anchor_images[:4].view(-1,1,256,256)), args.save_dir, 'sample_images_anchor.png')

    plot_sample_batch(torchvision.utils.make_grid(positive_images[:4].view(-1,1,256,256)), args.save_dir, 'sample_images_positive.png')

    plot_sample_batch(torchvision.utils.make_grid(negative_images[:4].view(-1,1,256,256)), args.save_dir, 'sample_images_negative.png')

    # Instantiate CNN, optimizer and loss function
    num_tiles = int((1500 / args.crop_size) ** 2)

    # model = ContrastiveClassifierAvgPoolCNN(model=args.model,
    #                                         num_classes=len(classes) if not args.classify_by_moa else len(classes_moa),
    #                                         num_channels=args.num_channels,
    #                                         num_features=args.num_features,
    #                                         pretrained=args.pretrained).to(device)
    model = ContrastiveClassifierAvgPoolCNN(model=args.model,
                                            num_classes=46,#len(classes),
                                            num_channels=args.num_channels,
                                            num_features=128,#args.num_features,
                                            pretrained=args.pretrained).to(device)

    if args.ckpt_path:
        if os.path.isdir(args.ckpt_path):
            try:
                ckpt_paths = glob.glob(os.path.join(args.ckpt_path, '*.tar'))
                args.ckpt_path = sorted(ckpt_paths, key=lambda s: os.path.basename(s).split('_')[2])[-1]
            except IndexError:
                args.ckpt_path = None
        print(f'Loading model from checkpoint {args.ckpt_path}...')
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        model.fc_head = nn.Linear(1280, args.num_features)
        model.clsf_head = nn.Linear(args.num_features, len(classes_moa))
        if args.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc_head.parameters():
                param.requires_grad = True
            for param in model.clsf_head.parameters():
                param.requires_grad = True

    print(model)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    for name, param in model.named_parameters():
        print(name,param.requires_grad)

    model = model.to(device)

    center_loss_fn = CenterLoss(num_classes=len(classes), feat_dim=args.num_features, use_gpu=True)
    params = list(model.parameters()) + list(center_loss_fn.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.l2)

    triplet_loss_fn = HardTripletLoss().to(device)

    class_weights = torch.Tensor(class_weights).to(device)
    print('Class weights', classes_moa, class_weights)
    classifier_loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    train(model=model,
          triplet_loss_fn=triplet_loss_fn,
          center_loss_fn=center_loss_fn,
          classifier_loss_fn=classifier_loss_fn,
          optimizer=optimizer,
          epochs=args.epochs,
          train_loader=train_loader,
          val_loader=val_loader,
          save_dir=args.save_dir,
          start_epoch=start_epoch,
          alpha=args.alpha,
          beta=args.beta,
          gamma=args.gamma,
          lr=args.lr,
          lr_cent=args.lr_cent,
          classify_by_moa=args.classify_by_moa)
