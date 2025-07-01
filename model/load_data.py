from torchvision import datasets, transforms
import torchvision
import torch
import os
import math
import numpy as np

from dataloaders import TripletDataset, ClassifierDataset
from utils import OverlappingCrop, OverlappingCropMultiChannel

def convert_to_list(x):
    return [x_i.unsqueeze(0) for x_i in x]

def dataset_loader(type='triplet'):
    if type == 'class':
        return ClassifierDataset
    elif type == 'triplet':
        return TripletDataset

def intersect_dicts(class_merge_dict, moa_dict):
    intrsct_dict = dict()
    for k, v in class_merge_dict.items():
        if v in moa_dict.keys():
            intrsct_dict[k] = moa_dict[v]

    return {**intrsct_dict, **moa_dict}

def load_data(root_dir,
              train_val_test_dir=[],
              dropped_classes=[],
              batch_size=32,
              val_split=0.2,
              crop_size=512,
              **kwargs):
    """
    Return train, val and test dataloaders
    """

    class_merge_dict = {'DMSO_0.5MIC': 'DMSO',
                        'DMSO_1MIC': 'DMSO',
                        'DMSO_5MIC': 'DMSO',
                        'DMSO_10MIC': 'DMSO',
                        'DMSO_0.125xIC50': 'Control', #DMSO
                        'DMSO_0.25xIC50': 'Control', #DMSO
                        'DMSO_0.5xIC50': 'Control', #DMSO
                        'DMSO_1xIC50': 'Control', #DMSO
                        'Water_0.5MIC': 'Water',
                        'Water_1MIC': 'Water',
                        'Water_5MIC': 'Water',
                        'Water_10MIC': 'Water',
                        'Water_1': 'Water',
                        'Water_2': 'Water',
                        'Water_3': 'Water',
                        'Water_4': 'Water',
                        'Water_5': 'Water',
                        'Water_6': 'Water',
                        'DMSO_1': 'DMSO',
                        'DMSO_2': 'DMSO',
                        'DMSO_3': 'DMSO',
                        'DMSO_4': 'DMSO',
                        'DMSO_5': 'DMSO',
                        'DMSO_6': 'DMSO'}
    if kwargs.get('use_e_coli_moa', False):
        e_coli_moa_dict = {'Avibactam_0.125xIC50': 'Cell wall (PBP 2)',
                          'Avibactam_0.25xIC50': 'Cell wall (PBP 2)',
                          'Avibactam_0.5xIC50': 'Cell wall (PBP 2)',
                          'Avibactam_1xIC50': 'Cell wall (PBP 2)',
                          'Aztreonam_0.125xIC50': 'Cell wall (PBP 3)',
                          'Aztreonam_0.25xIC50': 'Cell wall (PBP 3)',
                          'Aztreonam_0.5xIC50': 'Cell wall (PBP 3)',
                          'Aztreonam_1xIC50': 'Cell wall (PBP 3)',
                          'Cefepime_0.125xIC50': 'Cell wall (PBP 3)',
                          'Cefepime_0.25xIC50': 'Cell wall (PBP 3)',
                          'Cefepime_0.5xIC50': 'Cell wall (PBP 3)',
                          'Cefepime_1xIC50': 'Cell wall (PBP 3)',
                          'Cefsulodin_0.125xIC50': 'Cell wall (PBP 1)',
                          'Cefsulodin_0.25xIC50': 'Cell wall (PBP 1)',
                          'Cefsulodin_0.5xIC50': 'Cell wall (PBP 1)',
                          'Cefsulodin_1xIC50': 'Cell wall (PBP 1)',
                          'Ceftriaxone_0.125xIC50': 'Cell wall (PBP 3)',
                          'Ceftriaxone_0.25xIC50': 'Cell wall (PBP 3)',
                          'Ceftriaxone_0.5xIC50': 'Cell wall (PBP 3)',
                          'Ceftriaxone_1xIC50': 'Cell wall (PBP 3)',
                          'Chloramphenicol_0.125xIC50': 'Ribosome (50S)',
                          'Chloramphenicol_0.25xIC50': 'Ribosome (50S)',
                          'Chloramphenicol_0.5xIC50': 'Ribosome (50S)',
                          'Chloramphenicol_1xIC50': 'Ribosome (50S)',
                          'Ciprofloxacin_0.125xIC50': 'Gyrase',
                          'Ciprofloxacin_0.25xIC50': 'Gyrase',
                          'Ciprofloxacin_0.5xIC50': 'Gyrase',
                          'Ciprofloxacin_1xIC50': 'Gyrase',
                          'Clarithromycin_0.125xIC50': 'Ribosome (50S)',
                          'Clarithromycin_0.25xIC50': 'Ribosome (50S)',
                          'Clarithromycin_0.5xIC50': 'Ribosome (50S)',
                          'Clarithromycin_1xIC50': 'Ribosome (50S)',
                          'Clavulanate_0.125xIC50': 'Cell wall (beta-lac inhib)',
                          'Clavulanate_0.25xIC50': 'Cell wall (beta-lac inhib)',
                          'Clavulanate_0.5xIC50': 'Cell wall (beta-lac inhib)',
                          'Clavulanate_1xIC50': 'Cell wall (beta-lac inhib)',
                          'Colistin_0.125xIC50': 'Membrane integrity',
                          'Colistin_0.25xIC50': 'Membrane integrity',
                          'Colistin_0.5xIC50': 'Membrane integrity',
                          'Colistin_1xIC50': 'Membrane integrity',
                          'DMSO_0.125xIC50': 'Control', #DMSO
                          'DMSO_0.25xIC50': 'Control', #DMSO
                          'DMSO_0.5xIC50': 'Control', #DMSO
                          'DMSO_1xIC50': 'Control', #DMSO
                          'Doxycycline_0.125xIC50': 'Ribosome (30S)',
                          'Doxycycline_0.25xIC50': 'Ribosome (30S)',
                          'Doxycycline_0.5xIC50': 'Ribosome (30S)',
                          'Doxycycline_1xIC50': 'Ribosome (30S)',
                          'Kanamycin_0.125xIC50': 'Ribosome (30S)',
                          'Kanamycin_0.25xIC50': 'Ribosome (30S)',
                          'Kanamycin_0.5xIC50': 'Ribosome (30S)',
                          'Kanamycin_1xIC50': 'Ribosome (30S)',
                          'Levofloxacin_0.125xIC50': 'Gyrase',
                          'Levofloxacin_0.25xIC50': 'Gyrase',
                          'Levofloxacin_0.5xIC50': 'Gyrase',
                          'Levofloxacin_1xIC50': 'Gyrase',
                          'Mecillinam_0.125xIC50': 'Cell wall (PBP 2)',
                          'Mecillinam_0.25xIC50': 'Cell wall (PBP 2)',
                          'Mecillinam_0.5xIC50': 'Cell wall (PBP 2)',
                          'Mecillinam_1xIC50': 'Cell wall (PBP 2)',
                          'Meropenem_0.125xIC50': 'Cell wall (PBP 2)',
                          'Meropenem_0.25xIC50': 'Cell wall (PBP 2)',
                          'Meropenem_0.5xIC50': 'Cell wall (PBP 2)',
                          'Meropenem_1xIC50': 'Cell wall (PBP 2)',
                          'Norfloxacin_0.125xIC50': 'Gyrase',
                          'Norfloxacin_0.25xIC50': 'Gyrase',
                          'Norfloxacin_0.5xIC50': 'Gyrase',
                          'Norfloxacin_1xIC50': 'Gyrase',
                          'PenicillinG_0.125xIC50': 'Cell wall (PBP 1)',
                          'PenicillinG_0.25xIC50': 'Cell wall (PBP 1)',
                          'PenicillinG_0.5xIC50': 'Cell wall (PBP 1)',
                          'PenicillinG_1xIC50': 'Cell wall (PBP 1)',
                          'PolymyxinB_0.125xIC50': 'Membrane integrity',
                          'PolymyxinB_0.25xIC50': 'Membrane integrity',
                          'PolymyxinB_0.5xIC50': 'Membrane integrity',
                          'PolymyxinB_1xIC50': 'Membrane integrity',
                          'Relebactam_0.125xIC50': 'Cell wall (beta-lac inhib)',
                          'Relebactam_0.25xIC50': 'Cell wall (beta-lac inhib)',
                          'Relebactam_0.5xIC50': 'Cell wall (beta-lac inhib)',
                          'Relebactam_1xIC50': 'Cell wall (beta-lac inhib)',
                          'Rifampicin_0.125xIC50': 'RNA polymerase',
                          'Rifampicin_0.25xIC50': 'RNA polymerase',
                          'Rifampicin_0.5xIC50': 'RNA polymerase',
                          'Rifampicin_1xIC50': 'RNA polymerase',
                          'Sulbactam_0.125xIC50': 'Cell wall (beta-lac inhib)',
                          'Sulbactam_0.25xIC50': 'Cell wall (beta-lac inhib)',
                          'Sulbactam_0.5xIC50': 'Cell wall (beta-lac inhib)',
                          'Sulbactam_1xIC50': 'Cell wall (beta-lac inhib)',
                          'Trimethoprim_0.125xIC50': 'DNA synthesis',
                          'Trimethoprim_0.25xIC50': 'DNA synthesis',
                          'Trimethoprim_0.5xIC50': 'DNA synthesis',
                          'Trimethoprim_1xIC50': 'DNA synthesis'}

        class_merge_dict = intersect_dicts(class_merge_dict, e_coli_moa_dict)

    elif kwargs.get('use_h_pylori_moa', False) and kwargs.get('classify_by_moa', False):
        h_pylori_moa_dict = {'Avibactam_0.125xIC50': 'Cell wall (PBP 2)',
                            'Avibactam_0.25xIC50': 'Cell wall (PBP 2)',
                            'Avibactam_0.5xIC50': 'Cell wall (PBP 2)',
                            'Avibactam_1xIC50': 'Cell wall (PBP 2)',
                            'Aztreonam_0.125xIC50': 'Cell wall (PBP 3)',
                            'Aztreonam_0.25xIC50': 'Cell wall (PBP 3)',
                            'Aztreonam_0.5xIC50': 'Cell wall (PBP 3)',
                            'Aztreonam_1xIC50': 'Cell wall (PBP 3)',
                            'Cefepime_0.125xIC50': 'Cell wall (PBP 3)',
                            'Cefepime_0.25xIC50': 'Cell wall (PBP 3)',
                            'Cefepime_0.5xIC50': 'Cell wall (PBP 3)',
                            'Cefepime_1xIC50': 'Cell wall (PBP 3)',
                            'Cefsulodin_0.125xIC50': 'Cell wall (PBP 1)',
                            'Cefsulodin_0.25xIC50': 'Cell wall (PBP 1)',
                            'Cefsulodin_0.5xIC50': 'Cell wall (PBP 1)',
                            'Cefsulodin_1xIC50': 'Cell wall (PBP 1)',
                            'Ceftriaxone_0.125xIC50': 'Cell wall (PBP 3)',
                            'Ceftriaxone_0.25xIC50': 'Cell wall (PBP 3)',
                            'Ceftriaxone_0.5xIC50': 'Cell wall (PBP 3)',
                            'Ceftriaxone_1xIC50': 'Cell wall (PBP 3)',
                            'Chloramphenicol_0.125xIC50': 'Ribosome (50S)',
                            'Chloramphenicol_0.25xIC50': 'Ribosome (50S)',
                            'Chloramphenicol_0.5xIC50': 'Ribosome (50S)',
                            'Chloramphenicol_1xIC50': 'Ribosome (50S)',
                            'Ciprofloxacin_0.125xIC50': 'Gyrase',
                            'Ciprofloxacin_0.25xIC50': 'Gyrase',
                            'Ciprofloxacin_0.5xIC50': 'Gyrase',
                            'Ciprofloxacin_1xIC50': 'Gyrase',
                            'Clarithromycin_0.125xIC50': 'Ribosome (50S)',
                            'Clarithromycin_0.25xIC50': 'Ribosome (50S)',
                            'Clarithromycin_0.5xIC50': 'Ribosome (50S)',
                            'Clarithromycin_1xIC50': 'Ribosome (50S)',
                            'Clavulanate_0.125xIC50': 'Cell wall (beta-lac inhib)',
                            'Clavulanate_0.25xIC50': 'Cell wall (beta-lac inhib)',
                            'Clavulanate_0.5xIC50': 'Cell wall (beta-lac inhib)',
                            'Clavulanate_1xIC50': 'Cell wall (beta-lac inhib)',
                            'Colistin_0.125xIC50': 'Membrane integrity',
                            'Colistin_0.25xIC50': 'Membrane integrity',
                            'Colistin_0.5xIC50': 'Membrane integrity',
                            'Colistin_1xIC50': 'Membrane integrity',
                            'DMSO_0.125xIC50': 'Control', #DMSO
                            'DMSO_0.25xIC50': 'Control', #DMSO
                            'DMSO_0.5xIC50': 'Control', #DMSO
                            'DMSO_1xIC50': 'Control', #DMSO
                            'Doxycycline_0.125xIC50': 'Ribosome (30S)',
                            'Doxycycline_0.25xIC50': 'Ribosome (30S)',
                            'Doxycycline_0.5xIC50': 'Ribosome (30S)',
                            'Doxycycline_1xIC50': 'Ribosome (30S)',
                            'Kanamycin_0.125xIC50': 'Ribosome (30S)',
                            'Kanamycin_0.25xIC50': 'Ribosome (30S)',
                            'Kanamycin_0.5xIC50': 'Ribosome (30S)',
                            'Kanamycin_1xIC50': 'Ribosome (30S)',
                            'Levofloxacin_0.125xIC50': 'Gyrase',
                            'Levofloxacin_0.25xIC50': 'Gyrase',
                            'Levofloxacin_0.5xIC50': 'Gyrase',
                            'Levofloxacin_1xIC50': 'Gyrase',
                            'Mecillinam_0.125xIC50': 'Cell wall (PBP 2)',
                            'Mecillinam_0.25xIC50': 'Cell wall (PBP 2)',
                            'Mecillinam_0.5xIC50': 'Cell wall (PBP 2)',
                            'Mecillinam_1xIC50': 'Cell wall (PBP 2)',
                            'Meropenem_0.125xIC50': 'Cell wall (PBP 2)',
                            'Meropenem_0.25xIC50': 'Cell wall (PBP 2)',
                            'Meropenem_0.5xIC50': 'Cell wall (PBP 2)',
                            'Meropenem_1xIC50': 'Cell wall (PBP 2)',
                            'Norfloxacin_0.125xIC50': 'Gyrase',
                            'Norfloxacin_0.25xIC50': 'Gyrase',
                            'Norfloxacin_0.5xIC50': 'Gyrase',
                            'Norfloxacin_1xIC50': 'Gyrase',
                            'PenicillinG_0.125xIC50': 'Cell wall (PBP 1)',
                            'PenicillinG_0.25xIC50': 'Cell wall (PBP 1)',
                            'PenicillinG_0.5xIC50': 'Cell wall (PBP 1)',
                            'PenicillinG_1xIC50': 'Cell wall (PBP 1)',
                            'PolymyxinB_0.125xIC50': 'Membrane integrity',
                            'PolymyxinB_0.25xIC50': 'Membrane integrity',
                            'PolymyxinB_0.5xIC50': 'Membrane integrity',
                            'PolymyxinB_1xIC50': 'Membrane integrity',
                            'Relebactam_0.125xIC50': 'Cell wall (beta-lac inhib)',
                            'Relebactam_0.25xIC50': 'Cell wall (beta-lac inhib)',
                            'Relebactam_0.5xIC50': 'Cell wall (beta-lac inhib)',
                            'Relebactam_1xIC50': 'Cell wall (beta-lac inhib)',
                            'Rifampicin_0.125xIC50': 'RNA polymerase',
                            'Rifampicin_0.25xIC50': 'RNA polymerase',
                            'Rifampicin_0.5xIC50': 'RNA polymerase',
                            'Rifampicin_1xIC50': 'RNA polymerase',
                            'Sulbactam_0.125xIC50': 'Cell wall (beta-lac inhib)',
                            'Sulbactam_0.25xIC50': 'Cell wall (beta-lac inhib)',
                            'Sulbactam_0.5xIC50': 'Cell wall (beta-lac inhib)',
                            'Sulbactam_1xIC50': 'Cell wall (beta-lac inhib)',
                            'Temocillin_0.125xIC50': 'Cell wall (PBP 3)',
                            'Temocillin_0.25xIC50': 'Cell wall (PBP 3)',
                            'Temocillin_0.5xIC50': 'Cell wall (PBP 3)',
                            'Temocillin_1xIC50': 'Cell wall (PBP 3)'}
        class_merge_dict = intersect_dicts(class_merge_dict, h_pylori_moa_dict)

    elif kwargs.get('use_cglu_moa', False) and kwargs.get('classify_by_moa', False):
        cglu_moa_dict = {'Amoxicillin_0.5MIC': 'Cell wall (PBP)',
                        'Amoxicillin_1MIC': 'Cell wall (PBP)',
                        'Amoxicillin_5MIC': 'Cell wall (PBP)',
                        'Amoxicillin_10MIC': 'Cell wall (PBP)',
                        'Ampicillin_0.5MIC': 'Cell wall (PBP)',
                        'Ampicillin_1MIC': 'Cell wall (PBP)',
                        'Ampicillin_5MIC': 'Cell wall (PBP)',
                        'Ampicillin_10MIC': 'Cell wall (PBP)',
                        'BDM_0.5MIC': 'Gyrase',
                        'BDM_1MIC': 'Gyrase',
                        'BDM_5MIC': 'Gyrase',
                        'BDM_10MIC': 'Gyrase',
                        'Carbenicillin_0.5MIC': 'Cell wall (PBP)',
                        'Carbenicillin_1MIC': 'Cell wall (PBP)',
                        'Carbenicillin_5MIC': 'Cell wall (PBP)',
                        'Carbenicillin_10MIC': 'Cell wall (PBP)',
                        'Cefotaxim_0.5MIC': 'Cell wall (PBP)',
                        'Cefotaxim_1MIC': 'Cell wall (PBP)',
                        'Cefotaxim_5MIC': 'Cell wall (PBP)',
                        'Cefotaxim_10MIC': 'Cell wall (PBP)',
                        'Ciprofloxacin_0.5MIC': 'Gyrase',
                        'Ciprofloxacin_1MIC': 'Gyrase',
                        'Ciprofloxacin_5MIC': 'Gyrase',
                        'Ciprofloxacin_10MIC': 'Gyrase',
                        'Clarithromycin_0.5MIC': 'Ribosome',
                        'Clarithromycin_1MIC': 'Ribosome',
                        'Clarithromycin_5MIC': 'Ribosome',
                        'Clarithromycin_10MIC': 'Ribosome',
                        'Clofazimine_0.5MIC': 'Protein synthesis',
                        'Clofazimine_1MIC': 'Protein synthesis',
                        'Clofazimine_5MIC': 'Protein synthesis',
                        'Clofazimine_10MIC': 'Protein synthesis',
                        'DMSO_1': 'Control',
                        'DMSO_2': 'Control',
                        'DMSO_3': 'Control',
                        'DMSO_4': 'Control',
                        'DMSO_5': 'Control',
                        'DMSO_6': 'Control',
                        'Doxycycline_0.5MIC': 'Ribosome',
                        'Doxycycline_1MIC': 'Ribosome',
                        'Doxycycline_5MIC': 'Ribosome',
                        'Doxycycline_10MIC': 'Ribosome',
                        'Ethambutol_0.5MIC': 'Cell wall (Arabinose)',
                        'Ethambutol_1MIC': 'Cell wall (Arabinose)',
                        'Ethambutol_5MIC': 'Cell wall (Arabinose)',
                        'Ethambutol_10MIC': 'Cell wall (Arabinose)',
                        'Gepotidacin_0.5MIC': 'Gyrase',
                        'Gepotidacin_1MIC': 'Gyrase',
                        'Gepotidacin_5MIC': 'Gyrase',
                        'Gepotidacin_10MIC': 'Gyrase',
                        'Linezolid_0.5MIC': 'Ribosome',
                        'Linezolid_1MIC': 'Ribosome',
                        'Linezolid_5MIC': 'Ribosome',
                        'Linezolid_10MIC': 'Ribosome',
                        'Moxifloxacin_0.5MIC': 'Gyrase',
                        'Moxifloxacin_1MIC': 'Gyrase',
                        'Moxifloxacin_5MIC': 'Gyrase',
                        'Moxifloxacin_10MIC': 'Gyrase',
                        'Novobiocin_0.5MIC': 'Gyrase',
                        'Novobiocin_1MIC': 'Gyrase',
                        'Novobiocin_5MIC': 'Gyrase',
                        'Novobiocin_10MIC': 'Gyrase',
                        'Rifabutin_0.5MIC': 'RNA polymerase',
                        'Rifabutin_1MIC': 'RNA polymerase',
                        'Rifabutin_5MIC': 'RNA polymerase',
                        'Rifabutin_10MIC': 'RNA polymerase',
                        'Rifampicin_0.5MIC': 'RNA polymerase',
                        'Rifampicin_1MIC': 'RNA polymerase',
                        'Rifampicin_5MIC': 'RNA polymerase',
                        'Rifampicin_10MIC': 'RNA polymerase',
                        'Sulfamethoxazole_0.5MIC': 'DNA synthesis',
                        'Sulfamethoxazole_1MIC': 'DNA synthesis',
                        'Sulfamethoxazole_5MIC': 'DNA synthesis',
                        'Sulfamethoxazole_10MIC': 'DNA synthesis',
                        'Trimethoprim_0.5MIC': 'DNA synthesis',
                        'Trimethoprim_1MIC': 'DNA synthesis',
                        'Trimethoprim_5MIC': 'DNA synthesis',
                        'Trimethoprim_10MIC': 'DNA synthesis',
                        'Water_1': 'Control',
                        'Water_2': 'Control',
                        'Water_3': 'Control',
                        'Water_4': 'Control',
                        'Water_5': 'Control',
                        'Water_6': 'Control'}

        class_merge_dict = intersect_dicts(class_merge_dict, cglu_moa_dict)

    if kwargs.get('channels', None):
        ch_list = kwargs['channels']
    else:
        ch_list = [0,1,2]

    data_mean = 0.5
    data_std = 0.5

    # Generate n crops per image
    train_transforms_list = [transforms.RandomVerticalFlip(),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(90),
                             transforms.Lambda(lambda image: [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1)(image_slice) for image_slice in convert_to_list(image)]), #Apply color jitter to channels individually, since they're uncorrelated
                             transforms.Lambda(lambda image_list: torch.stack([image_slice for image_slice in image_list]).squeeze(1)),
                             transforms.CenterCrop(math.ceil(np.sqrt(0.5) * 2160)),
                             transforms.RandomCrop(1500), # = Slightly smaller random crop
                             OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
                             transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                             transforms.Resize((256, 256)),
                             transforms.Normalize(mean=data_mean, std=data_std)]

    train_transforms_list_png = [transforms.RandomVerticalFlip(),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(90),
                                 transforms.CenterCrop(math.ceil(np.sqrt(0.5) * 2160)), # = 1528
                                 transforms.RandomCrop(1500), # = Slightly smaller random crop
                                 OverlappingCrop(crop_size, crop_size, pad=False),
                                 transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                 transforms.Resize((256, 256)),
                                 transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                                 transforms.Lambda(lambda x: x[:,ch_list]),
                                 # transforms.Grayscale(num_output_channels=1),
                                 transforms.Normalize(mean=data_mean, std=data_std)]

    test_transforms_list = [transforms.CenterCrop(1500),
                            OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
                            transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                            transforms.Resize((256, 256)),
                            transforms.Normalize(mean=data_mean, std=data_std)]

    test_transforms_list_png = [transforms.CenterCrop(1500),
                                OverlappingCrop(crop_size, crop_size, pad=False),
                                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                transforms.Resize((256, 256)),
                                transforms.Lambda(lambda x: x[:,ch_list]),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.Normalize(mean=data_mean, std=data_std)]

    train_transforms =  transforms.Compose(train_transforms_list)
    test_transforms =  transforms.Compose(test_transforms_list)

    train_transforms_png =  transforms.Compose(train_transforms_list_png)
    test_transforms_png =  transforms.Compose(test_transforms_list_png)

    if len(train_val_test_dir) > 0:
        if isinstance(train_val_test_dir[0], str):
            root_train = [os.path.join(root_dir, train_val_test_dir[0])]
        else:
            print(train_val_test_dir[0])
            root_train = [os.path.join(root_dir, dir) for dir in train_val_test_dir[0]]

    train_dataset = dataset_loader(type=kwargs.get('loader_type', 'class'))(root=root_train,
                                                                        dropped_classes=dropped_classes,
                                                                        transform=train_transforms if kwargs.get('data_type', 'tiff') == 'tiff' else train_transforms_png,
                                                                        channels=kwargs.get('channels', None),
                                                                        class_merge_dict=class_merge_dict,
                                                                        subsampling_factor=kwargs.get('subsampling_factor', 1.),
                                                                        classify_by_moa=kwargs.get('classify_by_moa', False))
    if len(train_val_test_dir) > 0:
        if isinstance(train_val_test_dir[1], str):
            root_val = [os.path.join(root_dir, train_val_test_dir[1])]
        elif isinstance(train_val_test_dir[1], list):
            root_val = [os.path.join(root_dir, dir) for dir in train_val_test_dir[1]]
        elif train_val_test_dir[1] == None:
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    if 'val_dataset' not in locals():
        val_dataset = dataset_loader(type=kwargs.get('loader_type', 'class'))(root=root_val,
                                                                            dropped_classes=dropped_classes,
                                                                            transform=train_transforms if kwargs.get('data_type', 'tiff') == 'tiff' else train_transforms_png,
                                                                            channels=kwargs.get('channels', None),
                                                                            class_merge_dict=class_merge_dict,
                                                                            classify_by_moa=kwargs.get('classify_by_moa', False))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    if len(train_val_test_dir) > 0:
        if isinstance(train_val_test_dir[2], str):
            root_test = [os.path.join(root_dir, train_val_test_dir[2])]
        else:
            root_test = [os.path.join(root_dir, dir) for dir in train_val_test_dir[2]]

    test_dataset = dataset_loader(type=kwargs.get('loader_type', 'class'))(root=root_test,
                                                                        is_test=True,
                                                                        dropped_classes=dropped_classes,
                                                                        transform=test_transforms if kwargs.get('data_type', 'tiff') == 'tiff' else test_transforms_png,
                                                                        channels=kwargs.get('channels', None),
                                                                        class_merge_dict=class_merge_dict,
                                                                        classify_by_moa=kwargs.get('classify_by_moa', False))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    print('Train loader', len(train_loader.dataset), 'Val loader', len(val_loader.dataset), 'Test loader', len(test_loader.dataset))

    if kwargs.get('use_e_coli_moa', False) and kwargs.get('classify_by_moa', False):
        temp_dict = dict()
        for x in test_dataset.classes:
            temp_dict[x] = e_coli_moa_dict[x]
        class_weights = [1 - (list(temp_dict.values()).count(x) / len(temp_dict.values())) for x in test_dataset.classes_moa]
    elif kwargs.get('use_h_pylori_moa', False) and kwargs.get('classify_by_moa', False):
        temp_dict = dict()
        for x in test_dataset.classes:
            temp_dict[x] = h_pylori_moa_dict[x]
        class_weights = [1 - (list(temp_dict.values()).count(x) / len(temp_dict.values())) for x in test_dataset.classes_moa]
    elif kwargs.get('use_cglu_moa', False) and kwargs.get('classify_by_moa', False):
        temp_dict = dict()
        for x in test_dataset.classes:
            temp_dict[x] = cglu_moa_dict[x]
        class_weights = [1 - (list(temp_dict.values()).count(x) / len(temp_dict.values())) for x in test_dataset.classes_moa]
    else:
        class_weights = [1 - (6 / len(test_dataset.classes)) if c=='DMSO' or c=='Water' else 1 - (1 / len(test_dataset.classes)) for c in test_dataset.classes]

    print('Loader type: ', kwargs.get('loader_type', 'class'))
    if kwargs.get('loader_type', 'class') == 'class':
        return train_loader, val_loader, test_loader, test_dataset.classes, [], class_weights
    elif kwargs.get('loader_type', 'class') == 'triplet':
        return train_loader, val_loader, test_loader, test_dataset.classes, test_dataset.classes_moa, class_weights
