from datasets.irish import make_dataset, Irish
from datasets.irish_ext import make_dataset as make_dataset_ext
from datasets.danish import make_dataset as make_dataset_danish
from datasets.danish_ext import make_dataset as make_dataset_danishext
from datasets.danish import Danish
from mypath import Path
import os
import numpy as np
import torch

def irish(root=Path.db_root_dir('irish'), transform=None, transform_test=None):
    train_data, train_labels, val_data, val_labels = make_dataset(root=root)
    trainset = Irish(train_data, train_labels, transform=transform)
    testset = Irish(val_data, val_labels, transform=transform_test)
    return trainset, testset

def irish_ext(root=Path.db_root_dir('irish'), transform=None, transform_test=None, autom_only=False, no_pertu=False):
    if autom_only:
        train_labeled_data, train_labeled_labels =  np.array([]), torch.tensor([])
    else:
        train_labeled_data, train_labeled_labels, val_data, val_labels = make_dataset(root=root)
        
    train_unlabeled_data, train_unlabeled_labels, _, _ = make_dataset_ext(root=Path.db_root_dir('irish_ext'))
    clean_noisy = torch.tensor([1 for _ in range(len(train_labeled_data))] + [0 for _ in range(len(train_unlabeled_data))]).bool()
    
    noisy_indexes = range(len(train_labeled_data), len(train_unlabeled_data) + len(train_labeled_data))
    clean_indexes = range(len(train_labeled_data))

    train_data = train_labeled_data.tolist() + train_unlabeled_data.tolist()
    train_labels = torch.cat((train_labeled_labels, train_unlabeled_labels))
    
    trainset = Irish(train_data, train_labels, transform=transform, pertu=not no_pertu)
    testset = Irish(val_data, val_labels, transform=transform_test)

    trainset.clean_noisy = clean_noisy
    
    return trainset, testset, noisy_indexes, clean_indexes

def danish(root=Path.db_root_dir('danish'), transform=None, transform_test=None):
    train_data, train_labels, val_data, val_labels = make_dataset_danish(root=root)
    trainset = Danish(train_data, train_labels, transform=transform)
    testset = Danish(val_data, val_labels, transform=transform_test)
    return trainset, testset

def danish_ext(root=Path.db_root_dir('danish'), transform=None, transform_test=None, num_classes=None):
    train_labeled_data, train_labeled_labels, val_data, val_labels = make_dataset_danish(root=root)
    train_unlabeled_data, train_unlabeled_labels, _, _ = make_dataset_danishext(root=Path.db_root_dir('danish_ext'))

    noisy_indexes = range(len(train_labeled_data), len(train_labeled_data) + len(train_unlabeled_data))
    clean_indexes = range(len(train_labeled_data))
    train_data = train_labeled_data.tolist() + train_unlabeled_data.tolist()
    train_labels = torch.cat((train_labeled_labels, train_unlabeled_labels))
    trainset = Danish(train_data, train_labels, transform=transform, pertu=True)
    testset = Danish(val_data, val_labels, transform=transform_test)
    
    clean_noisy = torch.tensor([1 for _ in range(len(train_labeled_data))] + [0 for _ in range(len(train_unlabeled_data))]).bool()
    
    trainset.clean_noisy = clean_noisy
    
    return trainset, testset, noisy_indexes, clean_indexes
