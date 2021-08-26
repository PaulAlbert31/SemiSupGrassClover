from dataloaders.datasets import danish, irish
from torch.utils.data import DataLoader
import numpy as np
import torch

def make_data_loader(args, **kwargs):
    if args.dataset == 'danish':
        train_set = danish.CloverSegmentation(args, split='train')
        val_set = danish.CloverSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class, train_set.unorm
    elif args.dataset == 'irish':
        train_set = irish.CloverSegmentation(args, split='train')
        val_set = irish.CloverSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class, train_set.unorm    
    else:
        raise NotImplementedError

