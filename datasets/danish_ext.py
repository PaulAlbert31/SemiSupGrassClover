import torchvision as tv
import numpy as np
from PIL import Image
import time
from torch.utils.data import Dataset
import torch
import os
import csv
from tqdm import tqdm

def make_dataset(root):
    np.random.seed(42)
    img_paths = []
    labels = torch.tensor([])
    with open(os.path.join(root,"train.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            img_paths.append(os.path.join(root, 'images', row[0].replace('npy','jpg').replace('pred','')))
            biomass = torch.tensor([float(r) for r in row[1:]]).unsqueeze(0)
            #Grass, Clover, White, Red, Weeds
            labels = torch.cat((labels, biomass))

    img_paths = np.array(img_paths)
    labels = torch.tensor(labels)
    return img_paths, labels, None, None
