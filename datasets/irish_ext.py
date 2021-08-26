import numpy as np
import torch
import os
import csv

def make_dataset(root):
    np.random.seed(42)
    img_paths = []
    labels = torch.tensor([])
    with open(os.path.join(root,"train.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            if row[-1] == "missing data":
                continue
            img_paths.append(os.path.join(root, 'images', row[0].replace('npy','jpg').replace('pred','')))
            biomass = torch.tensor([float(r)/100 for r in row[1:4]])
            herbage_mass = torch.tensor([float(row[-2])])
            lab = torch.cat((herbage_mass, biomass), dim=0).view(1,-1)
            labels = torch.cat((labels, lab))
            
    #no validation data from the automatic labels
    img_paths = np.asarray(img_paths)
    return img_paths, labels, None, None
