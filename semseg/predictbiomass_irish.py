import argparse
import torch
import os
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch.nn.functional as F
import sys

parser = argparse.ArgumentParser('Biomass prediction')
parser.add_argument('--masks', type=str, required=True)
parser.add_argument('--gt', type=str, required=True)
parser.add_argument('--test', type=str)

args = parser.parse_args()

len_masks = len(os.listdir(args.masks))

masks = os.listdir(args.masks)
masks.sort()

torch.manual_seed(1)

histograms_dic = {}
#Loading semseg
for m in tqdm(masks):
    im = np.load(os.path.join(args.masks, m))
    h = [np.mean(im[-1, :, :])]
    im = im[:-1, :, :]
    am = np.argmax(im, axis=0).flatten()
    iam = np.bincount(am, minlength=4)
    i = np.mean(im, axis=(1,2))
    i = np.concatenate((iam/iam.sum(), i, h))
    
    histograms_dic[m.replace('pred','').replace('.npy','').replace('IMG_','')] = i

#Reading few labeled data
keys = []
gt_dic = {}
with open(args.gt) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for i, row in enumerate(csv_reader):
        if i < 2:
            print(row)
        else:
            im_name = row[0].replace('IMG_','').replace('.JPG','').replace('.jpg','')
            keys.append(im_name)
            if row[2] == "#REF!":
                continue
            if row[-1] != "missing data":
                gt_dic[im_name] = [float(r) for r in row[1:5]+[row[-1]]]
            else:
                gt_dic[im_name] = [float(r) for r in row[1:5]+[5.9]]
gt = []
histograms = []

for i, k in enumerate(keys):
    try:
        gt.append(gt_dic[k])
    except:
        print(k, "gt")
        continue
    try:
        histograms.append(histograms_dic[k])
    except:
        print(k, "hist")
        continue
       
gt = torch.tensor(gt)
histograms = torch.tensor(histograms)

ra = torch.randperm(len(gt_dic.keys()))
tio = int(.1 * len(ra))

#Train/val splits
with open("train.csv", 'w') as f_gt:
    writer = csv.writer(f_gt, delimiter=',')
    with open(args.gt) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue
            if i-1 in ra[:tio]:
                writer.writerow(row)
                
with open("val.csv", 'w') as f_gt:
    writer = csv.writer(f_gt, delimiter=',')
    with open(args.gt) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue
            if i-1 in ra[-2*tio:]:
                writer.writerow(row)
                
with open("test.csv", 'w') as f_gt:
    writer = csv.writer(f_gt, delimiter=',')
    with open(args.gt) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue
            if i-1 in ra[tio:-2*tio]:
                writer.writerow(row)


train = torch.tensor(ra[:tio])
val = torch.tensor(ra[-2*tio:])
print(len(train), len(val))

ratio = 3000
dim = len(gt[0])

def train_val(histograms, gt, train, val):
    histogram_0 = np.array([hist.tolist() for hist in histograms])
    gt_0 =  np.array([g[0].item() for g in gt])
    gt_1 =  np.array([g[1].item() for g in gt])
    gt_2 =  np.array([g[2].item() for g in gt])
    gt_3 =  np.array([g[3].item() for g in gt])

    gt_0 = gt_0 / ratio
    reg0 = LinearRegression(1).fit(histogram_0[train], gt_0[train])
    reg1 = LinearRegression(1).fit(histogram_0[train], gt_1[train])
    reg2 = LinearRegression(1).fit(histogram_0[train], gt_2[train])
    reg3 = LinearRegression(1).fit(histogram_0[train], gt_3[train])

    pred0 = reg0.predict(histogram_0[val])
    pred1 = reg1.predict(histogram_0[val])
    pred2 = reg2.predict(histogram_0[val])
    pred3 = reg3.predict(histogram_0[val])

    mse = torch.zeros(len(pred0), dim)
    mae = torch.zeros(len(pred0), dim)

    for i in range(len(pred0)):
        s = pred1[i] + pred2[i] + pred3[i]
        s = s / 100
        pred0[i] = pred0[i] if pred0[i] > 0 else 0
        pred1[i] = pred1[i]/s if pred1[i] > 0 else 0
        pred2[i] = pred2[i]/s if pred2[i] > 0 else 0
        pred3[i] = pred3[i]/s if pred3[i] > 0 else 0

        #Val metrics
        mse[i][0] = (pred0[i] - gt_0[val][i])**2# / ((gt_0[val][i]*ratio) **2)
        mse[i][1] = (pred1[i]/100*ratio*pred0[i] - gt_1[val][i]/100*ratio*gt_0[val][i]) ** 2
        mse[i][2] = (pred2[i]/100*ratio*pred0[i] - gt_2[val][i]/100*ratio*gt_0[val][i]) ** 2
        mse[i][3] = (pred3[i]/100*ratio*pred0[i] - gt_3[val][i]/100*ratio*gt_0[val][i]) ** 2

        #Biomass percentage error
        #mse[i][1] = (pred1[i] - gt_1[val][i]) ** 2
        #mse[i][2] = (pred2[i] - gt_2[val][i]) ** 2
        #mse[i][3] = (pred3[i] - gt_3[val][i]) ** 2

        #MAE
        mae[i][0] = np.absolute((pred0[i] - gt_0[val][i])) / gt_0[val][i]
        mae[i][1] = np.absolute(pred1[i]/100*ratio*pred0[i] - gt_1[val][i]/100*ratio*gt_0[val][i])
        mae[i][2] = np.absolute(pred2[i]/100*ratio*pred0[i] - gt_2[val][i]/100*ratio*gt_0[val][i])
        mae[i][3] = np.absolute(pred3[i]/100*ratio*pred0[i] - gt_3[val][i]/100*ratio*gt_0[val][i])


    l = torch.sqrt(mse)

    rmse = torch.sqrt(torch.mean(mse, 0))
    rmse[0] = rmse[0]
    mae = torch.mean(mae, 0)
    mae[0] = mae[0]

    print('RMSE', rmse, torch.mean(rmse[1:4]))
    print('MAE', mae, torch.mean(mae[1:4]))
    return reg0, reg1, reg2, reg3

from sklearn.linear_model import Ridge as LinearRegression
reg0, reg1, reg2, reg3 = train_val(histograms, gt, train, val)

masks = os.listdir(args.test)

histograms = []

for m in tqdm(masks):
    im = np.load(os.path.join(args.test, m))
    h = [np.mean(im[-1, :, :])]
    im = im[:-1, :, :]
    am = np.argmax(im, axis=0).flatten()
    iam = np.bincount(am, minlength=4)
    i = np.mean(im, axis=(1,2))
    i = np.concatenate((iam/iam.sum(), i, h))

    histograms.append(i)

histograms = np.array(histograms)

pred0 = reg0.predict(histograms)
pred1 = reg1.predict(histograms)
pred2 = reg2.predict(histograms)
pred3 = reg3.predict(histograms)

a = np.zeros(3)
#Automatic labels
with open('biomasscomposition.csv', mode='w') as csv_file:
    fieldnames = ['sample_id','grass_fraction','clover_fraction','weeds_fraction','herbage']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    
    for i in range(len(masks)):
        pred0[i] = pred0[i] if pred0[i] > 0 else 0
        pred1[i] = pred1[i] if pred1[i] > 0 else 0
        pred2[i] = pred2[i] if pred2[i] > 0 else 0
        pred3[i] = pred3[i] if pred3[i] > 0 else 0
        a[0] += pred1[i]
        a[1] += pred2[i]
        a[2] += pred3[i]
        s = (pred1[i] + pred2[i] + pred3[i]) / 100
        writer.writerow({'sample_id': '{}'.format(masks[i]),'grass_fraction':pred1[i]/s,'clover_fraction':pred2[i]/s,'weeds_fraction':pred3[i]/s, 'herbage': pred0[i]*ratio})

    print(a/i)
    
