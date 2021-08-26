import argparse
import torch
import os
import numpy as np
import csv
from tqdm import tqdm

def parse_num(num):
    num = str(num)
    while len(num) < 4:
        num = '0'+num
    return num

parser = argparse.ArgumentParser('Biomass prediction')
parser.add_argument('--masks', type=str, required=True)
parser.add_argument('--gt', type=str, required=True)
parser.add_argument('--test', type=str, required=True)

args = parser.parse_args()

len_masks = len(os.listdir(args.masks))

masks = ['semantic_segmentation_image_test_{}.png'.format(parse_num(i)) for i in range(len_masks)]
masks = ['predbiomass_image_train_{}.npy'.format(parse_num(i)) for i in range(len_masks)]

torch.manual_seed(2)

histograms = []
histograms_basic = []

for m in tqdm(masks):
    im = np.load(os.path.join(args.masks, m))
    am = np.argmax(im, axis=0).flatten()
    iam = np.bincount(am, minlength=5)
    i = np.mean(im, axis=(1,2))
    i = np.concatenate((iam/iam.sum(), i))
    histograms.append(i)

gt = []
indexs = []
basic_indexs = []
gt_basic = []
with open(args.gt) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for i, row in enumerate(csv_reader):
        if i == 0:
            print(row[-5:])
        else:
            if row[3] != 'basic':
                indexs.append(i-1)
            imp = row[-5:]
            gt.append([float(imp[3]), float(imp[0]), float(imp[2]), float(imp[1]), float(imp[4])]) #[3,0,2,1,4]                

histograms_basic = torch.tensor([histograms[i] for i in basic_indexs])
gt = torch.tensor(gt)
histograms = torch.tensor(histograms)
indexs = torch.tensor(indexs)
ra = indexs[torch.randperm(len(indexs))]
train = ra[:100]
val = ra[100:]
print(len(train), len(val))

with open("train.csv", 'w') as f_gt:
    writer = csv.writer(f_gt, delimiter=',')
    with open(args.gt) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue
            if i-1 in ra[:100]:
                writer.writerow(row)
                
with open("val.csv", 'w') as f_gt:
    writer = csv.writer(f_gt, delimiter=',')
    with open(args.gt) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue
            if i-1 in ra[100:]:
                writer.writerow(row)


dim =  len(gt[0])
from sklearn.linear_model import Ridge as LinearRegression

reg0 = LinearRegression().fit(histograms[train], gt[train, 0])
reg1 = LinearRegression().fit(histograms[train], gt[train, 1])
reg2 = LinearRegression().fit(histograms[train], gt[train, 2])
reg3 = LinearRegression().fit(histograms[train], gt[train, 3])
reg4 = LinearRegression().fit(histograms[train], gt[train, 4])

pred0 = reg0.predict(histograms[val])
pred1 = reg1.predict(histograms[val])
pred2 = reg2.predict(histograms[val])
pred3 = reg3.predict(histograms[val])
pred4 = reg4.predict(histograms[val])

mse = torch.zeros(len(pred0), 5)
mae = torch.zeros(len(pred0), 5)
a = np.zeros(4)
for i in range(len(pred0)):

    pred0[i] = pred0[i] if pred0[i] > 0 else 0
    pred1[i] = pred1[i] if pred1[i] > 0 else 0
    pred2[i] = pred2[i] if pred2[i] > 0 else 0
    pred3[i] = pred3[i] if pred3[i] > 0 else 0
    pred4[i] = pred4[i] if pred4[i] > 0 else 0

    s = pred0[i] + pred2[i] + pred3[i] + pred4[i]
    
    pred0[i] = pred0[i] / s
    pred2[i] = pred2[i] / s
    pred3[i] = pred3[i] / s
    pred4[i] = pred4[i] / s

    mse[i][0] = ((pred0[i] - gt[val, 0][i]) ** 2)
    mse[i][1] = ((pred1[i] - gt[val, 1][i]) ** 2)
    mse[i][2] = ((pred2[i] - gt[val, 2][i]) ** 2)
    mse[i][3] = ((pred3[i] - gt[val, 3][i]) ** 2)
    mse[i][4] = ((pred4[i] - gt[val, 4][i]) ** 2)
    
    mae[i][0] = np.absolute((pred0[i] - gt[val, 0][i]))
    mae[i][1] = np.absolute((pred1[i] - gt[val, 1][i]))
    mae[i][2] = np.absolute((pred2[i] - gt[val, 2][i]))
    mae[i][3] = np.absolute((pred3[i] - gt[val, 3][i]))
    mae[i][4] = np.absolute((pred4[i] - gt[val, 4][i]))

    a[0] += pred0[i]
    a[1] += pred2[i]
    a[2] += pred3[i]
    a[3] += pred4[i]
print(a / i)
rmse = torch.sqrt(torch.mean(mse, 0))

mae = torch.mean(mae, 0)

print('RMSE', rmse, torch.mean(rmse))
print('MAE', mae, torch.mean(mae))

#Predict autom labels
masks = os.listdir(args.test)
histograms_test = []

for m in tqdm(masks):
    im = np.load(os.path.join(args.test, m))
    am = np.argmax(im, axis=0).flatten()
    iam = np.bincount(am, minlength=5)
    i = np.mean(im, axis=(1,2))
    i = np.concatenate((iam/iam.sum(), i))
    histograms_test.append(i)

gt = []
indexs = []
histograms_test = torch.tensor(histograms_test)

pred0 = reg0.predict(histograms_test)
pred1 = reg1.predict(histograms_test)
pred2 = reg2.predict(histograms_test)
pred3 = reg3.predict(histograms_test)
pred4 = reg4.predict(histograms_test)

a = np.zeros(4)
with open('biomasscomposition.csv', mode='w') as csv_file:
    fieldnames = ['sample_id','grass_fraction','clover_fraction','white_clover_fraction','red_clover_fraction','weeds_fraction']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    
    for i in range(len(pred0)):
        pred0[i] = pred0[i] if pred0[i] > 0 else 0
        pred1[i] = pred1[i] if pred1[i] > 0 else 0
        pred2[i] = pred2[i] if pred2[i] > 0 else 0
        pred3[i] = pred3[i] if pred3[i] > 0 else 0
        pred4[i] = pred4[i] if pred4[i] > 0 else 0
        s = pred0[i] + pred4[i] + pred2[i] + pred3[i]
        a[0] += pred0[i]
        a[1] += pred2[i]
        a[2] += pred3[i]
        a[3] += pred4[i]
        writer.writerow({'sample_id': '{}'.format(masks[i]),'grass_fraction':pred0[i]/s,'clover_fraction':pred1[i],'white_clover_fraction':pred2[i]/s,'red_clover_fraction':pred3[i]/s,'weeds_fraction':pred4[i]/s})
    print(a/i)
