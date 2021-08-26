from PIL import Image, ImageEnhance, ImageFilter
import os
import sys
import random
import numpy as np
from tqdm import tqdm
import copy
import concurrent.futures
import torch
import torchvision
import time
from decode_segmap import decode_seg_map_sequence, decode_segmap


nimages = int(sys.argv[1])
isize = 2000
nelem = 800

root = "synthetic_images"

if not os.path.isdir(root):
    os.mkdir(root)

backgrounds = os.listdir('backgrounds')
croped = os.listdir('cropedbit')

clover = [os.path.join('cropedbit', c) for c in croped if 'clover' in c]
grass = [os.path.join('cropedbit', c) for c in croped if 'grass' in c]
weed = [os.path.join('cropedbit', c) for c in croped if 'weed' in c and c[0] != 'n']
items = [grass, clover, weed]
back = [os.path.join('backgrounds', b) for b in backgrounds]

print(len(clover), 'Clover')
print(len(grass), 'Grass')
print(len(weed), 'Weed')

def generate_syn(back, items, i):
    np.random.seed(int(i))
    random.seed(int(i))
    name = i+'.jpg'
    while len(name) < 9:
        name = '0'+name
        
    ind = int(random.random() * len(back))
    bg = back[ind]
    image = Image.open(bg).convert('RGB')
    image = image.resize((isize,isize))
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance((random.random() * .3) + .7)

    gt = np.zeros((isize, isize, 4))
    gt[:,:,0] = np.ones(image.size)
    
    ratio = 0
    d = np.random.dirichlet((9,2,1))
    o = np.argsort(-d)
    d = d[o]
    nel = int(max(d) * nelem)
    for _ in range(nel):
        #Random element (class balanced)
        r = random.random()
        if r < d[0]:
            cla = 0
        elif r < d[1]+d[0]:
            cla = 1
        else:
            cla = 2
        item = items[cla]
        gauss = gaussians[cla]
        mean, cov = gauss[int(random.random() * len(gauss))]
        #Random sample
        ind = int(random.random() * len(item))
        sample = item[ind]
        cr = Image.open(sample).convert('RGBA')
        
        alpha = np.array(cr.split()[-1])#alpha channel
        alpha[alpha > 0] = 255
        sample_gt = copy.deepcopy(alpha)
        sample_gt[sample_gt > 0] = (cla + 1)
        sample_gt = Image.fromarray(sample_gt).convert('RGBA')
        
        sample_gt.putalpha(Image.fromarray(alpha))
        cr.putalpha(Image.fromarray(alpha))
        
        #Random augmentation
        ##Rotation
        rot = random.random() * 360
        cr = cr.rotate(rot, expand=True)
        sample_gt = sample_gt.rotate(rot, expand=True)
        ##Blur
        radius = round(random.random() * 5)
        cr = cr.filter(ImageFilter.GaussianBlur(radius=radius))
        sample_gt = sample_gt.filter(ImageFilter.GaussianBlur(radius=radius))
        ##Brightness
        brightness = ImageEnhance.Brightness(cr)
        mini = .6
        cr = brightness.enhance((random.random() * .4) + mini)
        ##Resize
        #if cla == 2:
        #    cr = cr.resize((int(cr.size[0]*.5), int(cr.size[1]*.5)))
        ratio = (random.random() * 1) + .5
        cr = cr.resize((int(cr.size[0]*ratio), int(cr.size[1]*ratio)))
        sample_gt = sample_gt.resize((int(sample_gt.size[0]*ratio), int(sample_gt.size[1]*ratio)))
        #Random location
        s = cr.size
        pos = np.random.multivariate_normal(mean, cov)
        pos = (random.random() * image.size[0], random.random() * image.size[1])
        loc = (int(pos[0])-int(s[0]/2),int(pos[1])-int(s[1]/2))
        while loc[0] > image.size[0] or loc[1] > image.size[1]:
            pos = np.random.multivariate_normal(mean, cov)
            loc = (int(pos[0])-int(s[0]/2),int(pos[1])-int(s[1]/2))
        image.paste(cr, loc, cr)
        #Soft label
        mask = np.zeros(image.size)
        sample_gt = np.array(sample_gt)[:,:,-1]
        s = sample_gt.shape
        ms = mask.shape
        loc = (loc[1], loc[0])
        if loc[0] + s[0] > ms[0]:
            diff = loc[0] + s[0] - ms[0]
            sample_gt = sample_gt[:s[0]-diff,:]
            s = sample_gt.shape
        if loc[1] + s[1] > ms[1]:
            diff = loc[1] + s[1] - ms[1]
            sample_gt = sample_gt[:,:s[1]-diff]
            s = sample_gt.shape
        if loc[0] < 0:
            sample_gt = sample_gt[-loc[0]:,:]
            s = sample_gt.shape
            loc = (0, loc[1])
        if loc[1] < 0:
            sample_gt = sample_gt[:,-loc[1]:]
            s = sample_gt.shape
            loc = (loc[0], 0)
        mask[loc[0]:loc[0]+s[0], loc[1]:loc[1]+s[1]][sample_gt>0] = 1

        add = gt[mask > 0,:].max(axis=-1)
        add += 1
        gt[:, :, cla+1][mask > 0] += add
    
    image.save(os.path.join(root, name))
    norm = np.sum(gt, axis=-1, keepdims=True)
    gt = gt / norm
    np.savez_compressed(os.path.join(root, 'gt_'+name.replace('jpg', 'npz')), gt)
    if False: #Visualization
        norm = np.sum(gt, axis=-1, keepdims=True)
        gt = gt / norm
        brightness = 1 / gt[:, :, 0]
        brightness = brightness / 50
        brightness[brightness > 1] = 1
        r, g, b = np.ones((gt.shape[1], gt.shape[0])) * 255, np.ones((gt.shape[1], gt.shape[0])) * 255, np.ones((gt.shape[1], gt.shape[0])) * 255
        r, g, b = r * brightness, g * brightness, b * brightness
        class_img = np.zeros((gt.shape[0], gt.shape[1], 3))
        class_img[:, :, 0] = r
        class_img[:, :, 1] = g
        class_img[:, :, 2] = b
        class_img = Image.fromarray(class_img.astype(np.uint8))
        #im = Image.blend(image_s, Image.fromarray(class_img.astype(np.uint8)), alpha=.9)
        class_img.save(os.path.join(root, 'gt_height_'+name.replace('jpg','png')))
        gt = np.argmax(gt, axis=-1)
        pred = decode_segmap(gt, dataset='irish')*255.
        gt = pred.astype(np.uint8)
        gt = Image.fromarray(gt).convert('RGB')
        #gt = Image.blend(image, gt, alpha=.4)
        gt.save(os.path.join(root, 'gt_'+name.replace('jpg','png')))


def tqdm_parallel(futures):
    tbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))
    tbar.set_description('Generating images')
    for i, f in enumerate(tbar):
        r = f.result()
        if r is not None:
            print(r)

executor = concurrent.futures.ProcessPoolExecutor(12)
futures = [executor.submit(generate_syn, back, items, str(i)) for i in range(nimages)]
tqdm_parallel(futures)

