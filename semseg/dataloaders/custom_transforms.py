import torch
import torchvision
import random
import numpy as np
import copy
import torchvision.transforms.functional as F

from PIL import Image, ImageOps, ImageFilter
import time
import copy

from .utils import decode_segmap

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        
        img /= 255.0
        img -= self.mean
        img /= self.std

        sample['image'] = img
        sample['label'] = mask
                
        return sample

class ColorJitter(object):
    
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.jitter = torchvision.transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)        

    def __call__(self, sample):
       
        sample['image'] = self.jitter(sample["image"])
        return sample

class NormalizeImage(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.NUM_CLASSES=5
        
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        
        img = np.array(img).transpose((2, 0, 1))
        mask = np.array(mask)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        
        sample['image'] = img
        sample['label'] = mask
        
        return sample
    
class ToTensorImage(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        #No flip
        if random.random() < 0.5:
            return sample
        
        img = sample['image']
        mask = sample['label']
        
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        
        img = img.rotate(rotate_degree, Image.BILINEAR)
        #Can be changed to Image.BILINEAR for a soft label
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        sample['image'] = img
        sample['label'] = mask
        
        return sample

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            
        sample['image'] = img
        
        return sample


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.NUM_CLASSES = 5
        self.toPIL = torchvision.transforms.ToPILImage()

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        h, w = img.size

        #No resize
        if self.base_size == -1:
            self.base_size = min(h, w)

        if self.base_size > 0:
            if w < h:
                ow = random.randint(int(self.base_size), int(self.base_size * 1.33))
                oh = int(1.0 * ow * h / w)
            else:
                oh = random.randint(int(self.base_size), int(self.base_size * 1.33))
                ow = int(1.0 * oh * w / h)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        #Same random crop
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            img, output_size=(self.crop_size, self.crop_size))
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)


        sample["image"] = img
        sample["label"] = mask
        
        return sample


class CenterGT(object):
    def __init__(self, crop_size=400):
        self.crop_size = crop_size
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        h, w, c = img.shape
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        w, h = img.size
        delta_w = w - mask.size[0]
        delta_h = h - mask.size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_mask = ImageOps.expand(mask, padding)

        sample["image"] = img
        sample["label"] = new_mask
        
        return sample


class RandomAffine(object):
    def __init__(self, degrees=30, translate=[.3, .3], scale=None, shear=.3, resample=False, fillcolor=255):
        self.degrees = (-degrees, degrees)
        self.translate = translate
        self.scale = scale
        self.shear = (-shear, shear)
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        
        ret = torchvision.transforms.RandomAffine.get_params(degrees=self.degrees, translate=self.translate, scale_ranges=self.scale, shears=self.shear, img_size=img.size)
        img = F.affine(img, *ret, resample=self.resample, fillcolor=0)
        mask = F.affine(Image.fromarray(mask), *ret, resample=self.resample, fillcolor=self.fillcolor)

        sample["image"] = img
        sample["label"] = mask
        
        return sample

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        sample["image"] = img
        sample["label"] = mask
        
        return sample

class FixScaleCropImage(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        
        return img

class PaddedFixScale(object):
    def __init__(self, crop_size, fill=255):
        self.crop_size = crop_size
        self.fill = fill
        
    def __call__(self, img):
        w, h = img.size
        if w < h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
            
        return img, mask

    

class FixedResize(object):
    def __init__(self, size):
        self.size = size  # size: (h, w)

    def __call__(self, sample):
        if self.size == -1:
            return sample
        img = sample['image']
        mask = sample['label']
        
        #assert img.size == mask.size        
        w, h = img.size
        r = w / h
        if w > h:
            shape = (int(self.size*r), self.size)
        else:
            shape = (self.size, int(self.size/r))

        img = img.resize(shape, Image.BILINEAR)
        mask = mask.resize(shape, Image.NEAREST)
        
        sample["image"] = img
        sample["label"] = mask
        
        return sample


class RandomCrop(torchvision.transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        super(RandomCrop, self).__init__(size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        sample["image"] = img
        sample["label"] = mask
        
        return sample


def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)
