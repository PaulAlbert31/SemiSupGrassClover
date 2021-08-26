from __future__ import print_function, division
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms_soft as tr
from utils.visualization import UnNormalize
import torch.nn.functional as F
from PIL import Image

class CloverSegmentation(Dataset):
    NUM_CLASSES = 4 

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('irish'),
                 split='train'):
        """
        :param base_dir: path to clover dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = self._base_dir
        self._gt_dir = self._base_dir

        self.args = args

        self.unorm = UnNormalize(mean=(0.4353, 0.5188, 0.3149), std=(0.1647, 0.1679, 0.1589))

        self.split = split

        if split in ['train', 'val']:
            self._image_dir = os.path.join(self._image_dir, split)
            self._gt_dir = os.path.join(self._gt_dir, split)
        else:
            raise NotImplementedError

        im_n = int(len(os.listdir(self._image_dir)) / 2)
        self.images = []
        self.categories = []
        self.im_ids = []
        names = os.listdir(self._image_dir)
        names = [n for n in names if n[:2] != 'gt']
        for i, name in enumerate(names):
            _image, _gt = os.path.join(self._image_dir, name), os.path.join(self._gt_dir, 'gt_'+name.replace('jpg', 'npz'))

            self.im_ids.append(i)
            if os.path.isfile(_image) and os.path.isfile(_gt):
                self.images.append(_image)
                self.categories.append(_gt)
                
        print('Number of images in {}: {:d}'.format(split, len(self.images)))
        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'label': _target, 'id':index}
        if self.split == "train":
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)

        sample['label_height'] = 1-sample['label'][:, :, 0]
        sample['label'] = torch.argmax(sample['label'], dim=-1)
        
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index])
        _img.draft('RGB',(_img.size[0]//2, _img.size[1]//2))
        
        _target = np.load(self.categories[index])['arr_0']
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.base_size),
            tr.RandomCrop(self.args.crop_size),
            tr.ColorJitter(),
            tr.RandomHorizontalFlip(),
            tr.Normalize(mean=(0.4353, 0.5188, 0.3149), std=(0.1647, 0.1679, 0.1589)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.3153, 0.3760, 0.1896), std=(0.1629, 0.1968, 0.1151)),
            tr.ToTensor()])

        return composed_transforms(sample)
    
    def __str__(self):
        return 'Clover(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = CloverSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='clover')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


