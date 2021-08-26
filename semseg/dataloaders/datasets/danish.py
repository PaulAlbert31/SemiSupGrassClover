from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from utils.visualization import UnNormalize
from PIL import Image

class CloverSegmentation(Dataset):
    """
    GrassClover dataset https://vision.eng.au.dk/grass-clover-dataset/
    """
    NUM_CLASSES = 5

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('danish'),
                 split='train'):
        """
        :param base_dir: path to clover dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = self._base_dir
        self._gt_dir = self._base_dir.replace('imageslowres', 'gtlowres')
    
        self.args = args

        self.unorm = UnNormalize(mean=(0.3612, 0.4310, 0.2176), std=(0.1225, 0.1445, 0.1008))

        self.split = split

        if split in ['train', 'val']:
            self._image_dir = os.path.join(self._image_dir, split)
            self._gt_dir = os.path.join(self._gt_dir, split)
        else:
            raise NotImplementedError


        lines_im = os.listdir(self._image_dir)

        self.images = []
        self.categories = []
        
        for ii, lines in enumerate(lines_im):
            _image, _gt = os.path.join(self._image_dir, lines), os.path.join(self._gt_dir, lines[:-3]+'png')

            if os.path.isfile(_image) and os.path.isfile(_gt):
                self.images.append(_image)
                self.categories.append(_gt)

                assert os.path.isfile(_image)
                assert os.path.isfile(_gt)
                
        if split == 'train':
            self.images = self.images[:800]
            self.categories = self.categories[:len(self.images)]
        elif split == 'val':
            self.images = self.images[:200]
            self.categories = self.categories[:len(self.images)]
            
        #Normalization params
        '''
        std = np.zeros(3)
        mean = np.zeros(3)
        from tqdm import tqdm
        for i in tqdm(range(len(self.images))):
            _img, _ = self._make_img_gt_point_pair(i)
            _img = np.array(_img)
            mean += np.mean(_img, axis=(0,1))
            std += np.std(_img, axis=(0,1))
        mean = mean / i
        std = std / i

        print(mean, std)
        print(a)
        '''
        # Display stats
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
            
        sample['label'] = sample['label'].squeeze(0)
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index])
        _target = Image.open(self.categories[index]).convert('L')
        
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(self.args.base_size),
            tr.RandomCrop(self.args.crop_size),
            tr.ColorJitter(),
            tr.RandomHorizontalFlip(),
            tr.Normalize(mean=(0.3153, 0.3760, 0.1896), std=(0.1629, 0.1968, 0.1151)),
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
        return 'Irish(split=' + str(self.split) + ')'


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
            segmap = decode_segmap(tmp, dataset='danish')
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


