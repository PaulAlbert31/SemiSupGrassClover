import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
from PIL import Image
from dataloaders.utils import decode_segmap
from dataloaders import custom_transforms as tr
from dataloaders import make_data_loader

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.mioutab = []
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        self.kwargs = kwargs
        
        self.train_loader, self.val_loader, self.test_loader, self.nclass, self.unorm = make_data_loader(args, **kwargs)            
            
        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        freeze_bn=args.freeze_bn,
                        crop_size=args.crop_size,
                        train_height=self.args.train_height)
        
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Define Optimizer
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight, _ = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
                np.save(os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy'), weight)
            weight = torch.from_numpy(weight.astype(np.float32))
            print("Class weights:", weight)
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda, ignore_index=args.ignore_index).build_loss(mode=args.loss_type)
        self.mse = SegmentationLosses(weight=weight, cuda=args.cuda, ignore_index=args.ignore_index).build_loss(mode='mse')
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader), lr_step=args.lr_step)
        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()
        
        # Resuming checkpoint
        self.best_pred = 0.0
        self.best_height = float('Inf')
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']
            self.best_pred = checkpoint['best_pred']
            args.start_epoch = checkpoint['epoch']
            
            if 'module' in list(state_dict.keys())[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                state_dict = new_state_dict
         
            self.model.load_state_dict(state_dict, strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}), running best {}"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_pred']))
                

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        
        mean = torch.zeros(3)
        std = torch.zeros(3)
        
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                                   
            output, out_height = self.model(image)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            
            output_interp = F.interpolate(output, size=image.size()[2:], mode='bilinear', align_corners=True)
            
            loss = self.criterion(output_interp, target)
            
            if self.args.train_height:
                target_h = sample['label_height'].cuda()
                out_interp_height = F.interpolate(out_height, size=image.size()[2:], mode='bilinear', align_corners=True)
                loss_height = self.mse(torch.sigmoid(out_interp_height).squeeze(1), target_h)
                loss_red = loss.mean() + torch.sqrt(loss_height.mean())
            else:
                loss_height = torch.tensor(0).float()
                loss_red = loss.mean()
                
            loss_red.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            train_loss += loss_red.item()
            tbar.set_description('Train loss: {0:.4f}, height loss {1:.4f}'.format(train_loss / (i + 1), loss_height.item()))
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        
        # save checkpoint every epoch
        is_best = False
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),#'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)
        
    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        total = 0
        loss_height = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output, out_height = self.model(image)
                
            output = F.interpolate(output, size=image.size()[2:], mode='bilinear', align_corners=True)
            if self.args.train_height:
                target_h = sample['label_height'].cuda()
                out_interp_height = F.interpolate(out_height, size=image.size()[2:], mode='bilinear', align_corners=True)
                loss_height += self.mse(torch.sigmoid(out_interp_height).squeeze(1), target_h).sum()
                total += image.shape[0]
                
            loss_m = self.criterion(output, target)
            test_loss += loss_m.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            output = F.softmax(output, dim=1)

            pred = torch.argmax(output, dim=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())

        if self.args.train_height:
            loss_height = torch.sqrt(loss_height/total)
        else:
            loss_height = 0
            
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        
        print(self.evaluator.confusion_matrix / self.evaluator.confusion_matrix.sum(axis=1))
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}; height {}".format(Acc, Acc_class, mIoU, FWIoU, loss_height))
        print("Class IoU: {}".format(IoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        self.mioutab.append(new_pred)
        if new_pred > self.best_pred:
            self.best_weights = self.model.state_dict()#self.best_weights = self.model.module.state_dict()
            self.best_pred = new_pred
            self.best_height = loss_height
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),#'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'best_height': self.best_height
            }, True, filename='checkpoint_best.pth.tar')

    def predict(self, images):
        from torchvision import transforms
        """
        #getting normalisation params
        tbar = tqdm(images)
        std = np.zeros(3)
        mean = np.zeros(3)

        for i, imagen in enumerate(tbar):
            image = Image.open(imagen)
            image.draft('RGB',(image.size[0]//4, image.size[1]//4))
            image = np.array(image)
            mean += np.mean(image, axis=(0,1))
            std += np.std(image, axis=(0,1))
            if i > 2000:
                break

        mean = mean / i / 255.
        std = std / i / 255.
        print(mean, std)
        """
        
        #Irish
        if "irish" in self.args.dataset:
            mean = (0.41637952, 0.5502375,  0.2436111) 
            std = (0.190736, 0.21874362, 0.15318967)
        
        #Danish
        if "danish" in self.args.dataset:
            mean = (0.31345951, 0.4316692,  0.16106741)
            std  = (0.15147385, 0.18103442, 0.12314612)

        transf = transforms.Compose([
            transforms.Resize(1024),
            transforms.RandomCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        transf_flip = transforms.RandomHorizontalFlip(p=1)
        tbar = tqdm(np.random.choice(images, size=200), desc='\r')

        self.model.train() #BatchNorm parameters tuning
        image_batch = torch.tensor([])
        if self.args.cuda:
            image_batch = image_batch.cuda()
            
        mean_p = torch.zeros(self.nclass)
        for i, imagen in enumerate(tbar):
            image = Image.open(imagen)
            image = transf(image).unsqueeze(0)
            if self.args.cuda:
                image = image.cuda()
            image_batch = torch.cat((image_batch, image))
            if i%16 == 0 and i > 0:
                with torch.no_grad():
                    output, emb = self.model(image_batch)
                    image_batch = torch.tensor([])
                    if self.args.cuda:
                        image_batch = image_batch.cuda()
                    mean_p += F.softmax(output, dim=1).mean(dim=(0,2,3)).cpu()
        mean_p = mean_p / i * 16
        print(mean_p)

        transf = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        tbar = tqdm(images, desc='\r')
    
        self.model.eval()

        #To visualize the predictions
        visu = False
        if visu:
            self.toPIL = transforms.ToPILImage()

        #Perdictions
        for i, imagen in enumerate(tbar):
            image_ntf = Image.open(imagen)
            ori_size = image_ntf.size
    
            image = transf(image_ntf).unsqueeze(0)
            
            if self.args.cuda:
                image = image.cuda()
            
            with torch.no_grad():
                output, output_h = self.model(image)
                    
            output = F.interpolate(output, size=image.size()[2:], mode='bilinear', align_corners=True)
            if self.args.train_height:
                output_h = F.interpolate(output_h, size=image.size()[2:], mode='bilinear', align_corners=True)
                
            if self.args.train_height:
                output_h = torch.sigmoid(output_h)
                pred = torch.cat((output, output_h), dim=1)
            else:
                pred = output

            pred = pred.squeeze(0)
                                        
            max_pred = torch.max(pred, dim=0)[0]

            pred = pred.cpu()
            
            if visu:
                a_pred = torch.argmax(pred, dim=0)

            #To be used by the linear regression
            if not visu:
                np.save('preds/pred{}.npy'.format(imagen.split('/')[-1][:-4]), pred)
            elif visu:
                image_s = self.toPIL(self.unorm(image[0].cpu())).convert('L').convert('RGB') #so elegant ^^
                pred = decode_segmap(a_pred.numpy(), dataset=self.args.dataset)*255.
                segmap = pred.astype(np.uint8)
                pred_s = self.toPIL(segmap)
                im = Image.blend(image_s, pred_s, alpha=.4)
                im.save('preds/pred{}.png'.format(imagen.split('_')[-1][:-4]))
    
def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus for segmenting semantic clover images")
    parser.add_argument('--backbone', type=str, default='resnet34',
                        choices=['resnet34'],
                        help='backbone name (default: resnet34)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='irish',
                        choices=['danish', 'irish'],
                        help='dataset name (default: irish)')
    parser.add_argument('--workers', type=int, default=12,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'ls'],
                        help='loss func type (default: ce)')
    
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--lr-step', type=int, default=None,
                        help='epoch step for lr sheduler, (default: None)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    
    #Predict
    parser.add_argument('--predict', type=str, default=None, help='dir of images to predict')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    args.train_height = "irish" in args.dataset
    args.ignore_index = 255

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)

    print(args)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(int(np.exp(1) * 1000000))
    np.random.seed(int(np.exp(1) * 1000000))
    trainer = Trainer(args)

    if args.predict is not None:
        if not os.path.isdir('preds'):
            os.mkdir('preds')
        images = os.listdir(args.predict)
        images = [os.path.join(args.predict, image) for image in images if 'jpg' in image]
        images.sort()
        trainer.predict(images)
        return
    
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
           
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
            torch.save(trainer.mioutab, os.path.join('run', args.dataset, args.checkname, 'miou'))
            
if __name__ == "__main__":
   main()
