import argparse
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from utils import make_data_loader
import os
from torch.multiprocessing import set_sharing_strategy
set_sharing_strategy("file_system")

class Trainer(object):
    def __init__(self, args):
        self.args = args

        if args.net == "resnet18":
            from nets.resnet import ResNet18
            model = ResNet18(self.args.num_classes, pretrained=(self.args.pretrained=="imagenet"), dataset=self.args.dataset)
        else:
            raise NotImplementedError
        
        print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        self.model = nn.DataParallel(model).cuda()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.steps, gamma=self.args.gamma)

        self.criterion = nn.MSELoss()
        self.criterion_nored = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')
        
        self.kwargs = {"num_workers": 12, "pin_memory": False}        
        self.train_loader, self.val_loader = make_data_loader(args, **self.kwargs)
        
        #Herbage mass normalization
        if 'clover' in self.args.dataset:
            self.ratio = 3000 #(self.train_loader.dataset.train_labels[:, 0].max() - self.train_loader.dataset.train_labels[:, 0].min()) # 3000 works better
            self.train_loader.dataset.train_labels[:, 0] = self.train_loader.dataset.train_labels[:, 0] / self.ratio #between 0 & 1
            self.val_loader.dataset.train_labels[:, 0] = self.val_loader.dataset.train_labels[:, 0] / self.ratio #between 0 & 1
        
        self.best = float('Inf')
        self.best_mae = float('Inf')
        self.best_mass = float('Inf')
        self.best_epoch = 0
        self.acc = []
        self.train_acc = []
        
        self.epoch = 0
        
    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        self.epoch = epoch
        total = 0
        track_rmse = torch.zeros(self.args.num_classes)
        tbar.set_description("Training, train_loss {}".format(""))
 
        for i, sample in enumerate(tbar):
            img, target = sample["image"].cuda(), sample["target"].cuda()
            assert not torch.isnan(img).any()
            output = self.model(img)
            assert not torch.isnan(output).any()
            
            if "clover" in self.args.dataset:
                output = torch.cat((torch.sigmoid(output[:,0]).unsqueeze(-1), F.softmax(output[:,1:], dim=1)), dim=1)
            elif "danish" in self.args.dataset:
                output = F.softmax(output, dim=1)
                output = torch.cat((output[:, 0].unsqueeze(-1), (output[:, 1] + output[:, 2]).unsqueeze(-1), output[:, 1:]), dim=1) #Grass, Clover, White, Red, Weeds
                
            loss = self.criterion_nored(output, target)
            track_rmse += loss.detach().cpu().sum(dim=0)
            total += img.shape[0]
            loss = torch.sqrt(loss.mean())

            assert not torch.isnan(loss).any()

            if i % 5 == 0:
                tbar.set_description("Training, train loss {:.2f}, lr {:.3f}".format(loss.item(), self.optimizer.param_groups[0]['lr']))
                
            # compute gradient and do SGD step
            loss.backward()
                        
            self.optimizer.step()           
            self.optimizer.zero_grad()
            
        self.scheduler.step()
        print("Epoch: {0}".format(epoch))
        print("Training RMSE: {}, MEAN {}".format(torch.sqrt(track_rmse/total), torch.sqrt(track_rmse/total).mean()))
        #Checkpoints
        torch.save({'best':self.best, 'best_mae':self.best_mae, 'best_mass':self.best_mass, 'epoch':self.epoch, 'net':self.model.state_dict()}, os.path.join(self.args.save_dir, "last_model.pth.tar"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.args.save_dir, "last_optimizer.pth.tar"))
    
    def val(self, epoch):
        self.model.eval()
        acc = torch.zeros(self.args.num_classes)
        maes = torch.zeros(self.args.num_classes)
        mass = torch.zeros(3)
        vbar = tqdm(self.val_loader)
        total = 0
        rmses = torch.zeros(len(self.val_loader.dataset), self.args.num_classes-1)
        targets = torch.zeros(len(self.val_loader.dataset), self.args.num_classes)
        herbage_rmse = torch.zeros(1)
        biomass_rmse = torch.zeros(3)
        with torch.no_grad():
            for i, sample in enumerate(vbar):
                image, target, ids = sample["image"].cuda(), sample["target"].cuda(), sample["index"]
                output = self.model(image)
                
                if "clover" in self.args.dataset:
                    output = torch.cat((torch.sigmoid(output[:,0]).unsqueeze(-1), F.softmax(output[:,1:], dim=1)), dim=1) #Herbage mass, Grass, Clover, Weeds
                elif "danish" in self.args.dataset:
                    output_c = F.softmax(output, dim=1)
                    output = torch.cat((output_c[:, 0].unsqueeze(-1), (output_c[:, 1] + output_c[:, 2]).unsqueeze(-1), output_c[:, 1:]), dim=1) #Grass, Clover, White, Red, Weeds
                    
                targets[ids] = target.cpu()
                
                loss = self.criterion_nored(output, target)
                mae = self.mae(output, target)
                
                if "clover" in self.args.dataset:
                    herbage_rmse += loss[:, 0].sum(dim=0).cpu()
                    biomass_rmse += loss[:, 1:].sum(dim=0).cpu()
                    loss[:, 0] = loss[:, 0] #/ (output[:, 0] ** 2) #Uncomment for the relative herbage mass error
                    mae[:, 0] = mae[:, 0] #/ (output[:, 0])
                elif "danish" in self.args.dataset:
                    rmses += torch.cat((loss[:, 0].unsqueeze(-1), loss[:, 2:]), dim=1).detach().cpu().sum(dim=0)

                maes += mae.sum(dim=0).cpu()
                acc += loss.sum(dim=0).cpu()
                if "clover" in self.args.dataset:
                    #Per class herbage mass error (RMSE or MAE)
                    mass += ((self.ratio * output[:,0].unsqueeze(-1) * output[:,1:4] - self.ratio * target[:,0].unsqueeze(-1) * target[:,1:4])**2).sum(dim=0).cpu()
                    #mass += torch.absolute((self.ratio * output[:,0].unsqueeze(-1) * output[:,1:4] - self.ratio * target[:,0].unsqueeze(-1) * target[:,1:4])).sum(dim=0).cpu()
                    
                total += image.size(0)

                if i % 10 == 0:
                    vbar.set_description("Validation loss: {0:.2f}".format(loss.mean()))
                    
        final_acc = torch.sqrt(acc/total)
        final_mae = maes/total
        #Average the errors
        if "clover" in self.args.dataset:
            final_mass = torch.sqrt(mass / total)
        else:
            final_mass = torch.tensor([0,0], dtype=torch.float)
            
        print("[Epoch: {}, numImages: {}]".format(epoch, (len(self.val_loader)-1)*self.args.batch_size + image.shape[0]))
        self.acc.append(final_acc)
        herbage_rmse, biomass_rmse = torch.sqrt(herbage_rmse/total), torch.sqrt(biomass_rmse/total)

        #Choose the early stopping metric
        if "clover" in self.args.dataset:
            metric = final_mass.mean()
        elif "danish" in self.args.dataset:
            metric = torch.sqrt(rmses / total).mean()

        #Best model
        if metric <= self.best:
            self.best = metric
            self.best_mae = final_mae.mean()
            self.best_mass = final_mass.mean()
            self.best_epoch = epoch
            torch.save({'best':self.best, 'best_mae': self.best_mae, 'best_mass': self.best_mass, 'epoch':self.epoch, 'net':self.model.state_dict()}, os.path.join(self.args.save_dir, "best_model.pth.tar"))
            torch.save(self.optimizer.state_dict(), os.path.join(self.args.save_dir, "best_optimizer.pth.tar"))

        #Logging
        print("Validation Accuracy: {0:.4f}, best RMSE {1:.4f}, MAE {2:.4f}, MASS_RMSE {3:.4f} at epoch {4}".format(metric.item(), self.best, self.best_mae, self.best_mass, self.best_epoch))
        print("Detailed errors RMSE", final_acc, torch.sqrt(rmses/total).mean())
        print("MAE", final_mae)
        print("MASS_RMSE", final_mass)
        if "clover" in self.args.dataset:
            print("Herbage_RMSE", herbage_rmse*self.ratio)
        return final_acc
    
    def predict(self, images_folder):
        import csv
        from PIL import Image
        self.model.eval()
        images = [os.path.join(images_folder, im) for im in os.listdir(images_folder)]
        images.sort()
        
        if os.path.isfile('biomasscomposition.csv'):
            os.remove('biomasscomposition.csv')
        
        with open('biomasscomposition.csv', mode='a') as csv_file:
            fieldnames = ['sample_id','grass_fraction','clover_fraction','white_clover_fraction','red_clover_fraction','weeds_fraction']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
        tbar = tqdm(images)
        tbar.set_description("Predicting test images")
        for i in tbar:
            im = Image.open(i)
            im.draft('RGB',(im.size[0]//2, im.size[1]//2))
                     
            im = self.val_loader.dataset.transform(im).unsqueeze(0)
            with torch.no_grad():
                output = self.model(im)
                if "danish" in self.args.dataset:
                    output_c = F.softmax(output, dim=1)
                    output = torch.cat((output_c[:, 0].unsqueeze(-1), (output_c[:, 1] + output_c[:, 2]).unsqueeze(-1), output_c[:, 1:]), dim=1) #Grass, Clover, White, Red, Weeds
                    output = output.view(-1).tolist()
                elif "clover" in self.args.dataset:
                    output = torch.cat((torch.sigmoid(output[:,0]).unsqueeze(-1), F.softmax(output[:,1:], dim=1)), dim=1)
            print(i, output)
            if "danish" in self.args.dataset:
                with open('biomasscomposition.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
                    writer.writerow({'sample_id': '{}'.format(i.split('/')[-1].replace('.jpg', '')),'grass_fraction':output[0],'clover_fraction':output[1],'white_clover_fraction':output[2],'red_clover_fraction':output[3],'weeds_fraction':output[4]})
    
def main():


    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--net", type=str, default="resnet18",
                        choices=["resnet18"],
                        help="net name")
    parser.add_argument("--dataset", type=str, default="clover", choices=["clover", "clover_ext", "danish", "danish_ext"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--steps', type=int, default=[50,80], nargs='+', help='Epochs when to reduce lr')
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=0.5, help="Multiplicative factor for lr decrease, default .5")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="No cuda")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--no-eval", default=False, action='store_true')
    parser.add_argument("--pretrained", default=None, type=str)
    parser.add_argument("--predict", type=str, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--base-da", default=False, action='store_true')
    parser.add_argument("--no-TS", default=False, action='store_true')
    parser.add_argument("--no-pertu", default=False, action='store_true')
    parser.add_argument("--autom-only", default=False, action='store_true')
    
    args = parser.parse_args()
    #For reproducibility purposes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        
    args.cuda = not args.no_cuda
    
    torch.manual_seed(args.seed)
    
    _trainer = Trainer(args)
    start_ep = 0  
    
    if args.resume is not None:
        l = torch.load(args.resume)
        start_ep = l['epoch']
        _trainer.best = l['best']
        _trainer.best_mae = l['best_mae']
        _trainer.best_mass = l['best_mass']
        _trainer.best_epoch = l['epoch']
        _trainer.epoch = l['epoch']
        _trainer.model.load_state_dict(l['net'])
        _trainer.optimizer.load_state_dict(torch.load(args.resume.replace("model", "optimizer")))
        if not args.no_eval:
            _trainer.val(start_ep)
        if args.predict is not None:
            _trainer.predict(args.predict)
                
    for eps in range(start_ep, args.epochs):
        _trainer.train(eps)
        
        if not args.no_eval:
            _trainer.val(eps)

if __name__ == "__main__":
   main()
