import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, crop_size=513, freeze_bn=False, train_height=False):
        super(DeepLab, self).__init__()
        
        BatchNorm = nn.BatchNorm2d
            
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        if train_height:
            self.aspp_height = build_aspp(backbone, output_stride, BatchNorm)
            self.decoder_height = build_decoder(1, backbone, BatchNorm)

        self.crop_size = crop_size

        self.num_classes = num_classes
        self.train_height = train_height
        
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, use_crf=False):
        x, low_level_feat = self.backbone(input)
        crf_out = None
        x_a = self.aspp(x)
        x_a = self.decoder(x_a, low_level_feat)
        if self.train_height:
            x_height = self.aspp_height(x)
            x_height = self.decoder_height(x_height, low_level_feat)
            return x_a, x_height
        return x_a, None

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.Linear):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                            
    def get_discriminator_params(self):
        modules = [self.discriminator]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]#, self.backbone]#, self.decoder_real, self.aspp_real]#, self.convcrf]
        try:
            modules.append(self.aspp_pseudo)
            modules.append(self.decoder_pseudo)
        except:
            pass
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], (nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


