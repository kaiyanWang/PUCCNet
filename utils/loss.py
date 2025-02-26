import torch.nn as nn
import torch
from torchvision import models
from math import exp

#########################################################################
# 对比学习loss
class Resnet152(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet152, self).__init__()
        res_pretrained_features = models.resnet152(pretrained=True)
        self.slice1 = torch.nn.Sequential(*list(res_pretrained_features.children())[:-5])
        self.slice2 = torch.nn.Sequential(*list(res_pretrained_features.children())[-5:-4])
        self.slice3 = torch.nn.Sequential(*list(res_pretrained_features.children())[-4:-3])
        self.slice4 = torch.nn.Sequential(*list(res_pretrained_features.children())[-3:-2])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        return [h_relu1, h_relu2, h_relu3, h_relu4]


class ContrastLoss_res(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss_res, self).__init__()
        self.vgg = Resnet152().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [ 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ablation = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0
        for i in range(len(a_vgg)):
            a, p, n = a_vgg[i], p_vgg[i], n_vgg[i]
            d_ap = self.l1(a, p.detach())
            if not self.ablation:
                d_an = self.l1(a, n.detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss


import torch
import torch.nn.functional as F


#########################################################################
# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)

##########################################################################
# ca loss
class CALoss(nn.Module):
    def __init__(self):
        super(CALoss, self).__init__()
        self.l1loss=nn.L1Loss()
        self.act=nn.Sigmoid()

    def forward(self,x,y):

        loss=self.l1loss(torch.var(x,dim=1).view(x.size(0),1,x.size(2),x.size(3)),torch.var(y,dim=1).view(y.size(0),1,y.size(2),y.size(3)))
        return loss
