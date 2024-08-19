import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import math


from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D

from einops import rearrange
import copy


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def get_beta_schedule(beta_start=0.0001,
                      beta_end=0.02,
                      num_diffusion_timesteps=1000
                      ):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    betas = torch.from_numpy(betas).float().to('cuda')

    return betas

class diffgait_diffusion_branch(nn.Module):
    def __init__(self,
                 num_timesteps=1000,
                 ):
        super().__init__()
        self.betas = get_beta_schedule()
        self.num_timesteps = num_timesteps

    def forward(self, model, skes, sils):
        sils = rearrange(sils, 'n c s h w -> (n s) c h w')
        skes = rearrange(skes, 'n c s h w -> (n s) c h w')

        n, c, h, w = sils.size()
        e = torch.randn_like(sils)
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(n // 2 + 1,)
        ).to('cuda')
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        a = (1-self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        sil_t = sils * a.sqrt() + e * (1.0 - a).sqrt()
        preds = model(skes, sil_t, t.float())
        loss = (sils - preds).square().sum(dim=(1, 2, 3)).mean(dim=0)

        return loss, preds

class DFModel(nn.Module):
    def __init__(self,
                 ch=32, 
                 ch_mult=[1,2,4,4],
                 blocks_num=[1, 1, 1, 1],
                 ):
        super().__init__()

        self.inplanes = ch
        self.ch = ch
        self.ch_mult = ch_mult
        self.blocks_num = blocks_num

        # timestep embedding
        self.temb_ch = ch
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(32,
                            128),
            torch.nn.Linear(128,
                            128),
        ])
        self.layer0 = nn.Sequential(
           conv3x3(2, ch, 1),
           nn.BatchNorm2d(ch),
           nn.ReLU(inplace=True),
        )

        self.layer1 = self.make_layer(BasicBlock2D, ch*ch_mult[0], stride=[1, 1], blocks_num=blocks_num[0], mode='2d')
        self.layer2 = self.make_layer(BasicBlock2D, ch*ch_mult[1], stride=[2, 2],  blocks_num=blocks_num[1], mode='2d')
        self.layer3 = self.make_layer(BasicBlock2D, ch*ch_mult[2], stride=[2, 2],  blocks_num=blocks_num[2], mode='2d')
        self.layer4 = self.make_layer(BasicBlock2D, ch*ch_mult[3], stride=[1, 1],  blocks_num=blocks_num[3], mode='2d')

        self.gait_mapping = nn.Sequential(convbn(1, 64, 3, 2, 1, 1),
                                                  nn.ReLU(inplace=True),
                                                  convbn(64, 128, 3, 2, 1, 1),
                                                  nn.ReLU(inplace=True),)

        self.dres0 = nn.Sequential(convbn(ch*ch_mult[3], ch*ch_mult[3], 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn(ch*ch_mult[2], ch*ch_mult[2], 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(
            nn.ConvTranspose2d(ch*ch_mult[2], ch*ch_mult[1], 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(ch*ch_mult[1]))
        self.dres2 = nn.Sequential(
            nn.ConvTranspose2d(ch*ch_mult[1], ch*ch_mult[0], 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(ch*ch_mult[0]))

        self.classif0 = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):
       if max(stride) > 1 or self.inplanes != planes * block.expansion:
           if mode == '3d':
               downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
           elif mode == '2d':
               downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
           elif mode == 'p3d':
               downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
           else:
               raise TypeError('xxx')
       else:
           downsample = lambda x: x

       layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
       self.inplanes = planes * block.expansion
       s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
       for i in range(1, blocks_num):
           layers.append(
                   block(self.inplanes, planes, stride=s)
           )
       return nn.Sequential(*layers)

    def forward(self, x, y, t):
        temb = get_timestep_embedding(t, 32)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        gm = self.gait_mapping(y)

        latent0 = x4*gm + x4 + temb[:,:,None,None]
        latent1 = self.dres0(latent0) + x3
        latent2 = F.relu(self.dres1(latent1) + x2, inplace=True)
        latent3 = self.dres2(latent2)
        preds = self.classif0(latent3)
        preds = torch.clamp(((preds + 1.0) / 2.0), 0.0, 1.0)
        
        return preds


# ******* DiffGait *******

class DiffGait(BaseModel):
    def build_network(self, model_cfg):
        self.DFModel = DFModel()
        self.diffgait_diffusion_branch = diffgait_diffusion_branch()

    def inputs_pretreament(self, inputs):
        ### Ensure the same data augmentation for heatmap and silhouette
        pose_sils = inputs[0]
        new_data_list = []
        for pose, sil in zip(pose_sils[0], pose_sils[1]):
            sil = sil[:, np.newaxis, ...] # [T, 1, H, W]
            pose_h, pose_w = pose.shape[-2], pose.shape[-1]
            sil_h, sil_w = sil.shape[-2], sil.shape[-1]
            if sil_h != sil_w and pose_h == pose_w:
                cutting = (sil_h - sil_w) // 2
                pose = pose[..., cutting:-cutting]
            cat_data = np.concatenate([pose, sil], axis=1) # [T, 3, H, W]
            new_data_list.append(cat_data)
        new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
        return super().inputs_pretreament(new_inputs)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        
        pose = ipts[0]
        pose = pose.transpose(1, 2).contiguous()
        assert pose.size(-1) in [44, 48, 88, 96]
        maps = pose[:, :2, ...]
        sils = pose[:, -1, ...].unsqueeze(1)

        del ipts
        loss, sils = self.diffgait_diffusion_branch(self.DFModel, maps, sils)

        retval = {
            'loss_sum': loss,
            'visual_summary': {
                'image/sils': sils,
            },
        }

        return retval

