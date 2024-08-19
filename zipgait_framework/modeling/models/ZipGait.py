import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D

from einops import rearrange
import copy
import time

from .diffgait import DFModel


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

class diffgait_denoising_branch(nn.Module):
    def __init__(self,
                 num_timesteps=1000,
                 ):
        super().__init__()
        self.betas = get_beta_schedule()
        self.num_timesteps = num_timesteps
        self.timesteps = 5
        self.skip = self.num_timesteps // self.timesteps
        self.eta = 0.1
        self.seq = range(0, self.num_timesteps, self.skip)
        self.seq_next = [-1] + list(self.seq[:-1])

    def forward(self, model, skes):
        n, c, s, h, w = skes.size()
        skes = rearrange(skes, 'n c s h w -> (n s) c h w')
        
        sil_T = torch.randn(n*s, 1, h, w).cuda()
        out = 0
        sils = [sil_T]

        for i, j in zip(reversed(self.seq), reversed(self.seq_next)):
            t = (torch.ones(n*s) * i).to('cuda')
            next_t = (torch.ones(n*s) * j).to('cuda')
            at = compute_alpha(self.betas, t.long())
            at_next = compute_alpha(self.betas, next_t.long())
            sil_t = sils[-1]
            sil_0 = model(skes, sil_t, t)
            et = (sil_T - at.sqrt() * sil_0) / ((1 - at)).sqrt()
            out = out + sil_0*0.2
            c1 = (
                self.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            sil_t_next = at_next.sqrt() * sil_0 + c1 * torch.randn_like(sil_T) + c2 * et
            sils.append(sil_t_next)
        del sils
        torch.cuda.empty_cache()
        return rearrange(out, '(n s) c h w -> n c s h w', s=s)

# ******* DiffGaitPP *******
blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class DiffGaitPP(BaseModel):
    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d': 
            strides = [
                [1, 1], 
                [1, 2, 2], 
                [1, 2, 2], 
                [1, 1, 1]
            ]
        else: 
            strides = [
                [1, 1], 
                [2, 2], 
                [2, 2], 
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.sil_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(1, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        self.map_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(2, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        self.sil_layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode='2d'))
        self.map_layer1 = copy.deepcopy(self.sil_layer1)
        self.fusion = AttentionFusion(channels[0])   

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d': 
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        self.DFModel = DFModel()
        states = torch.load(
                model_cfg['pretrained_diffgait'],
                map_location='cuda',
            )
        model_state = {key.replace('DFModel.', ''): value for key, value in states['model'].items()}
        self.DFModel.load_state_dict(model_state, strict=True)    
        for param in self.DFModel.parameters():
            param.requires_grad = False
        self.diffgait_denoising_branch = diffgait_denoising_branch()

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

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        
        if len(ipts[0].size()) == 4:
            skes = ipts[0].unsqueeze(1)
        else:
            skes = ipts[0]
            skes = skes.transpose(1, 2).contiguous()
        assert skes.size(-1) in [44, 88]

        with torch.no_grad():
            start_time = time.time()
            sils = self.diffgait_denoising_branch(self.DFModel, skes)
            end_time = time.time()
        total_time = end_time - start_time
        print(f"Total inference time: {total_time} seconds")

        del ipts
        map0 = self.map_layer0(skes)
        map1 = self.map_layer1(map0)

        sil0 = self.sil_layer0(sils)
        sil1 = self.sil_layer1(sil0)

        out1 = self.fusion(sil1, map1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) # [n, c, s, h, w]


        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
                embed = embed_2
        else:
                embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval

class AttentionFusion(nn.Module): 
    def __init__(self, in_channels=64, squeeze_ratio=16):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv3x3(hidden_dim, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels * 2), 
            )
        )
    
    def forward(self, sil_feat, map_feat): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        c = sil_feat.size(1)
        feats = torch.cat([sil_feat, map_feat], dim=1)
        score = self.conv(feats) # [n, 2 * c, s, h, w]
        score = rearrange(score, 'n (d c) s h w -> n d c s h w', d=2)
        score = F.softmax(score, dim=1)
        retun = sil_feat * score[:, 0] + map_feat * score[:, 1]
        return retun

class PGI(nn.Module): 
    def __init__(self, in_channels=64, squeeze_ratio=16):
        super(PGI, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.align = SetBlockWrapper(
                    nn.Sequential(
                        conv1x1(in_channels * 2, hidden_dim), 
                        nn.BatchNorm2d(hidden_dim), 
                        nn.ReLU(inplace=True), 
                        conv3x3(hidden_dim, hidden_dim), 
                        nn.BatchNorm2d(hidden_dim), 
                        nn.Sigmoid(), 
                        conv1x1(hidden_dim, in_channels * 2), 
                    )
                )

    def forward(self, sil_feat, map_feat): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        feats = torch.concat((outs_sil, outs_hm), dim=1)
        align_score = self.align(feats)
        aligned_feats = align_score * feats + feats
        return aligned_feats

class PlusFusion(nn.Module): 
    def __init__(self):
        super(PlusFusion, self).__init__()

    def forward(self, sil_feat, map_feat): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        return sil_feat + map_feat
