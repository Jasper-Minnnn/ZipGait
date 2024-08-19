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
from torchvision.utils import save_image


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

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

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

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
        preds = []
        sils = [sil_T]

        for i, j in zip(reversed(self.seq), reversed(self.seq_next)):
            t = (torch.ones(n*s) * i).to('cuda')
            next_t = (torch.ones(n*s) * j).to('cuda')
            at = compute_alpha(self.betas, t.long())
            at_next = compute_alpha(self.betas, next_t.long())
            sil_t = sils[-1]
            sil_0 = model(skes, sil_t, t)
            preds.append(sil_0)
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
        return preds
        # return rearrange(out, '(n s) c h w -> n c s h w', s=s)

class DFModel(nn.Module):
    def __init__(self,
                 ch=32, 
                 ch_mult=[1,2,4],
                 blocks_num=[1, 1, 1, 1],
                 dropout=0.1
                 ):
        super().__init__()

        self.inplanes = ch
        self.ch = ch
        self.ch_mult = ch_mult
        self.blocks_num = blocks_num
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = 1

        # timestep embedding
        self.temb_ch = ch*4
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(32,
                            128),
            torch.nn.Linear(128,
                            128),
        ])

        #down
        self.conv_in = torch.nn.Conv2d(2,
                                       ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        curr_res = 64
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, True)
                curr_res = curr_res // 2
            self.down.append(down)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        1,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        self.gait_mapping = nn.Sequential(convbn(1, 64, 3, 2, 1, 1),
                                                  nn.ReLU(inplace=True),
                                                  convbn(64, 128, 3, 2, 1, 1),
                                                  nn.ReLU(inplace=True),)

    def forward(self, x, y, t):
        # timestep embedding
        temb = get_timestep_embedding(t, 32)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        gm = self.gait_mapping(y)
        h = hs[-1]
        h = h*gm + temb[:,:,None,None]

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        return h

# ******* DiffGait *******

class DiffGaitPP_Large(BaseModel):
    def build_network(self, model_cfg):
        self.DFModel = DFModel()
        # self.diffgait_diffusion_branch = diffgait_diffusion_branch()
        self.diffgait_denoising_branch = diffgait_denoising_branch()
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
        
        ### diffusion
        # del ipts
        # loss, sils = self.diffgait_diffusion_branch(self.DFModel, maps, sils)

        # retval = {
        #     'loss_sum': loss,
        #     'visual_summary': {
        #         'image/sils': sils,
        #     },
        # }

        # return retval

