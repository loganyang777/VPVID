# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from .wavelet_utils import layers, layerspp, normalization, dense_layer
import torch.nn as nn
import functools
import torch
import numpy as np
from .wavelet_utils.DWT_IDWT_layer import DWT_2D, IDWT_2D

from .shared import BackboneRegistry

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one

WaveletResnetBlockBigGAN = layerspp.WaveletResnetBlockBigGANpp_Adagn

Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
dense = dense_layer.dense


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
    

@BackboneRegistry.register("ncsnpp_wavelet")
class NCSNpp_Wavelet(nn.Module):
    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""
    
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--no-centered", dest="centered", action="store_false", help="The data is not centered [-1, 1]")
        parser.add_argument("--centered", dest="centered", action="store_true", help="The data is centered [-1, 1]")
        parser.set_defaults(centered=True)

        # 新增参数配置
        parser.add_argument("--scale-by-sigma", type=bool, default=True, help="Scale by sigma")
        parser.add_argument("--nonlinearity", type=str, default="swish", help="Nonlinearity type")
        parser.add_argument("--nf", type=int, default=128, help="Number of filters")
        parser.add_argument("--ch-mult", type=int, nargs="+", default=[1, 1, 2, 2, 2, 2, 2], help="Channel multiplier")
        parser.add_argument("--num-res-blocks", type=int, default=2, help="Number of residual blocks")
        parser.add_argument("--attn-resolutions", type=int, nargs="+", default=[16], help="Attention resolutions")
        parser.add_argument("--resamp-with-conv", type=bool, default=True, help="Resample with convolution")
        parser.add_argument("--conditional", type=bool, default=True, help="Use conditional model")
        parser.add_argument("--fir", type=bool, default=True, help="Use FIR")
        parser.add_argument("--fir-kernel", type=int, nargs="+", default=[1, 3, 3, 1], help="FIR kernel")
        parser.add_argument("--skip-rescale", type=bool, default=True, help="Skip rescale")
        parser.add_argument("--resblock-type", type=str, choices=["ddpm", "biggan", "wavelet"], default="wavelet", help="Resblock type")
        parser.add_argument("--progressive", type=str, choices=["none", "output_skip", "residual"], default="output_skip", help="Progressive type")
        parser.add_argument("--progressive-input", type=str, choices=["none", "input_skip", "residual", "wavelet_downsample"], default="wavelet_downsample", help="Progressive input type")
        parser.add_argument("--progressive-combine", type=str, choices=["sum", "cat"], default="sum", help="Progressive combine method")
        parser.add_argument("--init-scale", type=float, default=0.0, help="Initialization scale")
        parser.add_argument("--fourier-scale", type=int, default=16, help="Fourier scale")
        parser.add_argument("--image-size", type=int, default=256, help="Image size")
        parser.add_argument("--embedding-type", type=str, choices=["fourier", "positional"], default="fourier", help="Embedding type")
        parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
        parser.add_argument("--z-emb-dim", type=int, default=256, help="Z embedding dimension")
        parser.add_argument("--n-mlp", type=int, default=3, help="Number of MLP layers")
        parser.add_argument("--nz", type=int, default=128, help="Number of noise channels")
        return parser

    def __init__(self, 
                nonlinearity, 
                nf, 
                ch_mult, 
                num_res_blocks, 
                attn_resolutions, 
                dropout, 
                resamp_with_conv, 
                image_size,
                conditional, 
                centered, 
                scale_by_sigma,
                fir, 
                fir_kernel, 
                skip_rescale, 
                resblock_type, 
                progressive, 
                progressive_input, 
                progressive_combine,
                embedding_type, 
                init_scale, 
                fourier_scale,
                z_emb_dim,
                n_mlp,
                nz):
        super().__init__()
        
        self.act = act = get_act(nonlinearity)
        self.nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma

        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type.lower()
        self.progressive = progressive.lower()
        self.progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type.lower()
        self.nz = nz
        init_scale = init_scale
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)
        

        num_channels = 4  # x.real, x.imag, y.real, y.imag
        self.output_layer = nn.Conv2d(num_channels, 2, 1)

        modules = []
        # timestep/noise_level embedding
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            ))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
            init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir,
                fir_kernel=fir_kernel, with_conv=True)
        else:
            raise ValueError(f'progressive {progressive} unrecognized.')

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)
        elif progressive_input == 'wavelet_downsample':
            pyramid_downsample = functools.partial(layerspp.WaveletDownsample)
        else:
            raise ValueError(f'progressive input {progressive_input} unrecognized.')

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act,
                dropout=dropout, init_scale=init_scale,
                skip_rescale=skip_rescale, temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)
            
        elif resblock_type == 'wavelet':
            self.no_use_residual = False
            ResnetBlock = functools.partial(WaveletResnetBlockBigGAN,
                                                act=act,
                                                dropout=dropout,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=nf * 4,
                                                zemb_dim=z_emb_dim)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        hs_c2 = [nf]        # 高频信息

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                hs_c2.append(in_ch)
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                elif resblock_type == 'wavelet':
                    modules.append(ResnetBlock(
                            in_ch=in_ch, up=True, hi_in_ch=hs_c2.pop()))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                                    num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        
        mapping_layers = [PixelNorm(),
                          dense(self.nz, z_emb_dim),
                          self.act, ]
        for _ in range(n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)
        
        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")

    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        # Convert real and imaginary parts of (x,y) into four channel dimensions
        x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
                x[:,[1],:,:].real, x[:,[1],:,:].imag), dim=1)
        z = torch.randn(x.size(0), self.nz, device=x.device)
        zemb = self.z_transform(z)

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        # Input layer: Conv2d: 4ch -> 128ch
        hs = [modules[m_idx](x)]
        skipHs = []
        m_idx += 1

        # Down path in U-Net
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                # Attention layer (optional)
                if h.shape[-2] in self.attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            # Downsampling
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                elif self.resblock_type == 'wavelet':
                    h, skipH = modules[m_idx](h, temb, zemb)
                    skipHs.append(skipH)
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1] # actualy equal to: h = h
        if self.resblock_type == 'wavelet':
            h, hlh, hhl, hhh = self.dwt(h)
            h = modules[m_idx](h / 2., temb, zemb)
            h = self.iwt(h * 2., hlh, hhl, hhh)
        else:
            h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        
        h = modules[m_idx](h)  # Attention block
        m_idx += 1
        
        if self.resblock_type == 'wavelet':
            # forward on original feature space
            h = modules[m_idx](h, temb, zemb)
            h, hlh, hhl, hhh = self.dwt(h)
            h = modules[m_idx](h / 2., temb, zemb)  # forward on wavelet space
            h = self.iwt(h * 2., hlh, hhl, hhh)
        else:
            h = modules[m_idx](h, temb, zemb)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)  # Conv2D: 256 -> 4
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)  # Upsample
                        pyramid_h = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            # Upsampling Layer
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                elif self.resblock_type == 'wavelet':
                    h = modules[m_idx](h, temb, zemb, skipH=skipHs.pop())
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb, zemb)  # Upspampling
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules), "Implementation error"
        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        # Convert back to complex number
        h = self.output_layer(h)
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:,None, :, :]
        return h
