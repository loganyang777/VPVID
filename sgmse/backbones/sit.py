# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# 9.23 14.31保存 可以收敛但效果待改进
# -------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from .shared import BackboneRegistry
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 用于将整数或元组转换为元组
def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                              These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_size, mlp_hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(mlp_hidden_dim, hidden_size),
        # )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding with Variable Input Size Support"""
    def __init__(
        self,
        img_size=None,
        patch_size=16,
        in_channels=4,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        self.img_size = img_size  # Can be None for variable sizes
        self.patch_size = to_2tuple(patch_size)
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
            bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Compute grid size and number of patches dynamically
        H_patches = H // self.patch_size[0]
        W_patches = W // self.patch_size[1]
        num_patches = H_patches * W_patches

        # Apply convolutional projection
        x = self.proj(x)  # Shape: (B, embed_dim, H_patches, W_patches)

        # Flatten if required
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # Shape: (B, num_patches, embed_dim)

        # Apply normalization
        x = self.norm(x)

        return x, (H_patches, W_patches)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@BackboneRegistry.register("SIT")
class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--sit_config', type=str, default='SiT-B/2', choices=model_configs.keys(), help="Select SiT size from pre-defined configurations")
        parser.add_argument('--depth', type=int, default=None, help='覆盖默认的 depth 参数')
        parser.add_argument('--hidden_size', type=int, default=None, help='覆盖默认的 hidden_size 参数')
        parser.add_argument('--patch_size', type=int, default=None, help='覆盖默认的 patch_size 参数')
        parser.add_argument('--num_heads', type=int, default=None, help='覆盖默认的 num_heads 参数')
        parser.add_argument('--in_channels', type=int, default=4, help='输入通道数')
        parser.add_argument('--out_channels', type=int, default=2, help='输出通道数')
        parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP 比例')
        parser.add_argument('--class_dropout_prob', type=float, default=0.1, help='分类器 dropout 概率')
        parser.add_argument('--num_classes', type=int, default=1000, help='类别数量')
        parser.add_argument('--learn_sigma', action='store_true', help='是否学习 sigma')
        return parser
    
    def __init__(
        self,
        sit_config='SiT-B/2',
        depth=None, hidden_size=None, patch_size=None, num_heads=None,
        in_channels=32,
        out_channels=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
    ):
        super().__init__()
        
        if sit_config in model_configs:
            config = model_configs[sit_config]
            depth = depth if depth is not None else config['depth']
            hidden_size = hidden_size if hidden_size is not None else config['hidden_size']
            patch_size = patch_size if patch_size is not None else config['patch_size']
            num_heads = num_heads if num_heads is not None else config['num_heads']
        else:
            raise ValueError(f"未知的模型规模: {sit_config}")
        
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if learn_sigma else out_channels
        self.patch_size = to_2tuple(patch_size)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(patch_size=self.patch_size, in_channels=in_channels, embed_dim=hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)

        self.initialize_weights()


    def initialize_weights(self):
        # 有关其他版本的初始化方法，参考sit copy 5.py
        # Initialize all modules recursively
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                # Use Kaiming initialization for convolutional layers
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm weights and biases if elementwise_affine=True
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Initialize embedding layers
                nn.init.normal_(module.weight, std=0.02)

        # Initialize timestep embedding MLP
        for layer in self.t_embedder.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Initialize adaLN_modulation layers in SiT blocks
        for block in self.blocks:
            nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Initialize final_layer's adaLN_modulation
        nn.init.xavier_uniform_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)

        # Initialize final linear layer
        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)


    def unpatchify(self, x, H_patches, W_patches):
        """
        x: (N, T, patch_size[0]*patch_size[1]*C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p1, p2 = self.patch_size
        h = H_patches
        w = W_patches
        # x.shape[0]: batch size, x.shape[1]: number of patches, x.shape[2]: patch_size[0]*patch_size[1]*C

        x = x.reshape(shape=(x.shape[0], h, w, p1, p2, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p1, w * p2))
        return imgs

    def forward(self, x, t):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        """

         # Apply Patch Embedding
        x, (H_patches, W_patches) = self.x_embedder(x)

        # Generate positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.hidden_size,
            grid_size=(H_patches, W_patches)
        )
        pos_embed = torch.from_numpy(pos_embed).float().to(x.device).unsqueeze(0)

        x = x + pos_embed

        t = self.t_embedder(t)  # (N, D)
        c = t
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size[0]*patch_size[1]*out_channels)
        x = self.unpatchify(x, H_patches, W_patches)  # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)

        return x


    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t)
        # Apply classifier-free guidance
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: tuple of the grid height and width
    return:
    pos_embed: [grid_size[0]*grid_size[1], embed_dim]
    """
    grid_h, grid_w = grid_size
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_h, grid_w)
    grid = np.stack(grid, axis=0)  # (2, grid_h, grid_w)
    grid = grid.reshape(2, -1)  # (2, grid_h * grid_w)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # 使用一半的维度编码 grid_h，另一半编码 grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (grid_h * grid_w, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (grid_h * grid_w, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (grid_h * grid_w, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}


model_configs = {
    'SiT-XL/2': {'depth': 28, 'hidden_size': 1152, 'patch_size': 2, 'num_heads': 16},
    'SiT-XL/4': {'depth': 28, 'hidden_size': 1152, 'patch_size': 4, 'num_heads': 16},
    'SiT-XL/8': {'depth': 28, 'hidden_size': 1152, 'patch_size': 8, 'num_heads': 16},
    'SiT-L/2':  {'depth': 24, 'hidden_size': 1024, 'patch_size': 2, 'num_heads': 16},
    'SiT-L/4':  {'depth': 24, 'hidden_size': 1024, 'patch_size': 4, 'num_heads': 16},
    'SiT-L/8':  {'depth': 24, 'hidden_size': 1024, 'patch_size': 8, 'num_heads': 16},
    'SiT-B/2':  {'depth': 12, 'hidden_size': 768,  'patch_size': 2, 'num_heads': 12},
    'SiT-B/4':  {'depth': 12, 'hidden_size': 768,  'patch_size': 4, 'num_heads': 12},
    'SiT-B/8':  {'depth': 12, 'hidden_size': 768,  'patch_size': 8, 'num_heads': 12},
    'SiT-S/2':  {'depth': 12, 'hidden_size': 384,  'patch_size': 2, 'num_heads': 6},
    'SiT-S/4':  {'depth': 12, 'hidden_size': 384,  'patch_size': 4, 'num_heads': 6},
    'SiT-S/8':  {'depth': 12, 'hidden_size': 384,  'patch_size': 8, 'num_heads': 6},
}

