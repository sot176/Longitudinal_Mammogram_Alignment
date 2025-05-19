import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal

########################################################
# Networks
########################################################


class MammoRegNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        enc_channels: int = 8,
        dec_channels: int = 16,
        use_checkpoint: bool = True,
    ):
        super().__init__()

        self.Encoder = Conv_encoder(
            in_channels=in_channels,
            channel_num=enc_channels,
            use_checkpoint=use_checkpoint,
        )
        self.Decoder = Trans_decoder(
            in_channels=enc_channels,
            channel_num=dec_channels,
            use_checkpoint=use_checkpoint,
        )

        self.SpatialTransformer = SpatialTransformer_block(mode="bilinear")
        self.AffineTransformer = AffineTransformer_block(mode="bilinear")

    def forward(self, fixed, moving):
        x_fix = self.Encoder(fixed)
        x_mov = self.Encoder(moving)
        flow, affine_para = self.Decoder(x_fix, x_mov)
        warped = self.SpatialTransformer(moving, flow)
        affined = self.AffineTransformer(moving, affine_para)

        return warped, flow, affined, affine_para


########################################################
# Encoder/Decoder
########################################################


class Conv_encoder(nn.Module):
    def __init__(
        self, in_channels: int, channel_num: int, use_checkpoint: bool = False
    ):
        super().__init__()

        self.conv_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.conv_2 = Conv_block(channel_num, channel_num * 2, use_checkpoint)
        self.conv_3 = Conv_block(channel_num * 2, channel_num * 4, use_checkpoint)
        self.conv_4 = Conv_block(channel_num * 4, channel_num * 8, use_checkpoint)
        self.conv_5 = Conv_block(channel_num * 8, channel_num * 16, use_checkpoint)
        self.downsample = nn.AvgPool2d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.conv_1(x_in)

        x = self.downsample(x_1)
        x_2 = self.conv_2(x)

        x = self.downsample(x_2)
        x_3 = self.conv_3(x)

        x = self.downsample(x_3)
        x_4 = self.conv_4(x)

        x = self.downsample(x_4)
        x_5 = self.conv_5(x)

        return [x_1, x_2, x_3, x_4, x_5]


class Trans_decoder(nn.Module):
    def __init__(
        self, in_channels: int, channel_num: int, use_checkpoint: bool = False
    ):
        super().__init__()

        self.conv_1 = Conv_block(
            in_channels * 2 + channel_num, channel_num, use_checkpoint
        )
        self.trans_2 = SwinTrans_stage_block(
            embed_dim=channel_num * 2,
            num_layers=4,
            num_heads=channel_num // 8,
            window_size=[5, 5],
            use_checkpoint=use_checkpoint,
        )
        self.trans_3 = SwinTrans_stage_block(
            embed_dim=channel_num * 4,
            num_layers=4,
            num_heads=channel_num // 4,
            window_size=[5, 5],
            use_checkpoint=use_checkpoint,
        )
        self.trans_4 = SwinTrans_stage_block(
            embed_dim=channel_num * 8,
            num_layers=4,
            num_heads=channel_num // 2,
            window_size=[5, 5],
            use_checkpoint=use_checkpoint,
        )
        self.trans_5 = SwinTrans_stage_block(
            embed_dim=channel_num * 16,
            num_layers=4,
            num_heads=channel_num,
            window_size=[5, 5],
            use_checkpoint=use_checkpoint,
        )

        self.backdim_2 = nn.Conv2d(
            in_channels * 4 + channel_num * 2,
            channel_num * 2,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.backdim_3 = nn.Conv2d(
            in_channels * 8 + channel_num * 4,
            channel_num * 4,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.backdim_4 = nn.Conv2d(
            in_channels * 16 + channel_num * 8,
            channel_num * 8,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.backdim_5 = nn.Conv2d(
            in_channels * 32, channel_num * 16, kernel_size=1, stride=1, padding="same"
        )

        self.upsample_1 = PatchExpanding_block(embed_dim=channel_num * 2)
        self.upsample_2 = PatchExpanding_block(embed_dim=channel_num * 4)
        self.upsample_3 = PatchExpanding_block(embed_dim=channel_num * 8)
        self.upsample_4 = PatchExpanding_block(embed_dim=channel_num * 16)

        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint)
        self.reghead_3 = DeformHead_block(channel_num * 4, use_checkpoint)
        self.reghead_4 = DeformHead_block(channel_num * 8, use_checkpoint)
        self.reghead_5 = AffineHead_block(channel_num * 16)

        self.ResizeTransformer = ResizeTransformer_block(
            resize_factor=2, mode="bilinear"
        )
        self.SpatialTransformer = SpatialTransformer_block(mode="bilinear")

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = x_mov

        # Step 1
        x = torch.cat([x_fix_5, x_mov_5], dim=1)
        x = self.backdim_5(x)
        x_5 = self.trans_5(x)
        flow_5, affine_para = self.reghead_5(x_5)

        # Step 2
        flow_5_up = self.ResizeTransformer(flow_5)
        x_mov_4 = self.SpatialTransformer(x_mov_4, flow_5_up)

        x = self.upsample_4(x_5)

        x = torch.cat([x_fix_4, x, x_mov_4], dim=1)
        x = self.backdim_4(x)
        x_4 = self.trans_4(x)

        x = self.reghead_4(x_4)
        flow_4 = x + flow_5_up

        # Step 3
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)

        x = self.upsample_3(x_4)
        x = torch.cat([x_fix_3, x, x_mov_3], dim=1)
        x = self.backdim_3(x)
        x_3 = self.trans_3(x)

        x = self.reghead_3(x_3)
        flow_3 = x + flow_4_up

        # Step 4
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3)
        x = torch.cat([x_fix_2, x, x_mov_2], dim=1)
        x = self.backdim_2(x)
        x_2 = self.trans_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 5
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1, affine_para


########################################################
# Blocks
########################################################


class AffineTransformer_block(nn.Module):
    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, src, affine_para):
        """
        src: input image tensor with shape b,c,h,w (c = 1)
        """
        # generate affine matrix
        b = src.shape[0]
        self.id = torch.zeros((b, 2, 3)).to(src.device)

        # identity transform
        self.id[:, 0, 0] = 1
        self.id[:, 1, 1] = 1
        affine_matrix = (
            affine_para.reshape(b, 2, 3) + self.id
        )  # Reshape the parameters into a bx2x3 affine  matrix
        affine_grid = F.affine_grid(
            affine_matrix, src.shape, align_corners=True
        )  # shape b,h,w,2

        return F.grid_sample(src, affine_grid, align_corners=True, mode=self.mode)


class SpatialTransformer_block(nn.Module):
    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        """
        src: input image tensor with shape b,c,h,w (c = 1)
        flow: deformation field with shape b,c,h,w (c = 2) where the two channels represent (dx, dy) offsets.
        """
        shape = flow.shape[2:]  # h,w

        # create sampling grid
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(src.device)

        # new locations
        new_locs = grid + flow.to(src.device)

        # normlaize grid values to [-1,1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)  # shape b,h,w,c
        new_locs = new_locs[..., [1, 0]]  # shape b,h,w,2

        return F.grid_sample(
            src, new_locs, align_corners=True, mode=self.mode
        )  # applying the spatial transformations


class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor, mode="bilinear"):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )

        return x


class Conv_block(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding="same"
        )
        self.norm_1 = nn.InstanceNorm2d(out_channels)

        self.Conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding="same"
        )
        self.norm_2 = nn.InstanceNorm2d(out_channels)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def Conv_forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)

        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out

    def forward(self, x_in):
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)

        return x_out


class DeformHead_block(nn.Module):
    def __init__(self, in_channels: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.reg_head = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1, padding="same"
        )
        self.reg_head.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.reg_head.weight.shape)
        )
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x_in):
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.reg_head, x_in)
        else:
            x_out = self.reg_head(x_in)

        return x_out


class AffineHead_block(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.affine_head_1 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self.affine_head_2 = nn.Linear(in_channels // 2, 6, bias=False)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, x_in):
        x = torch.mean(x_in, dim=(2, 3))
        x = self.affine_head_1(x)
        x = self.ReLU(x)
        x = self.affine_head_2(x)
        affine_para = self.Tanh(x)

        b = x_in.shape[0]

        self.id = torch.zeros((b, 2, 3)).to(x_in.device)
        self.id[:, 0, 0] = 1
        self.id[:, 1, 1] = 1

        affine_matrix = (
            affine_para.reshape(b, 2, 3) + self.id
        )  # Reshape the parameters into a 1x2x3 affine  matrix

        affine_grid = F.affine_grid(
            affine_matrix, x_in.shape, align_corners=True
        )  # shape b,h,w,2

        shape = x_in.shape[2:]  # h,w
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(x_in.device)

        affine_grid = affine_grid[..., [1, 0]]
        affine_grid = affine_grid.permute(0, 3, 1, 2)
        for i in range(len(shape)):
            affine_grid[:, i, ...] = (affine_grid[:, i, ...] / 2.0 + 0.5) * (
                shape[i] - 1
            )

        flow = affine_grid - grid

        return flow, affine_para


class PatchExpanding_block(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(
            embed_dim, embed_dim // 2, kernel_size=2, stride=2
        )
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x_in):
        x = self.up_conv(x_in)
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x_out = einops.rearrange(x, "b h w c -> b c h w")

        return x_out


class SwinTrans_stage_block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        window_size: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTrans_Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=self.window_size,
                shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                use_checkpoint=use_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, x_in):
        b, c, h, w = x_in.shape
        window_size, shift_size = get_window_size(
            (h, w), self.window_size, self.shift_size
        )
        hp = int(np.ceil(h / window_size[0])) * window_size[0]
        wp = int(np.ceil(w / window_size[1])) * window_size[1]
        attn_mask = compute_mask([hp, wp], window_size, shift_size, x_in.device)

        x = einops.rearrange(x_in, "b c h w -> b h w c")
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)
        x_out = einops.rearrange(x, "b h w c -> b c h w")

        return x_out


class SwinTrans_Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: list,
        shift_size: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MSA_block(
            embed_dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_block(
            hidden_size=embed_dim, mlp_dim=int(embed_dim * mlp_ratio), dropout_rate=drop
        )

    def forward_part1(self, x_in, mask_matrix):
        x = self.norm1(x_in)

        b, h, w, c = x.shape
        window_size, shift_size = get_window_size(
            (h, w), self.window_size, self.shift_size
        )
        pad_l = pad_t = 0
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape
        dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)

        # W-MSA7SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)  # b,h,w,c

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x_out = torch.roll(
                shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2)
            )
        else:
            x_out = shifted_x

        if pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :h, :w, :].contiguous()

        return x_out

    def forward_part2(self, x_in):
        x = self.norm2(x_in)
        x_out = self.mlp(x)
        return x_out

    def forward(self, x_in, mask_matrix=None):
        if self.use_checkpoint and x_in.requires_grad:
            x = x_in + checkpoint.checkpoint(self.forward_part1, x_in, mask_matrix)
        else:
            x = x_in + self.forward_part1(x_in, mask_matrix)

        if self.use_checkpoint and x.requires_grad:
            x_out = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x_out = x + self.forward_part2(x)

        return x_out


class MSA_block(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: list,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads
            )
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_in, mask=None):
        """
        x_in: input features with shape of b,n,c
        mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        b, n, c = x_in.shape
        qkv = (
            self.qkv(x_in)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x_out = self.proj_drop(x)

        return x_out


class MLP_block(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        self.GELU = nn.GELU()

    def forward(self, x_in):
        x = self.linear1(x_in)
        x = self.GELU(x)
        x = self.drop1(x)

        x = self.linear2(x)
        x_out = self.drop2(x)

        return x_out


########################################################
# Functions
########################################################


def compute_mask(dims, window_size, shift_size, device):
    """
    Calculate attention mask for SW-MSA
    """
    cnt = 0
    h, w = dims
    img_mask = torch.zeros((1, h, w, 1), device=device)
    for h in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for w in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def window_partition(x_in, window_size):
    b, h, w, c = x_in.shape
    x = x_in.view(
        b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0] * window_size[1], c)
    )

    return windows


def window_reverse(windows, window_size, dims):
    b, h, w = dims
    x = windows.view(
        b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor
