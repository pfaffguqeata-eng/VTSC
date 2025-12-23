# --------------------------------------------------------#
# cited from https://github.com/jmliu206/LIC_TCM
# --------------------------------------------------------#
import random

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from torch import einsum
from collections import OrderedDict
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import time
from einops import rearrange
from einops.layers.torch import Rearrange
from vector_quantize_pytorch import ResidualVQ
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
import huffman
import torch
import torch.nn as nn
from torch import Tensor

class conv1x1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, stride: int = 1, padding: int = 0):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class conv3x3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

def subpel_conv1d(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """1D sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch * r**2, kernel_size=3, padding=1),  # 1D卷积
        nn.ConvTranspose1d(out_ch * r**2, out_ch, kernel_size=r, stride=r)  # 转置卷积进行上采样
    )


class NonNegativeParametrizer(nn.Module):
    def __init__(self, minimum: float = 1e-6):
        super().__init__()
        self.minimum = minimum

    def init(self, value: Tensor) -> Tensor:
        return torch.clamp(value, min=self.minimum)

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=self.minimum)

class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer for 1D data.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _ = x.size()  # 修改为一维数据的维度

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1)  # 一维卷积，因此在最后一个维度中添加1
        norm = F.conv1d(x**2, gamma) + beta.view(1, C, 1)  # 使用1D卷积

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out

class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv1d(in_ch, out_ch, upsample)  # Modify to 1D
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)  # Modify to 1D
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv1d(in_ch, out_ch, upsample)  # Modify to 1D

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)  # Apply sub-pixel convolution
        out = self.leaky_relu(out)
        out = self.conv(out)  # Apply the regular convolution
        out = self.igdn(out)  # Apply GDN
        identity = self.upsample(x)  # Upsample the identity
        out = identity + out# Residual connection
        return out
class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution for 1D data.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)  # 使用一维 GDN

        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = identity + out
        return out
class ResidualBlock1D(nn.Module):
    """Simple residual block with two 3x3 convolutions in 1D.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)  # Use 1D convolution
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)  # Use 1D convolution
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)  # Use 1D convolution
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
class AttentionBlock(nn.Module):
    """Self attention block for 1D data.

    Args:
        N (int): Number of channels.
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit for 1D data."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2, kernel_size=3),  # 修改为1D卷积
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out
class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[..., :-self.causal_padding]



def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
        module,
        buffer_name,
        state_dict_key,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
        module,
        module_name,
        buffer_names,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=1):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,  # 使用相同的公式来计算 padding
    )




class VSE(nn.Module):
    def __init__(self, conv_dim, trans_dim, dim, window_size, drop_path, type='W'):
        """ 1D SwinTransformer and Conv Block """
        super(VSE, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.dim = dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type#修改
        assert self.type in ['W', 'SW']

        # Transformer block, ensure it can handle 1D input
        self.GSE = Block1D(self.trans_dim, self.trans_dim, self.dim, self.drop_path, type=type)

        # Conv1D instead of Conv2D
        self.conv1_1 = nn.Conv1d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv1d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        self.relu = nn.ReLU()
        # Residual block, also need to adjust for 1D
        self.LSE = ResidualBlock1D(self.conv_dim, self.conv_dim)

    def forward(self, x):
        # Split along the channel dimension
        conv_x, trans_x = torch.split(self.relu(self.conv1_1(x)), (self.conv_dim, self.trans_dim), dim=1)#修改


        # Conv block
        conv_x = self.LSE(conv_x) + conv_x

        # Transformer block
        trans_x = Rearrange('b c l -> b l c')(trans_x)  # Change to (batch, sequence_length, channels)
        # print(trans_x.shape)
        trans_x = self.GSE(trans_x)  # Apply transformer
        trans_x = Rearrange('b l c -> b c l')(trans_x)  # Change back to (batch, channels, sequence_length)

        # Concatenate and apply second conv
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))

        # Residual connection
        x = x + res
        return x
class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwags):
        output = self.norm(x)
        output = self.fn(output, **kwags)
        return output
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(dim, hidden_dim)),
            ('ac1', nn.GELU()),
            ('dropout1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_dim, dim)),
            ('dropout2', nn.Dropout(dropout))
        ]))

    def forward(self, x):
        return self.mlp(x)
class Attention(nn.Module):

    def __init__(self, dim, heads=12, dim_head=64, dropout=0., type='W'):
        '''
        dim: dim of input(b,w,c)
        dim_head: dim of q, k, v
        '''
        super(Attention, self).__init__()
        self.type = type
        inner_dim = int(dim_head * heads)
        project_out = not(heads == 1 and dim_head == dim)
        self.lenth = 4
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.qkv = nn.Linear(dim, inner_dim * 3)

        self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.relative_position_params = nn.Parameter(torch.zeros((2 * self.lenth - 1), (2 * self.lenth -1), self.heads))
        #
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.transpose(2,1).transpose(0,1))

    def forward(self, x):


        b, n, _, h = *x.shape, self.heads
        lenth = x.size(1)
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(lenth // 2)), dims=1)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dim1 = dots.size(1)
        # print(x.shape, dots.shape, self.relative_embedding(lenth).shape, lenth)
        dots = dots + rearrange(self.relative_embedding(lenth), 'h i j -> 1 h i j')



        if self.type != "W":

            attn_mask = self.generate_mask(dim1//2, 2, self.lenth, self.lenth//2)[:, :, :lenth, :lenth]
            # print(attn_mask.shape)
            dots = dots.masked_fill_(attn_mask, float("-inf"))

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)
    def relative_embedding(self, lenth):
        cord = torch.tensor(np.array([[i, j] for i in range(self.lenth) for j in range(self.lenth)]))
        relation = cord[:lenth, None, :lenth] - cord[None, :lenth, :lenth] + self.lenth - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, type, dropout=0.):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, type=type)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x






class Block1D(nn.Module):
    def __init__(self, input_dim, output_dim, dim, drop_path, depth=2, heads=8, mlp_dim=256, type='W'):
        """ SwinTransformer Block for 1D inputs """
        super(Block1D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ln1 = nn.LayerNorm(input_dim)
        dim_head = dim/heads
        self.msa = Transformer(dim, depth, heads, dim_head, mlp_dim, type=type)  # Use 1D WMSA
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        # print("transformer:", x)
        x = x + self.drop_path(self.mlp(self.ln2(x)))


        return x
# class GRUNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(GRUNetwork, self).__init__()
#
#         # GRU layer
#         self.gru = nn.GRU(input_size=input_size,
#                           hidden_size=hidden_size,
#                           num_layers=num_layers,
#                           batch_first=True)
#
#         # Fully connected layer to produce output
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # Forward pass through the GRU layer
#         # x: (batch_size, seq_len, input_size)
#         out, _ = self.gru(x)
#
#         # We only want the output of the last time step (seq_len)
#         # out: (batch_size, seq_len, hidden_size)
#         out = out[:, -1, :]
#
#         # Pass the last time step's output through the fully connected layer
#         out = self.fc(out)
#
#         return out




class HaarWaveletTransform(nn.Module):
    def __init__(self, input_dim, drop_path=0.):
        super(HaarWaveletTransform, self).__init__()
        # 使用 nn.Conv1d 替代功能性的 conv1d
        self.input_dim = input_dim
        self.upsample1 = conv(input_dim, input_dim, stride=1, kernel_size=3)
        self.upsample2 = conv(input_dim, input_dim, stride=1, kernel_size=3)
        self.downsample1 = conv(input_dim, input_dim, stride=1, kernel_size=3)
        self.downsample2 = conv(input_dim, input_dim, stride=1, kernel_size=3)

        self.filter = Block1D(input_dim, input_dim, input_dim, drop_path)
        self.reconstructed = Block1D(input_dim, input_dim, input_dim, drop_path)
    def dwt(self, x):
        # 使用定义好的卷积层
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = Rearrange('b c l -> b l c')(x)  # Change to (batch, sequence_length, channels)
        x = self.filter(x)
        x = Rearrange('b l c -> b c l')(x)  # Change back to (batch, channels, sequence_length)
        return x


    def idwt(self, x):
        """
        离散小波逆变换（IDWT） - 重构信号。
        """
        # 恢复信号（逆变换）
        # 使用反卷积来进行逆变换，模拟 DWT 逆过程
        x = Rearrange('b l c -> b c l')(x)  # Change back to (batch, channels, sequence_length)
        x = self.reconstructed(x)
        x = Rearrange('b c l -> b l c')(x)  # Change to (batch, sequence_length, channels)
        x = self.downsample1(x)
        x = self.downsample2(x)

        # 返回低通和高通部分的合成信号
        return x
class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, dim, window_size, drop_path, inter_dim=298, type='W') -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, dim, window_size, drop_path, type=type)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, dim, window_size, drop_path, type=type)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)

        return out


class SwinBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dim: int, window_size: int, drop_path: float, type) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.block_1 = Block1D(input_dim, output_dim, dim, drop_path, type=type)
        self.block_2 = Block1D(input_dim, output_dim, dim, drop_path, type=type)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, channels, length]

        # Ensure the input is compatible with window size

        trans_x = Rearrange('b c l -> b l c')(x)  # Change to (batch, sequence_length, channels)
        # print(trans_x.shape)

        # Apply the first block
        trans_x = self.block_1(trans_x)

        # Apply the second block
        trans_x = self.block_2(trans_x)
        trans_x = Rearrange('b l c -> b c l')(trans_x)  # Change back to (batch, channels, sequence_length)
        return trans_x
class superior_block(nn.Module):
    def __init__(self, input_dim=6, output_dim=32):
        super(superior_block, self).__init__()
        self.head_size = 8
        # Frame-level layers
        self.tdnn1 = nn.Conv1d(input_dim, 512, kernel_size=3)
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3)
        self.tdnn3 = nn.Conv1d(512, output_dim, kernel_size=3)


        self.tdnn4 = nn.Conv1d(output_dim, 512, kernel_size=3)
        self.tdnn5 = nn.Conv1d(512, output_dim, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(output_dim*2, output_dim)  # 1500*2

    def forward(self, x, z):
        x = self.relu(self.tdnn1(x))
        x = self.relu(self.tdnn2(x))
        x = self.relu(self.tdnn3(x))

        z = self.relu(self.tdnn4(z))
        z = self.relu(self.tdnn5(z))
        x = torch.cat([x,z], dim=2)
        x = self.g_a(x)

        # Statistics pooling


        return x
# class WaveletTransform1D(torch.nn.Module):
#     def __init__(self, wavelet='haar'):
#         super(WaveletTransform1D, self).__init__()
#         self.wavelet = wavelet
#
#     def forward(self, x):
#         if self.wavelet == 'haar':
#             return self.haar_wavelet_transform(x)
#         else:
#             raise NotImplementedError("Only 'haar' wavelet is implemented.")
#
#     def haar_wavelet_transform(self, x):
#         # 这里实现一个简单的Haar小波变换
#         # 假设输入x是一个一维张量，长度为2的整数倍
#
#         # 执行小波分解：低频和高频部分
#         low_pass = (x[::2] + x[1::2]) / torch.sqrt(torch.tensor(2.0))
#         high_pass = (x[::2] - x[1::2]) / torch.sqrt(torch.tensor(2.0))
#
#         # 返回低频和高频部分
#         return low_pass, high_pass
class WaveletTransform1D(torch.nn.Module):
    def __init__(self, wavelet='haar'):
        # 简单选择小波类型，默认使用 Haar 小波
        super(WaveletTransform1D, self).__init__()

        self.wavelet = wavelet

    def haar_wavelet(self, signal):
        # Haar小波变换的实现
        assert signal.size(1) % 2 == 0, "Length of signal must be even"

        # 将信号拆分为低频和高频
        low_pass = (signal[:, ::2] + signal[:, 1::2]) / 2  # 偶数索引与奇数索引相加取平均
        high_pass = (signal[:, ::2] - signal[:, 1::2]) / 2  # 偶数索引减去奇数索引

        return low_pass, high_pass

    def forward(self, signal):
        if self.wavelet == 'haar':
            # signal的形状为 [batch_size, signal_length]
            low_pass, high_pass = self.haar_wavelet(signal)
            return low_pass, high_pass
        else:
            raise NotImplementedError(f"Wavelet '{self.wavelet}' is not supported.")
class VTSC(CompressionModel):

    def __init__(self):
        # super().__init__(entropy_bottleneck_channels=N)
        super().__init__()


        self.head_size = 8
        dim = 1
        self.dim = dim
        self.fram_num = 9


        self.max_support_slices = 9
        depth = 2
        N = 256
        M = 64
        M_semantic = 512-32*9
        self.z_lenth = M//4
        self.z_dim = M//4

        self.semantic_block = nn.Sequential(
            *[SWAtten(M * 9, M_semantic, 128, self.head_size, 0, inter_dim=128, type='SW')] +
             [SWAtten(M_semantic, M_semantic, 128, self.head_size, 0, inter_dim=128, type='W')]
        )
        self.WaveletTransform = WaveletTransform1D()
        # self.wav2vec2_model = nn.ModuleList(nn.Sequential(torchaudio.pipelines.WAV2VEC2_BASE.get_model(), superior_block(input_dim=4, output_dim=self.z_dim)) for i in range(self.max_support_slices))
        # self.wav2vec2_model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        self.superior_block = superior_block(input_dim=M, output_dim=M)#input_dim = wav2vec2模型的输出dim

        self.g_a = nn.Sequential(
                *[nn.Conv1d(dim, N//8, kernel_size=10, stride=5)] +
                 [ResidualBlockWithStride(N // 8, N // 2, stride=2)] +
                # [ResidualBlockWithStride(N//4, N//2, stride=2)] +
                [ResidualBlockWithStride(N//2, N, stride=2)] +
                [VSE(N // 2, N // 2, N // 2, self.head_size, 0., 'W' if not i % 2 else 'SW') for i in
                 range(depth)] +
                [ResidualBlockWithStride(N, N // 2, stride=2)] +
                [VSE(N // 4, N // 4, N // 4, self.head_size, 0., 'W' if not i % 2 else 'SW') for i in
                 range(depth)] +
                [ResidualBlockWithStride(N // 2, N // 4, stride=1)] +
                # [VSE(N // 8, N // 8, N // 8, self.head_size, 0., 'W' if not i % 2 else 'SW') for i in
                #  range(depth)] +
                [ResidualBlockWithStride(N // 4, M, stride=1)]


        )



        self.semantic_fc1 = nn.Linear(M*8, 1024)
        self.semantic_fc2 = nn.Linear(1024, 768)

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(M, M//2, 1)] + \
             [ResidualBlockWithStride(M//2, M//4, 1)]
        )


        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample(M//4, M//2, 1)] + \
             [ResidualBlockUpsample(M//2, M, 1)]
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout_t = nn.Dropout(p=0.1)
        self.quantizer = ResidualVQ(
            num_quantizers=4, dim=4, codebook_size=4096,
            kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
        )

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def add_gaussian_noise(self, signal, snr=0.1):

        signal_power = torch.mean(signal ** 2)

        # 计算噪声的功率，根据信噪比公式
        snr_linear = 10 ** (snr / 10)
        noise_power = signal_power / snr_linear

        # 计算噪声的标准差
        noise_std = torch.sqrt(noise_power)

        # 生成高斯噪声，均值为0，标准差为noise_std
        noise = torch.normal(mean=0.0, std=noise_std, size=signal.size()).to(device)

        # 添加噪声到信号
        noisy_signal = signal + noise
        return noisy_signal

    # 示例：向一维信号添加高斯噪声


    def forward(self, x):
        # x_transform, _ = self.wav2vec2_model(x)
        # snr = random.uniform(0, 0.1)
        # x = self.add_gaussian_noise(x, snr=snr)
        batch_size, signal_length = x.shape
        frame_length = 240
        frame_step = 180
        # 使用 unfold 进行滑动窗口分帧
        num_frames = (signal_length - frame_length) // frame_step + 1
        semantic_save = []
        y_slices = []
        for slice_index in range(num_frames):
            # 每个 slice 的处理可以通过矩阵操作一次性做
            # x_wav = x_transform[:, : (slice_index + 1) * self.dim, :]
            start_index = random.randint(0,10)
            if slice_index < (num_frames - 1) and slice_index != 0:
                x_slice = x[:, (slice_index * frame_step) + start_index: (slice_index * frame_step) + frame_length + start_index]  # 获取当前 slice 的数据 (batch_size, feature_size)
            elif slice_index == 0:
                x_slice = x[:, :frame_length]
            else:
                x_slice = x[:, -frame_length: ]  # 获取当前 slice 的数据 (batch_size, feature_size)
            # Wavelet_low, Wavelet_high = self.WaveletTransform(x_slice)
            # x_slice = torch.cat([x_slice, Wavelet_low, Wavelet_high], dim=1)
            x_slice = x_slice.unsqueeze(1)
            if slice_index == 0:
                codingstart_time = time.time()


            x_slice = torch.cat([x_slice], dim=2)
            y_slice = self.g_a(x_slice)
            quantized, indices, _ = self.quantizer(y_slice[:, :, :4])

            if slice_index == 0:
                codingend_time = time.time()

            y_slices.append(quantized)


        y = torch.cat(y_slices, dim=1)
        # semantic_surport = self.semantic_block(y)
        #
        # semantic_final = torch.cat([y, semantic_surport], dim=1)

        y = y.permute(0, 2, 1)

        # 应用全连接层
        semantic_final = self.semantic_fc1(y)
        output = self.semantic_fc2(semantic_final)

        # 调整回原始形状
        # if not x_hat.size(1)*x_hat.size(2)==1600:
        #     print(x_hat.size(1)*x_hat.size(2))
        # x_hat = x_hat.reshape(x.size(0), -1)[:, :1600]
        y_likelihoods = 0
        z_likelihoods = 0
        bitrate=indices.size(1)*indices.size(2)*num_frames*10
        codingtime = codingend_time - codingstart_time
        decodingtime = 0
        # print(codingtime, decodingtime)
        self.forward_output = {
            "output": output,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "bitrate": bitrate,
            "time": {"codingtime": codingtime, "decodingtime": decodingtime},
            "indices": indices
        }
        return self.forward_output
    # def frame_signal(self, signal, frame_length, frame_step):
    #     """
    #     对一维信号进行分帧处理
    #     Args:
    #         signal (torch.Tensor): 输入信号，形状为 [batch_size, signal_length]
    #         frame_length (int): 每帧的长度
    #         frame_step (int): 帧移（相邻帧之间的间隔）
    #     Returns:
    #         torch.Tensor: 分帧后的信号，形状为 [batch_size, num_frames, frame_length]
    #     """
    #
    #     return frames, num_frames  # 返回 [batch_size, num_frames, frame_length]
    def compress(self, x):
        batch_size, signal_length = x.shape
        frame_length = 240
        frame_step = 180
        # 使用 unfold 进行滑动窗口分帧
        num_frames = (signal_length - frame_length) // frame_step + 1
        semantic_save = []
        y_slices = []
        bitrate = 0

        for slice_index in range(num_frames):
            # 每个 slice 的处理可以通过矩阵操作一次性做
            # x_wav = x_transform[:, : (slice_index + 1) * self.dim, :]
            if slice_index < (self.fram_num - 1) and slice_index != 0:
                x_slice = x[:, (slice_index * frame_step): (slice_index * frame_step) + frame_length]  # 获取当前 slice 的数据 (batch_size, feature_size)
            elif slice_index == 0:
                x_slice = x[:, :frame_length]
            else:
                x_slice = x[:, -frame_length: ]  # 获取当前 slice 的数据 (batch_size, feature_size)
            # Wavelet_low, Wavelet_high = self.WaveletTransform(x_slice)
            # x_slice = torch.cat([x_slice, Wavelet_low, Wavelet_high], dim=1)
            x_slice = x_slice.unsqueeze(1)
            if slice_index == 0:
                codingstart_time = time.time()

            x_slice = torch.cat([x_slice], dim=2)
            y_slice = self.g_a(x_slice)
            # y_shape = y_slice.size
            # z = self.h_a(y_slice)
            # feat = torch.cat([y_slice, z], dim=1)
            # y_slice = y_slice.reshape(y_shape(0), -1)
            quantized, indices, _ = self.quantizer(y_slice[:, :, :4])
            indices_list = indices.flatten().tolist()  # 转换为列表

            # 统计索引频率
            freq_dict = {symbol: indices_list.count(symbol) for symbol in set(indices_list)}

            # 生成哈夫曼编码表
            codebook = huffman.codebook(freq_dict.items())

            # 进行哈夫曼编码
            encoded_indices = "".join(codebook[idx] for idx in indices_list)
            bitrate += len(encoded_indices)
            # y_slice = quantized[:, :y_shape(1), :]
            # z = quantized[:, y_shape(1):, :]
            # y_surport = self.h_mean_s(z)
            # y_slice = y_slice - y_surport
            # quantized = quantized.reshape(y_shape(0), y_shape(1), y_shape(2))
            if slice_index == 0:
                codingend_time = time.time()

            y_slices.append(quantized)

        y = torch.cat(y_slices, dim=1)
        # semantic_surport = self.semantic_block(y)
        #
        # semantic_final = torch.cat([y, semantic_surport], dim=1)

        y = y.permute(0, 2, 1)

        # 应用全连接层
        semantic_final = self.semantic_fc1(y)
        output = self.semantic_fc2(semantic_final)

        # 调整回原始形状
        # if not x_hat.size(1)*x_hat.size(2)==1600:
        #     print(x_hat.size(1)*x_hat.size(2))
        # x_hat = x_hat.reshape(x.size(0), -1)[:, :1600]
        y_likelihoods = 0
        z_likelihoods = 0
        codingtime = codingend_time - codingstart_time
        decodingtime = 0
        # print(codingtime, decodingtime)
        self.forward_output = {
            "output": output,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "bitrate": bitrate,
            "time": {"codingtime": codingtime, "decodingtime": decodingtime},
        }
        return self.forward_output



