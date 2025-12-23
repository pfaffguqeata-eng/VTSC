import torch
import torch.nn.functional as F
from torchvision import transforms
import math
import torch.nn as nn
import argparse
import logging
import matplotlib.pyplot as plt
from pytorch_msssim import MS_SSIM
from torchvision import transforms
from PIL import Image
import os
import datetime
def parse_args():
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-e",
        "--epochs",
        default=202,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        default='ccl',  # mse / ms-ssim
        type=str,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default=r'',
        type=str,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        default='./data/',
        type=str,
        help="Number of epochs (default: %(default)s)",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")  # 8
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )

    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    args = parser.parse_args()
    return args



def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    # print(h,w,new_h,new_w)
    x_padded = F.pad(x,(padding_left, padding_right, padding_top, padding_bottom),mode="constant",value=0)
    # print(x_padded.shape)


    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(x,(-padding[0], -padding[1], -padding[2], -padding[3]))




def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


# 定义综合损失函数

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')