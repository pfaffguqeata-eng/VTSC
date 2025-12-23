import random
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch
import torchaudio
import math
import torch.nn as nn
import librosa


def read_txt(txt_path, target_length=1600):
    # 读取文件内容
    with open(txt_path, 'r') as source_data:
        lines = [float(line.strip()) for line in source_data]

    # 转换为 NumPy 数组
    ret = np.array(lines, dtype=np.float32)
    current_length = len(ret)

    if current_length < target_length:
        # 计算填充长度
        pad_length = target_length - current_length
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        # 填充数组
        ret = np.pad(ret, (pad_left, pad_right), mode='mean')
    elif current_length > target_length:
        # 如果长度超过目标长度，裁剪数组
        start = (current_length - target_length) // 2
        ret = ret[start:start + target_length]

    return ret

    return ret

def label_fun(label_path):
    # Y_list = []
    # for labelname in label_path:
    label1 = label_path.split('_')[0]
    label2 = label1.split('\\')[2]
    label3 = label2[:-1] if label2[-1].isdigit() else label2
    # Y_list.append(label3)

    return label3

def read_npy_file(filename):
    data = np.load(filename)
    return data.astype(np.float32)


def crop_into_four_patches(image):
    img_width, img_height = image.size
    # 计算每个patch的大小
    patch_width = img_width // 2
    patch_height = img_height // 2

    patches = []

    for i in range(2):  # 2x2 grid
        for j in range(2):
            left = i * patch_width
            top = j * patch_height
            right = left + patch_width
            bottom = top + patch_height

            # 裁剪图像
            patch = image.crop((left, top, right, bottom))
            patches.append(patch)

    return patches

# def lowpass_filter(data, cutoff, fs, order=5):
#     nyquist = 0.5 * fs  # 奈奎斯特频率
#     normal_cutoff = cutoff / nyquist  # 归一化截止频率
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)  # 设计低通滤波器
#
#
#     b = b.astype(np.float32)  # 转换为float32
#     a = a.astype(np.float32)  # 转换为float32
#     data = data.astype(np.float32)  # 确保数据类型为float32
#     return lfilter(b, a, data)
class Datasets(Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.source_files_paths = sorted(glob(os.path.join(self.data_dir, 'TrainDataFile', 'source_tapping', "*.txt"))+glob(os.path.join(self.data_dir, 'TestDataFile', 'source_tapping', "*.txt")))
        self.target_files_paths = sorted(glob(os.path.join(self.data_dir, 'TrainDataFile', 'Target', "*.jpg"))+glob(os.path.join(self.data_dir, 'TestDataFile', 'Target', "*.jpg")))
        self.label_files_paths = sorted(glob(os.path.join('./data/LMT108/LMT_108_SurfaceMaterials_Database', "*.jpg")))
        self.source_files_paths_train, self.source_files_paths_test, self.target_files_paths_train, self.target_files_paths_test = train_test_split(self.source_files_paths, self.target_files_paths, test_size=0.3, random_state=42, shuffle=True)

        unique_jpg_files = []
        seen = set()
        for file in self.label_files_paths:
            file_name = os.path.splitext(os.path.basename(file))[0].split('_')[0][:2]
            if file_name not in seen:
                seen.add(file_name)
                unique_jpg_files.append(file_name)

        self.label_mapping = {file: idx for idx, file in enumerate(unique_jpg_files)}
        assert len(self.source_files_paths) == len(self.target_files_paths), \
            f'{len(self.source_files_paths)} != {len(self.target_files_paths)}'

        # self.transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Resize((256, 256))
        #      ])
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
        ])
    def __getitem__(self, item):


        # Apply downsampling with a shift:
        if self.mode == "train":
            source_file = self.source_files_paths_train[item]
            target_file = self.target_files_paths_train[item]
            part_item = 1
            # 检查 source_file 是否包含 ref_source_base，target_file 是否包含 ref_target_base
            filename = os.path.basename(target_file)
            source = read_txt(source_file)

            # source = lowpass_filter(source, cutoff, fs)
            source, cepstrum, fft_value = self.process_signals(source, part_item)
            cepstrum = self.normalize_min_max(cepstrum)
            cepstrum = torch.cat((cepstrum, torch.tensor(fft_value)), dim=1)
            target = Image.open(target_file).convert("RGB")
            # 将patches转换为Tensor或其他需要的格式
            target = self.transform(target)
            label_name = self._extract_class_from_filename(filename)[:2]
            label_num = self.label_mapping[label_name]
            data = {
                "source": source,
                "target": target,
                "cepstrum": cepstrum,
                "label": torch.tensor(label_num),
                "map": self.label_mapping,
            }
        else:
            source_file = self.source_files_paths_test[item]
            target_file = self.target_files_paths_test[item]
            # 检查 source_file 是否包含 ref_source_base，target_file 是否包含 ref_target_base
            filename = os.path.basename(target_file)
            source = read_txt(source_file)


            source, cepstrum, fft_value = self.process_signals(source)
            cepstrum = self.normalize_min_max(cepstrum)
            cepstrum = torch.cat((cepstrum, torch.tensor(fft_value)), dim=1)
            target = Image.open(target_file).convert("RGB")
            # 将patches转换为Tensor或其他需要的格式
            target = self.transform(target)
            label_name = self._extract_class_from_filename(filename)[:2]
            label_num = self.label_mapping[label_name]
            data = {
                "source": source,
                "target": target,
                "cepstrum": cepstrum,
                "label": torch.tensor(label_num),
                "map": self.label_mapping,
            }
        return data

        # return self.transform(source), self.transform(target)

    def __len__(self):
        if self.mode == "train":
            return len(self.source_files_paths_train)
        else:
            return len(self.source_files_paths_test)

    def _extract_class_from_filename(self, filename):
        # 提取文件名中类别的部分，例如 "G1EpoxyRasterPlate_train1.jpg" -> "G1EpoxyRasterPlate"
        return filename.split('_')[0]

    def normalize_min_max(self, cepstrum):
        # 确保 cepstrum 是 PyTorch 张量
        if not isinstance(cepstrum, torch.Tensor):
            cepstrum = torch.tensor(cepstrum, dtype=torch.float32)


        # 确保在正确的维度上进行最小值计算
        if cepstrum.dim() == 1:
            min_val = cepstrum.min()  # 一维情况
            max_val = cepstrum.max()
            normalized = (cepstrum - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
        elif cepstrum.dim() == 2:
            min_val = cepstrum.min(dim=1, keepdim=True)[0]  # 二维情况
            max_val = cepstrum.max(dim=1, keepdim=True)[0]
            normalized = (cepstrum - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
        else:
            raise ValueError("Unsupported tensor dimension")

        return normalized
    def time_shift(self, signal, shift_max=48000):
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(signal, shift)
    def random_time_reverse(self, signal, flip_prob=0.5):
        """
        随机翻转信号
        :param signal: 原始信号
        :param flip_prob: 信号翻转的概率，默认0.5表示50%的概率
        :return: 可能被翻转的信号
        """
        if np.random.rand() < flip_prob:  # 如果随机数小于概率值，则翻转信号
            return signal[::-1]
        else:
            return signal  # 否则返回原始信号

    def process_signals(self, signal, sr=48000, n_fft=1024, n_filters=64, n_coeffs=64, hop_length=512, win_length=1024):
        """
        计算音频信号的 LFCC（线性频率倒谱系数）

        参数:
        signal: 输入的一维音频信号
        sr: 采样率（默认32kHz）
        n_fft: FFT 点数（默认1000）
        n_filters: 滤波器数量（默认64）
        n_coeffs: 输出的倒谱系数个数（默认13）
        hop_length: 帧移（默认500）
        win_length: 窗口长度（默认1024）

        返回:
        LFCC 倒谱系数 (frames, n_coeffs)
        """
        # 定义 LFCC 转换器，使用传入的参数而不是硬编码
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

        signal = self.pre_emphasis(signal)
        signal = torch.tensor(signal)
        signal = signal.unsqueeze(0)
        lfcc_transform = torchaudio.transforms.LFCC(
        sample_rate=sr,
        n_lfcc=n_coeffs,
        n_filter=n_filters,
        speckwargs={"n_fft": n_fft, "hop_length": hop_length, },
        )

        # 计算 LFCC，输入的 `signal` 用作 waveform
        lfcc = lfcc_transform(signal)
        lfcc = lfcc.squeeze(0)
        signal = signal.squeeze(0)



        # 返回转置后的结果，以匹配 (frames, n_coeffs)
        return signal, self.liftering(librosa.power_to_db(lfcc).T), stft.T

    def liftering(self, mfcc, L=22):
        """
        对MFCC系数进行倒谱滤波（liftering）
        :param mfcc: MFCC矩阵 (shape: [num_frames, num_coeffs])
        :param L: liftering系数 (默认值为22)
        :return: 经过liftering处理的MFCC
        """
        num_frames, num_coeffs = mfcc.shape
        n = np.arange(num_coeffs)
        lifter = 1 + (L / 2) * np.sin(np.pi * n / L)
        return mfcc * lifter
    def pre_emphasis(self, signal, alpha=0.95):
        """
        对音频信号进行预加重处理
        :param signal: 原始音频信号 (numpy array)
        :param alpha: 预加重系数 (通常为0.95)
        :return: 预加重后的信号
        """
        emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
        return emphasized_signal
def get_loader(train_data_dir, batch_size):
    train_dataset = Datasets(train_data_dir, "train")
    test_dataset = Datasets(train_data_dir, "test")


    # print(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    # train_data = Datasets('./data/TrainDataFile')   (320, 480) (1601,)
    # val_data = Datasets('./data/TestDataFile')  (320, 480) (1581,)
    # d0 = val_data[0]
    # print(len(val_data))

    train_loader, test_loader = get_loader('./data', 16)
    for batch in train_loader:
        data = batch
        print(data["source"].shape, data["target"].shape, data["cepstrum"].shape)

