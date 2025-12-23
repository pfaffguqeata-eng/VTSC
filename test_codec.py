import os

import torch

from glob import glob
import shutil
import math
import random
import json
import time
from model_t import VTSC
from model_t import MFCM4, coding_feat_epoch_entropy_2
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from scipy.linalg import sqrtm
from dataset_tapping import Datasets
# from CompleterModel import *
from utils import *
from sklearn.metrics import f1_score
from ptflops import get_model_complexity_info

torch.cuda.empty_cache()

# def print_gradients(model):
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             continue
#         else:
#             print(f'梯度 - {name}: None')
def compute_bpp(z_strings, z_likelihoods):
    # 获取一维信号的大小
    num_points = sum([len(s) for s in z_strings])

    # 计算 bpp # 假设 'likelihoods' 是每个数据点的概率分布
    total_bpp = 0.0

    for likelihood in z_likelihoods:
        # 计算每个数据点的对数似然
        total_bpp += torch.log(likelihood).sum() / (-math.log(2) * num_points)

    return total_bpp.item()


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """计算FID值"""
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)

    # 如果结果有小的虚数部分，取实部
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def test_epoch(vtsc, classify_model, test_dataloader, val_path, criterion, device):
    vtsc.eval()  # 设置模型为评估模式

    total_loss = 0.0
    num_samples = 1
    coding_time = 0
    correct_before = 0
    correct_after = 0
    total = 0
    pre_label = []
    real_label = []
    fids = []
    cos_s = []
    with torch.no_grad():
        print("now testing")
        for data in tqdm(test_dataloader):

            # 将数据和标签移动到 GPU（如果可用）

            inputs_t = data["source"].float().to(device)


            labels = data["label"].to(device)



            output = vtsc.compress(inputs_t)




            output_feat = output["output"]
            bitrate = output["bitrate"]
            # 随机选择一个起始索引
            # 取出从 start_index 开始的 49 个通道
            output_class = classify_model(inputs_t)
            classify_result_before_slice = output_class["output_avg"]
            t_feat_, _ = classify_model.wav2vec2_model(inputs_t)
            t_feat_ = classify_model.wav_dropout(t_feat_)
            output_feat_np = output_feat.detach().cpu().numpy()
            output_feat_np = output_feat_np.reshape(-1, output_feat_np.shape[-1])
            t_feat_np = t_feat_.detach().cpu().numpy()
            t_feat_np = t_feat_np.reshape(-1, t_feat_np.shape[-1])
            mu1 = np.mean(output_feat_np, axis=0)
            sigma1 = np.cov(output_feat_np, rowvar=False)

            mu2 = np.mean(t_feat_np, axis=0)
            sigma2 = np.cov(t_feat_np, rowvar=False)

            fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
            fids.append(fid_score)
            cos_sim = F.cosine_similarity(output_feat, t_feat_, dim=1)
            cos_s.append(cos_sim)
            classify_result_after_slice, _ = classify_model.stat_(output_feat)
            _, predicted_before = torch.max(classify_result_before_slice, 1)
            _, predicted_after = torch.max(classify_result_after_slice, 1)
            correct_before += (predicted_before == labels).sum().item()
            correct_after += (predicted_after == labels).sum().item()
            pre_label.append(predicted_after.cpu().numpy())
            real_label.append(labels.cpu().numpy())
            codingtime = output["time"]["codingtime"]
            decodingtime = output["time"]["decodingtime"]

            total += labels.size(0)
            # if total>1 :
            #     coding_time += codingtime
            #     time_c = codingtime/(total-1)
            print(f"编码时间: {codingtime:.3f} 秒", f"解码时间: {decodingtime:.4f} 秒")

            num_samples += 1
            print_lenth = len(test_dataloader)
            if num_samples%print_lenth == 0:
                f1_macro = f1_score(real_label, pre_label, average='macro')
                print('\n---result_bpp(real)---')
                print(f'Bitrate: {bitrate:.3f}bpp')

                print(
                    f'classify_result: correct rate before coding:{correct_before * 100 / total:.3f},correct after coding:{correct_after*100 / total:.3f}, macro F1:{f1_macro}')

                # print(f'average_Bit-rate: {(Bit_rate/num_samples):.3f} bpp,classify_result: before{classify_result_before:.3f},after{classify_result_after:.3f}')
    avg_fid = np.mean(fids)
    print("平均 FID 分数为：", avg_fid)
    all_cos_sim = torch.cat(cos_s, dim=0)  # shape: [N * batch_size]
    avg_cos_sim = all_cos_sim.mean()
    print("平均余弦相似度：", avg_cos_sim.item())

    average_loss = total_loss / num_samples
    print(f"Average Test Loss: {average_loss:.4f}")
    return total_loss,average_loss


def main():
    device = "cuda"
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)









    test_dataset = Datasets(args.dataset, "test")
    # print(args.train_dataset)
    # print(args.test_dataset)

    # exit()
    args.batch_size = 8

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    #创建编码模型实例
    vtsc = VTSC()
    vtsc = vtsc.to(device)
    classify_model = MFCM4()
    classify_model.to(device)
    classify_model.eval()
    checkpoint = torch.load('./classift_tapping.pth', map_location=device)
    classify_model.load_state_dict(checkpoint["model_state_dict"])
    print(f'[*] Total Parameters = {sum(p.numel() for p in vtsc.parameters() if p.requires_grad)}')

    # flops, params = get_model_complexity_info(vtsc, (1, 1600), as_strings=True,
    #                                           print_per_layer_stat=True, verbose=True)
    # print('FLOPs:', flops)
    # print('Params:', params)

    # if torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)
    milestones = [195, 200]
    criterion = coding_feat_epoch_entropy_2()
    args.learning_rate = 1e-3
    optimizer = torch.optim.Adam(vtsc.parameters(), lr=args.learning_rate)
    args.checkpoint = 'logs/coding_tapping_20250304_091218/latest.pth'
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint, map_location=device)
        vtsc.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
        lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 1
    args.epochs = 810
    # best_psnr = 0
    for epoch in range(start_epoch, args.epochs):

        # logger.info(f"Epoch [{epoch + 1}], Total Loss: {loss_dict['total_loss'].item():.4f}, "
        #       f"l1loss_clsresult: {loss_dict['l1loss_clsresult'].item():.4f}, entropy_loss: {loss_dict['entropy_loss'].item():.4f},"
        #       f"Entropy Loss: {loss_dict['entropy_loss'].item():.4f}, ")
        save_dir = ""
        if epoch % 10 == 0 and epoch > 10:
            loss, avr_loss = test_epoch(vtsc, classify_model, test_dataloader, save_dir, criterion, device)



if __name__ == "__main__":
    main()
