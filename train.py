import os

import torch

from glob import glob
import shutil
import math
import random
import json
import time
from model_t import VTSC
from model_t import MFCM4, coding_feat_epoch_entropy
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2
from dataset_tapping import Datasets
# from CompleterModel import *
from utils import *
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

def train_one_epoch(vtsc, classify_model, criterion, train_dataloader, optimizer, epoch, clip_max_norm,
                    logger_train, tb_logger, current_step, args):

    vtsc.train()
    classify_model.eval()

    device = next(vtsc.parameters()).device
    vtsc.update()
    correct_before = 0
    correct_after = 0
    total = 0

    print(f"now training :Epoch:{epoch + 1}")
    for data in tqdm(train_dataloader):
        inputs_t = data["source"].float().to(device)
        labels = data["label"].to(device)
        output = vtsc.forward(inputs_t)
        output_feat = output["output"]
        z_likelihoods = output["likelihoods"]["z"]
        bitrate = output["bitrate"]
        optimizer.zero_grad()
        output_class = classify_model(inputs_t)
        classify_result_before_slice = output_class["output_avg"]
        t_feat_ = output_class["t_feat_"]
        classify_result_after_slice, _ = classify_model.stat_(output_feat)
        _, predicted_before = torch.max(classify_result_before_slice, 1)
        _, predicted_after = torch.max(classify_result_after_slice, 1)
        correct_before += (predicted_before == labels).sum().item()
        correct_after += (predicted_after == labels).sum().item()
        total += labels.size(0)
        loss_dict = criterion(output["indices"], z_likelihoods, labels, classify_result_before_slice, classify_result_after_slice, t_feat_, output_feat, epoch)
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(vtsc.parameters(), max_norm=0.1)
        optimizer.step()
        current_step += 1
        print_lenth = len(train_dataloader)
        logger = logging.getLogger('base')
        if (current_step-1) % print_lenth == 0:
            print(f"\n[Epoch:{epoch + 1}/iter:{current_step%len(train_dataloader)}], Total Loss: {loss_dict['total_loss'].item():.4f}, ",
                  f"cross_entropy_clsresult: {loss_dict['cross_entropy_clsresult'].item():.4f}, l1_loss:{loss_dict['l1_loss']}, kld_loss:{loss_dict['kld_loss']}, entropy_loss:{loss_dict['entropy_loss']}")
            logger.info(
                f"[Epoch:{epoch + 1}//iter{current_step%len(train_dataloader)}], Total Loss: {loss_dict['total_loss'].item():.4f}, "
                  f"cross_entropy_clsresult: {loss_dict['cross_entropy_clsresult'].item():.4f}, l1_loss:{loss_dict['l1_loss']}, kld_loss:{loss_dict['kld_loss']}, entropy_loss:{loss_dict['entropy_loss']}")
            print(f'classify_result: correct rate before coding:{correct_before*100/total:.3f},correct after coding:{correct_after*100/total:.3f}')
            logger.info(f'classify_result: correct rate before coding:{correct_before * 100 / total:.3f},correct after coding:{correct_after*100 / total:.3f},bitrate:{bitrate:.3f}')
            print(predicted_after, labels)
    return current_step, loss_dict



def test_epoch(vtsc, classify_model, test_dataloader, val_path, criterion, device):
    vtsc.eval()
    total_loss = 0.0
    num_samples = 1
    correct_before = 0
    correct_after = 0
    total = 0
    with torch.no_grad():
        print("now testing")
        for data in tqdm(test_dataloader):
            inputs_t = data["source"].float().to(device)
            labels = data["label"].to(device)
            output = vtsc.compress(inputs_t)
            output_feat = output["output"]
            bitrate = output["bitrate"]

            output_class = classify_model(inputs_t)
            classify_result_before_slice = output_class["output_avg"]
            classify_result_after_slice, _ = classify_model.stat_(output_feat)
            _, predicted_before = torch.max(classify_result_before_slice, 1)
            _, predicted_after = torch.max(classify_result_after_slice, 1)
            correct_before += (predicted_before == labels).sum().item()
            correct_after += (predicted_after == labels).sum().item()
            codingtime = output["time"]["codingtime"]
            decodingtime = output["time"]["decodingtime"]
            total += labels.size(0)
            num_samples += 1
            print_lenth = len(test_dataloader)
            logger = logging.getLogger('base')
            if num_samples%print_lenth == 0:
                print('\n---result_bpp(real)---')
                print(f'Bitrate: {bitrate:.3f}bpp')
                print(f"编码时间: {codingtime:.3f} 秒,解码时间: {decodingtime:.3f} 秒/帧")
                print(
                    f'classify_result: correct rate before coding:{correct_before * 100 / total:.3f},correct after coding:{correct_after*100 / total:.3f}')
                logger.info(f"编码时间: {codingtime:.3f} 秒,解码时间: {decodingtime:.3f} 秒/帧")
                logger.info(f'Bitrate: {bitrate:.3f}bpp\n'
                            f'classify_result: correct rate before coding:{correct_before * 100 / total:.3f},correct after coding:{correct_after*100 / total:.3f}')
                # print(f'average_Bit-rate: {(Bit_rate/num_samples):.3f} bpp,classify_result: before{classify_result_before:.3f},after{classify_result_after:.3f}')


    average_loss = total_loss / num_samples
    print(f"Average Test Loss: {average_loss:.4f}")
    return total_loss,average_loss


def main():
    device = "cuda"
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    date = str(datetime.datetime.now())
    date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    log_dir = os.path.join('./logs', f"coding_tapping_rc_{date}")
    os.makedirs(log_dir, exist_ok=True)

    summary_dir = os.path.join(log_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    tb_logger = SummaryWriter(logdir=summary_dir, comment='info')

    setup_logger('base', log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(f'[*] Start Log To {log_dir}')

    # copy code
    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
    os.makedirs(os.path.join(log_dir, 'codes'), exist_ok=True)
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        os.makedirs(os.path.join(log_dir, 'codes', to_make))

    pyfiles = glob("./*.py")
    for py in pyfiles:
        shutil.copyfile(py, os.path.join(log_dir, 'codes') + "/" + py)

    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        tmp_files = glob(os.path.join('./', to_make, "*.py"))
        for py in tmp_files:
            shutil.copyfile(py, os.path.join(log_dir, 'codes', py[2:]))

    with open(os.path.join(log_dir, 'setting.json'), 'w') as f:
        flags_dict = {k: vars(args)[k] for k in vars(args)}
        json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    train_dataset = Datasets(args.dataset, "train")
    test_dataset = Datasets(args.dataset, "test")
    # print(args.train_dataset)
    # print(args.test_dataset)

    logger.info(f'[*] Train File Account For {len(train_dataset)}, val {len(test_dataset)}')
    # exit()
    args.batch_size = 8
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

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
    logger.info(f'[*] Total Parameters = {sum(p.numel() for p in vtsc.parameters() if p.requires_grad)}')
    # if torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)
    milestones = [200, 300]
    criterion = coding_feat_epoch_entropy()
    args.learning_rate = 1e-3
    optimizer = torch.optim.Adam(vtsc.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    args.checkpoint = './logs/coding_tapping_20250409_154912/latest.pth'
    if args.checkpoint != '':
        logger.info("Loading %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        vtsc.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
        lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 1
    args.epochs = 800
    # best_psnr = 0
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step, loss_dict = train_one_epoch(
            vtsc,
            classify_model,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            logger,
            tb_logger,
            current_step,
            args
        )
        # logger.info(f"Epoch [{epoch + 1}], Total Loss: {loss_dict['total_loss'].item():.4f}, "
        #       f"l1loss_clsresult: {loss_dict['l1loss_clsresult'].item():.4f}, entropy_loss: {loss_dict['entropy_loss'].item():.4f},"
        #       f"Entropy Loss: {loss_dict['entropy_loss'].item():.4f}, ")
        save_dir = os.path.join(log_dir, 'val_images', '%03d' % (epoch + 1))
        if epoch % 10 == 0 and epoch > 10:
            loss, avr_loss = test_epoch(vtsc, classify_model, test_dataloader, save_dir, criterion, device)
        lr_scheduler.step()

        is_best = False
        avr_loss = 0
        best_loss = min(avr_loss, best_loss)
        logger.info(f"avr_loss: {avr_loss}, best_loss,{best_loss}")
        state = {
                "epoch": epoch,
                "state_dict": vtsc.state_dict(),
                "loss": avr_loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
        torch.save(state, os.path.join(log_dir, f"latest.pth"))
        if is_best:
            torch.save(state, os.path.join(log_dir, f"best.pth"))


if __name__ == "__main__":
    main()
