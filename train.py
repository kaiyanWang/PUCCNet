import os
import sys
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from tqdm import tqdm
from net.PUCCNet import PUCCNet
from options import Options
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus

# create loss_fig, psnr_fig, ssim_fig
def create_figs(start_plot_epoch, epoch, train_losses, valid_psnrs, valid_ssims):
    # 创建第一个图形对象，绘制训练损失曲线
    plt.figure(1)
    plt.plot(range(start_plot_epoch, epoch + 1), train_losses[start_plot_epoch - start_epoch:],
             label='Train Loss', color='tab:red')
    plt.legend(loc='upper left')
    plt.title(f'Training Loss (Epoch {epoch})')
    plt.pause(0.01)
    plt.savefig(os.path.join(opt.File_dir, 'training_loss.png'))
    plt.close()

    # 创建第二个图形对象，绘制PSNR曲线
    plt.figure(2)
    plt.plot(range(start_plot_epoch, epoch + 1, opt.VAL_AFTER_EVERY),
             valid_psnrs[(start_plot_epoch - start_epoch) // opt.VAL_AFTER_EVERY:],
             label='PSNR', color='tab:blue')
    plt.legend(loc='upper right')
    plt.title(f'Validation PSNR (Epoch {epoch})')
    plt.pause(0.01)
    plt.savefig(os.path.join(opt.File_dir, 'validation_psnr.png'))
    plt.close()

    # 创建第三个图形对象，绘制SSIM曲线
    plt.figure(3)
    plt.plot(range(start_plot_epoch, epoch + 1, opt.VAL_AFTER_EVERY),
             valid_ssims[(start_plot_epoch - start_epoch) // opt.VAL_AFTER_EVERY:],
             label='SSIM', color='tab:green')
    plt.legend(loc='upper right')
    plt.title(f'Validation SSIM (Epoch {epoch})')
    plt.pause(0.01)
    plt.savefig(os.path.join(opt.File_dir, 'validation_ssim.png'))
    plt.close()


# save checkpoint
def save_model(epoch, myNet_state_dict, optimizer_state_dict, scheduler_state_dict,
                       best_psnr, best_epoch, best_ssim, best_epoch_ssim, checkpoint_name):
    if checkpoint_name == 'model':
        checkpoint_name = checkpoint_name + f"_epoch_{epoch}.pth"
    else:
        checkpoint_name = checkpoint_name + ".pth"

    torch.save({'epoch': epoch,
                'state_dict': myNet_state_dict,
                'optimizer': optimizer_state_dict,
                'scheduler': scheduler_state_dict,
                'best_psnr': best_psnr,
                'best_epoch': best_epoch,
                'best_ssim': best_ssim,
                'best_epoch_ssim': best_epoch_ssim,
                }, os.path.join(opt.Model_dir, checkpoint_name))


# get epoch_loss and update checkpoint
def train(train_loader, myNet, criterion, optimizer):
    epoch_loss = 0
    torch.cuda.empty_cache()
    myNet.train().cuda()

    iters = tqdm(train_loader, file=sys.stdout)
    for i, data in enumerate(iters, 0):
        input, target = data[0].cuda(), data[1].cuda()

        output = myNet(input)
        out_dehaze = output.cuda()  # out_dehaze

        loss = 1.2 * criterion[0](out_dehaze, target) + 0.2 * (1 - criterion[1](out_dehaze, target)) + 5 * criterion[2](out_dehaze, target) / 0.2222223

        epoch_loss += loss.item()
        iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch, opt.NUM_EPOCHS, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_loss

def test(test_loader, myNet):
    PSNR = 0
    SSIM = 0
    MSE = 0
    L = 0

    torch.cuda.empty_cache()
    myNet.eval().cuda()

    with torch.no_grad():
        for index, (x, y, name) in enumerate(test_loader, 0):
            input = x.cuda()
            target = y.cuda()

            # input裁剪h、w都是4的倍数
            input = utils.crop_to_multiple_of_4(input)
            target = utils.crop_to_multiple_of_4(target)

            output = myNet(input)
            out_dehaze = output

            # 计算SSIM
            ssim_val = utils.ssim(out_dehaze, target).item()

            # 计算MSE
            mse_val = F.mse_loss(out_dehaze, target)

            # 计算PSNR
            psnr_val = 10 * torch.log10(1 / mse_val).item()

            MSE += mse_val.item()
            PSNR += psnr_val
            SSIM += ssim_val
            L = index+1

    epoch_avg_psnr = PSNR / L
    epoch_avg_ssim = SSIM / L
    epoch_avg_mse = MSE / L

    return epoch_avg_psnr, epoch_avg_ssim, epoch_avg_mse

if __name__ == "__main__":
    opt=Options()

    torch.cuda.is_available()
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    file_psnr = 'PSNR.txt'
    file_loss = 'LOSS.txt'

    set_seed(42)

    # Model
    myNet=PUCCNet().cuda()
    if len(device_ids) > 1:
        myNet = nn.DataParallel(myNet, device_ids=device_ids)

    # Loss
    criterion = []
    criterion.append(nn.L1Loss())
    criterion.append(utils.ssim)
    criterion.append(utils.CALoss())

    # optimizer
    # optimizer = optim.Adam(myNet.parameters(), lr=opt.Learning_Rate)
    optimizer = torch.optim.Adam(myNet.parameters(), lr=opt.Learning_Rate, betas=(0.9, 0.999))

    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.NUM_EPOCHS, eta_min=1e-8)

    start_epoch = 1
    best_psnr = 0
    best_epoch = 0
    best_ssim = 0
    best_epoch_ssim = 0

    if opt.RESUME:
        model_latest_path = os.path.join(opt.Model_dir, 'model_latest.pth')
        checkpoint = torch.load(model_latest_path)

        utils.load_myNet_state_dict(myNet, model_latest_path) # myNet.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_psnr = checkpoint['best_psnr']
        best_epoch = checkpoint['best_epoch']
        best_ssim = checkpoint['best_ssim']
        best_epoch_ssim = checkpoint['best_epoch_ssim']
        start_epoch = checkpoint['epoch'] + 1

        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate: ", scheduler.get_last_lr()[0])
        print("==> Resuming Training with Start Epoch: ", start_epoch)
        print('------------------------------------------------------------------------------')


    # DataLoaders
    train_dataset = utils.MyTrainDataSet(opt.Input_Path_Train, opt.Target_Path_Train, patch_size=opt.Patch_Size_Train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.Batch_Size_Train, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    datasetTest = utils.MyTestDataSet(opt.Input_Path_Test, opt.Target_Path_Test)
    test_loader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=opt.Num_Works, pin_memory=True)

    print('\n===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.NUM_EPOCHS))
    print('===> Training')

    writer = SummaryWriter(log_dir=opt.Log_dir)
    train_losses = []  # 记录每个epoch的训练损失
    valid_psnrs = []  # 记录每个验证周期的PSNR
    valid_ssims = []  # 记录每个验证周期的SSIM

    for epoch in range(start_epoch, opt.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        # 1、Train
        epoch_loss = train(train_loader, myNet, criterion, optimizer)
        writer.add_scalar('train_loss', epoch_loss, epoch)  # tensorboardX记录 1、loss
        train_losses.append(epoch_loss)  # 记录训练损失


        # 2、Evaluation
        if epoch % opt.VAL_AFTER_EVERY == 0:
            epoch_avg_psnr, epoch_avg_ssim, epoch_avg_mse = test(test_loader, myNet)
            writer.add_scalar('valid_psnr', epoch_avg_psnr, epoch)  # tensorboardX记录 2、epoch_avg_psnr
            valid_psnrs.append(epoch_avg_psnr)  # 记录PSNR
            valid_ssims.append(epoch_avg_ssim)  # 记录SSIM

            if epoch_avg_psnr > best_psnr:
                best_psnr = epoch_avg_psnr
                best_epoch = epoch
                # torch.save
                save_model(epoch, myNet.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                           best_psnr, best_epoch, best_ssim, best_epoch_ssim, checkpoint_name="model_best")

            if epoch_avg_ssim > best_ssim:
                best_ssim = epoch_avg_ssim
                best_epoch_ssim = epoch
                # torch.save
                save_model(epoch, myNet.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                           best_psnr, best_epoch, best_ssim, best_epoch_ssim, checkpoint_name="model_best_ssim")

            # 从第20个epoch开始绘制曲线
            start_plot_epoch = 20
            if epoch >= start_plot_epoch:
                create_figs(start_plot_epoch, epoch, train_losses, valid_psnrs, valid_ssims)

            # 输出：每VAL_AFTER_EVERY个epoch都输出和保存本次的日志信息--psnr
            print("\n------------------------------------------------------------------")
            format_str1 = 'Epoch: %d, PSNR: %.4f, SSIM: %.4f, MSE: %.4f, best_epoch: %d, Best_PSNR: %.4f, best_epoch_ssim: %d, Best_SSIM: %.4f'
            a = str(format_str1 % (epoch, epoch_avg_psnr, epoch_avg_ssim, epoch_avg_mse, best_epoch, best_psnr, best_epoch_ssim, best_ssim))
            print(a)
            PSNR_file = open(os.path.join(opt.File_dir, file_psnr), 'a+')
            PSNR_file.write(a)
            PSNR_file.write('\n')
            PSNR_file.close()
            print("------------------------------------------------------------------\n")

            writer.add_scalar('best_psnr', best_psnr, epoch)

            # # 每opt.VAL_AFTER_EVERY保存一次（其实保存best就好）
            # # torch.save
            # save_model(epoch, myNet.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
            #            best_psnr, best_epoch, best_ssim, best_epoch_ssim, checkpoint_name="model")

        # 3、change lr
        scheduler.step()
        epoch_end_time = time.time()

        # 输出：每一个epoch都输出和保存本次的日志信息--loss
        print("------------------------------------------------------------------")
        format_str = 'Epoch: %d, Time: %.4f, Loss: %.4f, LearningRate: %.8f'
        a = str(format_str % (epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]))
        print(a)
        loss_file = open(os.path.join(opt.File_dir,file_loss), 'a+')
        loss_file.write(a)
        loss_file.write('\n')
        loss_file.close()
        print("------------------------------------------------------------------")

        # torch.save
        save_model(epoch, myNet.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                   best_psnr, best_epoch, best_ssim, best_epoch_ssim, checkpoint_name="model_latest")