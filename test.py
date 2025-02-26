import sys
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from net.PUCCNet import PUCCNet
import utils
from options import Options
import torch
import os
from torchvision.utils import save_image

def test(test_loader, myNet):
    PSNR = 0
    SSIM = 0
    MSE = 0
    L = 0

    torch.cuda.empty_cache()
    myNet.eval().cuda()

    with torch.no_grad():
        for index, (x, y, name) in enumerate(tqdm(test_loader, desc='Testing !!! ', file=sys.stdout), 0):
            input = x.cuda()
            target = y.cuda()

            # # input裁剪h、w都是4的倍数
            # input = utils.crop_to_multiple_of_4(input)
            # target = utils.crop_to_multiple_of_4(target)

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

            print('---------------------------------------------------------')
            print(L, psnr_val, ssim_val, mse_val.item())  # current metrical scores
            format_str = 'L: %d, name: %s, psnr_val: %.6f, ssim_val: %.6f, mse_val: %.6f'
            a = str(format_str % (L, name[0], psnr_val, ssim_val, mse_val.item()))
            evaluation_file = open(os.path.join(opt.Img_dir, 'EVALUATION.txt'), 'a+')
            evaluation_file.write(a)
            evaluation_file.write('\n')
            evaluation_file.close()
            print('---------------------------------------------------------')

            save_image(out_dehaze.squeeze(0), os.path.join(opt.Img_dir, name[0]))

    print('\n---------------------------------------------------------')
    print(PSNR / L, SSIM / L, MSE / L)
    format_str = 'dataset: %s, Avg: PSNR: %.6f, SSIM: %.6f, MSE: %.6f'
    a = str(format_str % (opt.DATASET, PSNR / L, SSIM / L, MSE / L))
    evaluation_file = open(os.path.join(opt.Img_dir , 'EVALUATION.txt'), 'a+')
    evaluation_file.write(a)
    evaluation_file.write('\n')
    evaluation_file.close()
    print('---------------------------------------------------------')


if __name__ == '__main__':
    opt = Options()

    # Model
    myNet = PUCCNet().cuda()
    utils.load_myNet_state_dict(myNet, os.path.join(opt.Model_dir, 'model_best.pth'))

    # DataLoader
    datasetTest = utils.MyTestDataSet(opt.Input_Path_Test, opt.Target_Path_Test)
    test_loader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=opt.Num_Works, pin_memory=True)

    timeStart = time.time()
    test(test_loader, myNet)
    timeEnd = time.time()
    print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))