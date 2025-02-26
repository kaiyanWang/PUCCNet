import os
import torch
from collections import OrderedDict

def load_myNet_state_dict(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"],strict=False)
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict,strict=False)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


from torchvision.transforms import functional as ttf
#  将一张图像或tensor裁剪成h、w都是4的倍数
def crop_to_multiple_of_4(input_image):
    h, w = input_image.shape[-2:]  # 获取图像的高度和宽度
    new_h = h - (h % 4)  # 计算新的高度，使其是4的倍数
    new_w = w - (w % 4)  # 计算新的宽度，使其是4的倍数

    # 计算裁剪的上下左右边界
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    bottom = top + new_h
    right = left + new_w

    # 使用torchvision的crop函数裁剪图像
    cropped_image = ttf.crop(input_image, top, left, new_h, new_w)
    return cropped_image