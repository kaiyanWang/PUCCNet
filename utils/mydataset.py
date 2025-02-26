import os
import random
import torch
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])

class MyTrainDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=512):
        super(MyTrainDataSet, self).__init__()

        # 预加载数据集到内存
        haze_img_names = sorted(os.listdir(inputPathTrain))
        # # 排序  Haze4K
        # haze_img_names = sorted(os.listdir(inputPathTrain), key=lambda x: x.split('_')[0])
        self.haze_imgs = [
            Image.open(os.path.join(inputPathTrain, img)).convert("RGB")
            for img in haze_img_names if is_image_file(img)
        ]

        clear_img_names = sorted(os.listdir(targetPathTrain))
        self.clear_imgs = [
            Image.open(os.path.join(targetPathTrain, img)).convert("RGB")
            for img in clear_img_names if img.endswith(('.png', '.jpg', '.jpeg', '.PNG', 'JPG', 'JPEG'))
        ]

        print(f"Loaded {len(self.haze_imgs)} hazy images and {len(self.clear_imgs)} clear images into memory.")

        self.ps = patch_size

    def __len__(self):
        return len(self.haze_imgs)

    def __getitem__(self, index):

        ps = self.ps

        haze = self.haze_imgs[index]  # 直接从内存获取
        clear = self.clear_imgs[index]  # 直接从内存获取

        inputImage = ttf.to_tensor(haze)
        targetImage = ttf.to_tensor(clear)

        hh, ww = targetImage.shape[1], targetImage.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        aug = random.randint(0, 8)
        #
        input_ = inputImage[:, rr:rr+ps, cc:cc+ps]
        target = targetImage[:, rr:rr+ps, cc:cc+ps]

        if aug == 1:
            input_, target = input_.flip(1), target.flip(1)
        elif aug == 2:
            input_, target = input_.flip(2), target.flip(2)
        elif aug == 3:
            input_, target = torch.rot90(input_, dims=(1, 2)), torch.rot90(target, dims=(1, 2))
        elif aug == 4:
            input_, target = torch.rot90(input_, dims=(1, 2), k=2), torch.rot90(target, dims=(1, 2), k=2)
        elif aug == 5:
            input_, target = torch.rot90(input_, dims=(1, 2), k=3), torch.rot90(target, dims=(1, 2), k=3)
        elif aug == 6:
            input_, target = torch.rot90(input_.flip(1), dims=(1, 2)), torch.rot90(target.flip(1), dims=(1, 2))
        elif aug == 7:
            input_, target = torch.rot90(input_.flip(2), dims=(1, 2)), torch.rot90(target.flip(2), dims=(1, 2))

        return input_, target

class MyTestDataSet(Dataset):
    def __init__(self, inputPathTest, targetPathTest):
        super(MyTestDataSet, self).__init__()

        # 预加载数据集到内存
        self.haze_img_names = sorted(os.listdir(inputPathTest))
        # # 排序  Haze4K
        # self.haze_img_names = sorted(os.listdir(inputPathTest), key=lambda x: x.split('_')[0])
        self.haze_imgs = [
            Image.open(os.path.join(inputPathTest, img)).convert("RGB")
            for img in self.haze_img_names if is_image_file(img)
        ]

        clear_img_names = sorted(os.listdir(targetPathTest))
        self.clear_imgs = [
            Image.open(os.path.join(targetPathTest, img)).convert("RGB")
            for img in clear_img_names if is_image_file(img)
        ]
        print(f"Loaded {len(self.haze_imgs)} hazy images and {len(self.clear_imgs)} clear images into memory.")

    def __len__(self):
        return len(self.haze_imgs)

    def __getitem__(self, index):
        haze = self.haze_imgs[index]  # 直接从内存获取
        clear = self.clear_imgs[index]  # 直接从内存获取

        input_ = ttf.to_tensor(haze)
        target = ttf.to_tensor(clear)

        return input_, target, self.haze_img_names[index]