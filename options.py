import os
import utils

class Options():
    def __init__(self):
        super().__init__()

        # resume
        self.RESUME = False

        # dataset name
        self.DATASET = 'NH-HAZE'
        # train
        self.Input_Path_Train = '/root/NH-HAZE/train/hazy'
        self.Target_Path_Train = '/root/NH-HAZE/train/GT'
        # test
        self.Input_Path_Test = '/root/NH-HAZE/test/hazy'
        self.Target_Path_Test = '/root/NH-HAZE/test/GT'

        # save
        self.Model_Save_Path = './checkpoints/'
        self.Loss_File_Save_Path='./files/'
        self.Result_Save_Path = './results/'
        self.Logs_Save_Path = './logs/'
        self.Model_dir = os.path.join(self.Model_Save_Path, self.DATASET)
        self.File_dir = os.path.join(self.Loss_File_Save_Path, self.DATASET)
        self.Img_dir = os.path.join(self.Result_Save_Path, self.DATASET)
        self.Log_dir = os.path.join(self.Logs_Save_Path, self.DATASET)
        utils.mkdir(self.Model_dir)
        utils.mkdir(self.File_dir)
        utils.mkdir(self.Img_dir)

        self.NUM_EPOCHS = 1000
        self.VAL_AFTER_EVERY = 20
        self.Learning_Rate = 1e-3
        self.Batch_Size_Train = 10
        self.Patch_Size_Train = 512

        self.Num_Works = 4