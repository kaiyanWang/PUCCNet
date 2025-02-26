# An Efficient Dehazing Method Using Pixel Unshuffle and Color Correction(PUCCNet)

**Paper Link:** [[PUCCNet](https://www.sciencedirect.com/science/article/abs/pii/S0923596525000074)]

## Requirements

```
Python  3.8
PyTorch  1.11.0
Cuda  11.3
```

## Train

If you intend to conduct training on our proposed PUCCNet using your own datasets, it is imperative to initially ascertain the training and testing paths specified in `options.py`. Specifically, the paths should be provided in the manner illustrated below.

```python
# dataset name
self.DATASET = 'NH-HAZE'
# train
self.Input_Path_Train = '/root/NH-HAZE/train/hazy'
self.Target_Path_Train = '/root/NH-HAZE/train/GT'
# test
self.Input_Path_Test = '/root/NH-HAZE/test/hazy'
self.Target_Path_Test = '/root/NH-HAZE/test/GT'
```

Subsequently, you may attempt the execution of the following command in order to initiate the training process.

```python
python train.py
```

## Test

You can proceed with the testing phase and assess the performance of our proposed PUCCNet.

```
python test.py
```

## Dataset

| Dataset | Link                                                         | Dataset    | Link                                                         |
| ------- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
| O-HAZE  | [[O-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/)] | I-HAZE     | [[I-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18/i-haze/)] |
| NH-HAZE | [[NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)] | Dense-Haze | [[DENSE-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/)] |