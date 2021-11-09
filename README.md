# Ref

The repo comes from [this](https://github.com/WangYueFt/rfs/)


# RFS

Representations for Few-Shot Learning (RFS). This repo covers the implementation of the following paper:  

**"Rethinking few-shot image classification: a good embedding is all you need?"** [Paper](https://arxiv.org/abs/2003.11539),  [Project Page](https://people.csail.mit.edu/yuewang/projects/rfs/) 

If you find this repo useful for your research, please consider citing the paper  
```
@article{tian2020rethink,
  title={Rethinking few-shot image classification: a good embedding is all you need?},
  author={Tian, Yonglong and Wang, Yue and Krishnan, Dilip and Tenenbaum, Joshua B and Isola, Phillip},
  journal={arXiv preprint arXiv:2003.11539},
  year={2020}
}
```

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. However, it should be compatible with recent PyTorch versions >=0.4.0

## Download Data
The data we used here is preprocessed by the repo of [MetaOptNet](https://github.com/kjunelee/MetaOptNet), but we have
renamed the file. Our version of data can be downloaded from here:

[[DropBox]](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0)

## Pre-trained Models

[[DropBox]](https://www.dropbox.com/sh/6xt97e7yxheac2e/AADFVQDbzWap6qIGIHBXsA8ca?dl=0)

## Running

Exemplar commands for running the code can be found in `scripts/run.sh`.

For unuspervised learning methods `CMC` and `MoCo`, please refer to the [CMC](http://github.com/HobbitLong/CMC) repo.

## Contacts
For any questions, please contact:

Yonglong Tian (yonglong@mit.edu)  
Yue Wang (yuewang@csail.mit.edu)

## Acknowlegements
Part of the code for distillation is from [RepDistiller](http://github.com/HobbitLong/RepDistiller) repo.

---

## Custom Dataset For Medical Image Analysis

In order to apply our medical images for few-shot learning on this repo, we modified the codes in some cases. The usage is shown as the followings.


**（1）Generate the data with the specified format.**

Firstly, we need to generate the data with the specified format `.pickle` whose type is `dict`:
- data['data']: imgs (type: numpy.array, (batch_size, width, height, channels))
- data['labels']: labels (type: list) 

As `utils/create_dataset.py` shown, we split the data as `train.pickle`, `val.pickle`, `test.pickle`, `trainval.pickle`, the structure of the origin medical images should be like this:
```
directory/
├── class_x
│   ├── xxx.tif
│   ├── xxy.tif
│   └── ...
└── class_y
    ├── 123.tif
    ├── nsdf3.tif
    └── ...
    └── asd932_.tif
```

In the function `load_data`, the parameter `numPerClass` denotes the number of imgs each class sampling in the original medical images.

**（2）Choose the correct `Dataset` and `transform` for training:**

In `train_supervised.py`, `train_distillation.py`, `eval_fewshot.py`, we should set the value `customDataset` of `dataset` and set the value $n$  of `n-ways` in `args.parser`. 
**Note the `n` must be less than the number of total classes `N` in `train_dataset`.**

For medical images with different sizes, we should modify the `transform_E` in `dataset/transform_cfg.py`.

**（3）Set the appropriate top-k.**

In the function `train`, `validate` and `accuracy`, we should set the `topk=(1, k)`, the `k` must be less than the number of total classes `N` in `train_dataset`. Obviously, when `k == N`, `Top-k acc` must be 100% in the logs of the trainning phase, as `Top-1 acc` gives us accuracy in the usual sense.

**（4）Set `run.sh` to train**

Before runing the program, we should create the dirs, `checkpoints` and `tensorboardlogs` for `train_supervised.py`, `dis_checkpoints` and `dis_tensorboardlogs` for `train_distillation.py`.

`run.sh`: See code for specific parameters.

```python
# ======================
# exampler commands on custom datasets
# ======================

# supervised pre-training
python train_supervised.py --trial pretrain --model_path ./checkpoints --tb_path ./tensorboardlogs --data_root ./data

# distillation
# setting '-a 1.0' should give simimlar performance
# python train_distillation.py -r 0.5 -a 0.5 --path_t ./checkpoints/resnet12_customDataset_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --trial born1 --model_path ./dis_checkpoints --tb_path ./dis_tensorboardlogs --data_root ./data/

# evaluation
# python eval_fewshot.py --model_path ./dis_checkpoints/S:resnet12_T:resnet12_customDataset_kd_r:0.5_a:0.5_b:0_trans_A_born1/resnet12_last.pth --data_root ./data/customDataset/

```