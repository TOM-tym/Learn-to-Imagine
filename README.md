# Learn-to-Imagine
Official PyTorch implementation of CVPR2022 paper “Learning to Imagine: Diversify Memory for Incremental Learning using Unlabeled Data”
[paper](Learning_to_imagine.pdf)[Project Page](https://isee-ai.cn/~yuming/Learn_to_imagine.html)

## Environments

- Python: 3.6.9
- PyTorch: 1.2.0


## Training
### Training for CIFAR100
#### Data preparation
CIFAR100 dataset will be downloaded automatically to `data_path` specified in `CIFAR100/options/data/cifar100_3orders.yaml`.

For our proposed method, an extra unlabeled dataset is needed for feature generation. To cooperate with CIFAR100, we choose
the 32x32 down-sampled ImageNet as the auxiliary unlabeled dataset. Please download the dataset from [image-net.org](https://image-net.org/download.php)
and put it on the `data_path/imagenet_32`.
#### Training script
```
cd CIFAR100;
python -minclearn \
    --options options/Imagine/B50/CIFAR100_B50.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 10 --increment 10 --device YOUR_DEVICES_INDEX --label cifar_b50_step10 -w 0 --save task
```

### Training for ImageNet-Subset
#### Data preparation
ImageNet100/1000 dataset cannot be downloaded automatically, please download it from [image-net.org](https://image-net.org/download.php).
Place the dataset in `data_path` specified in `ImageNet/options/data/imagenet100_1order.yaml`.

In order to conduct incremental training, we also need to put imagenet split file `train_100.txt`, `val_100.txt` into
the data path. Symbolic link is recommended:
```
ln -s ImageNet/imagenet_split/train_100.txt data_path/imagenet1k/train_100.txt
ln -s ImageNet/imagenet_split/val_100.txt data_path/imagenet1k/val_100.txt
```

For ImageNet100 dataset, we use the rest ImagNet900 data from ImagNet1k as the auxiliary unlabeled dataset.
Likely, the corresponding split file should be linked to the `data_path`.
```
ln -s ImageNet/imagenet_split/900_100.txt data_path/imagenet1k/train_900.txt
```

In conclusion, the dataset should be organized like this
```
data_path
│  
│──imagenet1k
│   │
│   └───train
│       │   n01440764
│       │   n01443537 
│       │   ...
│   │
│   └───val
│       │   n01440764
│       │   n01443537
│       │   ...
│   │   
│   │ train_100.txt
│   │ train_900.txt
│   │ val_100.txt 
│   
└
```

#### Training script
```
 cd ImageNet;
 python -minclearn \
 --options options/Imagine/B50/ImageNet100_B50.yaml options/data/imagenet100_1order.yaml \
 --initial-increment 50 --increment 5 --label Imagine_ImageNet100 -w 4 --device YOUR_DEVICE_INDEX --save task
```

## Acknowledgement 
- This repository is heavily based on [incremental_learning.pytorch](https://github.com/arthurdouillard/incremental_learning.pytorch)
by [arthurdouillard](https://github.com/arthurdouillard).


- If you use this paper/code in your research, please consider citing us:
```
@inproceedings{tang2022learning,
  title={Learning to Imagine: Diversify Memory for Incremental Learning using Unlabeled Data},
  author={Tang, Yu-Ming and Peng, Yi-Xing and Zheng, Wei-Shi},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
