# OXnet: Deep Omni-supervised Thoracic DiseaseDetection from Chest X-rays
by [Luyang Luo](https://llyxc.github.io/), [Hao Chen](https://www.cse.ust.hk/admin/people/faculty/profile/jhc), [Yanning Zhou](https://ynzhou.netlify.app/), [Huangjing Lin](https://www.linkedin.com/in/huangjing-lin-3bb526a0/?originalSubdomain=hk), and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/)

Repository for MICCAI 2021 paper [OXnet: Deep Omni-supervised Thoracic DiseaseDetection from Chest X-rays](https://arxiv.org/abs/2104.03218)

![oxnet](https://github.com/LLYXC/LLYXC.github.io/blob/master/indexpics/miccai21_oxnet.png)

## Preliminary: pytorch-retinanet
This code is modified from [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet).
Please refer to [INSTALL.md](INSTALL.md) for preliminary requirements.
Please also make following directories to save the log files and checkpoints:
```
mkdir logs
mkdir checkpoints
```

## Dataset preparation
Due to security concern, we cannot release the dataset. 

To apply OXnet on a custom dataset: 
1. Note that the data shall follow the COCO style;
2. Put the the json and image file in a same root directory, e.g., root/of/json/and/image/files;
3. Modify the path to json file (line 62-63) and the path to image files (line 102-103) in [retinanet/dataloader.py](retinanet/dataloader.py);
4. Modify the ```set_name``` in [train_oxnet.py](train_oxnet.py) (line 92, 94, and 96) or [train_retinanet.py](train_retinanet.py) (line 72, 74, and 76).

The data tree is as follows:
``` 
    ├── Root
          ├── Images
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
          ├── Json
                ├── train.json
                ├── val.json
                ├── test.json
```

## Training
Note that when there are data of different types of annotations, in the json file, labeled data are listed at the beginning and unlabeled data are listed behind. To train OXnet, :
```
python train_oxnet.py --dataset coco --coco_path root/of/json/and/image/files --load_pth path/to/pretrained/model.pt --num_labeled_data 2725 --num_data 13964 --experiment_name oxnet --depth 101
```

To train RetinaNet as a baseline:
```
python train_retinanet.py --dataset coco --coco_path /root/of/json/and/image/files --experiment_name retinanet  --depth 101
```

## Citation
If you find the paper or the code helpful to your own work, please consider cite:
```
@inproceedings{luo2021oxnet,
      title={OXnet: Omni-supervised Thoracic Disease Detection from Chest X-rays}, 
      author={Luyang Luo and Hao Chen and Yanning Zhou and Huangjing Lin and Pheng-Ann Heng},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      year={2021}
}
```

## TODO list
1. Modify dataloader to load three types of data simultaneously: fully-annotated, weakly-annotated, and unlabeled.

## Questions

Please contact 'lyluo@cse.cuhk.edu.hk'
