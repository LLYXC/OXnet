# OXnet
Code for paper [OXnet: Deep Omni-supervised Thoracic DiseaseDetection from Chest X-rays](https://arxiv.org/abs/2104.03218)

## Preliminary: pytorch-retinanet
This code is modified from https://github.com/yhenon/pytorch-retinanet. Please read INSTALL.md for preliminary requirements.

## Dataset preparation
We use COCO style annotations. Due to security concern, we cannot release the dataset. 

To apply OXnet on a custom dataset, 

## Training
To train OXnet:
```
python train_oxnet.py --dataset coco --coco_path /root/of/json/and/image/files --load_pth /path/to/pretrained/model.pt --num_labeled_data 2725 --num_data 13964 --experiment_name debug  --depth 101
```


To train RetinaNet as a baseline:
```
python train_retinanet.py --dataset coco --coco_path /root/of/json/and/image/files --experiment_name debug  --depth 101
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
