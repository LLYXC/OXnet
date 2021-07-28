#python train_oxnet.py \
#--dataset coco --coco_path /research/d4/gds/lyluo/XR/data/Imsight \
#--load_pth /research/d4/gds/lyluo/XRdetection/code/retinanet_cxr/checkpoints/0629-res101-P2-lowerbound/coco_retinanet_22.pt \
#--num_labeled_data 2725 --num_unlabeled_data 13964 \
#--experiment_name debug  --depth 101


python train_retinanet.py \
--dataset coco --coco_path /research/d4/gds/lyluo/XR/data/Imsight \
--experiment_name debug  --depth 101