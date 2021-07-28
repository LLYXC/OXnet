#python train_oxnet.py \
#--dataset coco --coco_path /root/of/json/and/image/files \
#--load_pth /path/to/pretrained/model.pt \
#--num_labeled_data 2725 --num_unlabeled_data 13964 \
#--experiment_name debug  --depth 101


python train_retinanet.py \
--dataset coco --coco_path /root/of/json/and/image/files \
--experiment_name debug  --depth 101
