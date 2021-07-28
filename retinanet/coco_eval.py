from pycocotools.cocoeval import COCOeval
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_coco(dataset, model, threshold=0.05, experiment_name='tmp'):
    model.eval()

    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []
        global_gts = []
        global_preds = []
        local_preds = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            # run network
            if torch.cuda.is_available():
                scores, labels, boxes, global_out = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes, global_out = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))

            # global auc
            global_out = global_out[0].cpu().numpy()
            one_hot = np.zeros(global_out.shape)
            target = data['annot'][:, 4].long()
            for j in range(len(target)):
                if target[j] != -1:
                    one_hot[target[j]] = 1
            global_gts.append(one_hot)
            global_preds.append(global_out)

            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # for local auc
            local_output = torch.zeros(18)
            for i in range(18):
                idx = torch.where(labels == i)
                if len(idx[0]) != 0:
                    local_output[i] = scores[idx].sum() / len(idx[0])
            local_preds.append(local_output.numpy())

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        global_gts = np.stack(global_gts, 0)
        global_preds = np.stack(global_preds, 0)
        local_preds = np.stack(local_preds, 0)
        g_aucs = []
        l_aucs = []
        for i in range(18):
            if i in (1, 2, 4, 5, 6, 7, 14, 15, 17):
                try:
                    g_aucs.append(roc_auc_score(global_gts[:, i], global_preds[:, i]))
                    l_aucs.append(roc_auc_score(global_gts[:, i], local_preds[:, i]))
                except ValueError as e:
                    print('WARNING: {}. Check class {}.'.format(e, i))

        # write output
        json.dump(results, open('{}_{}_bbox_results.json'.format(experiment_name, dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_{}_bbox_results.json'.format(experiment_name, dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metric_names = [
            'AP', 'AP40', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10',
            'AR100', 'ARs', 'ARm', 'ARl', 'p_AP', 'p_AP40'
        ]
        eval_results = {
            metric_names[i]: coco_eval.stats[metric_names[i]]
            for i in range(len(metric_names))
        }

        model.train()

        return eval_results, g_aucs, l_aucs
