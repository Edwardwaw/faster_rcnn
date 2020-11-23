import os
import json
import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
import math
import yaml
from commons.augmentations import ScaleMinMax
from nets.faster_rcnn import FasterRCNN

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]




def write_coco_json():
    from pycocotools.coco import COCO
    img_root = "/home/wangchao/public_dataset/coco/images/val2017"
    with open('configs/faster_rcnn_coco.yml', 'r') as rf:
        cfg = yaml.safe_load(rf)
    model_cfg = cfg['model']
    model = FasterRCNN(cfg=model_cfg)
    weights = torch.load("weights/faster_rcnn_coco_best_map.pth")['ema']
    model.load_state_dict(weights)
    model.cuda().eval()

    data_cfg = cfg['data']
    basic_transform = ScaleMinMax(min_thresh=data_cfg['min_thresh'], max_thresh=data_cfg['max_thresh'])
    coco = COCO("/home/wangchao/public_dataset/coco/annotations/instances_val2017.json")

    coco_predict_list = list()
    for img_id in tqdm(coco.imgs.keys()):
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(img_root, file_name)
        img = cv.imread(img_path)
        # ori_img = img.copy()
        img, ratio = basic_transform.scale_img(img)
        h,w=img.shape[:2]
        max_h = int(math.ceil(h / 32) * 32)
        max_w = int(math.ceil(w / 32) * 32)
        dw, dh = max_w - w, max_h - h

        dw /= 2
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(103, 116, 123))
        img_out = img[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        img_out = ((img_out - np.array(rgb_mean)) / np.array(rgb_std)).transpose(2, 0, 1).astype(np.float32)
        img_out = torch.from_numpy(np.ascontiguousarray(img_out)).unsqueeze(0).float().cuda()
        predicts, _ = model(img_out)   # predicts (list,len=bs):  list(dets)  dets.shape=[num_box,6]  6==>x1,y1,x2,y2,score,label
        if predicts[0] is None:
            continue
        box = torch.cat(predicts)

        box[:, [0, 2]] = (box[:, [0, 2]]-left) / ratio
        box[:, [1, 3]] = (box[:, [1, 3]]-top) / ratio
        box = box.detach().cpu().numpy()
        # ret_img = draw_box(ori_img, box[:, [4, 5, 0, 1, 2, 3]], colors=coco_colors)
        # cv.imwrite(file_name, ret_img)
        coco_box = box[:, :4]
        coco_box[:, 2:] = coco_box[:, 2:] - coco_box[:, :2]
        for p, b in zip(box.tolist(), coco_box.tolist()):
            coco_predict_list.append({'image_id': img_id,
                                      'category_id': coco_ids[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})
    with open("predicts.json", 'w') as file:
        json.dump(coco_predict_list, file)


def coco_eval():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cocoGt = COCO("/home/wangchao/public_dataset/coco/annotations/instances_val2017.json")  # initialize COCO ground truth api

    cocoDt = cocoGt.loadRes("predicts.json")  # initialize COCO pred api
    imgIds = [img_id for img_id in cocoGt.imgs.keys()]
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds  # image IDs to evaluate
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def coco_val_eval():
    write_coco_json()
    coco_eval()


if __name__ == '__main__':
    coco_val_eval()