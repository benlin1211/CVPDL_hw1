# Please see 
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
# https://blog.csdn.net/llh_1178/article/details/114528795.
# # https://github.com/ultralytics/yolov3/issues/738
# # https://blog.csdn.net/IYXUAN/article/details/124339385
import json
from collections import defaultdict
import argparse
import os
import glob
from tqdm import tqdm
import numpy as np

def convert_coco_json(json_dir, prefix, save_dir):
     # output directory
    os.makedirs(save_dir, exist_ok=True) 

    # copy image
    image_dir = os.path.join(save_dir, 'images', prefix)
    os.makedirs(image_dir, exist_ok=True)
    for image in sorted(glob.glob(os.path.join(json_dir, '*.jpg'))):
        # print(image)
        os.system(f'cp {image} {image_dir}')

    # Import json
    save_json_dir = os.path.join(save_dir, 'labels', prefix)
    os.makedirs(save_json_dir, exist_ok=True)
    print(save_json_dir)

    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):

        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
            img = images['%g' % img_id]
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            for ann in anns:
                if ann['iscrowd']:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                #cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class
                cls = ann['category_id']
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            # print(os.path.join(save_json_dir, f"{f}.txt"))
            txt_name, _ = os.path.splitext(f)
            # print(txt_name)
            with open(os.path.join(save_json_dir, f"{txt_name}.txt"), 'a') as file:
                # print("Write")
                for i in range(len(bboxes)):
                    line = *(bboxes[i]),  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--coco_path', default="") #../../hw1_dataset
    parser.add_argument('--save_dir', default="") #../datasets/hw1_dataset_yolo_train
    # parser.add_argument('--save_dir_train', default="") #../datasets/hw1_dataset_yolo_train
    # parser.add_argument('--save_dir_valid', default="") #../datasets/hw1_dataset_yolo_valid
    args = parser.parse_args()

    train_json = os.path.join(args.coco_path, "train")
    val_json = os.path.join(args.coco_path, "valid")
    args = parser.parse_args()

    convert_coco_json(train_json,  # directory with *.json
                      'train',
                      args.save_dir)
    convert_coco_json(val_json,  # directory with *.json
                      'val',
                      args.save_dir)

    # zip results
    # os.system('zip -r ../coco.zip ../coco')
    print("done")
