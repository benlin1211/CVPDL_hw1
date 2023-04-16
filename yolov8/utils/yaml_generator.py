import json
import os
import yaml
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--coco_path', default="") #../../hw1_dataset/train/_annotations.coco.json
    parser.add_argument('--data_path', default="") #./hw1_dataset_yolo
    parser.add_argument('--out_path', default="") #../hw1_dataset_yolo.yaml
    parser.add_argument('--num_classes', default=8, type=int) #../hw1_dataset_yolo.yaml
    # parser.add_argument('--freeze', default=50, type=int) 
    args = parser.parse_args()

    result = {}
    root = os.getcwd()
    result['path'] = os.path.join(root, args.data_path) # dataset 路径
    result['train'] = 'images/train' #os.path.join( args.data_path,'images/train')  # 相对 path 的路径
    result['val'] = 'images/val' # same as train.
    result['test'] = 'images/test' # None
    result['nc'] = args.num_classes
    result['freeze'] = args.freeze
    result['names'] = {}

    print(args.coco_path)
    json_file = os.path.join(args.coco_path)
    
    with open(json_file) as f:
        data = json.load(f)
    
    for category in data["categories"]:
        # print(category["id"])
        # print(category["name"])
        id = category["id"]
        name = category["name"]
        result['names'][id] = name

    with open(args.out_path,"w") as file:
        yaml.dump(result, file, sort_keys=False)

    # Classes
    # print(result['names'])
    print("done")
