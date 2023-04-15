import json
import random
import argparse

def convert_and_save(file_contents, out_file):

    results = {}
    for pred in file_contents:
        #print(pred)
        #print(pred['image_id'])
        fname = pred['image_id'] + '.jpg'

        if fname not in results.keys():
            res = {} 
            res['boxes'] = [pred['bbox']] # ??
            res['labels'] = [pred['category_id']] # ??
            res['scores'] = [pred['score']] # ??
            results[fname] = res
        else:
            results[fname]['boxes'].append(pred['bbox']) # ??
            results[fname]['labels'].append(pred['category_id']) # ??
            results[fname]['scores'].append(pred['score']) # ??

    # for res in results.items():
    #     print(res)
    with open(out_file, 'w') as fp:
        json.dump(results, fp)
    print("over")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert submission format', add_help=False)
    parser.add_argument('--json_file',default="./runs/detect/val/predictions.json", type=str)
    parser.add_argument('--out_file', default="./pred.json", type=str)
    args = parser.parse_args()

    with open(args.json_file) as f:
        file_contents = json.load(f)

    convert_and_save(file_contents, args.out_file)
    
    #print(results.keys())

