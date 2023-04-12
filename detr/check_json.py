import json
import random
import argparse

parser = argparse.ArgumentParser('Check json format', add_help=False)
parser.add_argument('json_file', type=str)
args = parser.parse_args()

with open(args.json_file) as f:
    file_contents = json.load(f)

file_name = (random.choice(list(file_contents.keys())) )  


boxes = file_contents[file_name]['boxes']
# print(boxes)
labels = file_contents[file_name]['labels']
# print(labels)
scores = file_contents[file_name]['scores']
# print(scores)


print(file_name)
print(len(boxes))
print(len(labels))
print(len(scores))