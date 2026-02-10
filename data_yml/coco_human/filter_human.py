import os
from tqdm import tqdm

# note: copy the orignal first if u want

labels_dir = "/mnt/d/DATASETS/coco/labels/train2017"  # path to labels/train or labels/val

def filter_dir(labels_dir):
    for root, dirs, files in os.walk(labels_dir):
        for f in tqdm(files, total=len(files), desc=f"Processing dir: {labels_dir}"):
            if f.endswith(".txt"):
                path = os.path.join(root, f)
                with open(path, "r") as file:
                    lines = file.readlines()

                # keep only class 0 (person)
                filtered = [line for line in lines if line.startswith("0 ")]

                # overwrite file
                with open(path, "w") as file:
                    file.writelines(filtered)

filter_dir("/mnt/d/DATASETS/coco/labels/train2017")
filter_dir("/mnt/d/DATASETS/coco/labels/val2017")