import random
from collections import defaultdict
import json
from pathlib import Path
import shutil
from tqdm import tqdm

print('Loading original full json data file...')
with open('/mnt/d/DATASETS/text_img_pairs/final_flickr_separateGT_train_segm.json', 'r') as f:
    data = json.load(f)
    
# group images by original_img_id
groups = defaultdict(list)
for d in data['images']:
    groups[d["original_img_id"]].append(d)

# sample 100 unique original_img_id
sampled_ids = random.sample(list(groups.keys()), 100)

# collect images with those original_img_id
sampled_images = [d for k in sampled_ids for d in groups[k]]

# get corresponding image_ids
image_ids = {d["id"] for d in sampled_images}

# collect annotations that match any of those image_ids
sampled_annotations = [a for a in data['annotations'] if a["image_id"] in image_ids]

### OUTPUT ###

# Create new_data dict for json
new_data = data.copy()
new_data['images'] = sampled_images
new_data['annotations'] = sampled_annotations
print(f"images: {len(new_data['images'])}")
print(f"annotations: {len(new_data['annotations'])}")

# Prepare directories
sampled_data_dir = Path('/mnt/d/DATASETS/text_img_pairs/sampled_data')
img_source_path = Path('/mnt/d/DATASETS/text_img_pairs/flickr30k-images')
(sampled_data_dir / "annotations").mkdir(exist_ok=True, parents=True)
(sampled_data_dir / "full_images").mkdir(exist_ok=True, parents=True)

# (1) Save new json data file
with open(sampled_data_dir / "annotations/final_flickr_separateGT_train_segm.json", 'w') as f:
    json.dump(new_data, f, indent=4)

# (2) Extract file names to copy paste
file_names = [d["file_name"] for d in sampled_images]
for file_name in tqdm(file_names, desc='copying images'):
    shutil.copy(img_source_path / file_name, sampled_data_dir / "full_images" / file_name)
    

# from ultralytics.data.split import autosplit

# # (3) Automatically split the dataset
# autosplit(
#     path= sampled_data_dir / "full_images", # Directory containing all your image files
#     weights=(0.8, 0.1, 0.1),     # The desired split ratio (Train, Val, Test)
#     annotated_only=False          # Only use images that have a corresponding label file
# )