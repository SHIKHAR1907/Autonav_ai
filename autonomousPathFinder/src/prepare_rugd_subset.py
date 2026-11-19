"""
Extract and prepare a curated 100-image subset from RUGD for fine-tuning SegFormer-B0.
Reconstructs ground-truth semantic segmentation masks from DatasetNinja annotations.
"""

import os
import io
import json
import zlib
import base64
import tarfile
import numpy as np
from PIL import Image

CLASS_MAPPING = {
    'background': 0, 'void': 0, 'other': 0, 'container': 0, 'pole': 0, 'bridge': 0,
    'dirt': 1, 'sand': 1, 'gravel': 1, 'path': 1, 'road': 1,
    'grass': 2,
    'tree': 3, 'bush': 3, 'vegetation': 3,
    'rock': 4, 'rock-bed': 4, 'stone': 4,
    'water': 5,
    'sky': 6,
    'vehicle': 0
}

TARGET_CLASSES = [
    "background",
    "dirt_path",
    "grass",
    "vegetation",
    "rock",
    "water",
    "sky"
]

def prepare_subset(tar_path, output_dir, max_samples=100):
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"Reading index from {tar_path}...")
    with tarfile.open(tar_path, "r") as tar:
        ann_members = {}
        img_members = {}
        for m in tar:
            if "/ann/" in m.name and m.name.endswith(".png.json"):
                base_name = os.path.basename(m.name).replace(".png.json", "")
                ann_members[base_name] = m
            elif "/img/" in m.name and m.name.endswith(".png"):
                base_name = os.path.basename(m.name).replace(".png", "")
                img_members[base_name] = m
                
        common_keys = sorted(list(set(ann_members.keys()).intersection(set(img_members.keys()))))
        print(f"Found {len(common_keys)} matching image-annotation pairs.")
        
        selected_keys = common_keys[:max_samples]
        print(f"Extracting and reconstructing masks for {len(selected_keys)} samples...")
        
        count = 0
        for key in selected_keys:
            img_m = img_members[key]
            img_bytes = tar.extractfile(img_m).read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_w, img_h = img.size
            
            ann_m = ann_members[key]
            ann_json = json.loads(tar.extractfile(ann_m).read())
            
            seg_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            
            for obj in ann_json.get("objects", []):
                cls_title = obj.get("classTitle", "").lower()
                cls_id = CLASS_MAPPING.get(cls_title, 0)
                
                if cls_id > 0 and "bitmap" in obj and obj["bitmap"]:
                    origin = obj["bitmap"]["origin"]
                    col_x, row_y = origin[0], origin[1]
                    decomp = zlib.decompress(base64.b64decode(obj["bitmap"]["data"]))
                    patch = np.array(Image.open(io.BytesIO(decomp)))
                    if patch.ndim == 3:
                        patch = patch[:, :, 0]
                    h, w = patch.shape
                    
                    if col_x < img_w and row_y < img_h:
                        max_w = min(w, img_w - col_x)
                        max_h = min(h, img_h - row_y)
                        patch_slice = patch[:max_h, :max_w]
                        current_slice = seg_mask[row_y:row_y+max_h, col_x:col_x+max_w]
                        seg_mask[row_y:row_y+max_h, col_x:col_x+max_w] = np.where(
                            patch_slice > 0,
                            cls_id,
                            current_slice
                        )
                        
            out_img_path = os.path.join(img_dir, f"{key}.jpg")
            out_mask_path = os.path.join(mask_dir, f"{key}.png")
            
            img.save(out_img_path, quality=92)
            Image.fromarray(seg_mask).save(out_mask_path)
            
            count += 1
            if count % 20 == 0:
                print(f"Processed {count}/{len(selected_keys)} pairs (Sample unique classes: {np.unique(seg_mask)})...")
                
    print(f"Successfully prepared {count} paired training samples in {output_dir}")

if __name__ == "__main__":
    tar_file = "d:/5th semester/opencv/dataset/rugd-DatasetNinja.tar"
    dest = "d:/5th semester/opencv/autonomousPathFinder/datasets/RUGD_train"
    prepare_subset(tar_file, dest, max_samples=100)
