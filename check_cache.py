import json
import pickle
import torch
import numpy as np

index_path = 'dataset/cache/pseudo_label_cache/TR-CAMO+TR-COD10K/index.json'
try:
    with open(index_path, 'r') as f:
        idx_map = json.load(f)
    print(f"Found {len(idx_map)} items in pseudo label cache index.")
    
    # Check first 5 items
    for i in range(5):
        file_path = f'dataset/cache/pseudo_label_cache/TR-CAMO+TR-COD10K/{idx_map[str(i)]}'
        with open(file_path, 'rb') as f:
            mask = pickle.load(f)
            
        print(f"Item {i}: shape {mask.shape}, max {mask.max()}, min {mask.min()}, sum {mask.sum()}, dtype {mask.dtype}")
except Exception as e:
    print(f"Error checking cache: {e}")
