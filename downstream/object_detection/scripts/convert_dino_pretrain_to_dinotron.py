#!/usr/bin/env python

import pickle as pkl
import sys

import torch
import re

def adapt_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove SSL-specific components
        if any(x in k for x in ['head', 'mlp', 'projection']):
            continue

        new_key = k.replace("backbone.", "").replace("module.", "")

        # Convert BatchNorm tracking parameters
        if 'num_batches_tracked' in new_key:
            new_key = re.sub(r'\.num_batches_tracked', '', new_key)

        # Map layer blocks to Detectron2's architecture
        layer_map = {
            r'^stem\.': 'backbone.stem.',
            r'layer([1-4])': lambda m: f'backbone.res{int(m.group(1)) + 1}',
            r'bn([1-3])': r'conv\1.norm',
            r'downsample\.0': 'shortcut',
            r'downsample\.1': 'shortcut.norm'
        }

        for pattern, replacement in layer_map.items():
            new_key = re.sub(pattern, replacement, new_key)

        # Skip invalid FPN mapping attempts
        if 'fpn_lateral' not in new_key and 'backbone.' not in new_key:
            new_key = 'backbone.' + new_key

        new_state_dict[new_key] = v

    return new_state_dict


if __name__ == "__main__":
    filepath = sys.argv[1]
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
    else:
        raise KeyError("The checkpoint does not contain a 'student' key.")
    adapted_state_dict = adapt_state_dict(state_dict)

    res = {"model": adapted_state_dict, "__author__": "DINOTRON", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)

    print(f"DINOTRON Model saved as {sys.argv[2]}")