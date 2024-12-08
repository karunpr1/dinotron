#!/usr/bin/env python

import pickle as pkl
import sys

import torch


def adapt_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("backbone.", "").replace("module.", "")

        # key replacements for Detectron2 expected model keys
        if "layer" not in new_key:
            new_key = "backbone.stem." + new_key
        for t in [1, 2, 3, 4]:
            new_key = new_key.replace(f"layer{t}", f"backbone.res{t + 1}")
        for t in [1, 2, 3]:
            new_key = new_key.replace(f"bn{t}", f"conv{t}.norm")
        new_key = new_key.replace("downsample.0", "shortcut")
        new_key = new_key.replace("downsample.1", "shortcut.norm")

        # Add the adapted key-value pair to the new state dict
        new_state_dict[new_key] = v.detach().cpu()

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