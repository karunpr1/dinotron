import torch
import re
import argparse


def convert_key(key: str) -> str or None:
    """
    Convert a single key from DINO ResNet50 naming to the Detectron2 R_50 backbone naming.
    - Removes common prefixes ("module.", "backbone.").
    - Maps stem keys:
         conv1.weight -> backbone.stem.conv1.weight
         bn1.* -> backbone.stem.conv1.norm.*
    - Maps residual layers:
         layerX.Y.convZ.weight   -> backbone.res{X+1}.{Y}.convZ.weight
         layerX.Y.bnZ.*          -> backbone.res{X+1}.{Y}.convZ.norm.*
    - Maps downsample:
         layerX.Y.downsample.0.weight -> backbone.res{X+1}.{Y}.shortcut.weight
         layerX.Y.downsample.1.*        -> backbone.res{X+1}.{Y}.shortcut.norm.*
    - Drops any num_batches_tracked keys.
    """
    # Remove common prefixes
    for prefix in ("module.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix):]

    # Stem mapping
    if key.startswith("conv1.weight"):
        return "backbone.stem.conv1.weight"
    if key.startswith("bn1."):
        bn_attr = key[len("bn1."):]
        if bn_attr.startswith("num_batches_tracked"):
            return None
        return f"backbone.stem.conv1.norm.{bn_attr}"

    # Residual layers mapping: layerX.Y.something
    m = re.match(r"layer(\d+)\.(\d+)\.(.+)", key)
    if m:
        layer_num = int(m.group(1))  # e.g. 1,2,3,4
        block_idx = m.group(2)  # e.g. "0", "1", etc.
        rest = m.group(3)  # e.g. "conv1.weight" or "bn2.running_mean"
        new_layer = layer_num + 1  # layer1 -> res2, etc.

        # Standard convolution layers
        if rest.startswith("conv"):
            return f"backbone.res{new_layer}.{block_idx}.{rest}"
        # BatchNorm layers
        elif rest.startswith("bn"):
            m_bn = re.match(r"bn(\d+)\.(.+)", rest)
            if m_bn:
                conv_num = m_bn.group(1)
                bn_attr = m_bn.group(2)
                if bn_attr.startswith("num_batches_tracked"):
                    return None
                return f"backbone.res{new_layer}.{block_idx}.conv{conv_num}.norm.{bn_attr}"
        # Downsample branch
        elif rest.startswith("downsample"):
            parts = rest.split(".")
            if parts[1] == "0":
                return f"backbone.res{new_layer}.{block_idx}.shortcut.weight"
            elif parts[1] == "1":
                if parts[2].startswith("num_batches_tracked"):
                    return None
                remainder = ".".join(parts[2:])
                return f"backbone.res{new_layer}.{block_idx}.shortcut.norm.{remainder}"

    # For keys that do not match any of the above patterns, return them unchanged.
    return key


def convert_state_dict(dino_state_dict: dict) -> dict:
    new_state_dict = {}
    for old_key, val in dino_state_dict.items():
        new_key = convert_key(old_key)
        if new_key is not None:
            new_state_dict[new_key] = val
    return new_state_dict


def main(input_ckpt: str, output_ckpt: str, ckpt_key: str):
    # Load the checkpoint on CPU.
    dino_ckpt = torch.load(input_ckpt, map_location="cpu")

    # Determine which key to use.
    if ckpt_key is not None:
        if ckpt_key in dino_ckpt:
            dino_state_dict = dino_ckpt[ckpt_key]
            print(f"Using checkpoint key: {ckpt_key}")
        else:
            print(f"Warning: '{ckpt_key}' not found in checkpoint. Falling back to auto-detection.")
            if "teacher" in dino_ckpt:
                dino_state_dict = dino_ckpt["teacher"]
                print("Using 'teacher' key from checkpoint.")
            elif "student" in dino_ckpt:
                dino_state_dict = dino_ckpt["student"]
                print("Using 'student' key from checkpoint.")
            else:
                dino_state_dict = dino_ckpt
    else:
        # Auto-detect key: prefer teacher, then student.
        if "teacher" in dino_ckpt:
            dino_state_dict = dino_ckpt["teacher"]
            print("Using 'teacher' key from checkpoint.")
        elif "student" in dino_ckpt:
            dino_state_dict = dino_ckpt["student"]
            print("Using 'student' key from checkpoint.")
        else:
            dino_state_dict = dino_ckpt
            print("No specific key found; using the full checkpoint.")

    # Convert the state dict keys.
    adapted_sd = convert_state_dict(dino_state_dict)

    # Save the adapted state dict under the "model" key.
    res = {"model": adapted_sd, "__author__": "DINOTRON"}
    torch.save(res, output_ckpt)
    print(f"Converted weights saved to {output_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DINO pre-trained ResNet50 weights to Detectron2 naming convention."
    )
    parser.add_argument("--input_ckpt", "-i", required=True, help="Path to the DINO checkpoint.")
    parser.add_argument("--output_ckpt", "-o", required=True, help="Output path for the adapted checkpoint.")
    parser.add_argument("--ckpt_key", "-ck", default='teacher', help="Checkpoint key to use ('student' or 'teacher').")
    args = parser.parse_args()
    main(args.input_ckpt, args.output_ckpt, args.ckpt_key)
