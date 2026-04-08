#!/usr/bin/env python3
"""
Zero-shot evaluation of QoQ-Med-VL-7B on OASIS-3 Alzheimer's detection.

For each subject, presents every axial slice and asks "Does this brain MRI
slice show signs of Alzheimer's disease? Answer yes or no."

Bag-level prediction: positive if ANY slice is predicted positive (MIL assumption).

Usage:
    python evaluate_qoq_med_oasis3.py \
        --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy'
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score


PROMPT = "Does this brain MRI slice show signs of Alzheimer's disease? Answer yes or no."


def load_model(model_name):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def slice_to_pil(volume, idx):
    """Extract one axial slice from (H, W, D) volume as an RGB PIL image."""
    s = np.rot90(volume[:, :, idx], k=1)
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        s = ((s - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        s = np.zeros_like(s, dtype=np.uint8)
    return Image.fromarray(s, mode='L').convert('RGB')


def ask_slice(model, processor, image):
    """Ask the model about one slice. Returns 'yes', 'no', or 'unparseable'."""
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    response = processor.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip().lower()

    if 'yes' in response:
        return 'yes'
    elif 'no' in response:
        return 'no'
    return 'unparseable'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numpy_dir', required=True, type=str)
    parser.add_argument('--output_dir', default='./qoq_med_oasis3_results', type=str)
    parser.add_argument('--model_name', default='ddvd233/QoQ-Med-VL-7B', type=str)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'predictions.jsonl')

    labels_df = pd.read_csv(f'{args.numpy_dir}/labels.csv')
    labels_df = labels_df[labels_df['path'].apply(os.path.exists)]
    print(f"Subjects: {len(labels_df)}")

    completed = set()
    if args.resume and os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                completed.add(json.loads(line)['mr_id'])
        print(f"Resuming: {len(completed)} already done")

    model, processor = load_model(args.model_name)

    with open(results_file, 'a') as f:
        for i, (_, row) in enumerate(labels_df.iterrows()):
            mr_id = row.get('MR ID', str(i))
            if mr_id in completed:
                continue

            arr = np.load(row['path'])['arr_0']
            vol = arr[0] if arr.ndim == 4 else arr  # T1w channel, (H, W, D)
            n_slices = vol.shape[-1]

            slice_responses = []
            for s in range(n_slices):
                img = slice_to_pil(vol, s)
                answer = ask_slice(model, processor, img)
                slice_responses.append(answer)

            # MIL: positive if any slice says yes
            bag_pred = 1 if 'yes' in slice_responses else 0

            entry = {
                'mr_id': mr_id,
                'ground_truth': int(row["Alzheimer's"]),
                'bag_prediction': bag_pred,
                'slice_responses': slice_responses,
            }
            f.write(json.dumps(entry) + '\n')
            f.flush()

            yes_count = slice_responses.count('yes')
            print(f"[{i+1}/{len(labels_df)}] {mr_id}: gt={row['Alzheimer\\'s']}, pred={bag_pred}, yes={yes_count}/{n_slices}")

    # --- Metrics ---
    entries = []
    with open(results_file) as f:
        for line in f:
            entries.append(json.loads(line))

    y_true = np.array([e['ground_truth'] for e in entries])
    y_pred = np.array([e['bag_prediction'] for e in entries])

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = np.mean(y_true == y_pred)

    print(f"\n{'='*60}")
    print(f"Subjects:         {len(entries)}")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Balanced Acc:     {bal_acc:.4f}")
    print(f"GT distribution:  {dict(zip(*np.unique(y_true, return_counts=True)))}")
    print(f"Pred distribution: {dict(zip(*np.unique(y_pred, return_counts=True)))}")

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({'accuracy': float(acc), 'balanced_accuracy': float(bal_acc),
                   'n_subjects': len(entries)}, f, indent=2)


if __name__ == "__main__":
    main()
