"""
Analyze what THUMOS14 looks like if we crop each video to include
only snippets up to the end of the 2nd positive segment.

For videos with 1 segment: keep everything up to end of that segment
For videos with 2+ segments: keep everything up to end of 2nd segment
"""

import numpy as np

ann = '/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14/Thumos14reduced-Annotations'
segments = np.load(f'{ann}/segments.npy', allow_pickle=True, encoding='latin1')
seg_labels = np.load(f'{ann}/labels.npy', allow_pickle=True, encoding='latin1')
subsets = np.load(f'{ann}/subset.npy', allow_pickle=True, encoding='latin1')
classlist = [c.decode('latin1') for c in np.load(f'{ann}/classlist.npy', allow_pickle=True, encoding='latin1')]
duration = np.load(f'{ann}/duration.npy', allow_pickle=True, encoding='latin1')
features = np.load(
    '/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14/Thumos14reduced-I3D-JOINTFeatures.npy',
    allow_pickle=True, encoding='latin1'
)

decode = lambda v: v.decode('latin1') if isinstance(v, bytes) else str(v)

print(f"{'Class':<22s} {'Videos':>6s} {'OrigLen':>8s} {'CropLen':>8s} {'Reduction':>10s} {'Segs<=2':>7s} {'PosSnip':>8s} {'TotSnip':>8s} {'PosRate':>8s}")
print('-' * 95)

for cname in classlist:
    orig_lengths = []
    crop_lengths = []
    pos_snippets = []
    total_snippets = []

    for i, s in enumerate(subsets):
        if decode(s) not in ('validation', 'test'):
            continue

        T = features[i].shape[0]
        vid_dur = duration[i].item()
        snippet_dur = vid_dur / T

        # Get segments for this class in this video, sorted by start time
        class_segs = sorted(
            [seg for seg, lab in zip(segments[i], seg_labels[i]) if decode(lab) == cname],
            key=lambda x: x[0]
        )

        if not class_segs:
            continue

        orig_lengths.append(T)

        # Crop point: end of 2nd segment (or 1st if only one)
        n_keep = min(2, len(class_segs))
        crop_end_time = class_segs[n_keep - 1][1]

        # How many snippets to keep
        crop_T = int(np.ceil(crop_end_time / snippet_dur))
        crop_T = min(crop_T, T)
        crop_lengths.append(crop_T)

        # Count positive snippets in cropped region
        snippet_centers = np.array([(j + 0.5) * snippet_dur for j in range(crop_T)])
        labels = np.zeros(crop_T, dtype=int)
        for seg in class_segs[:n_keep]:
            mask = (snippet_centers >= seg[0]) & (snippet_centers <= seg[1])
            labels[mask] = 1
        pos_snippets.append(labels.sum())
        total_snippets.append(crop_T)

    if not orig_lengths:
        continue

    orig_mean = np.mean(orig_lengths)
    crop_mean = np.mean(crop_lengths)
    reduction = 1 - crop_mean / orig_mean
    segs_le2 = sum(1 for cl in crop_lengths if cl == cl)  # all are <=2 segs by design
    pos_rate = np.sum(pos_snippets) / np.sum(total_snippets)

    print(f"{cname:<22s} {len(orig_lengths):>6d} {orig_mean:>8.0f} {crop_mean:>8.0f} {reduction:>9.0%} "
          f"{'all':>7s} {np.sum(pos_snippets):>8d} {np.sum(total_snippets):>8d} {pos_rate:>8.1%}")

print()
print("OrigLen = mean snippets per video (full), CropLen = mean snippets after cropping")
print("Segs<=2 = all (by design), PosSnip = total positive snippets, PosRate = positive snippet rate")
