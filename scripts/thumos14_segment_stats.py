"""
Analyze temporal segment structure per video in THUMOS14.
Shows how many disjoint action segments each video has (stop-and-start patterns).

Usage:
    python scripts/thumos14_segment_stats.py
"""

import numpy as np

ann = '/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14/Thumos14reduced-Annotations'
segments = np.load(f'{ann}/segments.npy', allow_pickle=True, encoding='latin1')
seg_labels = np.load(f'{ann}/labels.npy', allow_pickle=True, encoding='latin1')
subsets = np.load(f'{ann}/subset.npy', allow_pickle=True, encoding='latin1')
videonames = np.load(f'{ann}/videoname.npy', allow_pickle=True, encoding='latin1')
duration = np.load(f'{ann}/duration.npy', allow_pickle=True, encoding='latin1')
features = np.load(
    '/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14/Thumos14reduced-I3D-JOINTFeatures.npy',
    allow_pickle=True, encoding='latin1'
)

decode = lambda v: v.decode('latin1') if isinstance(v, bytes) else str(v)

for split in ['validation', 'test']:
    print(f'\n=== {split} ===')
    seg_counts = []
    multi_segment_videos = []

    for i, s in enumerate(subsets):
        if decode(s) != split:
            continue

        vid_segs = segments[i]
        vid_labels = seg_labels[i]
        n_segs = len(vid_segs)
        seg_counts.append(n_segs)

        if n_segs > 1:
            T = features[i].shape[0]
            vid_dur = duration[i].item()
            # Group by class
            class_segs = {}
            for seg, lab in zip(vid_segs, vid_labels):
                l = decode(lab)
                if l not in class_segs:
                    class_segs[l] = []
                class_segs[l].append(seg)

            multi_segment_videos.append({
                'name': decode(videonames[i]),
                'n_segs': n_segs,
                'duration': vid_dur,
                'snippets': T,
                'classes': {c: len(s) for c, s in class_segs.items()},
            })

    seg_counts = np.array(seg_counts)
    print(f'  Videos: {len(seg_counts)}')
    print(f'  Segments per video: min={seg_counts.min()}, max={seg_counts.max()}, '
          f'mean={seg_counts.mean():.1f}, median={np.median(seg_counts):.0f}')
    print(f'  Videos with 1 segment:  {(seg_counts == 1).sum()}')
    print(f'  Videos with 2+ segments: {(seg_counts >= 2).sum()}')
    print(f'  Videos with 5+ segments: {(seg_counts >= 5).sum()}')
    print(f'  Videos with 10+ segments: {(seg_counts >= 10).sum()}')

    # Diving-specific (class 7)
    print(f'\n  --- Diving only ---')
    diving_seg_counts = []
    for i, s in enumerate(subsets):
        if decode(s) != split:
            continue
        diving_segs = [seg for seg, lab in zip(segments[i], seg_labels[i]) if decode(lab) == 'Diving']
        if diving_segs:
            diving_seg_counts.append(len(diving_segs))
    if diving_seg_counts:
        diving_seg_counts = np.array(diving_seg_counts)
        print(f'  Videos with Diving: {len(diving_seg_counts)}')
        print(f'  Diving segments per video: min={diving_seg_counts.min()}, max={diving_seg_counts.max()}, '
              f'mean={diving_seg_counts.mean():.1f}')
        print(f'  Videos with 1 Diving segment:  {(diving_seg_counts == 1).sum()}')
        print(f'  Videos with 2+ Diving segments: {(diving_seg_counts >= 2).sum()}')

    # Show some examples of multi-segment videos
    print(f'\n  Top 5 videos by segment count:')
    for v in sorted(multi_segment_videos, key=lambda x: x['n_segs'], reverse=True)[:5]:
        print(f"    {v['name']}: {v['n_segs']} segments, {v['duration']:.0f}s, {v['snippets']} snippets, classes={v['classes']}")
