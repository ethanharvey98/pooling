import numpy as np

ann = '/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14/Thumos14reduced-Annotations'
segments = np.load(f'{ann}/segments.npy', allow_pickle=True, encoding='latin1')
seg_labels = np.load(f'{ann}/labels.npy', allow_pickle=True, encoding='latin1')
subsets = np.load(f'{ann}/subset.npy', allow_pickle=True, encoding='latin1')
classlist = [c.decode('latin1') for c in np.load(f'{ann}/classlist.npy', allow_pickle=True, encoding='latin1')]
duration = np.load(f'{ann}/duration.npy', allow_pickle=True, encoding='latin1')

decode = lambda v: v.decode('latin1') if isinstance(v, bytes) else str(v)

print(f"{'Class':<22s} {'Videos':>6s} {'MeanSegs':>9s} {'Med':>5s} {'Max':>5s} {'1seg':>5s} {'2+':>5s} {'MeanDur(s)':>10s}")
print('-' * 75)

for cname in classlist:
    seg_counts = []
    seg_durations = []
    for i, s in enumerate(subsets):
        if decode(s) not in ('validation', 'test'):
            continue
        class_segs = [(seg, lab) for seg, lab in zip(segments[i], seg_labels[i]) if decode(lab) == cname]
        if class_segs:
            seg_counts.append(len(class_segs))
            for seg, _ in class_segs:
                seg_durations.append(seg[1] - seg[0])

    if not seg_counts:
        continue

    seg_counts = np.array(seg_counts)
    mean_dur = np.mean(seg_durations)
    print(f"{cname:<22s} {len(seg_counts):>6d} {seg_counts.mean():>9.1f} {np.median(seg_counts):>5.0f} "
          f"{seg_counts.max():>5d} {(seg_counts == 1).sum():>5d} {(seg_counts >= 2).sum():>5d} {mean_dur:>10.1f}")
