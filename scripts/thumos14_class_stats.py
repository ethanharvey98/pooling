import numpy as np

ann = '/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14/Thumos14reduced-Annotations'
labels_all = np.load(f'{ann}/labels_all.npy', allow_pickle=True, encoding='latin1')
subsets = np.load(f'{ann}/subset.npy', allow_pickle=True, encoding='latin1')
classlist = [c.decode('latin1') for c in np.load(f'{ann}/classlist.npy', allow_pickle=True, encoding='latin1')]
subsets = [s.decode('latin1') for s in subsets]

for split in ['validation', 'test']:
    mask = [s == split for s in subsets]
    indices = [i for i, m in enumerate(mask) if m]
    has_action = sum(1 for i in indices if len(labels_all[i]) > 0)
    no_action = sum(1 for i in indices if len(labels_all[i]) == 0)
    print(f'\n=== {split} ({len(indices)} videos) ===')
    print(f'Positive (any action): {has_action}')
    print(f'Negative (no action):  {no_action}')
    print(f'\nPer-class counts:')
    counts = {c: 0 for c in classlist}
    for i in indices:
        for label in labels_all[i]:
            l = label.decode('latin1') if isinstance(label, bytes) else label
            if l in counts:
                counts[l] += 1
    for c in sorted(counts, key=counts.get, reverse=True):
        print(f'  {c:20s}: {counts[c]:4d} videos')
