import numpy as np

ann = '/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14/Thumos14reduced-Annotations'
labels_all = np.load(f'{ann}/labels_all.npy', allow_pickle=True, encoding='latin1')
subsets = np.load(f'{ann}/subset.npy', allow_pickle=True, encoding='latin1')
videonames = np.load(f'{ann}/videoname.npy', allow_pickle=True, encoding='latin1')

decode = lambda v: v.decode('latin1') if isinstance(v, bytes) else str(v)

for split in ['validation', 'test']:
    multi = []
    single = 0
    for i, s in enumerate(subsets):
        if decode(s) != split:
            continue
        classes = [decode(l) for l in labels_all[i]]
        if len(classes) > 1:
            multi.append((decode(videonames[i]), classes))
        else:
            single += 1

    print(f'\n=== {split} ===')
    print(f'Single-class videos: {single}')
    print(f'Multi-class videos:  {len(multi)}')
    if multi:
        print(f'\nMulti-class examples:')
        for name, classes in multi[:10]:
            print(f'  {name}: {classes}')
