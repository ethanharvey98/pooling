"""
Recreate rsna_ich_subset_1149_complete.csv

Requirements:
- HuggingFace dataset: torchmil/RSNA_ICH_MIL (scan IDs and splits)
- Kaggle RSNA ICH: stage_2_train.csv (labels) and stage_2_train/*.dcm (DICOM files)

How it works:
- scan_id = DICOM StudyInstanceUID
- slice_id = DICOM SOPInstanceUID (filename without .dcm)
- z_position = DICOM ImagePositionPatient[2] (slice ordering)
- labels = Kaggle stage_2_train.csv
"""
import pandas as pd
import pydicom
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# === CONFIGURE THESE PATHS ===
RSNA_DICOM_DIR = '/media/M2SSD/gen_models_data/stage_2_train'
RSNA_LABELS_CSV = '/media/M2SSD/gen_models_data/stage_2_train.csv'

# 1. Get scan IDs from HuggingFace
print("1. Loading scan IDs from HuggingFace...")
hf_path = hf_hub_download(repo_id="torchmil/RSNA_ICH_MIL", filename="dataset/splits.csv", repo_type="dataset")
hf_df = pd.read_csv(hf_path)

# Check for data leak (scans in both train and test)
train_scans = set(hf_df[hf_df['split'] == 'train']['bag_name'])
test_scans = set(hf_df[hf_df['split'] == 'test']['bag_name'])
leak = train_scans & test_scans
if leak:
    print(f"   WARNING: {len(leak)} scan(s) in both train and test: {leak}")

# For scans in both splits, assign to test
splits = hf_df.groupby('bag_name')['split'].apply(
    lambda x: 'test' if 'test' in x.values else x.iloc[0]
).reset_index()
splits.columns = ['scan_id', 'split']
scan_ids = set(splits['scan_id'])
print(f"   {len(scan_ids)} scans")

# 2. Parse Kaggle labels
print("2. Loading Kaggle labels...")
kaggle = pd.read_csv(RSNA_LABELS_CSV)
kaggle['slice_id'] = kaggle['ID'].str.rsplit('_', n=1).str[0]
kaggle['label_type'] = kaggle['ID'].str.rsplit('_', n=1).str[1]
kaggle = kaggle.drop_duplicates(subset=['slice_id', 'label_type'])
labels = kaggle.pivot(index='slice_id', columns='label_type', values='Label').reset_index()
labels.columns = ['slice_id', 'label_any', 'label_epidural', 'label_intraparenchymal',
                  'label_intraventricular', 'label_subarachnoid', 'label_subdural']
print(f"   {len(labels)} slices")

# 3. Read DICOM metadata (slow - reads all files to find which belong to our scans)
print("3. Reading DICOM metadata (this takes ~30 min)...")
rows = []
for dcm_path in tqdm(list(Path(RSNA_DICOM_DIR).glob("*.dcm"))):
    slice_id = dcm_path.stem
    try:
        dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        scan_id = str(dcm.StudyInstanceUID)
        if scan_id in scan_ids:
            rows.append({
                'slice_id': slice_id,
                'scan_id': scan_id,
                'z_position': float(dcm.ImagePositionPatient[2])
            })
    except:
        pass

# 4. Build final dataframe
print("4. Building dataframe...")
df = pd.DataFrame(rows)
df = df.merge(labels, on='slice_id')
df = df.merge(splits, on='scan_id')

# Sort by scan_id, then descending z_position (superior to inferior)
df = df.sort_values(['scan_id', 'z_position'], ascending=[True, False])
df['slice_order'] = df.groupby('scan_id').cumcount() + 1
df['total_slices'] = df.groupby('scan_id')['scan_id'].transform('count')
df['scan_label'] = df.groupby('scan_id')['label_any'].transform('max')
df['instance_number'] = ''

df = df[['scan_id', 'slice_id', 'split', 'scan_label', 'slice_order', 'total_slices',
         'z_position', 'instance_number', 'label_any', 'label_epidural',
         'label_intraparenchymal', 'label_intraventricular', 'label_subarachnoid', 'label_subdural']]

df.to_csv('rsna_ich_subset_1149_complete_recreated.csv', index=False)
print(f"\nDone: {df['scan_id'].nunique()} scans, {len(df)} slices")
