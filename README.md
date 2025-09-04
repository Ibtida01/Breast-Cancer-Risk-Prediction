# Breast-Cancer-Risk-Prediction

## Install Dependencies
`!pip install timm pydicom pycox lifelines`

## Initialize
```
class BreastCancerDataset(Dataset):
    def __init__(self, meta_df, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Only keep rows where file exists in Batch_1/Batch_2
        valid_rows = []
        for i, row in meta_df.iterrows():
            filename = row["anon_filename"]
            found = False
            for batch in ["Batch_1", "Batch_2"]:
                candidate = os.path.join(root_dir, batch, batch, filename)
                if os.path.exists(candidate):
                    row["dcm_path"] = candidate
                    valid_rows.append(row)
                    found = True
                    break
            # skip if not found

        # keep only valid entries
        self.meta_df = pd.DataFrame(valid_rows).reset_index(drop=True)

        # normalize metadata features
        self.meta_df[METADATA_FEATURES] = (
            self.meta_df[METADATA_FEATURES]
            .apply(pd.to_numeric, errors="coerce")     # force numeric, convert bad values to NaN
            .fillna(0)                                # replace NaN with 0
        )
        
        # then standardize
        self.meta_df[METADATA_FEATURES] = (
            self.meta_df[METADATA_FEATURES] - self.meta_df[METADATA_FEATURES].mean()
        ) / (self.meta_df[METADATA_FEATURES].std() + 1e-6)

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        dcm_path = row["dcm_path"]

        # read dicom
        dicom = pydicom.dcmread(dcm_path)
        image = dicom.pixel_array.astype(np.float32)

        # normalize [0,1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-5)

        # convert grayscale → 3-channel
        image = np.stack([image, image, image], axis=-1)

        if self.transform:
            image = self.transform(image)

        # metadata tensor
        meta_features = torch.tensor(row[METADATA_FEATURES].astype(float).values, dtype=torch.float32)

        # label
        label = torch.tensor(row["x_case"], dtype=torch.float32)

        return image, meta_features, label

# confirm skipping missing files and load
dataset = BreastCancerDataset(
    meta_df=meta,
    root_dir=DATA_ROOT,
    transform=transform
)

print("✅ Dataset size after skipping missing files:", len(dataset))

# Test a batch
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
images, metas, labels = next(iter(dataloader))
print("Image batch shape:", images.shape)
print("Metadata batch shape:", metas.shape)
print("Labels:", labels[:5])

```
