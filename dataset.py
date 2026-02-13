import os
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from PIL import Image


class Luna16Dataset(Dataset):
    def __init__(self, root_dir, csv_path, transform=None, balance=True):

        self.root_dir = root_dir
        self.transform = transform

        full_df = pd.read_csv(csv_path)

        # ------------------------------
        # 1️⃣ Balanced Sampling
        # ------------------------------
        if balance:
            positives = full_df[full_df["class"] == 1]
            negatives = full_df[full_df["class"] == 0]

            # Sample negatives (2x positives)
            negatives = negatives.sample(
                n=len(positives) * 2,
                random_state=42
            )

            self.df = pd.concat([positives, negatives])
            self.df = self.df.sample(frac=1).reset_index(drop=True)

            print(f"Balanced dataset: {len(positives)} positives, {len(negatives)} negatives")

        else:
            self.df = full_df
            print(f"Full dataset used: {len(self.df)} samples")

        # ------------------------------
        # 2️⃣ Index all .mhd paths
        # ------------------------------
        self.mhd_paths = {}

        mhd_files = glob.glob(
            os.path.join(root_dir, "subset*", "*.mhd")
        )

        for path in mhd_files:
            uid = os.path.basename(path)[:-4]
            self.mhd_paths[uid] = path

        print("Indexed scans:", len(self.mhd_paths))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        uid = row["seriesuid"]
        label = row["class"]

        if uid not in self.mhd_paths:
            raise ValueError(f"MHD file not found for UID: {uid}")

        mhd_path = self.mhd_paths[uid]

        # ------------------------------
        # 3️⃣ Load CT + Metadata
        # ------------------------------
        itk_img = sitk.ReadImage(mhd_path)

        origin = np.array(itk_img.GetOrigin())      # (x, y, z)
        spacing = np.array(itk_img.GetSpacing())    # (x, y, z)

        volume = sitk.GetArrayFromImage(itk_img)    # (z, y, x)

        # ------------------------------
        # 4️⃣ World → Voxel Mapping
        # Formula: (world - origin) / spacing
        # ------------------------------
        world_coords = np.array([
            row["coordX"],
            row["coordY"],
            row["coordZ"]
        ])

        voxel_coords = np.rint(
            (world_coords - origin) / spacing
        ).astype(int)

        z_index = np.clip(
            voxel_coords[2],
            0,
            volume.shape[0] - 1
        )

        slice_2d = volume[z_index]

        # ------------------------------
        # 5️⃣ Lung Windowing [-1000, 400]
        # ------------------------------
        slice_2d = np.clip(slice_2d, -1000, 400)

        # ------------------------------
        # 6️⃣ Global Normalization
        # ------------------------------
        slice_2d = (slice_2d + 1000) / 1400.0
        slice_2d = (slice_2d * 255).astype(np.uint8)

        # ------------------------------
        # 7️⃣ Convert to RGB
        # ------------------------------
        slice_rgb = Image.fromarray(slice_2d).convert("RGB")

        if self.transform:
            slice_tensor = self.transform(slice_rgb)
        else:
            slice_tensor = torch.tensor(slice_2d).unsqueeze(0).float()

        return slice_tensor, torch.tensor(label, dtype=torch.long)
