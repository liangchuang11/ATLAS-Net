import os
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

class MRI_Dataset(Dataset):
    def __init__(self, excel_file, fmri_root, smri_root):
        self.df = pd.read_excel(excel_file)
        self.tabular_cols = [
            'gen', 'Weight', 'Age', 'FDG', 'AV45','ABETA','TAU','PTAU'
        ]
        self.df.dropna(subset=self.tabular_cols, inplace=True)

        scaler = StandardScaler()
        self.df[self.tabular_cols] = scaler.fit_transform(self.df[self.tabular_cols])

        self.df.reset_index(drop=True, inplace=True)

        print(f"✅ 清洗后样本数: {len(self.df)}")

        required_columns = ['subID','dir', 'DX']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Excel 文件必须包含 '{col}' 列！")
        self.fmri_root = fmri_root
        self.smri_root = smri_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        site_name_f = str(self.df.loc[idx, 'subID'])
        site_name_s = str(self.df.loc[idx, 'dir'])
      #  StudyID = str(self.df.loc[idx, 'SUB_ID'])
        label = self.df.loc[idx, 'DX']

        folder_name_f = f"{site_name_f}"
        folder_name_s = f"{site_name_s}"
        fmri_path = os.path.join(self.fmri_root, folder_name_f + "_fALFF.nii")
        smri_path = os.path.join(self.smri_root, folder_name_s, "rsmwc1T1.nii")

        fmri_img = nib.load(fmri_path).get_fdata()
        smri_img = nib.load(smri_path).get_fdata()

        fmri_img = (fmri_img - np.mean(fmri_img)) / (np.std(fmri_img) + 1e-6)
        smri_img = (smri_img - np.mean(smri_img)) / (np.std(smri_img) + 1e-6)

        #  [1, D, H, W]
        fmri_tensor = torch.tensor(fmri_img, dtype=torch.float32).unsqueeze(0)
        smri_tensor = torch.tensor(smri_img, dtype=torch.float32).unsqueeze(0)

        # [2, D, H, W]
        img = torch.cat([fmri_tensor, smri_tensor], dim=0)

        #  64×64×52
        _, D, H, W = img.shape
        pad_d = max(0, 64 - D)
        pad_h = max(0, 64 - H)
        img = F.pad(img, (0, 0, 0, pad_h, 0, pad_d), "constant", 0)

        # 13, 26, 39
        slice_indices = [13, 26, 39]
        slices = img[:, :, :, slice_indices]  # [2, 64, 64, 3]

        # [2,64,64,3] -> [6,64,64]
        slices = slices.permute(0, 3, 1, 2).reshape(-1, 64, 64)
        label = torch.tensor(label, dtype=torch.long)
        tabular_data = self.df.loc[idx, self.tabular_cols].values.astype(np.float32)
        tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32)

        return slices, label, tabular_tensor, folder_name_f
