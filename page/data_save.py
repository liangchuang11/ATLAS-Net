
from data_pre import *
from hyper import *

excel_path = r"/home/test/qsl/LC/pycharm/ADNI/AD_HC_no_order_new.xlsx"
fmri_root = r"/home/test/qsl/LC/matlab/Quality_screening/AD/fALFF/"
smri_root = r"/home/qi/data/ADNI/Data_BIDS/Raw_Data/"

full_dataset = MRI_Dataset(excel_path, fmri_root, smri_root)

cached_list = []
for i in range(len(full_dataset)):
    slices, label, tabular, folder = full_dataset[i]
    cached_list.append({
        "slices": slices,           # [6,64,64]
        "label": label,             # tensor
        "tabular": tabular,         # [9]
        "scan_id": folder
    })

torch.save(cached_list, "/home/test/qsl/LC/pycharm/ADNI/AD_HC.pt")
print("缓存保存完成，共保存样本:", len(cached_list))
