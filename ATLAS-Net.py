import torch.nn as nn
import copy
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score
from einops import rearrange
from page.vit import ViT
from page.bottleneck_layer import Bottleneck
from page.decoder import SignleConv
from page.Attention import DisTransBlock, NonSingleModalBlock, SE
from page.data_pre import *
from page.hyper import *

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, output_dim=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class Resnet_VIT_Hyper(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 train_loader=None, GPU=None,
                 vit_blocks=12,
                 vit_heads=6,
                 vit_dim_linear_mhsa_block=1024,
                 patch_size=8,
                 vit_transformer_dim=768,
                 vit_transformer=None,
                 ):
        super().__init__()
        self.inplanes = 64
        self.patch_size1 = patch_size
        self.patch_size2 = patch_size // 2
        self.patch_size3 = patch_size // 4
        self.patch_size4 = patch_size // 4
        self.vit_transformer_dim = vit_transformer_dim
        self.in_channels = in_channels

        vit_channels1 = self.inplanes  #128
        vit_channels2 = self.inplanes * 2 #256
        vit_channels3 = self.inplanes * 4 #512
        vit_channels4 = self.inplanes * 8  # 1024
        self.img_dim_vit1 = img_dim
        self.img_dim_vit2 = img_dim // 2
        self.img_dim_vit3 = img_dim // 4
        self.img_dim_vit4 = img_dim // 8

        weights_init_method = "embedding_variance"
        embd_tab_out_size = 6
        tabular_input_dim = 8

        self.mlp1 = TabularMLP(tabular_input_dim)
        self.mlp2 = TabularMLP(tabular_input_dim)
        self.mlp3 = TabularMLP(tabular_input_dim)
        self.mlp_cov1= TabularMLP(tabular_input_dim)
        self.mlp_cov2 = TabularMLP(tabular_input_dim)
        self.mlp_VIT1 = TabularMLP(tabular_input_dim)
        self.mlp_VIT2 = TabularMLP(tabular_input_dim)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=train_loader,
        )

        in_conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=1, padding=3,bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.vit_conv1 = SignleConv(in_ch=vit_channels1, out_ch=self.inplanes * 2)

        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.vit_conv2 = SignleConv(in_ch=vit_channels2, out_ch=self.inplanes * 4)

        self.conv3 = HyperPreactivResBlock2D_TTT(in_channels=self.inplanes * 4, out_channels=self.inplanes * 8, stride=2, embedding_model=self.mlp_VIT1,
                                    **general_hyper_kwargs)

        self.vit_conv2 = SignleConv(in_ch=vit_channels3, out_ch=self.inplanes * 8)

        self.conv4 = HyperPreactivResBlock2D_TTT(in_channels=self.inplanes * 8, out_channels=self.inplanes * 16, stride=2, embedding_model=self.mlp_VIT2,
                                    **general_hyper_kwargs)
        self.vit_conv2 = SignleConv(in_ch=vit_channels4, out_ch=self.inplanes * 16)

        assert (self.img_dim_vit3 % patch_size == 0), "Vit patch_dim not divisible"

        self.vit1 = ViT(img_dim=self.img_dim_vit1,
                       in_channels=vit_channels1,  # input features' channels (encoder)
                       patch_dim=patch_size,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer

        self.vit2 = ViT(img_dim=self.img_dim_vit2,
                       in_channels=vit_channels2,  # input features' channels (encoder)
                       patch_dim=4,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer

        self.vit3 = ViT(img_dim=self.img_dim_vit3,
                       in_channels=vit_channels3,  # input features' channels (encoder)
                       patch_dim=2,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer
        self.vit4 = ViT(img_dim=self.img_dim_vit4,
                       in_channels=vit_channels4,  # input features' channels (encoder)
                       patch_dim=2,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer
        # to project patches back - undoes vit's patchification
        token_dim1 = vit_channels1 * (patch_size ** 2)
        token_dim2 = vit_channels2 * ((patch_size//2) ** 2)
        token_dim3 = vit_channels3 * ((patch_size//4) ** 2)
        token_dim4 = vit_channels4 * ((patch_size//4) ** 2)

        self.project_patches_back1 = nn.Linear(vit_transformer_dim, token_dim1)
        self.project_patches_back2 = nn.Linear(vit_transformer_dim, token_dim2)
        self.project_patches_back3 = nn.Linear(vit_transformer_dim, token_dim3)
        self.project_patches_back4 = nn.Linear(vit_transformer_dim, token_dim4)

        # CNN-stage1
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # stage2
        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )

        # stage3
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        )
        self.stage4_cov = HyperPreactivResBlock2D_TTT(in_channels=256, out_channels=512, stride=2, embedding_model=self.mlp_cov1,**general_hyper_kwargs)
        # stage4
        self.stage4 = nn.Sequential(
#            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512)
        )
        self.stage5_cov = HyperPreactivResBlock2D_TTT(in_channels=512, out_channels=1024, stride=2, embedding_model=self.mlp_cov2,
                                    **general_hyper_kwargs)
        # stage5
        self.stage5 = nn.Sequential(
            BasicBlock(1024, 1024),
            BasicBlock(1024, 1024)
        )
        self.delchl1 = nn.Conv2d(self.inplanes * 4, self.inplanes * 2, 1)
        self.delchl2 = nn.Conv2d(self.inplanes * 8, self.inplanes * 4, 1)
        self.delchl3 = nn.Conv2d(self.inplanes * 16, self.inplanes * 8, 1)
        self.delchl4 = nn.Conv2d(self.inplanes * 32, self.inplanes * 16, 1)
        self.trans_se1 = SE(self.inplanes * 2, 8)
        self.trans_se2 = SE(self.inplanes * 4, 8)
        self.trans_se3 = SE(self.inplanes * 8, 8)
        self.trans_se4 = SE(self.inplanes * 16, 8)
        self.mutual_trans1 = DisTransBlock(self.inplanes * 2, dimension=2)
        self.mutual_trans2 = DisTransBlock(self.inplanes * 4, dimension=2)
        self.mutual_trans3 = DisTransBlock(self.inplanes * 8, dimension=2)
        self.mutual_trans4 = NonSingleModalBlock(self.inplanes * 16, dimension=2)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = LinearLayer(2048, 1024, embedding_model=self.mlp1, **general_hyper_kwargs)
        self.bn1 = nn.BatchNorm1d(1024)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = LinearLayer(1024, 512, embedding_model=self.mlp2, **general_hyper_kwargs)
        self.bn2 = nn.BatchNorm1d(512)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = LinearLayer(512, n_class, embedding_model=self.mlp3, **general_hyper_kwargs)

        self.embedding_to_vit1 = nn.Linear(6, 512)  # E -> C
        self.embedding_to_conv1 = nn.Linear(6, 512)  # E -> C
        self.embedding_to_vit2 = nn.Linear(6, 1024)  # E -> C
        self.embedding_to_conv2 = nn.Linear(6, 1024)  # E -> C

    def forward(self, x, tabular_tensor):
        x_fmri = x[:, 0:3, :, :]
        x_smri = x[:, 3:6, :, :]
        tabular = tabular_tensor
        trans_x1 = self.init_conv(x_fmri)
        conv_x1 = self.stage1(x_smri)
        trans_vit1 = self.vit1(trans_x1)
        trans_vit1 = self.project_patches_back1(trans_vit1)
        trans_vit1 = rearrange(trans_vit1, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                               x=self.img_dim_vit1 // self.patch_size1, y=self.img_dim_vit1 // self.patch_size1,
                               patch_x=self.patch_size1, patch_y=self.patch_size1)
        trans_x2 = self.conv1(trans_vit1)
        conv_x2 = self.stage2(conv_x1)

        fuse1 = torch.cat([conv_x2, trans_x2], dim=1)
        fuse1 = self.delchl1(fuse1)
        fuse1 = self.trans_se1(fuse1)
        fuse1_fmri = self.mutual_trans1(trans_x2, fuse1)
        fuse1_smri = self.mutual_trans1(conv_x2,fuse1)


        trans_vit2 = self.vit2(fuse1_fmri)
        trans_vit2 = self.project_patches_back2(trans_vit2)
        trans_vit2 = rearrange(trans_vit2, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                               x=self.img_dim_vit2 // self.patch_size2, y=self.img_dim_vit2 // self.patch_size2,
                               patch_x=self.patch_size2, patch_y=self.patch_size2)
        trans_x3 = self.conv2(trans_vit2)
        conv_x3 = self.stage3(fuse1_smri)
        fuse2 = torch.cat([conv_x3,  trans_x3], dim=1)
        fuse2 = self.delchl2(fuse2)
        fuse2 = self.trans_se2(fuse2)
        fuse2_fmri = self.mutual_trans2(trans_x3, fuse2)
        fuse2_smri = self.mutual_trans2(conv_x3,fuse2)


        trans_vit3 = self.vit3(fuse2_fmri)
        trans_vit3 = self.project_patches_back3(trans_vit3)
        trans_vit3 = rearrange(trans_vit3, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                               x=self.img_dim_vit3 // self.patch_size3, y=self.img_dim_vit3 // self.patch_size3,
                               patch_x=self.patch_size3, patch_y=self.patch_size3)
        trans_x4, embedding_features_trans_x4 = self.conv3((trans_vit3, tabular))
        # print("trans_x4 shape:", trans_x4.shape)
        # print("embedding_features_trans_x4 shape:", embedding_features_trans_x4.shape)
        conv_x4, embedding_features_conv_x4 = self.stage4_cov((fuse2_smri, tabular))
        conv_x4 = self.stage4(conv_x4)
        # print("conv_x4 shape:", conv_x4.shape)
        # print("embedding_features_conv_x5 shape:", embedding_features_conv_x4.shape)

#=======================
        # embedding_features: [B, 6]
        emb_proj_trans_x4 = self.embedding_to_vit1(embedding_features_trans_x4)  # [B, 512]
        emb_proj_conv_x4 = self.embedding_to_conv1(embedding_features_conv_x4)  # [B, 512]
        # embedding_features_conv_x4: [B, 6]
        emb_proj_trans_x4 = emb_proj_trans_x4.view(emb_proj_trans_x4.size(0), emb_proj_trans_x4.size(1), 1, 1)  # [B, 512, 1, 1]
        emb_proj_conv_x4 = emb_proj_conv_x4.view(emb_proj_conv_x4.size(0), emb_proj_conv_x4.size(1), 1,1)  # [B, 512, 1, 1]
        # channel-wise multiplication
        trans_x4 = trans_x4 * emb_proj_trans_x4  # [B, 512, H, W
        conv_x4 = conv_x4 * emb_proj_conv_x4  # [B, 512, H, W
#=======================
        fuse3 = torch.cat([conv_x4, trans_x4], dim=1)
        fuse3 = self.delchl3(fuse3)
        fuse3 = self.trans_se3(fuse3)
        fuse3_fmri = self.mutual_trans3(trans_x4, fuse3)
        fuse3_smri = self.mutual_trans3(conv_x4,fuse3)


        trans_vit4 = self.vit4(fuse3_fmri)
        trans_vit4 = self.project_patches_back4(trans_vit4)
        trans_vit4 = rearrange(trans_vit4, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                               x=self.img_dim_vit4 // self.patch_size4, y=self.img_dim_vit4 // self.patch_size4,
                               patch_x=self.patch_size4, patch_y=self.patch_size4)

        trans_x5, embedding_features_trans_x5 = self.conv4((trans_vit4, tabular))  # [16, 512, 16, 16]
        # print("trans_x4 shape:", trans_x4.shape)
        # print("embedding_features_trans_x4 shape:", embedding_features_trans_x4.shape)
        conv_x5, embedding_features_conv_x5 = self.stage5_cov((fuse3_smri, tabular))
        conv_x5 = self.stage5(conv_x5)#[16, 1024, 8, 8]
        # print("conv_x4 shape:", conv_x4.shape)
        # print("embedding_features_conv_x5 shape:", embedding_features_conv_x4.shape)
##+++++++++++++++++++++++++++
        emb_proj_trans_x5 = self.embedding_to_vit2(embedding_features_trans_x5)  # [B, 512]
        emb_proj_conv_x5 = self.embedding_to_conv2(embedding_features_conv_x5)  # [B, 512]
        # embedding_features_conv_x4: [B, 6]
        emb_proj_trans_x5 = emb_proj_trans_x5.view(emb_proj_trans_x5.size(0), emb_proj_trans_x5.size(1), 1, 1)  # [B, 512, 1, 1]
        emb_proj_conv_x5 = emb_proj_conv_x5.view(emb_proj_conv_x5.size(0), emb_proj_conv_x5.size(1), 1,1)  # [B, 512, 1, 1]
        # channel-wise multiplication
        trans_x5 = trans_x5 * emb_proj_trans_x5  # [B, 512, H, W
        conv_x5 = conv_x5 * emb_proj_conv_x5  # [B, 512, H, W
##++++++++++++++++++++++++++++++++


        conv_x5, trans_vit4 = self.mutual_trans4(conv_x5, trans_x5)
        x_fuse = torch.cat([conv_x5, trans_vit4],dim=1)
        x = self.avgpool(x_fuse)
   #     print(x.shape)
        x = x.view(x.size(0), -1)
        out, feature1 = self.fc1((x, tabular))
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.dropout1(out)

        out, feature2 = self.fc2((out, tabular))
        out = self.bn2(out)
        out = self.prelu2(out)
        out = self.dropout2(out)

        out, feature = self.fc3((out, tabular))
        return out
# -------------------------
class CachedDataset(Dataset):
    def __init__(self, cached_path):
        self.data = torch.load(cached_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        slices = item["slices"]              # tensor [6,64,64]
        label = item["label"]                # tensor
        tabular = item["tabular"]            # tensor size [9]
        scan_id = item["scan_id"]            # string

        return slices, label, tabular, scan_id
## -------------------------
cached_path = "/home/test/qsl/LC/pycharm/ADNI/11111.pt"
# Use cached dataset instead of MRI_Dataset
full_dataset = CachedDataset(cached_path)
print(f"✅ Loaded cached dataset with {len(full_dataset)} samples")

num_0, num_1 = 0, 0
for i in range(len(full_dataset)):
    _, label, _, _ = full_dataset[i]
    if label == 0:
        num_0 += 1
    elif label == 1:
        num_1 += 1

print(f"Label 0 count: {num_0}")
print(f"Label 1 count: {num_1}")

# -------------------------
# K-Fold Configuration
# -------------------------
batch_size = 12
lr = 1e-4
num_epochs = 150
n_class = 2
patience = 50
k_folds = 10
val_ratio = 0.2  # Use 20% of training data as validation set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold_results = {}

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# -------------------------
# Save Directory and File
# -------------------------
save_dir = "/home/test/qsl/LC/pycharm/ADNI"
os.makedirs(save_dir, exist_ok=True)
result_file = os.path.join(save_dir, "AD_HC.txt")

fold_results = []

# -------------------------
# Start K-Fold Training
# -------------------------
for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
    print(f"\n===== Fold {fold + 1}/{k_folds} =====")

    # Create train/test subsets
    train_subset = Subset(full_dataset, train_idx)
    test_subset = Subset(full_dataset, test_idx)

    # Split validation set from training set
    train_size = len(train_subset)
    val_size = int(train_size * val_ratio)
    train_size = train_size - val_size

    train_subset, val_subset = random_split(
        train_subset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed random seed for reproducibility
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(
        f"Train set size: {len(train_subset)}, Validation set size: {len(val_subset)}, Test set size: {len(test_subset)}")

    # Initialize model
    model = Resnet_VIT_Hyper(in_channels=3, img_dim=64, vit_blocks=6,
                             vit_dim_linear_mhsa_block=1024, classes=2, train_loader=train_loader)
    device_count = torch.cuda.device_count()
    print(f"🎯 Detected {device_count} GPU(s)")
    device_ids = [1]
    device = torch.device(f"cuda:{device_ids[0]}")
    model = model.to(device)
    if device_count > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_acc = 0.0
    early_stop_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    # -------------------------
    # Training Loop (using validation set for early stopping)
    # -------------------------
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels, tabular_tensor, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            tabular_tensor = tabular_tensor.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, tabular_tensor)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase (for early stopping)
        model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels, tabular_tensor, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                tabular_tensor = tabular_tensor.to(device)
                outputs = model(imgs, tabular_tensor)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        # Early stopping based on validation accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            print(f"  ✓ Validation accuracy improved to {val_acc:.4f}, saving model")
        else:
            early_stop_counter += 1
            print(
                f"  ✗ Validation accuracy not improved ({val_acc:.4f} vs {best_val_acc:.4f}), early stop count: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"🛑 Early stopping triggered! Stopping training at epoch {epoch + 1}")
                break

    # -------------------------
    # Evaluate best model on test set
    # -------------------------
    model.load_state_dict(best_model_wts)
    model.eval()
    test_ids, test_preds, test_labels_all, test_probs = [], [], [], []

    with torch.no_grad():
        for imgs, labels, tabular_tensor, scan_id in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            tabular_tensor = tabular_tensor.to(device)
            outputs = model(imgs, tabular_tensor)

            # Binary classification
            _, predicted = outputs.max(1)
            probs = F.softmax(outputs, dim=1)[:, 1]  # Positive class probability for binary classification

            test_ids.extend(scan_id)
            test_preds.extend(predicted.cpu().numpy())
            test_labels_all.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    # -------------------------
    # Calculate metrics
    # -------------------------
    acc = accuracy_score(test_labels_all, test_preds)
    precision = precision_score(test_labels_all, test_preds)
    recall = recall_score(test_labels_all, test_preds)
    f1 = f1_score(test_labels_all, test_preds)
    cm = confusion_matrix(test_labels_all, test_preds)
    auc = roc_auc_score(test_labels_all, test_probs)

    fold_results.append({
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "ids": test_ids,
        "preds": test_preds,
        "labels": test_labels_all,
        "probs": test_probs,
        "best_val_acc": best_val_acc  # Record best validation accuracy
    })

    # -------------------------
    # Save fold results to txt file
    # -------------------------
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(f"\n===== Fold {fold + 1} =====\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write("Test Set Prediction Results:\n")
        for sid, pred, label, prob in zip(test_ids, test_preds, test_labels_all, test_probs):
            f.write(f"Scan ID: {sid}, Pred: {pred}, True: {label}, Prob: {prob:.4f}\n")

# -------------------------
# Save average metrics
# -------------------------
mean_acc = np.mean([r["acc"] for r in fold_results])
std_acc = np.std([r["acc"] for r in fold_results])

mean_precision = np.mean([r["precision"] for r in fold_results])
std_precision = np.std([r["precision"] for r in fold_results])

mean_recall = np.mean([r["recall"] for r in fold_results])
std_recall = np.std([r["recall"] for r in fold_results])

mean_f1 = np.mean([r["f1"] for r in fold_results])
std_f1 = np.std([r["f1"] for r in fold_results])

mean_auc = np.mean([r["auc"] for r in fold_results])
std_auc = np.std([r["auc"] for r in fold_results])

mean_val_acc = np.mean([r["best_val_acc"] for r in fold_results])

with open(result_file, "a", encoding="utf-8") as f:
    f.write("\n===== Mean Metrics Across Folds =====\n")
    f.write(f"Best Validation Accuracy (mean): {mean_val_acc:.4f}\n")
    f.write(f"Test Accuracy: {mean_acc:.4f}  (std: {std_acc:.4f})\n")
    f.write(f"Test Precision: {mean_precision:.4f}  (std: {std_precision:.4f})\n")
    f.write(f"Test Recall: {mean_recall:.4f}  (std: {std_recall:.4f})\n")
    f.write(f"Test F1-score: {mean_f1:.4f}  (std: {std_f1:.4f})\n")
    f.write(f"Test AUC: {mean_auc:.4f}  (std: {std_auc:.4f})\n")

print(f"\n📁 All fold metrics saved to: {result_file}")