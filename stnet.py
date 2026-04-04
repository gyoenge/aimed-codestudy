# manualy implemneted st-net

# densenet 
import densenet 
# stnet 
import torch 
import torch.nn as nn 
from torchvision import models 
# data 
import os
import json
import pandas as pd
from hest.bench.st_dataset import H5PatchDataset, load_adata
from torchvision import transforms 
# train test 
import torch.optim as optim
import numpy as np
import random
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
# logging
import logging
from datetime import datetime


class STNet(nn.Module):
    def __init__(
        self,
        # backbone_name='densenet121',
        num_genes=250,
        pretrained=True,
    ):
        super(STNet, self).__init__()

        # backbone == 'densenet121' option 
        if pretrained:
            self.backbone = models.densenet121(weights='DEFAULT').features  # DEFAULT = IMAGENET1K_V1, feature extracter만
        else:
            self.backbone = densenet._densenet121().feature  # 직접 정의한 _densnet121 (random init)

        self.pool = nn.AdaptiveAvgPool2d((1,1)) # (B, C, H, W) -> (B, C, 1, 1)
        # 입력 크기가 어떻든 간에(adaptive), 평균을 내어(average), 원하는 출력 크기를 만든다(pooling).

        self.classifier = nn.Linear(in_features=1024, out_features=num_genes)

    def forward(self, x):
        x = self.backbone(x)        # (Batch, 3, 224, 224) -> (Batch, 1024, 7, 7)
        x = self.pool(x)            # (Batch, 1024, 7, 7) -> (Batch, 1024, 1, 1)
        x = torch.flatten(x, 1)     # (Batch, 1024, 1, 1) -> (Batch, 1024)
        output = self.classifier(x) # (Batch, 1024) -> (Batch, 250)
        return output


class STNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bench_data_root,
        gene_list_path,
        split_csv_path=None,
        split_df=None,
        transforms=None,
    ):
        """ 
        split_csv_path 또는 split_df 둘 중 하나를 받아
        이미지-유전자 발현 쌍을 생성
        """
        assert (split_csv_path is not None) ^ (split_df is not None), \
            "Provide exactly one of split_csv_path or split_df."

        if split_df is not None:
            self.split_df = split_df.reset_index(drop=True).copy()
        else:
            self.split_df = pd.read_csv(split_csv_path)

        with open(gene_list_path, 'r') as f:
            self.genes = json.load(f)['genes']

        self.images = []
        self.targets = []
        self.transforms = transforms

        for _, row in self.split_df.iterrows():
            patches_h5_path = os.path.join(bench_data_root, row['patches_path'])
            expr_path = os.path.join(bench_data_root, row['expr_path'])

            assert os.path.isfile(patches_h5_path), f"Patch file not found: {patches_h5_path}"
            assert os.path.isfile(expr_path), f"Expr file not found: {expr_path}"

            patch_dataset = H5PatchDataset(patches_h5_path)

            slide_imgs = []
            slide_barcodes = []

            for i in range(len(patch_dataset)):
                chunk_imgs = patch_dataset[i]['imgs']
                chunk_barcodes = patch_dataset[i]['barcodes']

                if isinstance(chunk_imgs, torch.Tensor):
                    chunk_imgs = chunk_imgs.numpy()
                if isinstance(chunk_barcodes, torch.Tensor):
                    chunk_barcodes = chunk_barcodes.numpy()

                if chunk_imgs.ndim == 3:
                    chunk_imgs = np.expand_dims(chunk_imgs, axis=0)
                    chunk_barcodes = [chunk_barcodes]

                for barcode, img in zip(chunk_barcodes, chunk_imgs):
                    barcode_str = barcode.decode('utf-8') if isinstance(barcode, bytes) else str(barcode)
                    slide_barcodes.append(barcode_str)
                    slide_imgs.append(img)

            adata_df = load_adata(
                expr_path,
                genes=self.genes,
                barcodes=slide_barcodes,
                normalize=True
            )

            for j in range(len(slide_imgs)):
                self.images.append(slide_imgs[j])
                self.targets.append(
                    torch.tensor(adata_df.iloc[j].values, dtype=torch.float32)
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]

        # if transforms exists 
        if self.transforms:
            img = self.transforms(img)

        # convert into torch tensor
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(np.array(img))

        # channel dim position correction
        if len(img.shape) == 3 and img.shape[-1] == 3: 
            img = img.permute(2, 0, 1)  # (H,W,C)->(C,H,W)

        # normalize 
        if img.max() > 1.0:
            img = img.float() / 255.0
        else: 
            img = img.float()

        return img, target      


### 

def train_fold(model, train_loader, device, num_epochs=50):
    model.to(device)
    model.train()
    ctiterion = nn.MSELoss()
    # ST-Net: SGD (lr=1e-6, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for imgs, targets in train_loader: 
            imgs, targets = imgs.to(device), targets.to(device) 

            optimizer.zero_grad() 
            preds = model(imgs)
            loss = ctiterion(preds, targets)
            loss.backward() 
            optimizer.step() 
            epoch_loss += loss.item() 
        
        if logger: 
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    return model


def eval_fold(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, targets in test_loader: 
            imgs = imgs.to(device)
            preds = model(imgs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, G) 
    all_targets = np.concatenate(all_targets, axis=0)  # (N, G)

    # Following calculation of Pearson Correlation & R2 from HEST-Bench (also ST-Net)
    pearson_corrs = []
    for i in range(all_targets.shape[1]):
        corr, _ = pearsonr(all_targets[:, i], all_preds[:, i])
        pearson_corrs.append(0.0 if np.isnan(corr) else corr)
    # 모든 patch에서 gene i의 실제값 vs 예측값 correlation. (각 gene마다 (patch-wise))
    # PCC --> 패턴(variation)을 얼마나 잘 맞추는지가 중요 (gene-level 성능을 평가) 

    return np.mean(pearson_corrs), pearson_corrs


### 

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(num_genes=250, pretrained=True):
    return STNet(num_genes=num_genes, pretrained=pretrained)

def train_one_epoch(model, train_loader, device, optimizer, criterion):
    model.train()
    epoch_loss = 0.0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def select_best_epoch(
    train_df,
    bench_data_root,
    gene_list_path,
    device,
    num_genes,
    pretrained=True,
    max_epochs=50,
    n_inner_folds=4,
    batch_size=32,
    seed=42,
):
    """
    train_df를 inner folds로 나누고,
    epoch 1~50에 대해 validation Pearson 평균이 가장 좋은 epoch를 선택
    """
    loo = LeaveOneOut()
    fold_indices = list(loo.split(train_df))

    epoch_scores = {epoch: [] for epoch in range(1, max_epochs + 1)}

    for inner_fold, (tr_idx, val_idx) in enumerate(fold_indices):
        if logger: 
            logger.info(f"\n  [Inner Fold {inner_fold + 1}/{len(fold_indices)}]")

        inner_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        inner_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_transform = transforms.Compose([
            transforms.ToPILImage(),

            transforms.RandomChoice([
                transforms.RandomRotation((0,0)),
                transforms.RandomRotation((90,90)),
                transforms.RandomRotation((180,180)),
                transforms.RandomRotation((270,270)),
            ]),
            transforms.RandomHorizontalFlip(p=0.5),

            transforms.ToTensor() 
        ])

        inner_train_dataset = STNetDataset(
            bench_data_root=bench_data_root,
            gene_list_path=gene_list_path,
            split_df=inner_train_df,
            transforms=train_transform,
        )

        inner_val_dataset = STNetDataset(
            bench_data_root=bench_data_root,
            gene_list_path=gene_list_path,
            split_df=inner_val_df,
        )

        inner_train_loader = torch.utils.data.DataLoader(
            inner_train_dataset, batch_size=batch_size, shuffle=True
        )
        inner_val_loader = torch.utils.data.DataLoader(
            inner_val_dataset, batch_size=batch_size, shuffle=False
        )

        model = build_model(num_genes=num_genes, pretrained=pretrained).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, inner_train_loader, device, optimizer, criterion)
            val_pearson, _ = eval_fold(model, inner_val_loader, device)

            epoch_scores[epoch].append(val_pearson)
            if logger: 
                logger.info(
                    f"    Epoch {epoch:02d}/{max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Pearson: {val_pearson:.4f}"
                )

    mean_epoch_scores = {
        epoch: float(np.mean(scores)) for epoch, scores in epoch_scores.items()
    }

    best_epoch = max(mean_epoch_scores, key=mean_epoch_scores.get)
    best_score = mean_epoch_scores[best_epoch]

    if logger: 
            logger.info("\n  [Epoch Selection Summary]")
    for epoch in range(1, max_epochs + 1):
        if logger: 
            logger.info(f"    Epoch {epoch:02d}: Mean Val Pearson = {mean_epoch_scores[epoch]:.4f}")

    if logger: 
        logger.info(f"  >>> Selected Best Epoch = {best_epoch} (Mean Val Pearson = {best_score:.4f})")
    return best_epoch, mean_epoch_scores


def retrain_full_train(
    train_df,
    bench_data_root,
    gene_list_path,
    device,
    num_genes,
    pretrained=True,
    num_epochs=50,
    batch_size=32,
):
    full_train_dataset = STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=gene_list_path,
        split_df=train_df,
    )
    full_train_loader = torch.utils.data.DataLoader(
        full_train_dataset, batch_size=batch_size, shuffle=True
    )

    model = build_model(num_genes=num_genes, pretrained=pretrained).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, full_train_loader, device, optimizer, criterion)
        if logger: 
            logger.info(f"  [Retrain] Epoch {epoch:02d}/{num_epochs} | Train Loss: {train_loss:.4f}")

    return model


def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log") # 

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 기존 handler 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 포맷 정의
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 출력
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 파일 저장
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    print(f"[LOG] Logging to {log_path}")

    return timestamp, logger


### 

if __name__ == "__main__":
    seed_everything(42)
    timestamp, logger = setup_logger() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench_data_root = "/root/workspace/hest_data/eval/bench_data/IDC"
    task_name = "IDC"

    backbones = ['densenet121'] # ['densenet121', 'resnet18', 'alexnet', 'vgg11']
    num_genes_list = [250] # [50, 100, 200, 300, 400, 500]
    genes_criteria_list = ['var', 'mean'] # ['var', 'mean'] 
    # ST-Net: mean_250.  

    outer_folds = [0, 1, 2, 3]
    max_epochs = 50 # ST-Net: 50 epoch  
    batch_size = 32 # ST-Net: 32 batch 

    results_grid = []

    logger.info("=== Start Nested CV Evaluation for IDC ===")

    for backbone in backbones:
        for num_genes in num_genes_list:
            for genes_criteria in genes_criteria_list:
                logger.info(f"\n[Setting] Backbone: {backbone} | Genes: {num_genes} | Criteria: {genes_criteria}")

                gene_list_path = os.path.join(
                    bench_data_root, f'{genes_criteria}_{num_genes}genes.json'
                )
                assert os.path.isfile(gene_list_path), f"Gene list file not found: {gene_list_path}"

                fold_test_scores = []

                for outer_fold in outer_folds:
                    logger.info(f"\n================ OUTER FOLD {outer_fold} ================")

                    train_csv = os.path.join(bench_data_root, f'splits/train_{outer_fold}.csv')
                    test_csv = os.path.join(bench_data_root, f'splits/test_{outer_fold}.csv')

                    assert os.path.isfile(train_csv), f"Train CSV not found: {train_csv}"
                    assert os.path.isfile(test_csv), f"Test CSV not found: {test_csv}"

                    outer_train_df = pd.read_csv(train_csv)
                    outer_test_df = pd.read_csv(test_csv)

                    # 1) inner validation으로 best epoch 선택
                    # best_epoch, epoch_score_dict = select_best_epoch(
                    #     train_df=outer_train_df,
                    #     bench_data_root=bench_data_root,
                    #     gene_list_path=gene_list_path,
                    #     device=device,
                    #     num_genes=num_genes,
                    #     pretrained=True,
                    #     max_epochs=max_epochs,
                    #     n_inner_folds=4,
                    #     batch_size=batch_size,
                    #     seed=42,
                    # )
                    best_epoch = 50 

                    # 2) best epoch로 outer train 전체 재학습
                    final_model = retrain_full_train(
                        train_df=outer_train_df,
                        bench_data_root=bench_data_root,
                        gene_list_path=gene_list_path,
                        device=device,
                        num_genes=num_genes,
                        pretrained=True,
                        num_epochs=best_epoch,
                        batch_size=batch_size,
                    )

                    # save checkpoint 
                    save_dir = f"/root/workspace/impl/stnet/checkpoints/run_{timestamp}"
                    os.makedirs(save_dir, exist_ok=True)

                    torch.save(
                        final_model.state_dict(),
                        os.path.join(save_dir, f"stnet_full_fold{outer_fold}_{backbone}_{genes_criteria}{num_genes}.pth")
                    )

                    torch.save(
                        final_model.backbone.state_dict(),
                        os.path.join(save_dir, f"stnet_backbone_fold{outer_fold}_{backbone}_{genes_criteria}{num_genes}.pth")
                    )

                    logger.info(f"Model saved to {save_dir} for fold {outer_fold}")

                    # 3) outer test 평가
                    outer_test_dataset = STNetDataset(
                        bench_data_root=bench_data_root,
                        gene_list_path=gene_list_path,
                        split_df=outer_test_df,
                    )
                    outer_test_loader = torch.utils.data.DataLoader(
                        outer_test_dataset, batch_size=batch_size, shuffle=False
                    )

                    test_mean_pearson, test_gene_corrs = eval_fold(final_model, outer_test_loader, device)

                    logger.info(
                        f"[Outer Fold {outer_fold}] "
                        f"Best Epoch = {best_epoch} | Test Mean Pearson = {test_mean_pearson:.4f}"
                    )

                    fold_test_scores.append(test_mean_pearson)

                    results_grid.append({
                        "backbone": backbone,
                        "num_genes": num_genes,
                        "genes_criteria": genes_criteria,
                        "outer_fold": outer_fold,
                        "best_epoch": best_epoch,
                        "test_mean_pearson": test_mean_pearson,
                    })

                avg_outer_score = float(np.mean(fold_test_scores))
                std_outer_score = float(np.std(fold_test_scores))

                logger.info(
                    f"\n>>> Final Summary | Backbone: {backbone} | Genes: {num_genes} | "
                    f"Criteria: {genes_criteria} | Mean Outer Test Pearson: {avg_outer_score:.4f} ± {std_outer_score:.4f}"
                )

    df_results = pd.DataFrame(results_grid)
    df_results.to_csv("/root/workspace/impl/stnet/nested_cv_eval_results.csv", index=False)
    logger.info("\nAll Nested CV Evaluation Completed and Saved!")
