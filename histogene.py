# HisToGene 의 핵심 두가지: 
# (i) ViT로 patch 간 관계 학습 → gene expression 예측. 
# (ii) 학습이후 dense sampling + averaging으로 super-resolution 예측 가능. 

# source: https://github.com/maxpmx/HisToGene/blob/main/vis_model.py 

###

import os 
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import pytorch_lightning as pl 
# PyTorch Lightning: PyTorch 코드에서 훈련 루프(boilerplate)를 제거해주는 고수준 프레임워크 
# for epoch for batch --> 코드 길어지고 / 실수 많고 / 재사용 어렵고 / distributed 처리 복잡하다는 문제 해결 
from torchmetrics.functional import accuracy # 맞춘 개수 / 전체 개수 -> acc = accuracy(preds, target, task="binary")
from transformer import ViT 
from torch.optim.lr_scheduler import ReduceLROnPlateau # plateau(loss 잘 안내려가고 멈추는 곳) -> lr 감소 스케쥴링 


# patch + spatial position → Transformer → gene expression

class HisToGene(pl.LightningModule):
    def __init__(self, 
        patch_size=112,
        n_layers=4, 
        n_genes=1000,
        dim=1024, 
        learning_rate=1e-4,
        dropout=0.1,
        n_pos=64
    ):
        super().__init__()

        """
        [patch image] + [x,y 위치]
                ↓
        embedding
                ↓
        Transformer (ViT)
                ↓
        MLP
                ↓
        gene vector
        """

        # self.save_hyperparameters()

        self.learning_rate = learning_rate
        patch_dim = 3*patch_size*patch_size

        self.patch_embedding = nn.Linear(patch_dim, dim)
        # 이미지 patch → 벡터 (1024 차원)
        # CNN 대신 Linear projection 사용
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        # (x 좌표, y 좌표) → embedding
        # position embedding 추가

        self.vit = ViT(
            dim=dim, 
            depth=n_layers,
            heads=16, 
            mlp_dim=2*dim,
            dropout=dropout, 
            emb_dropout=dropout,
        )
        # patch 간 관계 학습 (attention)
        # "이 patch는 주변 patch와 어떻게 관계있는가"

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )
        # 각 patch → gene vector (ex: 1000 genes)

    def forward(self, patches, centers):
        # (B, N, patch_dim) → (B, N, dim)
        patches = self.patch_embedding(patches) 
        # centers: (B, N, 2) → (x, y)
        # --> (B, N, dim)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        # image 정보 + spatial 정보 결합
        x = patches + centers_x + centers_y
        # patch 간 interaction 학습
        h = self.vit(x)
        # 각 patch → gene vector
        # --> (B, N, n_genes)
        x = self.gene_head(h)
        return x
    
    def training_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('train_loss', loss)
        return loss
    
    def validtion_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('valid_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('test_loss', loss) 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    @staticmethod 
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    a = torch.rand(1, 4000, 3*122*122)
    p = torch.ones(1, 4000, 2).long()
    model = HisToGene()
    print(count_parameters(model))
    x = model(a,p)
    print(x.shape)


"""
Usage: 

import torch
from vis_model import HisToGene

model = HisToGene(
    n_genes=1000, 
    patch_size=112, # HER2+ BC dataset patch size 
    n_layers=4, 
    dim=1024, 
    learning_rate=1e-5, 
    dropout=0.1, 
    n_pos=64
)

# flatten_patches: [N, 3*W*H]
# coordinates: [N, 2]

pred_expression = model(flatten_patches, coordinates)  # [N, n_genes]

"""


# super-resolution prediction logic

from tqdm import tqdm
import anndata as ann

def sr_predict(
        model, 
        test_loader, 
        attention=True, 
        device=torch.device('cpu')
    ):
        """
        이 함수 자체가 super-resolution을 "만드는" 건 아니고,
        super-resolution prediction을 "수용하는 구조"다.
        즉, 핵심은 이 함수가 아니라 test_loader 설계에 있다. 

        논문에서 super-resolution은: 
        1. patch를 더 촘촘히 sampling
        2. 각 patch에서 gene 예측
        3. overlapping patch 평균 → 더 높은 해상도

        test_loader가 어떤 patch를 주느냐 = resolution 결정
        (patch 개수 != ST spot 개수) -- patch를 훨씬 더 촘촘히 샘플링.

        왜 super-resolution용이냐?
        preds = torch.cat((preds, pred), dim=0)
        --> "모든 patch의 prediction을 그대로 다 모은다"
        --> dense sampling 결과를 그대로 유지
        (preds.shape = (N_dense_patches, genes)) -- N_dense_patches >> N_spots
        """
        # patch 단위 모델 예측 → 전체 spatial gene map으로 합치고 → AnnData로 변환 
        model.eval()
        model = model.to(device)
        preds = None
        with torch.no_grad():
            for patch, position, center in tqdm(test_loader):
                # position: (x, y) 좌표
                # center: 실제 spatial 좌표 (나중에 mapping용)
                
                patch, position = patch.to(device), position.to(device)
                pred = model(patch, position)
                
                if preds is None:
                    preds = pred.squeeze()
                    ct = center
                else:
                    preds = torch.cat((preds,pred),dim=0)
                    ct = torch.cat((ct,center),dim=0)
                # 결과 누적 (모든 patch 합치기 (cat)) 

        preds = preds.cpu().squeeze().numpy()
        ct = ct.cpu().squeeze().numpy()
        adata = ann.AnnData(preds)
        adata.obsm['spatial'] = ct

        return adata


"""
Usage: 

dataset_sr = ViT_HER2ST(train=False,sr=True,fold=fold)
test_loader_sr = DataLoader(dataset_sr, batch_size=1, num_workers=2)
adata_sr = sr_predict(model,test_loader_sr,attention=False)
adata_sr = comp_tsne_km(adata_sr,4)

g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
adata_sr.var_names = g
sc.pp.scale(adata_sr)
adata_sr

"""


# utils: marker genes
# 세포 타입별 marker gene 목록을 정의하고, 분석에 사용할 gene 리스트를 구성. 

BCELL = ['CD19', 'CD79A', 'CD79B', 'MS4A1'] # B cell marker genes
TUMOR = ['FASN'] # tumor 관련 gene 
# FASN → 암 세포에서 lipid synthesis 증가 (논문에서도 중요 gene)
CD4T = ['CD4'] # T cell subtype 구분 -- CD4 → helper T cell
CD8T = ['CD8A', 'CD8B'] # T cell subtype 구분 -- CD8 → cytotoxic T cell
DC = ['CLIC2', 'CLEC10A', 'CD1B', 'CD1A', 'CD1E'] # DC (Dendritic cells) -- antigen presenting cell marker
MDC = ['LAMP3'] # MDC (Mature DC) -- 성숙 dendritic cell marker
CMM = ['BRAF', 'KRAS'] # CMM (암 관련) # melanoma / oncogene
# BRAF → mutation 많음, KRAS → 대표 oncogene 

IG = {
 'B_cell': BCELL,
 'Tumor': TUMOR,
 'CD4+T_cell': CD4T,
 'CD8+T_cell': CD8T,
 'Dendritic_cells': DC,
 'Mature_dendritic_cells': MDC,
 'Cutaneous_Malignant_Melanoma': CMM
} # 세포 타입 → marker gene 리스트 

MARKERS = []
for i in IG.values():
    MARKERS += i
# MARKERS 생성 : 모든 marker gene을 하나의 리스트로 합침

LYM = {'B_cell':BCELL, 'CD4+T_cell':CD4T, 'CD8+T_cell':CD8T}
# LYM 딕셔너리 : 림프구 관련 세포만 따로 모음

"""
--> vis 가능
: sc.pl.spatial(adata, color=MARKERS)
: marker gene expression visualization

--> 모델 평가 시 marker gene만 따로 평가 가능
--> clustering 해석 용이 
"""

# utils: adata (gene expression) preprocessing logic 

import scanpy as sc, anndata as ad
# scanpy (sc) : 분석 도구 (전처리, 시각화, clustering)
# anndata (ad) : 데이터 구조 (행렬 + 메타데이터 저장)

"""
AnnData 객체: 
adata = ad.AnnData(X)

adata
 ├── X        (핵심 데이터: gene expression matrix)
 ├── obs      (row metadata → cell / spot 정보)
 ├── var      (column metadata → gene 정보)
 ├── obsm     (embedding, spatial coords 등)
 └── layers   (여러 버전의 데이터 저장)

예시: 
adata.X.shape = (1000, 20000)
adata.obs['cell_type']
adata.var_names # gene 이름들 
adata.obsm['spatial'] # spatial 좌표 저장 
"""
"""
scanpy는 분석 함수 모음

전처리: 
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

feature selection: 
sc.pp.highly_variable_genes(adata)

차원 축소: 
sc.tl.pca(adata)
sc.tl.umap(adata)

clustering: 
sc.tl.leiden(adata)

시각화: 
sc.pl.umap(adata, color='cell_type')
sc.pl.spatial(adata, color='CD4')

"""

import numpy as np
from sklearn import preprocessing 

def preprocess(
        adata, 
        n_keep=1000, # 사용할 gene 개수 (HVG 개수)
        include=LYM, # marker gene dict (ex: LYM)
        g=True       # 특정 gene list 사용할지 여부
    ):
    # AnnData를 모델 학습용으로 전처리

    adata.var_names_make_unique() # gene 이름 중복 제거 
    sc.pp.normalize_total(adata)  # cell별 총 expression을 같게 맞춤. library size normalization.  
    sc.pp.log1p(adata)  # log(1 + x). 분포 안정화. 

    if g:
        # 미리 정의된 gene list만 사용
        # custom gene set (예: 250 genes)
        b = list(np.load('data/skin_a.npy', allow_pickle=True))
        adata = adata[:, b]
    
    elif include:
        # marker aggregation
        exp = np.zeros((adata.X.shape[0], len(include)))
        for n, (i,v) in enumerate(include.items()):
            # 각 cell type마다: marker gene 평균 → 하나의 feature 
            tmp = adata[:,v].X
            tmp = np.mean(tmp, 1).flatten() 
            exp[:, n] = tmp
            # gene → cell-type feature로 축소 
            # : dimension reduction + biological 의미 유지
        adata = adata[:, :, len(include)]
        adata.X = exp
        adata.var_names = list(include.keys())
        # gene이 아니라 cell type feature가 됨

    else:
        # default (HVG): variation 높은 gene만 선택
        sc.pp.highly_variable_genes(adata, n_top_genes=n_keep, subset=True)
    
    # spatial 좌표 normalization (평균 0, 분산 1로 normalization)  
    c = adata.obsm['spatial']
    scalar = preprocessing.StandardScalar().fit(c)
    c = scalar.transform(c)
    adata.obsm['position_norm'] = c

    return adata


# utils: compute umap or tsne logic
# prediction 결과를 시각화하고, spatial domain이 잘 나오는지 확인하는 단계
# “고차원 gene expression → 저차원 embedding + clustering” 하는 전형적인 분석 파이프라인 

def comp_umap(adata):
    # gene expression → PCA → graph → UMAP → clustering
    # : biological cluster (cell type / region) 찾기 
    sc.pp.pca(adata) # PCA: 고차원 gene (ex: 1000개) → 저차원 (ex: 50개). noise 제거 + 계산 효율 ↑. 
    sc.pp.neighbors(adata) # neighbors graph: 각 cell/spot 간 “가까운 이웃” 찾음. graph 기반 구조 생성 (k-NN graph). 
    sc.tl.umap(adata) # 2D로 projection. 비슷한 cell은 가까이, 다른 cell은 멀리. --> adata.obsm['X_umap'] 
    sc.tl.leiden(adata, key_added="clusters") # Leiden clustering. graph 기반 clustering. 비슷한 cell끼리 그룹화. --> adata.obs['clusters']
    return adata


from sklearn.cluster import KMeans

def comp_tsne_km(adata,k=10):
    # gene expression → PCA → t-SNE → KMeans clustering
    sc.pp.pca(adata) # PCA
    sc.tl.tsne(adata) # 2D embedding (local 구조 강조, 느림, 시각화용.) --> adata.obsm['X_tsne']
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(adata.obsm['X_pca']) # PCA 공간에서 clustering 수행 
    adata.obs['kmeans'] = kmeans.labels_.astype(str) # --> adata.obs['kmeans']
    return adata
