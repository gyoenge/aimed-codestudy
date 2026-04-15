# Hist2ST source : https://github1s.com/biomed-AI/Hist2ST/blob/main/HIST2ST.py 

import numpy as np
from collections import defaultdict # 키가 없어도 자동으로 기본값을 만들어주는 dict
import torch 
from torch import einsum
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn import init 
from torch.autograd import Variable # 예전 PyTorch에서 자동 미분(gradient 계산)을 위해 사용하던 래퍼 클래스를 import하는 코드. Tensor를 감싸서 gradient 추적 (autograd) 가능하게 해줌. (지금은 거의 사용 안함. deprecated)
from torch.autograd.variable import *
from einops import rearrange
import pytorch_lightning as pl # PyTorch 코드를 더 간단하고 구조적으로 만들어주는 고수준 프레임워크
import torchvision.transforms as tf
import scanpy as sc
import anndata as ann
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

### gs_block: GraphSAGE 스타일의 그래프 이웃 집계 블럭 ### 

class gs_block(nn.Module):
    """
        입력 노드 feature x와 adjacency matrix Adj를 받아서, 
        각 노드의 새 embedding을 만드는 역할을 한다. 
        즉, 자신의 feature와 이웃 feature를 모아서, 선형변환 + 정규화로 새로운 node representation을 만든다. 
    """
    def __init__(
        self, 
        feature_dim, # 각 노드의 입력 feature 차원 
        embed_dim, # 출력 embedding 차원 
        policy='mean', # 이웃 feature를 어떻게 모을지 결정 (mean: 평균, max: max pooling)
        gcn=False, # GraphSAGE 방식 처럼 self + neighbor concat 할지, GCN처럼 neighbor aggregate만 쓸지 결정 
        num_sample=10, # 지금 코드에서는 실제로 안 쓰임. 아마 원래는 이웃 샘플링하려고 넣어둔 값으로 추정. 
    ): 
        super().__init__()
        self.gcn = gcn 
        self.policy = policy 
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.num_sample = num_sample
        # weight 파라미터: 학습되는 선형변환 weight 
        self.weight = nn.Parameter(
            torch.FloatTensor(
                embed_dim, 
                self.feat_dim if self.gcn else 2*self.feat_dim, 
            )
        )
        init.xavier_uniform_(self.weight)

    def forward(self, 
        x, 
        Adj, 
    ): 
        # aggregate(x, Adj)로 이웃 feature 집계 
        # 즉, 각 노드마다 "이웃으로부터 집계된 feature"가 하나씩 생김. 
        neigh_feats = self.aggregate(x, Adj)
        # 옵션에 따라, 자기 자신 feature와 concat 
        if not self.gcn: 
            # gcn이 아니면 자신과 concat
            combined = torch.cat([x, neigh_feats], dim=1)
        else: 
            # gcn이면 이웃 정보만 반영 
            combined = neigh_feats 
        # weight로 선형변환 + ReLU + L2 normalize 
        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined, 2, 1)
        # 최종 embedding 
        return combined

    def aggregate(self, 
        x, 
        Adj, 
    ): 
        """
            이웃 feature 집계 
        """
        adj = Variable(Adj).to(Adj.device) # 여기서 Variable은 예전 PyTorch 방식이라 지금은 사실상 불필요. 
        # adj = Adj.to(Adj.device) 로 써도 된다. 
        
        # self-loop 제거 여부 
        # GraphSAGE 스타일일 때 자기 자신을 이웃 집계에서 제외하려는 의도
        # 왜냐하면, GraphSAGE에서는 보통: self feature는 따로 사용. neighbor aggragate은 self 제외. 그래서 나중에 [self;neighbors]를 concat함
        if not self.gcn: 
            n = len(adj)
            adj = adj - torch.eye(n).to(adj.device) 
        # 반대로 gcn=True일 때는 self-loop를 포함한 aggregate를 쓰는 GCN 스타일. 
        if self.policy == 'mean':
            # 각 노드의 이웃 feature 평균을 구하는 코드 
            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)
            to_feats = mask.mm(x)
        elif self.policy == 'max':
            # 각 노드의 이웃 feature들 중에서 feature-wise max pooling을 하는 것. 
            """
                예를 들어 어떤 노드의 이웃 feature가:
                    [1, 5, 2]
                    [3, 2, 4]
                    [0, 7, 1]
                이면 max aggregate는:
                    [3, 7, 4]
            """
            indexs = [i.nonzero() for i in adj==1]
            to_feats = []
            for feat in [x[i.squeeze()] for i in indexs]:
                if len(feat.size()) == 1: 
                    to_feats.append(feat.view(1, -1)) # view: 1행짜리 벡터로 펼쳐라 (열은 자동 계산). 
                else: 
                    to_feats.append(torch.max(feat, 0)[0].view(1, -1))
            to_feats = torch.cat(to_feats, 0)
        return to_feats 


### NB_module: loss 계산 ### 
# count data를 모델링하기 위한 Negative Binomial(NB) / Zero-Inflated Negative Binomial(ZINB) 관련 모듈
# 주로 scRNA-seq gene expression, UMI count, sparse count matrix 같은 데서 많이 사용. 
"""
왜 이런 loss를 쓰는가: 일반 MSE는 count data에 잘 안 맞는다. 
예를 들어 gene count는: 0이 매우 많고, 분산이 평균보다 훨씬 크고, 연속값이 아니라 count임. 
그래서 보통:
- Poisson: 기본 count 분포
- NB: over-dispersion 반영
- ZINB: 0이 유난히 많은 sparse count 반영
을 사용한다. 
"""

class MeanAct(nn.Module):
    # MeanAct: 평균값 파라미터를 양수로 만들기 위한 activation (NB/ZINB의 mean 파라미터를 만드는 activation)
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    # DispAct: dispersion 파라미터를 양수로 만들기 위한 activation
    # dispersion: NB 분포는 평균만 있는 게 아니라 분산을 조절하는 파라미터인 dispersion이 있다. 이 dispersion 덕분에 Poisson보다 더 유연하게 분산을 표현 가능하다. 
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4) # softplus(x) = log(1 + exp(x)). 항상 양수, exp보다 완만해서 더 안정적. dispersion처럼 양수이면서 너무 급격하지 않게 만들고 싶은 값에 사용. 

def NB_loss(x, h_r, h_p):
    # NB_loss: Negative Binomial likelihood 기반 loss (NB 분포의 log-likelihood를 기반으로 한 loss를 계산)
    """
    x: 실제 count 데이터
    h_r: NB의 count/dispersion 관련 latent parameter의 log-space 표현
    h_p: 성공확률 또는 로짓 비슷한 latent parameter

    모델이 예측한 NB 분포 아래에서 실제 count x가 얼마나 그럴듯한지 계산하고,
    그 log-likelihood를 최대화한다. 
    """
    ll = torch.lgamma(torch.exp(h_r)+x) - torch.lgamma(torch.exp(h_r))
    ll += h_p * x - torch.log(torch.exp(h_p)+1) * (x+torch.exp(h_r))
    loss = -torch.mean(torch.sum(ll, axis=-1))
    return loss

def ZINB_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
    # ZINB_loss: zero inflation까지 고려한 loss (Zero-Inflated Negative Binomial loss)
    # ZINB는 NB에다가 “추가적인 zero 생성 과정”을 하나 더 붙인 모델
    """
    즉, 0이 나오는 이유가 2가지라고 본다. 
        1. 원래 NB에서 우연히 0이 나옴
        2. 별도의 dropout/zero inflation mechanism 때문에 0이 나옴
    scRNA-seq에서 굉장히 자주 쓰는 관점이다. 
    """

    # 샘플별 스케일 차이를 반영 (예를 들어 scRNA-seq에서는 cell마다 total count가 달라서, mean을 그냥 쓰지 않고 scale factor를 곱해 조정)
    eps = 1e-10
    if isinstance(scale_factor,float):
        scale_factor=np.full((len(mean),),scale_factor)
    scale_factor = scale_factor[:, None]
    mean = mean * scale_factor

    # NB 부분 계산 (“기본 NB에서 이 관측값이 나올 비용”을 계산)
    t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
    t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
    nb_final = t1 + t2

    # non-zero case (mixing probability (1 - pi)를 반영해서 likelihood를 조정)
    nb_case = nb_final - torch.log(1.0-pi+eps)
    # zero case 
    zero_nb = torch.pow(disp/(disp+mean+eps), disp) 
    zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps) 
    # (i) zero inflation branch에서 0이 나옴: 확률 pi
    # (ii) NB branch에서 우연히 0이 나옴: 확률 (1-pi) * P_NB(0)

    # 각 원소가 0인지 아닌지에 따라 loss를 다르게 계산 
    # 0인 관측값은 ZINB식, 0이 아닌 관측값은 NB식. 
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

    # ridge penalty: pi가 너무 커지는 걸 막기 위한 regularization. 필요한 이유는 모델이 너무 쉽게 “다 zero inflation 때문이야”라고 해버릴 수 있기 때문. 
    # 그래서 pi를 무작정 크게 만드는 걸 억제하려고 L2 penalty를 넣는다. 
    if ridge_lambda > 0:
        ridge = ridge_lambda*torch.square(pi)
        result += ridge
    
    # 최종 평균 (전체 batch와 feature에 대해 평균 loss를 반환)
    result = torch.mean(result)
    return result


### transformer ### 
# skip detailed comments 

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    # @get_local('attn')
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class attn_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attn=PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff=PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


### Hist2ST ###

class convmixer_block(nn.Module): 
    # ConvMixer 구조 
    """
        depthwise conv + pointwise conv 로 
        spatial + channel mixing 을 분리해서 수행하는 것. 

        spatial 정보는 depthwise conv로, 
        channel 정보는 pointwise conv로 섞는다. 

        ConvMixer의 아이디어: CNN+Transformer의 장점 결합 
        - CNN: local pattern 잘 잡음. channel mixing 약함. <- depthwise 
        - Transformer: globl mixing 잘함, channel mixing 강함. <- pointwise
    """
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.dw = nn.Sequential(
            # group=dim : depthwise convolution 
            # 각 채널을 독립적으로 처리 (channel 간 mixing 없음)
            # 공간 정보 (spatial information) 학습 
            nn.Conv2d(dim, dim, kernel_size, group=dim, padding="same"),
            nn.BatchNorm2d(dim),
            nn.GELU(), 
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.BatchNorm2d(dim), 
            nn.GELU(), 
        ) 
        self.pw = nn.Sequential(
            # kernel_size=1 : pointwise conv
            # channel간 정보 mixing. 채널 간 상호작용 학습. 
            # feature combination. cross-channel correlation. 
            nn.Conv2d(dim, dim, kernel_size=1), 
            nn.GELU(), 
            nn.BatchNorm2d(dim), 
        )

    def forward(self, x):
        x = self.dw(x) + x # skip connection, feature preservation. depthwise conv는 정보 손실이 생기기 쉬워서, residual로 보완. 
        x = self.pw(x) 
        return x  


class mixer_transformer(nn.Module):
    """ 
    CNN(ConvMixer) + Transformer attention + Graph layer(GraphSAGE/GCN류) + Jumping Knowledge aggregation을 한 모델. 
    : 이미지 patch feature를 먼저 뽑고, context/token 정보와 attention으로 섞고, 그래프 구조로 노드 간 관계까지 반영해서 최종 embedding g를 만드는 구조
    """ 
    def __init__(self, 
        channel=32, # ConvMixer 쪽 feature map channel 수
        kernel_size=5, # ConvMixer depthwise conv 커널 크기
        dim=1024, # attention, graph block에서 쓰는 feature 차원
        depth1=2, # ConvMixer block 개수
        depth2=8, # Transformer attention block 개수
        depth3=4, # graph block 개수
        heads=8, # multi-head attention 설정
        dim_head=64, # multi-head attention 설정
        mlp_dim=1024, #  # Transformer block 내부 MLP 차원
        dropout=0., 
        policy='mean', # graph aggregation 방식
        gcn=True, # graph block에서 GCN 스타일로 할지 여부
    ):  
        super().__init__()
        # layer1: ConvMixer stage
        # 입력 x에 대해: local spatial pattern 추출, channel mixing, residual conv mixing.
        # 이미지 feature map 정제 단계. ConvMixer를 통해 patch 내부의 texture, morphology, local structure를 더 잘 섞어줌. 
        self.layer1 = nn.Sequential(
            *[convmixer_block(channel, kernel_size) for i in range(depth1)]
        )
        # layer2: attention stage
        # ConvMixer가 local spatial feature를 잘 다룬다면, attention은 global interaction / token 간 관계를 다루는 쪽. 
        # 즉, 각 샘플/patch/node 간 관계, context token과의 상호작용, long-range dependency를 반영하는 단계. 
        self.layer2 = nn.Sequential( 
            *[attn_block(dim, heads, dim_head, mlp_dim, dropout) for i in range(depth2)]
        )
        # layer3: graph stage
        # attention으로 한 번 contextualized된 feature를, 이번에는 adjacency matrix 기반 그래프 관계로 다시 업데이트하는 단계. 
        # 연결된 이웃 노드의 정보 집계, graph-aware node embedding 생성을 수행. 
        self.layer3 = nn.ModuleList(
            [gs_block(dim, dim, policy, gcn) for i in range(depth3)]
        )
        # jknet: Jumping Knowledge 스타일 집계. 
        """
            graph layer를 여러 번 거치면:
            얕은 layer는 local 정보가 많고, 
            깊은 layer는 더 넓은 neighborhood 정보가 반영됨.
            그래서 각 graph layer 출력들을 다 모아두고,
            LSTM으로 순차적으로 읽어서 더 좋은 통합 표현을 만들려는 것. 
            여러 레이어의 representation을 버리지 않고 합치는 방식. 
        """
        self.jknet = nn.Sequential(
            nn.LSTM(dim, dim, 2), 
            SelectItem(0), # 여기서 SelectItem(0)은 아마 output만 꺼내려는 모듈일 것. 
        )
        # down: 차원 정리
        # ConvMixer 출력 feature map을 낮은 채널 수로 줄이고 1차원으로 펴는 단계. 
        self.down = nn.Sequential(
            nn.Conv2d(channel, channel//8, 1, 1), # 1x1 conv: channel reduction
            nn.Flatten(), # Flatten: attention/graph에 넣기 좋게 vector화
        )

    def forward(self, x, ct, adj):
        # x (image-like feature map)
        # → layer1: ConvMixer blocks
        # → down: 채널 축소 + flatten
        x = self.down(
            self.layer1(x)
        )
        # → ct와 더해서 Transformer attention
        g = x.unsqueeze(0)
        g = self.layer2(g + ct).squeeze(0)
        # → layer3: graph blocks 여러 번 통과
        # → jknet(LSTM)으로 graph layer들의 출력을 통합
        # → mean(0)
        jk = []
        for layer in self.layer3:
            g = layer(g, adj)
            jk.append(g.unsqueeze(0))
        g = torch.cat(jk, 0)
        g = self.jknet(g).mean(0)
        # → 최종 graph-aware representation 반환
        return g 


class ViT(nn.Module):
    """
    backbone 
    Dropout → mixer_transformer 호출만 하는 간단한 wrapper 
    """
    def __init__(self, channel=32,kernel_size=5,dim=1024, 
                 depth1=2, depth2=8, depth3=4, 
                 heads=8, mlp_dim=1024, dim_head = 64, dropout = 0.,
                 policy='mean',gcn=True
                ):
        super().__init__()
        # 입력 feature에 dropout 적용
        self.dropout = nn.Dropout(dropout)
        # 핵심 모델
        """
        내부적으로:
            ConvMixer (local feature)
            Attention (global relation)
            Graph block (adj 기반 관계)
            JKNet (layer aggregation)
        이 다 수행됨
        """
        self.transformer = mixer_transformer(
            channel, kernel_size, dim, 
            depth1, depth2, depth3, 
            heads, dim_head, mlp_dim, dropout,
            policy,gcn,
        )

    def forward(self,x,ct,adj):
        x = self.dropout(x)
        x = self.transformer(x,ct,adj)
        return x


class Hist2ST(pl.LightningModule):
    """
    PyTorch Lightning 기반 spatial transcriptomics 예측 모델. 
    입력으로 histology patch 이미지 + patch 위치(center) + 그래프 adjacency를 받고, 
    출력으로 gene expression을 예측. 
    필요하면 NB/ZINB 분포 기반 보조 loss와 augmentation distillation까지 같이 쓰도록 설계되어 있다. 
    """
    """
    pl.LightningModule을 상속했기 때문에, 단순 모델 정의뿐 아니라
    forward, training_step, validation_step, configure_optimizers
    까지 포함해서 학습 전체를 관리하는 모델 클래스 
    """
    def __init__(self, learning_rate=1e-5, fig_size=112, label=None, 
                 dropout=0.2, n_pos=64, kernel_size=5, patch_size=7, n_genes=785, 
                 depth1=2, depth2=8, depth3=4, heads=16, channel=32, 
                 zinb=0, nb=False, bake=0, lamb=0, policy='mean', 
                ):
        super().__init__()
        # self.save_hyperparameters()
        dim=(fig_size//patch_size)**2*channel//8
        self.learning_rate = learning_rate
        
        # 옵션들은 loss와 auxiliary branch를 제어
        """
        nb: NB loss 쓸지 여부
        zinb: ZINB/NB 보조 loss 가중치처럼 사용
        bake: augmentation distillation 횟수
        lamb: bake loss 가중치
        label: validation 때 clustering metric 계산용 label
        """
        self.nb=nb
        self.zinb=zinb
        self.bake=bake
        self.lamb=lamb
        self.label=label

        # patch embedding 
        self.patch_embedding = nn.Conv2d(3,channel,patch_size,patch_size)
        # position embedding 
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        # backbone 
        self.vit = ViT(
            channel=channel, kernel_size=kernel_size, heads=heads,
            dim=dim, depth1=depth1,depth2=depth2, depth3=depth3, 
            mlp_dim=dim, dropout = dropout, policy=policy, gcn=True,
        )

        # settings 
        self.channel=channel
        self.patch_size=patch_size
        self.n_genes=n_genes

        # ZINB/NB branch
        if self.zinb>0:
            if self.nb:
                self.hr=nn.Linear(dim, n_genes)
                self.hp=nn.Linear(dim, n_genes)
            else:
                self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
                self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
                self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())
        # bake branch
        # augmentation을 여러 번 적용한 결과를 가중합(distillation)할 때 쓰는 coefficient predictor
        # 즉 여러 augmentation 결과 중 어떤 게 더 믿을 만한지 weight를 주려는 구조 
        if self.bake>0:
            self.coef=nn.Sequential(
                nn.Linear(dim,dim),
                nn.ReLU(),
                nn.Linear(dim,1),
            )

        # gene head 
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )

        # augmentation distillation 용 transform 
        self.tf=tf.Compose([
            tf.RandomGrayscale(0.1),
            tf.RandomRotation(90),
            tf.RandomHorizontalFlip(0.2),
        ])
    
    def forward(self, patches, centers, adj, aug=False):
        # patch image
        B,N,C,H,W=patches.shape
        patches=patches.reshape(B*N,C,H,W)
        # → patch embedding
        patches = self.patch_embedding(patches)
        # → 위치 임베딩(center x,y)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        ct=centers_x + centers_y
        # → ViT / mixer-transformer 계열 backbone
        h = self.vit(patches,ct,adj)
        # → gene head
        # → gene expression 예측
        x = self.gene_head(h)
        # (+ 선택적으로 NB/ZINB 파라미터도 예측)
        extra=None
        if self.zinb>0:
            if self.nb:
                r=self.hr(h)
                p=self.hp(h)
                extra=(r,p)
            else:
                m = self.mean(h)
                d = self.disp(h)
                p = self.pi(h)
                extra=(m,d,p)
        if aug:
            h=self.coef(h)
        return x,extra,h

    def aug(self,patch,center,adj):
        bake_x=[]
        for i in range(self.bake):
            new_patch=self.tf(patch.squeeze(0)).unsqueeze(0)
            x,_,h=self(new_patch,center,adj,True)
            bake_x.append((x.unsqueeze(0),h.unsqueeze(0)))
        return bake_x

    def distillation(self,bake_x):
        new_x,coef=zip(*bake_x)
        coef=torch.cat(coef,0)
        new_x=torch.cat(new_x,0)
        coef=F.softmax(coef,dim=0)
        new_x=(new_x*coef).sum(0)
        return new_x
    
    def training_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_ = batch
        adj=adj.squeeze(0)
        exp=exp.squeeze(0)
        pred,extra,h = self(patch, center, adj)
        
        mse_loss = F.mse_loss(pred, exp)
        self.log('mse_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)
        bake_loss=0
        if self.bake>0:
            bake_x=self.aug(patch,center,adj)
            new_pred=self.distillation(bake_x)
            bake_loss+=F.mse_loss(new_pred,pred)
            self.log('bake_loss', bake_loss,on_epoch=True, prog_bar=True, logger=True)
        zinb_loss=0
        if self.zinb>0:
            if self.nb:
                r,p=extra
                zinb_loss = NB_loss(oris.squeeze(0),r,p)
            else:
                m,d,p=extra
                zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
            self.log('zinb_loss', zinb_loss,on_epoch=True, prog_bar=True, logger=True)
            
        loss=mse_loss+self.zinb*zinb_loss+self.lamb*bake_loss
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_ = batch
        def cluster(pred,cls):
            sc.pp.pca(pred)
            sc.tl.tsne(pred)
            kmeans = KMeans(n_clusters=cls, init="k-means++", random_state=0).fit(pred.obsm['X_pca'])
            pred.obs['kmeans'] = kmeans.labels_.astype(str)
            p=pred.obs['kmeans'].to_numpy()
            return p
        
        pred,extra,h = self(patch, center, adj.squeeze(0))
        if self.label is not None:
            adata=ann.AnnData(pred.squeeze().cpu().numpy())
            idx=self.label!='undetermined'
            cls=len(set(self.label))
            x=adata[idx]
            l=self.label[idx]
            predlbl=cluster(x,cls-1)
            self.log('nmi',nmi_score(predlbl,l))
            self.log('ari',ari_score(predlbl,l))
        
        loss = F.mse_loss(pred.squeeze(0), exp.squeeze(0))
        self.log('valid_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        
        pred=pred.squeeze(0).cpu().numpy().T
        exp=exp.squeeze(0).cpu().numpy().T
        r=[]
        for g in range(self.n_genes):
            r.append(pearsonr(pred[g],exp[g])[0])
        R=torch.Tensor(r).mean()
        self.log('R', R, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict

