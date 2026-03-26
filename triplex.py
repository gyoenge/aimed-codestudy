# just model impl practice 
# TRIPLEX source: https://github.com/NEXGEM/TRIPLEX 

# TRIPLEX 포인트: target/neighbor/global 관점을 종합적으로 판단한다. 
""" 
예를 들어, 
target: tumor처럼 보임
neighbor: 정상 조직 많음
global: 전체적으로 normal 
→ fusion: "이건 tumor가 아닐 수도 있음"
""" 

import itertools

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from einops import rearrange 
# einops.rearrange = 텐서의 shape을 직관적인 문자열 패턴으로 바꾸는 함수 (reshape + permute + view를 한 번에) 
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
# Flash Attention을 실행하기 위한 고속 attention 함수 2개
# flash_attn_qkvpacked_func(qkv, ...): Q/K/V가 하나로 합쳐진 상태에서 사용한다. (더 빠름)
# flash_attn_func(q, k, v, ...): 일반적인 Q, K, V separate 입력용. 이미 Q/K/V를 따로 만든 경우 사용한다. 
"""
Flash Attention 내부:
block-wise:
Q block
K block
V block
→ partial softmax
→ 누적 계산
**특징: QK^T 전체 matrix 안 만듦. 
"""

try: 
    import MinkowskiEngine as ME
    HAS_MINKOWSKI = True
except ImportError: 
    HAS_MINKOWSKI = False 
# MinkowskiEngine: sparse convolution 라이브러리
# 희소 데이터 (sparse) -> 빈 공간 계산 안 함, 메모리 효율 up.  
# 사용 예) point cloud, 3d voxel, irregular grid (WSI/ST 일부 상황)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# FutureWarning(앞으로 바뀔 예정인 기능 경고)을 출력하지 않도록 숨기는 코드 

# ...

### Sub Modules 

class PreNorm(nn.Module):
    # 입력을 LayerNorm으로 먼저 정규화한 뒤, 실제 연산(fn: attention/MLP 등)을 수행하는 wrapper 모듈 
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn 
    def forward(self, x, **kwargs):
        x = self.norm(x)

        # 이건 cross-attention을 위한 처리. 
        # cross-attention에서 query = x, key/value = x_kv인데, 둘 다 scale 맞춰야 안정적이다. 
        # 따라서, query, key/value 모두 정규화한다. 
        if 'x_kv' in kwargs.keys():
            kwargs['x_kv'] = self.norm(kwargs['x_kv'])

        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    # Transformer에서 attention 다음에 붙는 블록 
    # 각 token을 **독립적으로 처리해서 feature를 더 풍부하게 만든다. 
    # attention만 있으면 정보 섞기만 하고, 표현력 부족. 비선형 변환으로 더 복잡한 패턴 학습 가능하도록 한다. 
    def __init__(self, emb_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),  # feature 확장 
            nn.GELU(),  # ReLU보다 부드러운 활성화 함수 
            nn.Dropout(dropout),  # overfitting 방지 
            nn.Linear(hidden_dim, emb_dim), 
            # GELU? --> 최종 출력이기 때문에 비선형을 넣지 않는다. 출력도 비선형으로 왜곡되는 것을 방지. 
            # FFN의 목적: 두 번째 Dropout = 최종 출력(feature) 정규화. 
            nn.Dropout(dropout)  # 추가 regularization 
        ) 
    def forward(self, x): 
        return self.net(x) 

class MultiHeadAttention(nn.Module):
    # 공부 포인트: 최적화 어떻게 했는지 
    """
    Multi-Head Attention = 여러 개의 attention을 병렬로 수행해서 다양한 관계를 동시에 학습하는 방식
    같은 입력을 여러 개로 나눠서 각각 다른 관점에서 본다

    입력 x (B, N, D)
        ↓
    Linear → Q, K, V 생성
        ↓
    head별로 쪼갬
        ↓
    각 head에서 attention 수행
        ↓
    concat
        ↓
    Linear projection
    
    """
    def __init__(self, emb_dim, heads=4, dropout=0., attn_bias=False, resolution=(5,5), flash_attn=False): 
        super().__init__()

        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'
        # 각 head에 동일한 차원을 나눠주기 위해서, emb_dim이 heads로 나누어 떨어져야 함. 

        self.flash_attn = flash_attn
        # Flash Attention = GPU 메모리와 연산을 최적화한 빠른 attention 알고리즘 
        # 일반 attention은 메모리를 많이 쓰기 때문에, attention을 블록 단위로 효율적으로 계산하도록 한다. 
        # GPU 환경에서 sequence 길이 크거나 batch가 큰 경우 사용하면 좋다. 

        dim_head = emb_dim // heads 
        project_out = not (heads == 1)  # head가 1개면 projection 안 하고, 2개 이상이면 projection한다. 

        self.heads = heads 
        self.drop_p = dropout  # attention 계산에서의 dropout 
        self.scale = dim_head ** -0.5  # 즉, self.scale = 1 / sqrt(dim_head). attention score 계산 후 scaling에 사용. 
        self.attend = nn.Softmax(dim = -1)  # 마치막 차원에 대해 softmax 적용. 

        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias = False)
        # 입력 x로부터 Q, K, V를 한 번에 만드는 레이어. 나중에 쪼개서 사용 
        # (B, N, 3 * emb_dim) -> (B, N, emb_dim) -> 각 (B, N, 3 * emb_dim) for Q/K/V 
        # 입력 token을 attention 계산용 표현으로 바꾸는 projection layer이다. 

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  
        # attention 결과는 여러 head에서 나온 값을 합쳐서 다시 (B, N, emb_dim)으로 만들게 되는데,
        # 그 뒤에 마지막으로 한 번 더 linear projection을 해준다.

        self.attn_bias = attn_bias 
        if attn_bias: 
            points = list(itertools.product(range(resolution[0]), range(resolution[1])))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])
            # attention_offsets: 두 token 간 상대 위치. 
            self.attention_biases = torch.nn.Parameter(torch.zeros(heads, len(attention_offsets)))
            # 각 attention head마다, 상대 위치(또는 관계)에 따라 다른 가중치를 주기 위한 learnable bias 테이블 
            # shape: (heads, num_offsets) 
            self.register_buffer('attention_bias_idxs', 
                                 torch.LongTensor(idxs).view(N, N),
                                 persistent=False) 
            # 각 token pair에 대해 어떤 offset index를 써야 하는지 담은 lookup table
            # bias index는 buffer로 등록한다. 

    @torch.no_grad()
    def train(self, mode=True):
        if self.attn_bias:
            super().train(mode) 
            # nn.Module의 train() 함수: 
            # model.train() → mode=True, model.eval() → mode=False.  
            if mode and hasattr(self, 'ab'):  # train 시 
                del self.ab
                # ab = attention bias를 미리 계산해서 저장해둔 캐시 (cached attention bias) 
                # del self.ab = 객체(self)에서 ab라는 속성(attribute)을 삭제하는 것
            else:  # inference 시 
                self.ab = self.attention_biases[:, self.attention_bias_idxs]
    # train/eval 모드에 따라 attention bias를 동적으로 처리해서 성능을 최적화하는 로직. 
    # 학습 중에는 매번 bias를 동적으로 계산하고, 
    # 추론 시에는 미리 계산해서 캐싱(ab)해 속도를 높인다. 
    # [Q.]

    def forward(self, x, mask=None, return_attn=False):

        qkv = self.to_qkv(x)  # b x n x d*3

        if self.flash_attn:
            qkv = rearrange(qkv, 'b n (h d a) -> b n a h d', h = self.heads, a=3)
            # 하나로 합쳐진 QKV 벡터를 → (Q, K, V)로 나누고 + multi-head 형태로 분해 
            # (h d a): emb_dim * 3 = heads × dim_head × 3 
            # a h d : (QKV 구분) → head → feature 
            # 결과 shape: (B, N, 3, heads, dim_head)
            out = flash_attn_qkvpacked_func(qkv, self.drop_p, softmax_scale=None, causal=False)
            # QKV가 합쳐진 텐서를 입력으로 받아, Flash Attention으로 빠르게 attention 결과를 계산하는 함수
            # self.drop_p : attention weight에 dropout 적용. softmax 결과 일부를 랜덤하게 제거. 
            # softmax_scale=None : scaling 값 관련. 직접 지정하지 않고, 자동으로 sqrt(d)로 scaling 적용한다는 의미.  
            # causal=False : 모든 token이 서로 볼 수 있다. (True면 i번째 token은 i이후 못 본다는 의미). 
            out = rearrange(out, 'b n h d -> b n (h d)')

        else: 
            qkv = qkv.chunk(3, dim=-1)
            # 마지막 차원을 기준으로 텐서를 3등분해서 (Q, K, V)로 나누는 코드
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
            # Q, K, V를 multi-head attention용 shape로 변환 
            # qkv 각각을 (B, N, D) → (B, heads, N, dim_head) 형태로 

            qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            # 각 token끼리 얼마나 비슷한지(유사도)를 계산한 attention score 
            # k.transpose(-1, -2): 마지막 두 차원을 바꾼다.  
            # qk shape: (B, heads, N, N) 
            if self.attn_bias: 
                qk += (self.attention_biases[:, self.attention_bias_idxs]
                       if self.training else self.ab) 
            # bias 추가 

            if mask is not None: 
            # mask 처리 
                fill_value = torch.finfo(torch.float16).min  # 엄청 작은 값 (-무한대처럼 사용)
                ind_mask = mask.shape[-1]  # mask의 마지막 차원 크기 = mask 길이 (mask 대상 토큰 수)
                qk[:, :, -ind_mask:, -ind_mask:] = qk[:, :, -ind_mask:, -ind_mask:].masked_fill(mask==0, fill_value) 
                # qk에서 마지막 M query/key 영역에 대해서만 마스킹 (뒤쪽 토큰들끼리의 attention만 제한)
                # mask == 0 (False 위치 찾기), 그 위치를 fill_value (-무한대) 으로 채운다. 
                # attention score (qk)에 mask를 적용해서 특정 위치의 attention을 강제로 막는 부분이다. 

            attn_weights = self.attend(qk)  # b h n n 
            # softmax 적용 (attention matrix) --> 각 query에 대해 합이 1임 (확률 분포로 변환됨)
            if return_attn: 
                attn_weights_averaged = attn_weights.mean(dim=1) 
            # 여러 head 평균 (dim=1: head dimension) 
            # query가 각 key를 얼마나 참고할지 비율 --> 분석용 attention map 

            out = torch.matmul(attn_weights, v)  # value 곱해줌  (b,h,n,n)@(b,h,n,d)->(b,h,n,d)
            out = rearrange(out, 'b h n d -> b n (h d)') # emb_dim = h*d 

            if return_attn: 
                return self.to_out(out), attn_weights_averaged[:,0]
            # self.to_out(out)은 head들을 concat한 뒤 한 번 더 선형변환하는 단계. 

        return self.to_out(out) 


class MultiHeadCrossAttention(nn.Module):
    # Q 와 K/V 가 다름 
    def __init__(self, emb_dim, heads=4, dropout=0., flash_attn=False):
        super().__init__()

        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'

        self.flash_attn = flash_attn 

        dim_head = emb_dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5 
        self.attend = nn.Softmax(dim = -1)
        
        # cross-attn이기 때문에 Q / KV 무조건 따로 
        self.to_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_kv = nn.Linear(emb_dim, emb_dim*2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), 
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() 

    def forward(self, x_q, x_kv, mask=None, return_attn=False):
        q = self.to_q(x_q)
        kv = self.to_kv(x_kv).chunk(2, dim=-1)

        if self.flash_attn:
            q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), kv)

            out = flash_attn_func(q, k, v)
            # multi-head로 나눈 q,k,v를 flash attention 
            out = rearrange(out, 'b n h d -> b n (h d)')
            # 다시 하나의 embedding으로 합침 

        else: 
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

            qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            if mask is not None: 
                fill_value = torch.finfo(torch.float16).min
                ind_mask = mask.shape[-1]
                qk[:,:,-ind_mask:,-ind_mask:] = qk[:,:,-ind_mask:,-ind_mask:].masked_fill(mask==0, fill_value)

            attn_weights = self.attend(qk)  # b h n n 
            if return_attn:
                attn_weights_averaged = attn_weights.mean(dim=1)

            out = torch.matmul(attn_weights, v)
            out = rearrange(out, 'b h n d -> b n (h d)')

            if return_attn:
                return self.to_out(out), attn_weights_averaged[:0] 
            
        return self.to_out(out)


class PosMLP(nn.Module):
    # MLP Positional Encoding 
    def __init__(self, input_dim=2, embed_dim=1024, hidden_dim=512, grid_size=None):
        super(PosMLP, self).__init__() 
        self.grid_size = grid_size  # (H, W) or None  
        # 좌표를 grid로 변환할 때 사용. 없으면 나중에 자동 추정. 

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),  # inplace=True: 메모리 절약용. 입력 tensor를 바로 덮어씀. 
            nn.Linear(hidden_dim, embed_dim)  # 출력 벡터 크기를 transformer embedding 차원과 맞춤 
        )
        # PosMLP는 정보가 아니라 좌표(구조)를 넣는 것이기 때문에 dropout을 잘 안쓴다. 

    def forward(self, x, pos):
        B = x.shape[0]
        device = x.device 

        if self.grid_size is None: 
            self.grid_size = self.infer_grid_size(pos, rounding_factor=20) 
            # 좌표(pos)로부터 grid 크기를 자동으로 추정 (WSI/ST처럼 불규칙 좌표를 격자(grid)로 바꾸기 위함). 
        W, H = self.grid_size 

        pos_min = pos.min(dim=0, keepdim=True)[0]  # keepdim=True: 연산 후에도 차원을 유지해서 shape을 맞춰주는 옵션. 
        pos_max = pos.max(dim=0, keepdim=True)[0]  # (N, 2)->(1,2) 유지. (2,)가 되는 걸 방지.  
        pos_norm = (pos - pos-min) / (pos_max - pos_min + 1e-5)
        grid_pos = pos_norm * torch.tensor([W-1, H-1], device=device)
        grid_pos = grid_pos.round() 
        # pos를 grid 좌표로 변환 

        pos_emb = self.mlp(grid_pos) 
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (N,D)->(1,N,D)-> (B,N,embed_dim)
        # grid_pos를 embedding으로 (MLP) 

        out = x + pos_emb 
        # feature + 위치 정보 결합. (B, N, D)

        return out 

    def infer_grid_size(self, pos, rounding_factor=None):
        if rounding_factor is None:
            rounding_factor = self.dynamic_rounding_factor(pos)
        # grid를 만들 때 좌표를 얼마나 묶을지 (해상도), 자동으로 결정하는 것. 
        # [Q.] 구현 어디에?? 

        pos_rounded = (pos / rounding_factor).round() * rounding_factor 
        # 좌표를 일정 간격으로 묶어서(반올림해서) grid로 만듦. --> x축과 y축의 칸 개수를 구한다.
        unique_x = torch.unique(pos_rounded[:, 0]) 
        unique_y = torch.unique(pos_rounded[:, 1]) 
        # x/y 좌표들 중 중복 제거된 값들 
        W = unique_x.numel() 
        H = unique_y.numel() 
        # numel() = number of elements (원소 개수)
        return (W, H) 

    @staticmethod
    def dynamic_rounding_factor(pos, base=1, scale_ref=1000.0):
        pos_range = (pos.max(0)[0] - pos.min(0)[0]).max().item()
        scale_ratio = pos_range / scale_ref
        rounding_factor = base * scale_ratio
        return rounding_factor


class TransformerEncoder(nn.Module): 
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., attn_bias=False, resolution=(5,5), flash_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(emb_dim, MultiHeadAttention(emb_dim, heads=heads, dropout=dropout, attn_bias=attn_bias, resolution=resolution, flash_attn=flash_attn)),
                    PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
                ])
            )
        # ModuleList로 관리하는 이유: 여러 layer를 저장하면서, 실행 흐름은 직접 제어하기 위해. (self/Sequential 부적합)
        # PreNorm: 연산 전에 정규화해서 학습을 안정화, 요즘 표준. (PostNorm에서는 gradient가 residual을 통과하면서 깨진다 -> 깊어질수록 학습 불안정)
    
    def forward(self, x, mask=None, return_attn=False):
        for attn, ff in self.layers:
            # self.layers 안에 [attn, ff] 형태로 저장되어 있어서 for문에서 자동으로 두 개로 unpack 된다. 
            if return_attn: 
                attn_out, attn_weights = attn(x, mask=mask, return_attn=return_attn)
                x += attn_out  # residual connection after attention 
                x = ff(x) + x  # residual connection after feed forward net 
            
            else: 
                x = attn(x, mask=mask) + x  # residual connection after attention 
                x = ff(x) + x  # residual connection after feed forward net 
            
        if return_attn:
            return x, attn_weights 
        else: 
            return x 


class CrossEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0.): 
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(emb_dim, MultiHeadCrossAttention(emb_dim, heads=heads, dropout=dropout)),
                    PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
                ])
            )

    def forward(self, x_q, x_kv, mask=None, return_attn=False):
        for attn, ff in self.layers:
            if return_attn: 
                attn_out, attn_weights = attn(x_q, x_kv=x_kv, mask=mask, return_attn=return_attn)
                x_q += attn_out  # residual connection after attention 
                x_q = ff(x_q) + x_q  # residual connection after feed forward net 
            else: 
                x_q = attn(x_q, x_kv=x_kv, mask=mask) + x_q
                x_q = ff(x_q) + x_q  # residual connection after feed forward net

        if return_attn:
            return x_q, attn_weights
        else: 
            return x_q 


class APEG(nn.Module):
    # APEG: Atypical Position Encoding Generator 
    # 좌표 기반으로 feature를 2D grid에 올린 뒤, Conv로 공간 정보를 반영하고 다시 토큰으로 가져오는 모듈
    def __init__(self, dim=512, kernel_size=3, grid_size=None, use_sparse=False, sparse_resolution=128):
        super(APEG, self).__init__() 
        self.use_sparse = use_sparse 
        self.grid_size = grid_size  # (H,W) or None 
        self.sparse_resolution = sparse_resolution  

        if self.use_sparse:
            if not HAS_MINKOWSKI: 
                raise ImportError("MinkowskiEngine is not installed. Install it via 'pip install MinkowskiEngine.") 
            self.proj = ME.MinkowskiConvolution(
                in_channels=dim, 
                out_channels=dim, 
                kernel_size=kernel_size, 
                stride=1, 
                dimension=2, 
                bias=True 
            )
            # [Q.]
        else: 
            self.proj = nn.Conv2d(
                dim, dim, 
                kernel_size=kernel_size, 
                padding=kernel_size//2,
                bias=True, 
                groups=dim  # depthwise 
            )
            # 입력 (B, dim, H, W) -- grid 형태 feature map
            # 출력 (B, dim, H, W) -- spatial 정보 반영된 feature 
            # depthwise : groups=in_channels=dim : 각 채널을 독립 처리 (일반 conv는 channel간 서로 섞이는 것과 대비됨.)
    
    def infer_grid_size(self, pos, rounding_factor=None):
        if rounding_factor is None:
            rounding_factor = self.dynamic_rounding_factor(pos)
        
        pos_rounded = (pos / rounding_factor).round() * rounding_factor
        unique_x = torch.unique(pos_rounded[:, 0])
        unique_y = torch.unique(pos_rounded[:, 1])
        W = unique_x.numel()
        H = unique_y.numel()
        return (W, H)
    
    @staticmethod
    def dynamic_rounding_factor(pos, base=1, scale_ref=1000.0):
        pos_range = (pos.max(0)[0] - pos.min(0)[0]).max().item()
        scale_ratio = pos_range / scale_ref
        rounding_factor = base * scale_ratio
        return rounding_factor
    
    def forward(self, x, pos):
        # x: (1,N,dim). pos: (N,2)
        # x_out: (1,N,dim)

        B, N, C = x.shape
        device = x.device 

        if self.use_sparse: 
            # pos (연속좌표)
            pos_min = pos.min(dim=0, keepdim=True)[0]
            pos_max = pos.max(dim=0, keepdim=True)[0]
            pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-5)
            # normalize 좌표 

            discrete_coords = (pos_norm * (self.sparse_resolution -1)).round().int()
            # discrete grid 좌표  

            batch_indices = torch.zeros((N,1), dtype=torch.int, device=device)
            coords = torch.cat([batch_indices, discrete_coords], dim=1)  # (N, 1+2)
            # (batch + 좌표) 결합
            
            sparse_input = ME.SparseTensor(
                features=x.squeeze(0),
                coordinates=coords,
            )
            # SparseTensor 생성 

            sparse_output = self.proj(sparse_input)
            # sparse conv 적용 (결과: 좌표+feature)
            # 그러나 순서가 원래 token 순서랑 다르다. 
            # 따라서 좌표 기준으로 다시 찾아서 정렬해야 한다. 

            # Matching input coords and output coords
            out_features = sparse_output.features  # (N_out, C) 
            out_coords = sparse_output.coordinates[:, 1:]  # (N_out, 2) 
            # output 분리 

            match_idx = [] 
            for i in range(N):
                match = ((out_coords == discrete_coords[i]).all(dim=1)).nonzero(as_tuple=True)[0]
                match_idx.append(match.item()) 
            # 이 좌표에 해당한느 feature가 어디에 있는지 찾는다. 

            match_idx = torch.tensor(match_idx, device=device, dtype=torch.long)
            matched_features = out_features[match_idx]
            # tensor로 변환 후 feature 재정렬 (N, C)

            x_out = matched_features.unsqueeze(0) 
            # 결과 (B, N, C) (batch 차원 복구)
            return x_out 
        
        else: 
            if self.grid_size is None: 
                self.grid_size = self.infer_grid_size(pos, rounding_factor=20)  # (W, H) inferred

            W, H = self.grid_size

            pos_min = pos.min(dim=0, keepdim=True)[0]
            pos_max = pos.max(dim=0, keepdim=True)[0]
            pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-5)
            grid_pos = pos_norm * torch.tensor([W - 1, H - 1], device=device)
            grid_pos = grid_pos.round().long()

            gx, gy = grid_pos[:, 0], grid_pos[:, 1]  # (N,), (N,)

            # Flatten 2D grid into 1D index for scatter 
            idx_1d = gy * W + gx  # (N,)

            dense_x = torch.zeros((B, C, H*W), device=device)
            dense_mask = torch.zeros((B, 1, H*W), device=device)

            # Scatter add (optimized)
            for b in range(B):
                dense_x[b] = dense_x[b].scatter_add(1, idx_1d.unsqueeze(0).expand(C, -1), x[b].transpose(0,1))
                dense_mask[b] = dense_mask[b].scatter_add(1, idx_1d.unsqueeze(0), torch.ones(1, N, device=device))

            dense_x = dense_x.view(B, C, H, W)
            dense_mask = dense_mask.view(B, 1, H, W)

            dense_mask = dense_mask.clamp(min=1.0)
            dense_x = dense_x / dense_mask

            x_pos = self.proj(dense_x)

            mask = (dense_mask > 0)
            x_pos = x_pos * mask + dense_x * (~mask)

            # Sampling
            x_out = []
            for b in range(B):
                sampled_feat = x_pos[b, :, grid_pos[:, 1], grid_pos[:, 0]].transpose(0,1)  # (N, C)
                x_out.append(sampled_feat)

            x_out = torch.stack(x_out, dim=0)  # (B, N, C)
            return x_out 


class FourierPositionalEncoding(nn.Module):
    # Fourier Positional Encoding 
    pass 


class SpatialFormerBlock(nn.Module):
    pass 


### Main Modules 

class GlobalEncoder(nn.Module):
    # 공부 포인트: TransformerEncoder, SpatialFormerBlock + PE 

    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., kernel_size=3, pos_method='APEG', flash_attn=True):
        super().__init__() 

        # positional encoding: transformer는 순서/위치 정보 없기 때문에 위치 정보 없으면 공간 관계를 모른다
        # 모두 같은 목적 -- (x,y)좌표를 모델이 이해할 수 있는 embedding으로 변환 
        # TRIPLEX는 feature만 중요한 게 아니라, 어디에 있는 feature인가가 핵심이기 때문에, positional encoding이 성능에 큰 영향을 준다. 
        self.pos_method = pos_method
        if pos_method == 'MLP':
            # 좌표를 MLP 신경망으로 직접 학습해서 embedding으로 변환  
            self.pos_layer = PosMLP(input_dim=2, embed_dim=emb_dim, hidden_dim=emb_dim//2) 
        elif pos_method == 'APEG':
            # 좌표를 직접 embedding하지 않고, feature를 convolution으로 보정 
            # 주변 patch를 보고 위치 느낌을 학습하는 느낌. 
            # local spatial structure를 implicit하게 잘 반영한다. 
            self.pos_layer = APEG(dim=emb_dim, kernel_size=kernel_size)
        elif pos_method == 'None':
            # 위치 정보 아예 사용 안함. spatial 정보 완전히 손실. 
            self.pos_layer = None
        elif pos_method == 'spatialformer':
            # 좌표를 sin/cos 기반 주파수 공간으로 변환. 좌표를 주파수 패턴으로 표현하는 것. 
            # deterministic하고, generalization에 좋지만, 학습 유연성이 낮다. 
            self.pos_layer = FourierPositionalEncoding(coord_dim=2, embed_dim=emb_dim)
            self.spatial_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        else:
            raise ValueError(f"Unknown pos_layer: {pos_method}. Choose 'MLP' or 'APEG' or 'None'.")
        
        if pos_method == 'APEG':
            assert depth > 1, "APEG requires depth > 1."
            # APEG는 보통 attention --> APEG --> attention 처럼, 중간에 끼워넣는 구조이기 때문. 
            
            # APEG requires a grid size for the convolutional layer 
            self.layer1 = TransformerEncoder(emb_dim, 1, heads, mlp_dim, dropout, flash_attn=flash_attn)
            self.layer2 = TransformerEncoder(emb_dim, depth-1, heads, mlp_dim, dropout, flash_attn=flash_attn)

        elif pos_method == 'spatialformer': 
            self.layers = nn.ModuleList([
                SpatialFormerBlock(emb_dim, heads, mlp_dim//emb_dim, dropout, flash_attn=flash_attn)
                for _ in range(depth)
            ])

        else: 
            self.layer = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout, flash_attn=flash_attn)

        self.norm = nn.LayerNorm(emb_dim)
        # 각 token의 feature(emb_dim)를 정규화해서 학습을 안정화 

    def foward_features(self, x, pos=None): 

        if self.pos_method == 'APEG': 
            # Translayer x 1 
            x = self.layer1(x)
            x = self.pos_layer(x, pos)
            # Translayer x (depth-1)
            x = self.layer2(x)

        elif self.pos_method == 'spatialformer':
            # 좌표 -> positional embedding (spatial_pos) -> spatial token 으로 만들고,
            # -> 각 transformer layer에서 feature (x) 와 함께 업데이트하면서 spatial context를 학습하는 구조 
            spatial_pos = self.pos_layer(pos)
            spatial_tokens = spatial_pos + self.spatial_token  # spatial token 은 위치 embedding에 더해지는 학습 가능한 global bias 벡터

            for layer in self.layers:
                x, spatial_tokens = layer(x, spatial_tokens)
        
        else: 
            if self.pos_method == 'MLP':
                x = self.pos_layer(x, pos) 
            
            x = self.layer(x)
        
        x = self.norm(x)

        return x 
    
    def forward(self, x, position):
        x = self.foward_features(x, position)

        return x 


class NeighborEncoder(nn.Module):
    # 공부 포인트: TransformerEncoder + resolution & mask

    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., resolution=(5,5)):
        super().__init__()

        self.layer = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout, attn_bias=True, resolution=resolution)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):

        if mask != None:  # [Q.] if mask is not None? 
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        # Translayer
        x = self.layer(x, mask=mask)
        x = self.norm(x)

        return x


class FusionEncoder(nn.Module):
    # 공부 포인트: CrossEncoder + mask 

    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        self.fusion_layer = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout) 
        # 서로 다른 feature들끼리 attention (일반 Transformer는 self-attention으로 자기 자신끼리만 관계 학습) 
        # Fusion이므로, target-global/neighbor-global끼리 서로서로 보게 만든다. 
        self.norm = nn.LayerNorm(emb_dim) 

    def forward(self, x_t=None, x_n=None, x_g=None, mask=None):
        
        if mask != None: 
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        # Target token ; global이 target을 봄 
        fus1 = self.fusion_layer(x_g.unsqueeze(1), x_t)

        # Neighbor token ; global이 neighbor을 봄
        # mask 적용: 유효한 neighbor만 attention한다. 
        fus2 = self.fusion_layer(x_g.unsqueeze(1), x_n, mask=mask)

        # 둘 다 global이 중심으로 global에 target/neighbor정보를 반영하는 것임. 

        fusion = (fus1 + fus2).unsqueeze(1)
        # target 정보와 neighbor 정보를 합침. 
        fusion = self.norm(fusion)

        return fusion 


def load_model_weights(ckpt: str):
    pass 

class TRIPLEX(nn.Module):
    def __init__(self,
                 num_genes=250,
                 emb_dim=512,
                 depth1=2,  # depth of FusionEncoder 
                 depth2=2,  # depth of GlobalEncoder 
                 depth3=2,  # depth of NeighborEncoder # target과 가장 가까운 걸 의미 
                 num_heads1=8,
                 num_heads2=8, 
                 num_heads3=8, 
                 mlp_ratio1=2.0,
                 mlp_ratio2=2.0,
                 mlp_ratio3=2.0,
                 dropout1=0.1,
                 dropout2=0.1,
                 dropout3=0.1,
                 kernel_size=3, 
                 res_neighbor=(5,5),
                 pos_layer='APEG',
                 max_batch_size=1024,
                 ):
        super().__init__() 

        self.alpha = 0.3  # fusion output과 개별 encoder output 사이의 distillation 비율을 조절하는 하이퍼파라미터 
        self.emb_dim = emb_dim
        self.max_batch_size = max_batch_size 

        # Target Encoder 
        resnet18 = load_model_weights("tenpercent_resnet18.ckpt")  # load_model_weights helper 
        module = list(resnet18.children())[:-2]  # 마지막 avgpool, fc 제거 -- feature map만 뽑는 backbone 
        self.target_encoder = nn.Sequential(*module)  # forward할 수 있는 형태로 
        self.fc_target = nn.Linear(emb_dim, num_genes)  # gene 예측 (output head)
        self.target_linear = nn.Linear(512, emb_dim)  # feature 차원 변환 (encoder용)

        # Neighbor Encoder 
        self.neighbor_encoder = NeighborEncoder(emb_dim,  # patch token의 feature 크기 (transformer 기본 hidden size)
                                                depth3,   # attention block 개수 
                                                num_heads3,  # multi-head attention haed 개수 
                                                int(emb_dim*mlp_ratio3),  # transformer 내구 MLP hidden dimension 
                                                dropout=dropout3,  # overfitting 방지, attention/MLP에서 사용 
                                                resolution=res_neighbor)  # neighbor grid 크기 
        self.fc_neighbor = nn.Linear(emb_dim, num_genes)  # gene 예측 (output head)

        # Global Encoder 
        self.global_encoder = GlobalEncoder(emb_dim,
                                            depth2,
                                            num_heads2,
                                            int(emb_dim*mlp_ratio2),
                                            dropout2,
                                            kernel_size,
                                            pos_layer)
        self.fc_global = nn.Linear(emb_dim, num_genes)  # gene 예측 (output head)

        # Fusion Layer 
        self.fusion_encoder = FusionEncoder(emb_dim,
                                            depth1,
                                            num_heads1,
                                            int(emb_dim*mlp_ratio1),
                                            dropout1)
        self.fc = nn.Linear(emb_dim, num_genes)  # gene 예측 (output head)


    def forward(self,
                img,  # target input 
                mask,  # neighbor input 
                neighbor_emb,  # neighbor input 
                position=None,  # global input 
                global_emb=None,  # global input 
                pid=None,  # global input 
                sid=None,  # global input 
                **kwargs):
        
        if 'dataset' in kwargs:
            # Training (* pid, kwargs['dataset'], kwargs['label'])
            return self._process_training_batch(img, mask, neighbor_emb, pid, sid, kwargs['dataset'], kwargs['label'])
        else:
            # Inference (* position, global_emb)
            return self._process_inference_batch(img, mask, neighbor_emb, position, global_emb, sid)
        
    def _process_training_batch(self, img, mask, neighbor_emb, pid, sid, dataset, label):
        global_emb, position = self.retrieve_global_emb(pid, dataset)  # batch에 포함된 각 샘플(pid)에 대해, 해당 patient의 global embedding과 위치 정보를 dataset에서 꺼내오는 과정 
        # global_emb : 전체 조직(WSI)의 각 위치(spot)의 feature (N_spots, emb_dim)
        # position : 그 spot이 조직에서 어디에 있는지 (좌표) 

        fusion_token, target_token, neighbor_token, global_token = \
            self._encode_all(img, mask, neighbor_emb, position, global_emb, pid, sid)
        # 3가지 정보(target/neighbor/global)를 각각 encoding한 뒤, fusion해서 최종 representation을 만드는 단계 

        return self._get_outputs(fusion_token, target_token, neighbor_token, global_token, label)
        # 각 representation으로 gene 예측을 만든 뒤, multi-branch loss + distillation을 계산해서 반환 

    def _process_inference_batch(self, img, mask, neighbor_emb, position, global_emb, sid=None):
        if sid is None and img.shape[0] > self.max_batch_size:
            # batch가 너무 크면, 여러 작은 batch로 쪼개서 순차적으로 처리하겠다 
            imgs = img.split(self.max_batch_size, dim=0)
            neighbor_embs = neighbor_emb.split(self.max_batch_size, dim=0)
            masks = mask.split(self.max_batch_size, dim=0)
            # split해서 나눈 batch를 다시 정확히 매칭시키기 위한 장치
            # 전체 batch의 index를 만든 다음, split된 각 mini-batch에 맞게 sid도 같이 나누는 것 
            sid = torch.arange(img.shape[0]).to(img.device)
            sids = sid.split(self.max_batch_size, dim=0)

            # split된 mini-batch들을 하나씩 처리해서 다시 합치는 inference 로직 
            # _encode_all(...)은 (fusion_token, t~~, n~~, g~~)반환하는데, fusion_token만 사용 
            pred = [self.fc(self._encode_all(img, mask, neighbor_emb, position, global_emb, sid=sid)[0]) \
                    for img, neighbor_emb, mask, sid in zip(imgs, neighbor_embs, masks, sids)] 
            pred = torch.clamp(torch.cat(pred, dim=0), 0) 

            return {'logits': pred}  # 최종 예측값. regression이지만 관습적으로 logits라고 명명 
        else: 
            fusion_token, _, _, _ = self._encode_all(img, mask, neighbor_emb, position, global_emb, sid=sid)
            pred = torch.clamp(self.fc(fusion_token), 0)
        return {'logits': pred}

    def _encode_all(self, img, mask, neighbor_emb, position, global_emb, pid=None, sid=None):
        # 각 인코더 전체 forward 
        target_token = self.encode_target(img)
        neighbor_token = self.neighbor_encoder(neighbor_emb, mask)
        global_token = self.encode_global(global_emb, position, pid, sid) 

        fusion_token = self.fusion_encoder(target_token, neighbor_token, global_token, mask=mask)

        return fusion_token, target_token, neighbor_token, global_token 
    
    def encode_global(self, global_emb, position, pid=None, sid=None):
        # Global tokens 
        if isinstance(global_emb, dict):
            # global_token을 담을 빈 공간. batch 크기만큼의 global feature를 담을 텐서를 0으로 초기화해서 생성 
            global_token = torch.zeros((sid.shape[0], self.emb_dim)).to(sid.device)
            # patient별로 global 정보를 꺼내서 batch에 맞게 정렬 
            for _id, x_g in global_emb.items():  # 각 patient 
                batch_idx = pid == _id  # 해당 patient에 속한 batch 샘플만 고름 
                # batch_idx는 현재 patient에 해당하는 batch 위치에 global feature를 정확히 넣기 위한 마스크 
                pos = position[_id]  # 그 patient의 global embedding + position 가져옴 
                g_token = self.global_encoder(x_g, pos).squeeze() # N x 512 
                global_token[batch_idx] = g_token[sid[batch_idx]] # B x D 
                # x_cond_encoded = self.encode_cond(x_g, pos[id_]) # N x D 
                # [Q.] 잘 이해안됨.  
        else: 
            global_token = self.global_encoder(global_emb, position).squeeze() # N x 512  
            if sid is not None: 
                global_token = global_token[sid] 

        return global_token 
    
    def _get_outputs(self, fusion_token, target_token, neighbor_token, global_token, label):
        # 최종 예측 (Training)
        output = torch.clamp(self.fc(fusion_token), 0)  # B x num_genes
        out_target = torch.clamp(self.fc_target(target_token.mean(1)), 0)  # B x num_genes 
        out_neighbor = torch.clamp(self.fc_neighbor(neighbor_token.mean(1)), 0)  # B x num_genes 
        out_global = torch.clamp(self.fc_global(global_token), 0)  # B x num_genes 

        preds = (output, out_target, out_neighbor, out_global)

        loss = self.calculate_loss(preds, label)

        return {'loss': loss, 'logits': output}  # Loss 포함, 개별 인콬더 out 포함 
    
    def calculate_loss(self, preds, label): 
        loss = F.mse_loss(preds[0], label)  # Supervised loss for Fusion 
        # fusion 학습 

        for i in range(1, len(preds)):
            loss += F.mse_loss(preds[i], label) * (1-self.alpha)  # Supervised loss 
            loss += F.mse_loss(preds[0], preds[i]) * self.alpha  # Distillation loss 
        # 각 branch를 동시에 학습시키면서, branch들이 fusion 결과를 따라가도록 만든다 

        return loss 
    
    def retrieve_global_emb(self, pid, dataset):
        # batch에 필요한 global 정보만 patient 단위로 가져오는 전처리 단계 
        # 각 patient의 전체 global embedding + 위치 정보를 딕셔너리로 만들어 반환 
        # [Q.] 잘 이해안됨.  
        device = pid.device 
        unique_pid = pid.unique()

        global_emb = {}
        pos = {}
        for pid in unique_pid:
            pid = int(pid)
            _id = dataset.int2id[pid]

            global_emb[pid] = dataset.global_embs[_id].clone().to(device).unsqueeze(0) 
            pos[pid] = dataset.pos_dict[_id].clone().to(device) 

        return global_emb, pos 


