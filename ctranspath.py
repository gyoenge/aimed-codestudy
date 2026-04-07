# CTransPath impl for practices
# source: https://github.com/Xiyue-Wang/TransPath 

###############################################################
# Modeling - CTransPath (hybrid backbone of Swin Transformer and ResNet)
###############################################################

# ConvStem: Swin Transformer 앞단의 patch embedding을 Conv 기반으로 바꾼 구조. 
"""
입력 이미지
 → ConvStem (CNN으로 feature 추출 + downsampling)
 → flatten → token (B, N, C)
 → Swin Transformer
"""

import torch 
import torch.nn as nn 
import timm 
from timm.models.layers.helpers import to_2tuple
# timm: PyTorch 이미지 모델링 라이브러리. 다양한 모델과 레이어, 유틸리티 함수를 제공하는 라이브러리.
# to_2tuple: 입력을 무조건 (height, width) 형태의 tuple로 만들어주는 함수. (숫자든 tuple이든 입력을 (H, W) 형태로 통일해주는 함수.)

class ConvStem(nn.Module):
    def __init__(self, 
        img_size=224,
        patch_size=4, 
        in_chans=3, 
        embed_dim=768, 
        norm_layer=None, 
        flatten=True,
    ): 
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Conv 총 3번 
        # 이미지 다운샘플링하면서 feature를 점점 확장 → 마지막에 embed_dim으로 맞춤. 
        stem = []
        input_dim, output_dim = 3, embed_dim // 8 # embed_dim = 768 → output_dim = 96
        for i in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False)) # stride=2 : downsampling 효과. feature map 크기가 절반으로 줄어듦.
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) # (B, embed_dim, H/4, W/4)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # (B, N, embed_dim)
        x = self.norm(x)
        return x

def ctranspath():
    model = timm.create_model(
        'swin_tiny_patch4_window7_224', 
        embed_layer=ConvStem, 
        pretrained=False
    )
    # timm에 swin_tiny_patch4_window7_224 모델이 이미 구현되어 있음.
    # embed_layer=ConvStem: patch embedding 레이어로 ConvStem을 사용하겠다는 의미.
    return model        


### Swin Transformer 직접 구현 연습 ### 

class PatchEmbed(nn.Module):
    # Swin의 patch4는 보통 Conv2d 하나로 시작. 
    """
    Swin의 기본 patch embedding: 
    Conv2d(kernel_size=patch_size, stride=patch_size)로 
    이미지를 non-overlapping patch token으로 변환 
    """
    def __init__(
        self, 
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size, 
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, N, C)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}, {W}) must match model img_size {self.img_size}"
        
        x = self.proj(x) # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x 


# Window utilities 

def window_partition(x, window_size):
    # Swin의 window partitioning 함수. 
    # : Swin은 전체 토큰끼리 attention 하지 않고, 작은 local window 안에서만 attention한다. 
    # : 예를 들어 (B, 56, 56, C) feature map이 있으면, window_size=7 기준으로: 56/7=8 총 8x8=64개 window. 
    #   각 window는 (7,7,C)가 되고, attention은 이 안에서만 일어난다. 그래서 계산량이 크게 준다. 
    """
    x: (B, H, W, C)
    return: windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, 
        H // window_size, window_size,
        W // window_size, window_size,
        C, 
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C 
    )
    """
    - (permute) --> (B, H//ws, W//ws, ws, ws, C)
    - .permute(...).contiguous().view(...)
        : permute로 축 순서 바꾸고, contiguous로 메모리를 다시 연속적으로 정렬한 뒤, view로 reshape
    - (view) --> (B * (H//ws) * (W//ws), ws, ws, C) --> 각 window를 배치처럼 일렬로 펼지는 것 
        : view의 첫번째 -1은 그냥 이렇게 생각하면 된다: "앞에 있는 모든 축을 다 합치는 것". (주의: -1은 하나만 쓸 수 있음.)
    """
    return windows 


def window_reverse(windows, window_size, H, W):
    # Swin의 window reverse 함수. 
    # : window_partition의 정확한 역연산 
    # : 쪼개졌던 (window 단위) 텐서를 다시 (전체 feature map)으로 되돌리는 과정. 
    """
    windows: (num_windows*B, window_size, window_size, C)
    return: x: (B, H, W, C)
    """
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    # B == num_windows*B / ((H//ws) * (W//ws))
    x = windows.view(
        B, 
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    # : window > grid 구조로 복원 
    # --> (B, num_window_rows, num_window_cols, ws, ws, C) == (B, H//ws, W//ws, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # : 원래 이미지 구조로 재배열 
    # --> (B, H//ws, ws, W//ws, ws, C)


# Window attention 

class WindowAttention(nn.Module):
    # Swin의 window attention 모듈. 
    """
    Window-based multi-head self attention (W-MSA).
    간단한 학습용 구현: 
    - qkv projection
    - scaled dot-product attention 
    - optional attention mask (shifted window 용).
    """
    def __init__(
        self, 
        dim, 
        window_size, 
        num_heads, 
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0, 
    ):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        x: (num_windows*B, N, C), where N = window_size * window_size 
        mask: (num_windows, N, N) or None 
        """
        B_, N, C = x.shape 
        # B_ : num_windows * B 

        qkv = self.qkv(x)  # (B_, N, 3C)
        # 3C : each q,k,v 
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim)
        # C (embedding) --> multi-head split 
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)
        # permute --> [0 - qkv], [3 -- multi-head], [4 -- N]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Q,K,V from linear projection of input tokens 

        q = q * self.scale
        attn = q @ k.transpose(-2, -1) # (B_, heads, N, N)
        # attention 
        # : 모든 window를 batch처럼 펼쳐서 attention 계산한 상태 

        # Swin에서 제일 헷갈리는 핵심 로직 (shifted window + mask)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) # B_ --> B_//nw & nw 
            # --> attn(flatten된 상태)와 mask의 축을 맞춤 
            # attn: (B, nW, num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0) # attention mask 
            # --> window / shifted window + mask 
            # mask --> unsqueeze(1) → (nW, 1, N, N)
            #      --> unsqueeze(0) → (1, nW, 1, N, N)
            attn = attn.view(-1, self.num_heads, N, N) # (B_, heads, N, N)
            # --> 다시 (배치)
        """
            - 왜 이게 shifted window 인가? 
                : window마다 다른 mask를 가지기 때문. 
            - mask는 보통 0(허용) 또는 -100/-inf(금지) 값을 가지고, 금지된 위치는 softmax 0으로 수렴 
                : attention 하면 안되는 토큰 쌍은 score를 -inf로 만들어서 무시한다. 
            - shift된 window 안에서는: 원래 다른 window였던 토큰들이 같은 window 안으로 들어올 수 있음. 
                그 상태로 attention하면: 서로 원래 unrelated한 토큰끼리 attention함. 
                mask로: 다른 영역 토큰 → -inf (attention 못 하게 막음) 
        """
        
        attn = attn.softmax(dim=-1) # attention score
        attn = self.attn_drop(attn) # attention dropout 

        x = attn @ v  # (B_, heads, N, head_dim) # value와 곱 
        x = x.transpose(1, 2).reshape(B_, N, C) # 다시 token 형태로 복원 
        x = self.proj(x) # projection 
        x = self.proj_drop(x) # projection dropout 
        return x 


# MLP 

class MLP(nn.Module):
    # MLP 모듈. 
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x 



# DropPath = residual branch를 통째로 랜덤하게 끄는 regularization (Stochastic Depth)
# : batch 단위로 residual branch를 살릴지(1) 끌지(0) 랜덤하게 결정하고, 기대값을 유지하기 위해 스케일링

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    # Stochastic depth.
    if drop_prob == 0.0 or not training: 
        return x
    # DropPath는 training & drop_prob 설정할 때만 동작
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast to all dims except batch
    # (B, 1, 1, ..., 1) --> batch마다 하나의 랜덤 값만 쓰기 위해
    # 예시로 x: (B, N, C)를 넣으면 (B, 1, 1) shape 만들어짐 
    """
    ** 그래서 neuron 단위가 아니라 batch 단위
        DropPath = "sample 단위로 전체 경로를 끊는다" 
    """
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) # keep_prob + rand → [keep_prob, 1 + keep_prob)
    random_tensor.floor_() # Bernoulli 만들기 
    # rand ≥ (1 - keep_prob) → 1 
    # random_tensor : 끌거냐 살릴거냐 (결과 0 or 1)
    return x.div(keep_prob) * random_tensor 
    # scaling(x / keep_prob --> 기대값 유지 목적) + masking
    # drop 적용하면: E[x] = 0.8 * x (줄어듦) --> x / 0.8 → 평균 맞춰줌 --> 결과: E[x] = x. 

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

# SwinTransformer Block 

class SwinTransformerBlock(nn.Module):
    # Swin Transformer의 기본 블록. 
    """
    Swin의 핵심 block: 
        - LayerNorm
        - window attention
        - shifted window (optional)
        - residual 
        - MLP 
    """
    def __init__(self, 
        dim, 
        input_resolution,
        num_heads,
        window_size=7, 
        shift_size=0, # default=0(no shift)
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_prob=0.0,
        ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        H, W = input_resolution
        if min(H, W) <= window_size: 
            self.window_size = min(H,W)
            self.shift_size = 0
        # window가 H/W보다 작으면 min(H,W)를 size로 재설정하여 처리 

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim, 
            window_size=self.window_size,
            num_heads=num_heads, 
            qkv_bias=qkv_bias,
            attn_drop=attn_drop, 
            proj_drop=drop, 
        )

        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity() 

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio),
            out_features=dim, 
            drop=drop,
        )

        if self.shift_size > 0:
            self.register_buffer(
                "attn_mask", 
                self.create_attn_mask(H, W, self.window_size, self.shift_size),
                persistent=False, 
            )
            # shifted window가 있을 때만 mask를 생성해서 모델에 “buffer”로 저장한다. 
        else:
            self.attn_mask = None 

    @staticmethod
    def create_attn_mask(H, W, window_size, shift_size): 
        # Shifted window에서 window 경계 넘어가는 attention을 막기 위한 mask. 

        img_mask = torch.zeros((1, H, W, 1))  # (1, H, W, 1)
        # : 각 픽셀에 "어느 영역에 속하는지" 라벨을 붙일 예정 
        cnt = 0

        # 영역 나누기 
        # : 이미지를 3x3 영역으로 나누는 것. shift 후에 섞일 영역을 구분하기 위해 
        h_slices = (
            slice(0, -window_size), # 영역 1: [ 0 ~ H-ws ]
            slice(-window_size, -shift_size), # 영역 2: [ H-ws ~ H-shift ]
            slice(-shift_size, None), # 영역 3: [ H-shift ~ H ]
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )

        # 영역별 번호 부여 
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        """
        예: 
        0 0 0 | 1 1 1 | 2 2
        0 0 0 | 1 1 1 | 2 2
        ------|--------|---
        3 3 3 | 4 4 4 | 5 5
        3 3 3 | 4 4 4 | 5 5
        ------|--------|---
        6 6 6 | 7 7 7 | 8 8

        같은 숫자 → 같은 원래 window  
        다른 숫자 → 다른 window
        """

        mask_windows = window_partition(img_mask, window_size) # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        # : 윈도우로 나누기(각 window안에 이 토큰이 원래 어느 영역에서 왔는지 정보) & flatten -> (nW, N) 
        # N = ws * ws, 예시: [0,0,1,1,0,0,1,1,...] --> 각 토큰의 영역 ID 
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) 
        # : 각 token pair가 같은 영역인지 비교 (모든 token pair(i,j)에 대해 같은 영역인지 비교하려고)
        """
        예시: 
            mask_windows = [0, 0, 1, 1]
            mask_windows.unsqueeze(1) = [[0],[0],[1],[1]]
            mask_windows.unsqueeze(2) = [[0, 0, 1, 1]]
            빼기 연산 시 broadcasting 
            --> 
                    t0 t1 t2 t3
                t0 →  0  0 -1 -1
                t1 →  0  0 -1 -1
                t2 →  1  1  0  0
                t3 →  1  1  0  0
            --> 
                같은영역==0, 다른영역!=0. 
        """
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        # : mask 값 변환 
        # mask = "같은 원래 window였던 애들끼리만 attention 허용"
        return attn_mask 
    
    def forward(self, x):
        # x: (B, H*W, C)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, expected={H*W}"

        shortcut = x # residual 
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, 
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1,2),
            )
            # 텐서를 특정 방향으로 “밀어서(순환시키면서)” 이동시키는 함수.
            # 핵심은 끝에서 밀려난 값이 반대편으로 다시 들어온다 (circular shift)는 것.
            # dims: 어떤 축 기준으로 이동할지 (2d에서 0행방향 1열방향)(3d에서 1행방향 2열방향)
        else:
            shifted_x = x 

        # partition windows 
        x_windows = window_partition(shifted_x, self.window_size) # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA 
        attn_windows = self.attn(x_windows, mask=self.attn_mask) # (nW*B, ws*ws, C)

        # merge windows 
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift 
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, 
                shifts=(self.shift_size, self.shift_size),
                dims=(1,2),
            )
        else: 
            x = shifted_x

        # residual + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x 


# Patch Merging 

class PatchMerging(nn.Module):
    # Swin Transformer의 patch merging 모듈. 
    # 2x2 patch를 하나로 합쳐 resolution을 줄이고 channel을 늘림. 
    # (B, H*W, C) --> (B, H/2 * W/2, 2C)
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        self.norm = nn.LayerNorm(4*dim)
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, expected={H*W}"
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for PatchMerging"

        x = x.view(B, H, W, C)

        # downsample (2x2 patch를 하나로 합침)
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right

        x = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)                  # (B, H/2*W/2, 4C)

        x = self.norm(x)
        x = self.reduction(x)                     # (B, H/2*W/2, 2C)
        return x


# Basic Layer 

class BasicLayer(nn.Module):
    # Swin의 stage 하나. (Swin Transformer의 기본 레이어.)
    # 여러 개의 Swin block(SwinTransformerBlock과) + optional downsampling(PatchMerging)
    """
    짝수 block → normal window
    홀수 block → shifted window
    """
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_probs=None,
        downsample=None,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        if drop_path_probs is None:
            drop_path_probs = [0.0] * depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                # 짝수 홀수에 따라 shift 여부 
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path_prob=drop_path_probs[i],
            )
            for i in range(depth)
        ])

        self.downsample = downsample(input_resolution, dim=dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


# Mini Swin Transformer 

class MiniSwinTransformer(nn.Module):
    # Swin Transformer의 작은 버전. 
    # 이미지를 patch → window attention → downsampling 반복 → classification 하는 구조
    """
    이미지
    → PatchEmbed
    → 여러 Stage (BasicLayer x 4)
        → SwinBlock 반복
        → PatchMerging (해상도↓, 채널↑)
    → Norm
    → Global Pooling
    → FC (classification)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )

        patches_resolution = (
            img_size // patch_size,
            img_size // patch_size,
        )
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        self.layers = nn.ModuleList()
        cur = 0
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            resolution = (
                patches_resolution[0] // (2 ** i_layer),
                patches_resolution[1] // (2 ** i_layer),
            )

            layer = BasicLayer(
                dim=dim,
                input_resolution=resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_probs=dpr[cur:cur + depths[i_layer]],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)
            cur += depths[i_layer]
        
        """
        예시(stage별 변화):
            depths=(2, 2, 6, 2)
            num_heads=(3, 6, 12, 24)

                    (H, W, C)
            stage1) (56,56,96)
            stage2) (28,28,192)
            stage3) (14,14,384)
            stage4) (7,7,768)

            --> CNN pyramid 구조랑 완전히 동일. 
        """

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # 이미지 → patch embedding
        x = self.patch_embed(x)   # (B, N, C)
        x = self.pos_drop(x)

        # stage 반복
        for layer in self.layers:
            x = layer(x)    
            
        # 마지막 feature 추출
        x = self.norm(x)                  # (B, N, C)
        x = x.transpose(1, 2)             # (B, C, N)
        x = self.avgpool(x).squeeze(-1)   # (B, C)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # 최종 classification -- (B, C) → (B, num_classes) 
        x = self.head(x)
        return x


def mini_swin_tiny(num_classes=1000):
    # Mini Swin Tiny 모델 생성 함수.
    return MiniSwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    )


###############################################################
# SRCL (Semantically-Relevant Contrastive Learning)
###############################################################

# SRCL: "같은 semantic (병리학적 의미)끼리 positive로 묶는다"

# ..

