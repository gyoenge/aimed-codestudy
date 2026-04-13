import gc  # garbage collector (파이썬에서 안 쓰는 객체 메모리를 정리할 때 사용. 보통 대규모 학습 코드에서 메모리 관리할 때 가끔 씀.) 
import math 
from typing import Dict, Mapping, Optional, Tuple, Any, Union  # for type hint 

import torch
import numpy as np
from torch import nn, Tensor 
import torch.distributed as dist # 분산 학습용 
import torch.nn.functional as F 
from torch.nn import TransformerEncoder, TransformerEncoderLayer # Transformer 모듈
from torch.distributions import Bernoulli # 베르누이 확률분포. 0/1 샘플링할 때 사용. 이 코드에서는 explicit zero probability가 있을 때: 어떤 gene이 zero일 확률을 예측하고 그 확률로 샘플링할 때 사용됨.
from tqdm import trange # 진행률 표시. progress bar. 

# FlashAttention (더 빠른 attention 사용) 
try: 
    from flash_attn.flash_attention import FlashMHA
    flash_attn_available = True 
except ImportError:
    import warnings
    warnings.warn("flash_attn is not installed")
    flash_attn_available = False
    # 없으면 기본 PyTorch transformer 사용

"""scGPT integration에서 batch correction 핵심 기능"""
# DSBN (batch/domain마다 다른 BatchNorm 적용, 입력 분포 차이 보정)
from .dsbn import DomainSpecificBatchNorm1d
# DAR (forward는 그대로, backward는 gradient 반전. adversarial batch correction용)
from .grad_reverse import grad_reverse 


class TransformerModel(nn.Module):
    def __init__(
        self, 
        # Transformer model architecture 
        ntoken: int, # gene token 개수 (vocab size)
        d_model: int, # embedding dimension (Transformer hidden size) (cell_emb dimension = d_model)
        nhead: int, # multi-head attention head 수 
        d_hid: int, # feed-forward layer hidden dimension (FFN: d_model → d_hid → d_model)
        nlayers: int, # Transformer encoder layer 개수
        # Classification/CLS 관련 
        nlayers_cls: int=3, # CLS head (classifier)의 layer 수
        n_cls: int=1, # classification output dimension 
        # 입력 관련 
        vocab: Any=None, # gene token mapping 
        dropout: float=0.5, # (dropout -> regularization)
        pad_token: str="<pad>", # padding token 이름 
        pad_value: int=0, # padding 값 (expression 값 쪽)
        # 학습 Objectives 관련 - MVC, DAB, DSBN 
        do_mvc: bool=False, # GEPC 활성화 여부 (cell_emb → gene expression 예측)
        do_dab: bool=False, # Batch Correction - DAR 활성화 (grad_reverse + discriminator 활성화)
        use_batch_labels: bool=False, # batch label 사용 (batch embedding 추가할지 여부) (사용되면: embedding + batch embedding concat)
        num_batch_labels: Optional[int]=None, # batch label 사용 (batch 개수)
        domain_spec_batchnorm: Union[bool, str]=False, # Batch Correction - DSBN 활성화 (입력 분포 정리 (low-level))
        # 입력 embedding 방식 (continuous: 실수값>MLP, category: binning후 embed, scaling: gene embedding * value)
        input_emb_style: str="continuous", 
        n_input_bins: Optional[int]=None, # category 방식일 때 bin. ㅐ수 
        # Cell embedding 방식 
        cell_emb_style: str="cls", # (cls: 첫 token 사용, avg-pool: 평균 pooling, w-pool: weighted pooling)
        # GEPC (MVC) 관련 
        mvc_decoder_style: str="inner product", # (inner product: dot product 방식, concat: concat 후 MLP, sum: 합 기반) (대부분 inner product --> 효율 + 성능)
        # ECS (contrastive) 관련 
        ecs_threshold: float=0.3, # embedding similarity threshold (의미: cosine similarity > threshold → positive)
        # Zero modeling 
        explicit_zero_prob: bool=False, # gene expression이 0일 확률을 따로 모델링 (이유: scRNA → zero inflation problem. True이면 value + zero_prob 둘 다 예측.)
        # Others 
        use_fast_transformer: bool=False, # FlashAttention 등 사용 여부 (flash: FlashAttention, linear: linear attention)
        fast_transformer_backend: str="flash", 
        pre_norm: bool=False, # normalization 방식 (pre-norm, post-norm) (pre-norm → 안정적 (deep 모델), post-norm → original Transformer)
    ): 
        super().__init__()
        pass 

