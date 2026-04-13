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


