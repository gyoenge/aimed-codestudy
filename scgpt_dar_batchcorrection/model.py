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
        
        # 기본 설정 저장 
        self.model_type = "Transformer"
        self.d_model = d_model # hidden size 
        self.do_dab = do_dab # DAR 사용 여부 
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style 
        self.cell_emb_style = cell_emb_style 
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        # value embedding 방식 (입력 설정 검증)
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        # cell embedding 추출 방식 (입력 설정 검증)
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        # fast transformer 입력 설정 검증 (flash attention 사용 가능 여부 확인)
        if use_fast_transformer:
            if not flash_attn_available: 
                warnings.warn(
                    "flash-attn is not installed, using pytorch transformer instead. "
                    "Set use_fast_transformer=False to avoid this warning. "
                    "Installing flash-attn is highly recommended."
                )
                use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer

        # TODO: add dropout in the GeneEncoder 
        # GeneEncoder: gene token id를 embedidng vector로 바꾸는 모듈 
        # 입력 (batch, seq_len) -> 출력 (batch, seq_len, d_model)
        # NLP에서 word embedding에 해당. gene 이름을 벡터로 바꾸는 단계 
        self.encoder = GeneEncoder(
            ntoken, 
            d_model, 
            padding_idx=vocab[pad_token], 
        )

        # Value Encoder, NOTE: the scaling style is also handled in _encode method 
        # ValueEncoder: expression value를 embedding space로 옮김. 
        if input_emb_style == "continuous":
            # 실수값을 MLP로 projection. scRNA expression 같은 연속값 처리. 
            self.value_encoder = ContinuousValueEncoder(
                d_model, 
                dropout, 
            )
        elif input_emb_style == "category":
            # 값을 binning한 뒤 embedding lookup
            # discrete category처럼 취급 
            assert n_input_bins > 0 
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, 
                d_model, 
                padding_idx=pad_value,
            )
        else: 
            # scaling: 별도 encoder 없이 Identity 
            # 나중에 _encode()에서 gene embedding과 곱함. 
            # 즉 이 부분은: gene token embedding + value embedding을 만들기 위한 준비. 
            self.value_encoder = nn.Identity()  
            # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling 

        # Batch Encoder 
        # 역할: batch label도 embedding으로 바꾸는 모듈. 
        # batch 정보를 모델이 명시적으로 참고하게 할 수 있음. 
        # batch 정보를 decoder 쪽에 concat할 때 사용 가능. 예를 들어 expr/mvc decoder. 
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(
                num_batch_labels, 
                d_model, 
            )
        
        # DSBN or 일반 BN 설정 
        # 역할: Transformer에 들어가기 전 embedding 분포를 정리. 
        if domain_spec_batchnorm is True or domain_spec_batchnorm == "dsbn":
            # DSBN: batch마다 다른 BN 사용. low-level batch correction. 
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                d_model, num_batch_labels, eps=6.1e-5, affine=use_affine, 
            )
        elif domain_spec_batchnorm == "batchnorm":
            # 일반 BN: 모든 batch에 공통 BN. 
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)
        # 아무 것도 안쓰면 normalization 생략. 

        # Transformer Encoder 생성. 
        # 역할: 모델의 backbone. 이 부분이 실제로 gene들 사이 관계를 학습하는 핵심 블록. 
        # Fast/Flash/Normal Transformer setting 
        if use_fast_transformer: 
            if fast_transformer_backend == "linear":
                # linear fast attention 기반. 
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, 
                    nhead, 
                    d_hid, 
                    nlayers, 
                    dropout, 
                )
            elif fast_transformer_backend == "flash":
                # FlashAttention 기반 
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model, 
                    nhead, 
                    d_hid, 
                    dropout, 
                    batch_first=True, 
                    norm_scheme=self.norm_scheme, 
                )
                self.transformer_encoder = TransformerEncoder(
                    encoder_layers, 
                    nlayers, 
                )
        else: 
            # PyTorch 기본 TransformerEncoderLayer 
            encoder_layers = TransformerEncoderLayer(
                d_model, 
                nhead, 
                d_hid, 
                dropout,
                batch_first=True, 
            )
            self.transformer_encoder = TransformerEncoder(
                encoder_layers, 
                nlayers, 
            )

        # Expr Decoder setting 
        # 역할: Transformer output으로부터 gene expression 예측. 
        # 논문 기준으로 GEP/MLM 쪽 decoder 
        self.decoder = ExprDecoder(
            d_model, 
            explicit_zero_prob=explicit_zero_prob, # 값뿐 아니라 "0일 확률"도 같이 예측. 
            use_batch_labels=use_batch_labels, # batch embedding을 concat해서 decoder에 넣음. 
        )

        # CLS Decoder setting
        # 역할: cell embedding --> class logits. cell type classification 같은 task용.  
        self.cls_decoder = ClsDecoder(
            d_model, n_cls, 
            nlayers=nlayers_cls, 
        )

        # MVC Decoder setting 
        # 역할: GEPC/MVC objective 용 decoder. 
        # cell embedding 기반으로 gene expression을 다시 맞추는 head. 
        # cell_emb 가 진짜 biological state를 잘 담도록 강제. 
        if do_mvc: 
            self.mvc_decoder = MVCDecoder(
                d_model, 
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob, 
                use_batch_labels=use_batch_labels, 
            )

        # DAB setting 
        # DAB / DAR discriminator 생성 
        # 역할: DAR의 핵심. cell embedding으로 batch label을 예측하는 discriminator. 
        # 내부에서 x = grad_reverse(x)를 거쳐서 encoder 쪽 gradient를 뒤집는다. 
        # 결과적으로, discriminator는 batch를 맞추려고 하고 encoder는 batch를 못 맞추게 만드는 방향으로 학습이 된다. 
        # 즉: batch-invariant representation을 만들게 된다. 
        if do_dab: 
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model, 
                n_cls=num_batch_labels, 
                reverse_grad=True, 
            )

        # Similarity: cosine similarity / temperature scaling. contrastive objective 계산용. 
        self.sim = Similarity(temp=0.5) # TODO: auto set temp 
        # CCE Loss 계산용 cross entropy. 즉 CCE/ECS 류 objective를 위해 미리 준비. 
        self.creterion_cce = nn.CrossEntropyLoss()

        # weight 초기화. 앞에서 만든 모듈들의 일부 weight 초기화. 예를 들어 gene embedding weight를 uniform 초기화. 
        self.init_weights() 

    def init_weights(self) -> None: 
        """
            모델의 일부 파라미터를 초기화하는 함수
            보통 embedding weight나 linear layer weight를 특정 범위로 초기화할 때 사용
            신경망은 초기값에 따라 학습 안정성이 달라질 수 있음. 
            특히 embedding layer는 직접 초기화하는 경우가 많음. 
        """
        initrange = 0.1 
        # TODO: check if this initialization is helpful and shall we apply to all? 
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange) 
        # gene embedding weight를 [-0.1, 0.1] 범위의 균등분포로 초기화

    def _encode(
        self, 
        src: Tensor, # gene token id. (batch, seq_len)
        values: Tensor, # 각 gene에 해당하는 expression 값. (batch, seq_len)
        src_key_padding_mask: Tensor, # padding 위치를 표시하는 mask. (batch, seq_len). 보통 True면 padding. 
        batch_labels: Optional[Tensor] = None, # 각 샘플의 batch/domain label. (batch,). DSBN 또는 batch embedding 할 때 필요.  
    ) -> Tensor: # 반환: (batch, seq_len, d_model). 즉: Transformer encoder output. token별 representation을 반환한다. 
        """
            입력 gene token + value를 Transformer에 넣기 전처리하고 encoder를 통과시키는 함수
            즉, 모델의 핵심 인코딩 단계  
        """
        self._check_batch_labels(batch_labels)

        # gene encoding 
        src = self.encoder(src) # (batch, seq_len, d_model)
        self.cur_gene_token_embs = src 
        
        # value encoding 
        values = self.value_encoder(values) # (batch, seq_len, d_model)
        
        # gene embedding(src)과 value embedding(values)을 어떻게 결합하느냐를 결정
        # : expression 값을 embedding에 어떻게 반영할 것인가. 
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values 
            # gene embedding을 "크기(scale)"로 조절 
        else: 
            total_embs = src + values
            # gene 정보 + expression 정보 더하기. 
            # gene identity + expression 정보를 독립적으로 유지. 더 expressive 함. 

        # batch normalization 
        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(
                total_embs.permute(0, 2, 1), 
                batch_label, 
            ).permute(0, 2, 1)
            # the batch norm always works on dim 1 
        elif getattr(self, "bn", None) is not None: 
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        # transformer encoding 
        # 앞에서 만든 total_embs가 여기서 문맥(context)을 학습한 representation으로 변환
        # gene embedding + value 정보를 Transformer에 넣어서 gene 간 관계를 학습하는 단계
        output = self.transformer_encoder(
            total_embs, # 각 gene token이 이미 value까지 반영된 embedding 상태
            src_key_padding_mask=src_key_padding_mask, # padding token은 attention에서 제외하는 마스크. 
        )
        return output # (batch, seq_len, d_model )

    def _get_cell_emb_from_layer(
        self, 
        layer_output: Tensor, # Transformer encoder의 출력. (batch, seq_len, d_model)
        weights: Tensor=None, # weighted pooling할 때 사용하는 가중치 (batch, seq_len). cell_emb_style == "w-pool"일 때만 필요. 
    ) -> Tensor: # 반환: (batch, d_model). 즉: 샘플 하나당 하나의 cell embedding. 
        """
            Transformer output에서 cell-level embedding 하나를 뽑는 함수. 
            즉: token-level embedding들 --> cell embedding 1개 
        """
        pass 

    def _check_batch_labels(self, 
        batch_labels: Tensor # batch/domain label tensor 
    ) -> None: # 반환값 없음. 잘못되면 assert 또는 ValueError 
        """
            역할: 현재 설정에서 batch_labels가 필요한지/불필요한지 검사하는 함수. 
            왜 필요하냐: 예를 들어: use_batch_labels=True 또는 DSBN=True인데 batch_labels=None이면 문제. 
                        반대로 batch 기능 안 쓰는데 batch_labels를 넣어도 이상함. 
        """
        pass 

    def generate(
        self, 
        cell_emb: Tensor, # cell-level embedding. (batch, d_model). 
        src: Tensor, # gene token ids. (batch, seq_len). 
        values: Optional[Tensor]=None, # gene expression value 입력. 없을 수도 있음. 
        src_key_padding_mask: Optional[Tensor]=None, # padding mask. 없으면 내부에서 전부 non-padding으로 처리할 수도 있음. 
        gen_iters: int=1, # generation iteration 횟수. interation generation 확장용 인자. 
        batch_labels: Optional[Tensor]=None, # batch label. DSBN/batch embedding 필요 시 사용. (batch, )
    ) -> Tensor: # 반환: (batch, seg_len). gene 별 예측 expression 값. 
        """
            역할: 주어진 cell embedding을 바탕으로 gene expression을 생성/예측 
            논문에서 말하는 generation 쪽과 연결되는 함수. 
        """
        pass 

    def forward(
        self, 
        src: Tensor, # gene token ids. (batch, seq_len)
        values: Tensor, # gene expression values. (batch, seq_len)
        src_key_padding_mask: Tensor, # padding mask. (batch, seq_len)
        batch_labels: Optional[Tensor]=None, # batch/domain labels. (batch,)
        # objective 관련 boolean 옵션 
        CLS: bool=False, # cell type classification head 사용할지 여부 
        CCE: bool=False, # contrastive cell embedding objective 사용할지 여부 
        MVC: bool=False, # GEPC/MVC decoder 사용할지 여부 
        ECS: bool=False, # elastic cell similarity loss 계산할지 여부 
        do_sample: bool=False, # zero probability가 있을 때 실제 sampling할지 여부 
    ) -> Mapping[str, Tensor]: # 반환: dictionary 형태. 
        """
            역할: 모델의 메인 forward 함수. 
            어떤 objective를 켤지 옵션으로 받아서 필요한 output들을 dictionary로 반환. 
            학습 시 가장 많이 쓰이는 핵심 함수. 
            여러 head/output을 한 번에 관리. 
        """
        pass 

    def encode_batch(
        self, 
        srd: Tensor, # 전체 gene token ids. (N, seq_len)
        values: Tensor, # 전체 expression values. (N, seq_len)
        src_key_padding_mask: Tensor, # 전체 padding mask. (N, seq_len)
        batch_size: int, # 한 번에 몇 개씩 encode 할지. 
        batch_labels: Optional[Tensor]=None, # 전체 batch labels. (N,)
        output_to_cpu: bool=True, # 결과를 CPU로 옮길지. 큰 embedding 추출할 때 GPU 메모리 절약용. 
        time_step: Optional[int]=None, # 특정 token 위치의 output만 뽑을지. 예: 0이면 CLS token만 추출 가능. 
        return_np: bool=False, # 결과를 numpy array로 반환할지 여부 
    ) -> Tensor: # (N, seq_len, d_model). 또는 time_step 지정 시 (N, d_model).
        """
            역할: 큰 데이터셋을 batch 단위로 나눠서 인코딩하는 함수 
            inference/embedding 추출용. 
            즉: 전체 데이터 --> 여러 mini-batch로 쪼개서 _encode 실행 --> 결과 합치기 
        """
        pass 


### Sub Modules ### 

class FastTransformerEncoderWrapper(nn.Module):
    pass

class FlashTransformerEncoderLayer(nn.Module):
    pass

class GeneEncoder(nn.Module):
    pass 

class PositionalEncoding(nn.Module):
    pass 

class ContinuousValueEncoder(nn.Module):
    pass 

class CategoryValueEncoder(nn.Module):
    pass

class BatchLabelEncoder(nn.Module):
    pass 

class Similarity(nn.Module):
    pass 

class ExprDecoder(nn.Module):
    pass 

class ClsDecoder(nn.Module):
    pass 

class MVCDecoder(nn.Module):
    pass 

class AdversarialDiscriminator(nn.Module):
    pass 

### 