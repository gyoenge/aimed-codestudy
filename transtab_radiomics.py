# reference:: Radiomics Retrieval, TransTab. 
# Radiomcis Retrieval source: https://github.com/nainye/RadiomicsRetrieval/blob/main/source/transtab.py 
# The original implementation of TransTab : https://github.com/RyanWangZf/transtab 
# ...

import os 
import math 
import collections 
import json 
from typing import Dict, Optional, Any, Union, Callable, List 

from loguru import logger 
import torch 
from torch import nn 
from torch import Tensor 
import torch.nn.init as nn_init 
import torch.nn.functional as F 

from transformers import BertTokenizer, BertTokenizerFast 
# 문장을 토큰(id)으로 바꿔주는 BERT용 토크나이저 (TransTab은 NLP처럼 동작) 

import numpy as np 
import pandas as pd 

import constants 

"""
흐름 느낌: 
DataFrame
  ↓ (FeatureExtractor)
Token IDs (text처럼 변환)
  ↓ (FeatureProcessor)
Embedding
  ↓ (CLS token 추가)
Sequence
  ↓ (Transformer)
Contextual embedding
  ↓
[CLS] → classification or projection
"""


### 

class TransTabWordEmbedding(nn.Module):
    # embedding (일반 BERT embedding 느낌)
    # token id → embedding vector
    # "age" → [0.12, -0.8, ...]
    # "tumor_size" → [0.3, 0.5, ...]
    def __init__(self,
        vocab_size,  # 총 단어(여기선 column 종류) 개수 (ex: age, tumor_size, ...)
        hidden_dim,  # embedding vetor 차원 
        padding_idx=0,  # padding = 빈칸 채우기용 토큰 
        hidden_dropout_prob=0,  # dropout 비율 
        layer_norm_eps=1e-5,  # LayerNorm 안정성용 작은 값 
        ) -> None: 
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)  # ReLU에서 gradient가 죽지 않도록 weight을 초기화하는 방법.
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings 


class TransTabNumEmbedding(nn.Module):
    # embedding = column_embedding x value + bias 
    def __init__(self, hidden_dim) -> None: 
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim))  # add bias 
        # num_bias는 숫자값이 0이어도 정보가 사라지지 않도록 하는 “기본 embedding”. 
        # num_feat_emb = num_col_emb * value + num_bias 
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        # num_bias를 [-1/√d, +1/√d] 범위의 균등분포로 초기화해서 값의 스케일을 안정적으로 유지 

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        # (컬럼 개수, dim) → (batch, 컬럼 개수, dim)으로 확장해서 모든 샘플이 같은 column embedding을 쓰도록 
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias 
        # x_num_ts.unsqueeze(-1).float() : 숫자값을 embedding 공간으로 확장하는 과정
        # 예시 [5] → [5, 5, 5, ..., 5] (128차원으로 broadcast) 
        return num_feat_emb


class TransTabFeatureExtractor:
    # DataFrame -> tokenize input 
    # tabular data를 텍스트처럼 바꿔서 BERT tokenizer 사용 
    def __init__(self, 
        categorical_columns=None,  # 범주형 feature 목록 
        numerical_columns=None,  # 숫자형 feature 목록 
        binary_columns=None,  # 0/1 feature 
        disable_tokenizer_parallel=False,  # tokenizer 병렬 처리 여부 
        ignore_duplicate_cols=False,  # 같은 컬럼이 여러 타입에 들어갈 경우 처리 방식 
        **kwargs, 
        ) -> None: 

        if os.path.exists('./transtab/tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('./transtab/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('./transtab/tokenizer')
        self.tokenizer.__dict__['model_max_length'] = 512 
        # tokenizer가 처리할 수 있는 최대 길이를 512로 강제한다 
        if disable_tokenizer_parallel: 
            os.eviron["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id 

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

        if categorical_columns is not None: 
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        # check if column exists overlap 
        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def __call__(self, x, shuffle=False) -> Dict:
        # x: pd.DataFrame 
        # returns: encoded_inputs (dict)
        encoded_inputs = {
            'x_num': None,  # tensor contains numerical features 
            'num_col_input_ids': None,  # tensor contains numerical column tokenized ids 
            'x_cat_input_ids': None,  # tensor contains categorical column + feature ids 
            'x_bin_input_ids': None,  # tensor contains binary column + feature ids 
        }
        # tabular 데이터를 타입별로 나눠서 모델 입력 형태로 담아두는 딕셔너리 
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c in self.categorical_columns] if self.categorical_columns is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns] if self.numerical_columns is not None else []
        bin_cols = [c for c in col_names if c in self.binary_columns] if self.binary_columns is not None else []

        if len(cat_cols + num_cols + bin_cols) == 0:
            # take all columns as categorical columns 
            cat_cols = col_names
        # 타입 정보 없으면 전부 텍스트로 처리하는 안전장치 
        
        if shuffle: 
            np.random.shuffle(cat_cols)
            np.random.shuffle(num_cols)
            np.random.shuffle(bin_cols)

        # mask out NaN values like done in binary columns 
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0)  # fill Nan with zero 
            x_num_ts = torch.tensor(x_num.values, dtype=float)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            # add_special_tokens=False : [CLS], [SEP] 안 넣음. column name만 필요. 
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask']  # mask out attention 
            # num_att_mask: Transformer에서 “어디를 볼지 / 안 볼지”를 결정하는 매우 중요한 정보
            # 컬럼 이름 토큰 중에서 “유효한 토큰만 attention 하도록 알려주는 mask” 
            # tokenizer가 만들어주는 기본 출력 중 하나

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_mask = (~pd.isna(x_cat)).astype(int)
            x_cat = x_cat.fillna('')
            x_cat = x_cat.apply(lambda x: x.name + ' ' + x) * x_mask  # mask out nan features 
            x_cat_str = x_cat.agg(' '.join, axis=1).values.tolist() 
            # 각 행의 여러 컬럼 값을 하나의 문자열(문장)로 합치는 코드
            """
            x_cat: 
            (DataFrame)
            col1        col2        col3
            --------------------------------
            "age 30"    "stage II"  "male"
            "age 25"    "stage I"   "female"

            .agg(' '.join, axis=1)
            : 각 row의 값을 " "로 이어붙이기 
            """
            x_cat_ts = self.tokenizer(x_cat_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

            encoded_inputs['x_cat_input_ids'] = x_cat_ts['input_ids']
            encoded_inputs['cat_att_mask'] = x_cat_ts['attention_mask']

        if len(bin_cols) > 0:
            x_bin = x[bin_cols]  # x_bin should already be integral (Binary values in 0 & 1)
            x_bin_str = x_bin.apply(lambda x: x.name + ' ') * x_bin
            # 값이 1인 binary 컬럼만 “컬럼 이름”을 남기고, 0은 제거하는 코드 
            x_bin_str = x_bin_str.agg(' '.join, axis=1).values.tolist()
            x_bin_ts = self.tokenizer(x_bin_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            if x_bin_ts['input_ids'].shape[1] > 0:  # not all false
                encoded_inputs['x_bin_input_ids'] = x_bin_ts['input_ids']
                encoded_inputs['bin_att_mask'] = x_bin_ts['attention_mask']
            # binary feature가 “아예 없는 경우(= 전부 0)”를 처리하는 안전장치
            # binary feature가 하나도 활성화(1)되지 않았으면 입력에서 아예 제외하는 조건

        return encoded_inputs
    
    def save(self, path):
        # save the feature extractor configuration to local dir. 
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save tokenizer 
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        # save other configurations 
        coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
        col_type_dict = {
            'categorical': self.categorical_columns,
            'binary': self.binary_columns,
            'numerical': self.numerical_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(col_type_dict))

    def load(self, path):
        # load the feature extractor configuration from local dir. 
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        coltype_path = os.path.join(path, constants.EXTRACTOR_STATE_NAME)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, 'r', encoding='utf-8') as f:
            col_type_dict = json.loads(f.read())

        self.categorical_columns = col_type_dict['categorical']
        self.numerical_columns = col_type_dict['numerical']
        self.binary_columns = col_type_dict['binary']
        logger.info(f'load feature extractor from {coltype_path}')

    def update(self, cat=None, num=None, bin=None):
        # update cat/num/bin column maps. 
        if cat is not None: 
            self.categorical_columns.extend(cat)
            self.categorical_columns = list(set(self.categorical_columns))

        if num is not None:
            self.numerical_columns.extend(num)
            self.numerical_columns = list(set(self.numerical_columns))

        if bin is not None: 
            self.binary_columns.extend(bin)
            self.binary_columns = list(set(self.binary_columns))

        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_colds` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else: 
            self._solve_duplicate_cols(duplicate_cols)

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        if org_length == 0:
            logger.warning('No cat/num/bin cols specified, will take ALL columns as categorical! Ignore this warning if you specify the `checkpoint` to load the model.')
            return True, []  
        # 컬럼 타입이 하나도 없으면, 경고를 띄우고 "문제 없음(True)"으로 처리하면서 중복 검사도 건너뛴다. 
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1] 
        # 리스트에서 2번 이상 등장한 요소(중복 컬럼)만 추출하는 코드.
        """
        collections.Counter(all_cols)
        예시: 
        {
            "age": 2,
            "tumor_size": 1,
            "stage": 1
        }
        Counter(...).items() 하면: 
        [
            ("age", 2),
            ("tumor_size", 1),
            ("stage", 1)
        ]
        """ 
        return org_length == unq_length, duplicate_cols
    
    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')


class TransTabFeatureProcessor(nn.Module):
    # 모든 feature를 하나로 합침 
    # num + cat + bin --> concat 
    # embedding: (bs, seq_len, hidden_dim)
    # 모델 입력으로 바로 넣을 수 있는 형태로 만드는 과정. 
    def __init__(self,
        vocab_size=None, 
        hidden_dim=128, 
        hidden_dropout_prob=0,
        pad_token_id=0,
        device='cuda:0'
        ) -> None: 
        super().__init__()
        self.word_embedding = TransTabWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id,
        )
        self.num_embedding = TransTabNumEmbedding(hidden_dim)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.device = device 

    def _avg_embedding_by_mask(self, embs, att_mask=None):
        # padding을 제외하고 token embedding들의 평균을 구하는 함수 
        # embs: (batch_size, token_len, hidden_dim) 
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask==0] = 0
            embs = embs.sum(1) / att_mask.sum(1, keepdim=True).to(embs.device)
            return embs 
        
    def forward(self,
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        x_cat_input_ids=None,
        cat_att_mask=None, 
        x_bin_input_ids=None,
        bin_att_mask=None,
        **kwargs, 
        ) -> Tensor: 
        # x: pd.DataFrame 
        # shuggle, num_mask.
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None

        # "컬럼 의미(이름) + 숫자값(value)"를 결합해서 최종 numerical embedding을 만드는 과정. 
        if x_num is not None and num_col_input_ids is not None: 
            num_col_emb = self.word_embedding(num_col_input_ids.to(self.device))  # (number of cat col, num of tokens, embedding size) 
            x_num = x_num.to(self.device)
            num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)  # (#columns, token_len, hidden_dim) -> (#columns, token_len, hidden_dim) 
            # token embedding → 평균 → 하나의 column vector
            # ["tumor", "size"] → 평균 → "tumor_size embedding" 
            num_feat_embedding = self.num_embedding(num_col_emb, x_num)
            num_feat_embedding = self.align_layer(num_feat_embedding)

        # categorical 텍스트를 그대로 embedding 
        if x_cat_input_ids is not None: 
            cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device))
            cat_feat_embedding = self.align_layer(cat_feat_embedding)

        # binary feature를 embedding으로 만들되, "전부 0인 특수 케이스"까지 고려하는 로직. 
        if x_bin_input_ids is not None:
            if x_bin_input_ids.shape[1] == 0:  # all false, pad zero 
                x_bin_input_ids = torch.zeros(x_bin_input_ids.shape[0], dtype=int)[:, None]
            # 값이 전부 0이면 dummy 입력으로 처리 (batch, 0) shape. 
            bin_feat_embedding = self.word_embedding(x_bin_input_ids.to(self.device))
            bin_feat_embedding = self.align_layer(bin_feat_embedding)

        # Concatenate all available feature embeddings (numberical, categorical, binary)
        emb_list = []
        att_mask_list = []
        if num_feat_embedding is not None:
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1])]
            # transformer에 넣을 attention mask를 만드는 부분이고, 특히 numerical feature용 mask를 추가하는 코드이다. 
            # shape[0] = batch_size, shape[1] = feature 개수
            # 따라서, 생성되는 mask는 (batch_size, num_features) 크기의 1 tensor이고, 
            # 이는 "모든 위치가 valid token이다"라는 의미이다. 
        if cat_feat_embedding is not None:
            emb_list += [cat_feat_embedding]
            att_mask_list += [cat_att_mask]
        if bin_feat_embedding is not None: 
            emb_list += [bin_feat_embedding]
            att_mask_list += [bin_att_mask]
        if len(emb_list) == 0: raise Exception('no feature found belonging into numerical, categorical, or binary, check your data!')
        all_feat_embedding = torch.cat(emb_list, 1).float() 
        # num/cat/bin embedding과 mask를 하나로 이어 붙여 transformer 입력을 만든다. 
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        # attention mask도 동일하게 처리. 
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "selu":
        return F.selu
    elif activation == "leakyrelu":
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class TransTabTransformerLayer(nn.Module):
    # self-attention + FFN (with gating)
    # 특이사항: Gating이 있음. --> feature importance 조절, noise suppression 
    """
    g = sigmoid(Wx)
    h = FFN(x)
    output = h * g
    """
    __constants__ = ['batch_first', 'norm_first']
    # __constants__ 는 TorchScript(JIT) 컴파일 시 "상수로 취급할 attribute 목록"을 지정하는 것. 
    # PyTorch에서 torch.jit.script() 또는 torch.jit.trace()로 모델을 컴파일할 때: 
    # 어떤 attribute는 "변하지 않는 값(상수)"로 취급하면 최적화가 가능함
    # 그래서 이 두 변수는 항상 고정된 값이라고 TorchScript에게 알려주는 것. 
    def __init__(self,
        d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
        layer_norm_eps=1e-5, batch_first=True, norm_first=False, 
        device=None, dtype=None, use_layer_norm=True
        ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # PyTorch layer 생성 시 공통 옵션 전달용. 
        # 모든 레이어를 동일한 device / dtype으로 생성하기 위함. 
        super().__init__() 

        # MultiheadAttention
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            nhead, 
            batch_first=batch_first,  # batch_first=True: (B, N, D)를 말함. (batch size, feature token count, embedding dim)
            **factory_kwargs)
        
        # Implementation of Feedforward model (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Implementation of gates 
        """
        TransTab의 커스텀 (gating):
        각 feature/token에 대해 중요도 weight를 학습. 
        
        실제 forward에서: 
        g = sigmoid(Wx)     # (B, N, 1)
        h = linear1(x)      # (B, N, dim_ff)
        h = h * g           # gate 적용

        이는 "이 feature를 얼마나 반영할지"를 조절한다. 
        g ≈ 1: 중요 --> 그대로 사용 
        g ≈ 0: 덜 중요 --> 거의 무시
        """
        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid() 

        # norm_first : layernorm을 언제 적용할지 결정 (post-norm vs. pre-norm)
        # post -- x → Attention → Add → LayerNorm
        # pre -- x → LayerNorm → Attention → Add (요즘 많이 씀, gradient 흐름 안정, deep model에 좋음)
        # 안정성과 학습 성능을 결정하는 블록임. 
        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else: 
            self.activation = activation

    # self-attention block 
    # residual은 밖에서 처리 
    def _sa_block(self,
        x: Tensor,  # (B, N, D)
        attn_mask: Optional[Tensor],  # python type hint: Tensor 또는 None 가능 
        key_padding_mask: Optional[Tensor]
        ) -> Tensor:
        src = x 
        key_padding_mask = ~key_padding_mask.bool()  
        s = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           )[0]  #(attn_output, attn_weights) 중 output만 사용함  
        return self.dropout1(s)
    
    # feed forward block 
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g  # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h) 

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)
    # __setstate__는 객체가 로드될 때 상태(state)를 복원하는 함수 
    # 파이썬 객체 직렬화(pickle) + PyTorch 모델 로딩 안정성과 관련된 코드. 
    """
    이 함수는 보통 여기서 자동 호출된다:
    torch.load(...)
    pickle.load(...)
    즉, 모델을 저장했다가 다시 불러올 때 실행된다. 
    """
    # 버전 호환성 때문에 추가됨. 

    def forward(self, 
        src, # 입력 데이터 (B, N, D)
        src_mask=None, # token 간 관계를 제한하는 mask 
        src_key_padding_mask=None, # padding된 token을 무시하기 위한 mask (B, N) 
        is_causal=None, # causal attention인지? (GPT처럼 미래 못 보게) (TransTab에서는 보통 사용 안 함)
        **kwargs
        ) -> Tensor: 
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        # 입력 sequence(src)에 self-attention + FFN을 적용해 출력 embedding을 만드는 함수. 
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))
        else: # do not use layer norm
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(x)
        """
        transformer가 잘 학습되려면, residual이 있어야 한다. 
        projection(linear), softmax, nonlinear 변환 등의 과정에서, 원래 입력 정보가 완전히 변형되거나 사라질 수 있기 때문이다. 
        또한 gradient 흐름도 개선한다. 최소한 1이 항상 존재하므로, gradient가 절대 0으로 사라지지 않는다. deep network 안정, 학습 속도 향상의 효과. 
        """
        return x   


###

class TransTabInputEncoder(nn.Module):
    # build a feature encoder that maps inputs tabular samples to embeddings. 
    # x: pd.DataFrame
    # returns embedding 
    # feature_extractor, feature_processor 종합하는 역할. 
    def __init__(self,
        feature_extractor, 
        feature_processor, 
        device='cuda:0',
        ): 
        super().__init__() 
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    def forward(self, x):
        # x: pd.DataFrame
        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized)
        return embeds 
    
    def load(self, ckpt_dir):
        # load feature extractor
        self.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))

        # load embedding layer
        model_name = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')


class TransTabEncoder(nn.Module):
    # transformer model 
    """
    구조: 
    Custom Transformer Layer 스택 (n개) [Q.]
    모든 레이어에 feature importance를 반영하는 특별한 layer를 사용한다. 
    feature 중요도 다르고 noise 많기 때문에 gate를 사용해 feature selection을 수행하는 것. 
    """
    def __init__(self, 
        hidden_dim=128,
        num_layer=2, 
        num_attention_head=2, 
        hidden_dropout_prob=0,
        ffn_dim=256, 
        activation='relu',
        ): 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.num_attention_head = num_attention_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.ffn_dim = ffn_dim
        self.transformer_encoder = nn.ModuleList(
            [
            TransTabTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5, 
                norm_first=False, 
                use_layer_norm=True, 
                activation=activation,
            )
            ]
        )
        if num_layer > 1: 
            encoder_layer = TransTabTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5, 
                norm_first=False, 
                use_layer_norm=True, 
                activation=activation,
            )
            stacked_transformer = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=num_layer-1, 
            )
            self.transformer_encoder.append(stacked_transformer)
        # 왜 ModuleList + TransformerEncoder 같이 쓰냐?
        # : ModuleList - 서로 다른 모듈 저장, TransformerEncoder - 반복 구조 효율적 훈련 
        """
        layer 2에서:
        입력
        ↓
        [0] TransTabTransformerLayer (gate 포함)
        ↓
        [1] TransformerEncoder (n-1 layers)
        ↓
        출력
        """
        """
        ModuleList 구조: 
        self.transformer_encoder = [
            Layer1: TransTabTransformerLayer,
            Layer2: TransformerEncoder
        ]
        TransformerEncoder 내부: 
        TransformerEncoder = [
            Layer2,  # 각각이 TransTabTransformerLayer (clone)
            Layer3,
            Layer4,
            ...
        ]
        """
    
    def forward(self, 
        embedding, # (bs, num_token, hidden_dim). token = feature (각 feature의 embedding) 
        attention_mask=None, 
        **kwargs, 
        ) -> Tensor: 
        outputs = embedding # init
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        # 순차 통과 
        return outputs 


class TransTabLinearClassifier(nn.Module):
    # linear classifier 
    # Transformer 출력 → 최종 예측값(logits)으로 바꾸는 classifier 헤드
    def __init__(self,
        num_class, 
        hidden_dim=128,
        ) -> None:   
        super().__init__()
        if num_class <= 2:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        # x : after transformer 
        x = x[:,0,:]  # take the cls token embedding 
        x = self.norm(x)
        logits = self.fc(x)
        return logits 


class TransTabProjectionHead(nn.Module):
    # projection head 
    # Transformer embedding을 contrastive learning용 벡터로 변환하는 linear projection layer
    # contrastive learning에서 매우 중요한 역할을 하는 부분.
    # embedding을 “비교하기 좋은 공간(projection space)”으로 변환하는 층이다. 
    # contrastive learning에서는 representation space ≠ similarity space 
    """
    여기서 특히, 
    encoder output (CLS) -- downstream task (classification) 용
    projection output -- projection output 용
    을 서로 분리하기 위해 필요하다. 
    """
    def __init__(self, 
        hidden_dim=128, 
        projection_dim=128, 
        ):
        super().__init__()
        self.dense = nn.Linear(
            hidden_dim,
            projection_dim,
            bias=False, 
        ) 
    
    def forward(self, x) -> Tensor:
        h = self.dense(x)
        return h 


class TransTabCLSToken(nn.Module):
    # embedding = [CLS] + embedding 
    # transformer output에서 x[:,0,:] --> 전체 representation
    # add a learnable cls token embedding at the end of each sequence. 
    """
    CLS token을 직접 만들어 붙이는 모듈
    학습 가능한(global summary token)을 시퀀스 앞에 추가한다. 

    원래: (B, N, D)
    CLS:  (B, 1, D)

    → 결과: (B, N+1, D)

    CLS는: 모든 feature 정보를 받아서 요약하는 역할. 
    """
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        # 왜 expand를 따로 만들었나? -- batch마다 동일한 CLS, 메모리 효율적 (copy 아님, view 기반). 
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)
    
    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        # CLS token을 batch 크기에 맞게 만든 뒤, 기존 embedding 앞에 붙인다 
        """
        CLS:       (B, 1, D)
        embedding: (B, N, D)
        → 결과:   (B, N+1, D)
        """
        outputs = {'embedding': embedding}
        if attention_mask is not None: 
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask])
        outputs['attention_mask'] = attention_mask
        return outputs 


class ContrastiveToken(nn.Module):
    # contrastive token 
    # add a learnable contrastive token embedding at the end of each sequence. 
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([embedding, self.expand(len(embedding), 1)], dim=1)
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0],1).to(attention_mask.device)], 1)
        outputs['attention_mask'] = attention_mask
        return outputs


### 

"""
TransTabModel: 공통 기본 인코더
TransTabClassifier: 분류용 모델
TransTabForRadiomics: radiomics용 contrastive learning + 분류 모델

모두 같은 기반 구조를 공유하지만, forward에서 무엇을 출력하느냐와 위에 어떤 head를 얹느냐가 다르다. 

TransTabModel: x -> embedding -> transformer -> CLS -> output
TransTabClassifier: CLS -> Linear -> logits 
TransTabForRadiomics: radiomics feature -> embedding -> contrastive token 추가 
            -> projection head -> 최종 출력 (bs, num_partition, projection_dim) 
"""

class TransTabModel(nn.Module):
    '''
    The base TransTab model for downstream tasks like contrastive learning, binary classification, etc. 
    All modles subclass this basemodel and usually rewrite the `forward` function. 
    Refer to the source code of `transtab.modeling_transtab.TransTabClassifier` or `transtab.modeling_transtab.TransTabForCL` for the implementation details. 

    Parameters: 
        categorical_columns: a list of categorical feature names. (list)
        numerical_columns: a list of numerical feature names. (list)
        binary_columns: a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1). (list)
        feature_extractor: a feature extractor to tokenize the input tables. if not passed the model will build itself. (FeatureExtractor)
        hidden_dim: the dimension of hidden embeddings. (int)
        num_layer: the number of transformer layers used in the encoder. (int)
        num_attention_head: the number of heads of multihead self-attention layer in the transformers. (int)
        hidden_dropout_prob: the dropout ratio in the transformer encoder. (float)
        ffn_dim: the dimension of feed-forward layer in the transformer layer. (int)
        activation: the name of used activation functions, support `"relu"`, `"gelu"`, `"selu"`, `"leakyrelu"`.
        device: the device, `"cpu"` or `"cuda:0"`. 

    Returns: 
        A TransTabModel model. 
    '''

    def __init__(self,
        categorical_columns=None, 
        numerical_columns=None, 
        binary_columns=None, 
        feature_extractor=None, 
        hidden_dim=128, 
        num_layer=2, 
        num_attention_head=8, 
        hidden_dropout_prob=0.1, 
        ffn_dim=256, 
        activation='relu',
        device='cuda:0',
        **kwargs, 
        ) -> None: 

        super().__init__() 

        # TransTabFeatureExtractor 
        self.categorical_columns=categorical_columns
        self.numerical_columns=numerical_columns
        self.binary_columns=binary_columns
        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        if feature_extractor is None: 
            feature_extractor = TransTabFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns, 
                binary_columns=self.binary_columns,
                **kwargs, 
            )
        
        # TransTabFeatureProcessor
        feature_processor = TransTabFeatureProcessor(
            vocab_size=feature_extractor.vocab_size, 
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim, 
            hidden_dropout_prob=hidden_dropout_prob,
            device=device, 
        )

        # (i) TransTabInputEncoder <-- TransTabFeatureExtractor & TransTabFeatureProcessor 
        self.input_encoder = TransTabInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
        )

        # (ii) TransTabEncoder 
        self.encoder = TransTabEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation, 
        )

        # token 
        self.cls_token = TransTabCLSToken(hiddendim=hidden_dim)
        ### 
        self.device = device 
        self.to(device)

    def forward(self, x, y=None):
        '''
        Extract the embeddings based on input tables. 

        Parameters: 
            x: a batch of samples stored in pd.DataFrame. (pd.DataFrame)
            y: the corresponding labels for each sample in `x`. ignored for the basemodel. (pd.Series)

        Returns: 
            final_cls_embedding: the [CLS] embedding at the end of transformer encoder. (torch.Tensor)
        '''
        
        embeded = self.input_encoder(x)
        embeded = self.cls_token(**embeded)

        # go through transformers, get final cls embedding
        # Pass through transformer layers to obtain final CLS token embedding 
        encoder_output = self.encoder(**embeded) 

        # get cls token
        final_cls_embedding = encoder_output[:,0,:]
        return final_cls_embedding 

    def load(self, ckpt_dir):
        '''
        Load the model state_dict and feature_extractor configuration from the `ckpt_dir`.

        Parameters: ckpt_dir.
        Returns: None. 
        '''
        # load model weight state dict 
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor 
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extracto.binary_columns 
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        '''
        Save the model state_dict and feature_extractor configuration to the `ckpt_dir`. 

        Parameteres: ckpt_dir.
        Returns: None. 
        '''
        # Save model weights (state_dict)
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)
        
        # save the input encoder separately 
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None 

    def update(self, config):
        '''
        Update the configuration of feature extractor's column map for cat, num, and bin cols.
        Or update the number of classes for the output classifier layer. 
        - 모델을 새 데이터셋에 맞게 재설정하는 함수: (i) 컬럼 구조 업데이트 (ii) 분류 클래스 수 업데이트 

        Parameters:
            config: a dict of configurations (dict): 
                keys cat:list, num:list, bin:list are to specify the new colun names;
                key num_class:int is to specifiy the number of classes for finetuning on a new dataset. 

        Returns: None. 
        '''
        
        # config 중에서 column 정보와 관련된 것만 필터링 
        col_map = {}
        for k,v in config.items():
            if k in ['cat','num','bin']: col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns 
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns 

        # 분류 클래스 관련 
        if 'num_class' in config: 
            num_class = config['num_class']
            self._adapt_to_new_num_class(num_class)

        return None
    
    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        '''
        Column 중복 검사 
        '''
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        '''
        중복 Column 이름 해결 
        '''
        for col in duplicate_cols:
            logger.warning('Fine duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_colums: 
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')
    
    def _adapt_to_new_num_class(self, num_class):
        '''
        Classifier 구조 변경 (task 변경 대응)
        '''
        if num_class != self.num_class:
            self.num_class = num_class 
            self.clf = TransTabLinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            if self.num_class > 2:
                self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.') 


class TransTabClassifier(TransTabModel):
    '''
    The classifier model subclass from `transtab.modeling_transtab.TransTabModel`. 

    Parameters: 
        categorical_columns, numerical_columns binary_columns, 
        feature_extractor, 
        num_class, hidden_dim, num_layer, num_attention_head, hidden_dropout_prob, ffn_dim, 
        activation, device.
    
    Retures: 
        TransTabClassifier model. 
    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )
        self.num_class = num_class 
        self.clf = TransTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        if self.num_class > 2: 
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)

    def forward(self, x, y=None):
        '''
        Make forward pass given the input feature `x` and label `y` (optional).

        Parameters: 
            x: 
                (pd.DataFrame) a batch of raw tabular samples.
                (dict) the output of TransTabFeatureExtractor. 
            y: 
                (pd.Series) the corresponding labels for each sample in `x`. 
                if label is given, the model will return the classification loss by `self.loss_fn`. 

        Returns:
            logits: the [CLS] embedding at the end of transformer encoder (torch.Tensor).
            loss: the classification loss. (torch.Tensor or None)
        '''
        pass


class TransTabForRadiomics(TransTabModel):
    '''
    The contrastive learning model subclass from `transtab.modeling_transtab.TransTabModel`.

    Parameters: 
        categorical_columns, numerical_columns binary_columns, 
        feature_extractor, 
        num_class, hidden_dim, num_layer, num_attention_head, hidden_dropout_prob, ffn_dim, 
        projection_dim,
        num_sub_cols, 
        activation, device.
    
    Retures: 
        TransTabRadiomcis model. 

    '''

    def __init__(self, 
        categorical_columns=None,
        numerical_columns=None, 
        binary_columns=None, 
        feature_extractor=None, 
        # [?] about inner arch 
        num_class=2,  # [+][?] cancer / healthy 
        hidden_dim=128, 
        num_layer=2, 
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128, # [+]
        num_sub_cols=[72, 54, 36, 18, 9, 3, 1],  # [+] radiomics feature를 몇 개씩 뽑아서 subset(view)을 만들지 정하는 리스트 
                                                 # 즉, 하나의 샘플로부터 7개의 다른 view(positive)를 생성한다. 
        gpe_drop_rate=0.1,  # [+]
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None: 
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )
        self.projection_head = TransTabProjectionHead(hidden_dim, projection_dim)
        self.num_sub_cols = num_sub_cols
        self.num_class = num_class 
        self.clf = TransTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        self.contrastive_token = ContrastiveToken(hidden_dim=hidden_dim)
        self.gpe_drop_rate = gpe_drop_rate
        self.projection_dim = projection_dim
        self.activation = activation 
        self.device = device 
        self.to(device)  # 모델 전체가 device 위에 올라가도록 (*이때 pytorch에서 tensor와 model은 같은 device에 있어야 함에 주의)

    def forward(self, x, gpe=None): 
        '''
        Make forward pass given the input feature `x` and the global positional embeddings `gpe` (optional).

        Parameters: 
            x: a batch of raw tabular samples. (pd.DataFrame)
            gpe: a batch of global positional embeddings of the same size as x. (pd.DataFrame)

        Returns: 
            feat_x_multiview: the embeddings of the input tabular samples. (torch.Tensor)
            logits: the classification logits. (torch.Tensor)
        '''

        # Perform positive sampling with multiple radiomics subsets 
        feat_x_list = []
        feat_x_for_cl = None 
        if isinstance(x, pd.DataFrame):
            sub_x_list = self._build_sub_x_list_random(x, self.num_sub_cols)

            # concatenate with the gpes with a certain drop rate 
            if gpe is not None and np.random.rand() > self.gpe_drop_rate: 
                for i in range(len(sub_x_list)):
                    sub_x_list[i] = pd.concat([sub_x_list[i], gpe], axis=1)
                
            if gpe is not None: 
                sub_x_list.append(gpe)
            
            # [Q.] gpe  

            for i, sub_x in enumerate(sub_x_list):
                # encode each subset(view) feature sample into embedding
                # gonna construct pair  
                feat_x = self.input_encoder(sub_x)
                feat_x = self.contrastive_token(**feat_x)  # 여기서 input_encoder의 출력으로 dictionary가 반환되고, token 함수 입력으로는 이를 함수 인자로 풀어서 넣는다. 
                feat_x = self.cls_token(**feat_x)
                feat_x = self.encoder(**feat_x)
                if i == 0: 
                    # cl(classification)은 정보가 많이 필요하기 때문에 full feature 사용 
                    feat_x_for_cl = feat_x 
                feat_x_proj = feat_x[:,1,:]  # contrastive token의 embedding만 꺼내는 것 (batch_size, num_tokens, hidden_dim)의 num_tokens=1번째. 
                feat_x_proj = self.projection_head(feat_x_proj) # bs, projection_dim 
                feat_x_list.append(feat_x_proj)
        else:
            raise ValueError(f'expect input x to be pd.DataFrame, get {type(x)} instead')

        logits = self.clf(feat_x_for_cl)

        feat_x_multiview = torch.stack(feat_x_list, axis=1)  # bs, num_partition, projection_dim  
        # 여러 subset(view)의 embedding을 view 차원으로 묶어서 하나의 tensor로 만드는 것.  
        # (bs, dim)이 num_partition개 있었다면, 결고로 (bs, num_partition, dim)이 됨. 

        return feat_x_multiview, logits 
    
    def _build_sub_x_list_random(self, x, num_sub_cols):
        '''
        x: DataFrame with 72 radiomics feature columns 
        Returns: A list of sub-DataFrames, each containing a random subset of columns 
                with lengths [72, 54, 36, 18, 9, 3, 1] respectively.  
        ''' 
        cols = x.columns.tolist()
        total_cols = len(cols)

        if total_cols != 72: 
            raise ValueError(f'expect 72 columns, get {total_cols} instead')
        
        sub_x_list = [] 
        for count in num_sub_cols: 
            # select count columns randomly 
            if count == total_cols: 
                selected_cols = cols
            else: 
                indices = np.random.choice(total_cols, count, replace=False)
                selected_cols = [cols[i] for i in indices]
            sub_x = x.copy()[selected_cols]
            sub_x_list.append(sub_x)
        
        return sub_x_list 
    
    def forward_withSubX(self, sub_x_list, gpe=None):
        '''
        Make forward pass given the input feature `x` and the global positional embeddings `gpe` (optional).
        - forward: 모델 내부에서 subset을 랜덤 생성.  
        - forward_withSubX: 외부에서 만든 subset을 그대로 사용. 

        Parameters: x, gpe
        Returns: feat_x_multiview, logits 
        '''

        # do positive sampling 
        feat_x_list = []
        feat_x_for_cl = None
        # sub_x_list = self._build_sub_x_list_random(x, self.num_sub_cols)

        # concatenate with the gpes with a certain drop rate
        if gpe is not None and np.random.rand() > self.gpe_drop_rate:
            for i in range(len(sub_x_list)):
                sub_x_list[i] = pd.concat([sub_x_list[i], gpe], axis=1)

        if gpe is not None:
            sub_x_list.append(gpe)

        for i, sub_x in enumerate(sub_x_list):
            # encode two subset feature samples 
            feat_x = self.input_encoder(sub_x)
            feat_x = self.contrastive_token(**feat_x) 
            feat_x = self.cls_token(**feat_x)
            if i == 0:
                feat_x_for_cl = feat_x
            feat_x_proj = feat_x[:,1,:]  # take the contrastive token embedding 
            feat_x_proj = self.projection_head(feat_x_proj) # bs, projection_dim 
            feat_x_list.append(feat_x_proj)
        
        logits = self.clf(feat_x_for_cl)

        feat_x_multiview = torch.stack(feat_x_list, axis=1) # bs, num_partition, projection_dim 

        return feat_x_multiview, logits 

    def load(self, ckpt_dir):
        '''
        Load the model state_dict and feature_extractor configuration from the `ckpt_dir`.

        Parameters: ckpt_dir.
        Returns: None. 
        '''
        # Load model weights (state_dict) 
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')  # (*checkpoint는 항상 CPU로 먼저 로드한 다음, 필요하면 GPU로 옮긴다. 안전+호환성 때문)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor 
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns 
        self.categorical_columns = self.input_encoder.feature_extractor.numerical_columns 

    def save(self, ckpt_dir):
        '''
        Save the model state_dict and feature_extractor configuration to the `ckpt_dir`.

        Parameters: ckpt_dir. 
        Returns: None. 
        '''
        # save model weight state dict 
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict() 
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None: 
            self.input_encoder.feature_extractor.save(ckpt_dir)
        # save model parameters 
        model_params = {
            'categorical_columns': self.input_encoder.feature_extractor.categorical_columns,
            'numerical_columns': self.input_encoder.feature_extractor.numorical_columns, 
            'binary_columns': self.input_encoder.feature_extractor.binary_columns, 
            'num_class': self.num_class, 
            'hidden_dim': self.encoder.hidden_dim, 
            'num_layer': self.encoder.num_layer, 
            'num_attention_head': self.encoder.num_attention_head, 
            'hidden_dropout_prob': self.encoder.hidden_dropout_prob, 
            'ffn_dim': self.encoder.ffn_dim, 
            'projection_dim': self.projection_dim, 
            'num_sub_cols': self.num_sub_cols, 
            'gpe_drop_rate': self.gpe_drop_rate, 
            'activation': self.activation, 
        }
        with open(os.path.join(ckpt_dir, constants.TRANSTAB_PARAMS_NAME), 'w') as f:
            json.dump(model_params, f, indent=4)

        # save the input encoder separately 
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None 


### 

def build_extractor(
    categorical_columns=None,  # a list of categorical feature names
    numerical_columns=None,  # a list of numerical feature names 
    binary_columns=None,  # a list of binary feature names 
    ignore_duplicate_cols=False,  
    disable_tokenizer_parallel=False, 
    checkpoint=None,  # the directory of the predefiend TransTabFeatureExtractor
    **kwargs, 
    ) -> TransTabFeatureExtractor:
    # build a feature extractor for TransTab model. 
    feature_extractor = TransTabFeatureExtractor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        ignore_duplicate_cols=ignore_duplicate_cols,
        disable_tokenizer_parallel=disable_tokenizer_parallel,
    )
    if checkpoint is not None:
        extractor_path = os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR)
        if os.path.exists(extractor_path):
            feature_extractor.load(extractor_path)
        else:
            feature_extractor.load(checkpoint)
    return feature_extractor


def build_classifier(
    categorical_columns=None, 
    numerical_columns=None, 
    binary_columns=None,
    feature_extractor=None,
    num_class=2,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8, 
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device='cuda:0',
    checkpoint=None,
    **kwargs
    ) -> TransTabClassifier:
    # build a :class: `transtab.modeling_transtab.TransTabClassifier.`
    model = TransTabClassifier(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        device=device,
        **kwargs,
    )

    if checkpoint is not None:
        model.load(checkpoint)

    return model 


def build_radiomics_learner(
        categorical_columns=None, 
        numerical_columns=None, 
        binary_columns=None,
        feature_extractor=None,
        num_class=2, 
        hidden_dim=128, 
        num_layer=2,
        num_attention_head=8, 
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128,
        num_sub_cols=[72, 54, 36, 18, 9, 3, 1],
        gpe_drop_rate=0.1,
        activation='relu',
        device='cuda:0',
        checkpoint=None,
        ignore_duplicate_cols=True,
        **kwargs, 
    ):
    # build a contrastive learning and classification model for radiomics feature extraction.  
    model = TransTabForRadiomics(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor=feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        num_sub_cols=num_sub_cols,
        gpe_drop_rate=gpe_drop_rate,
        activation=activation,
        device=device,
    )
    if checkpoint is not None:
        model.load(checkpoint)

    return model


### 

# contrastive learning 부분 공부.   
"""
RaciomidsRetrieval/train_RadiomicsRetrieval_NSCLC_Img+Rad_withAPE.py
(3D 이미지 모델 + TransTab radiomics 모델을 동시에 학습하면서
segmentation + contrastive + classification을 같이 최적화하는 구조)
에서, IMG emb <-> Rad emb 만 따와서 공부. 
""" 
"""
흐름: 

radiomics_features, ape_df
    → model_radiomics(...)
    → radiomics_embeddings   # (B, n_rad, D)

images, points, ape_map
    → image_encoder
    → prompt_encoder
    → mask_decoder
    → radiomics_token_out    # (B, D)
    → projection_head
    → image_patch_embeddings # (B, D)

image_patch_embeddings + radiomics_embeddings
    → multimodal contrastive loss

"""

def simclr_nt_xent_loss_multi_pos(
    embeddings: torch.Tensor,  # (M, D)
    idxes,  # (M,)
    temperature: float = 0.07, 
    ) -> torch.Tensor:
    # 같은 idx를 가진 샘플들을 positive으로 보는 multi-positive SimCLR loss. 

    device = embeddings.device 

    if not isinstance(idxes, torch.Tensor):
        idxes = torch.tensor(idxes, device=device, dtype=torch.long)

    # L2 normalize 
    z = F.normalize(embeddings, dim=1)  # (M, D)

    # similarity matrix 
    sim_matrix = torch.mm(z, z.t()) / temperature  # (M, M)

    # 자기 자신은 제외 
    diag_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(diag_mask, -1e4)

    # 같은 idx면 positive 
    pos_mask = (idxes.unsqueeze(1) == idxes.unsqueeze(0)) & (~diag_mask)

    # log-softmax 형태 
    logsumexp = torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    log_prob = sim_matrix - logsumexp

    # positive log-prob 평균 
    pos_log_prob_sum = (pos_mask * log_prob).sum(dim=1)
    num_pos = pos_mask.sum(dim=1).clamp_min(1)
    pos_log_prob_mean = pos_log_prob_sum / num_pos

    # loss
    loss = -pos_log_prob_mean()
    return loss 


def compute_multimodal_contrastive_loss_singleSimCLR(
    image_token_embedding: torch.Tensor,  # (B, D)
    radiomics_token_embedding: torch.Tensor,  # (B, N_rad, D)
    idxes, 
    temperature: float = 0.07, 
    ) -> torch.Tensor:
    # image embedding 1개 + radiomics multi-view embedding 여러 개를 
    # 하나의 pool로 묶어서 SimCLR loss 계산. 
    device = image_token_embedding.device
    B, n_rad, D = radiomics_token_embedding.shape 

    # radiomics: (B, N_rad, D) -> (B*N_rad, D)
    rad_all = radiomics_token_embedding.reshape(B * n_rad, D)

    # image + radiomics를 하나로 합침
    combined = torch.cat([image_token_embedding, rad_all], dim=0)
    """
    예시: 
    
    """

    # idx도 radiomics view 수만큼 확장 
    if isinstance(idxes, torch.Tensor):
        idxes = idxes.to(device)
        idxes_rad = idxes.repeat_interleave(n_rad)  # (B*N_rad,)
        combined_idxes = torch.cat([idxes, idxes_rad], dim=0)
    else: 
        idxes_rad = []
        for x in idxes:
            idxes_rad.extend([x] * n_rad)
        combined_idxes = list(idxes) + idxes_rad

    loss = simclr_nt_xent_loss_multi_pos(
        embeddings=combined,
        idxes=combined_idxes,
        temperature=temperature,
    )
    return loss


def main_contrastive_only(
    batch, 
    model, # image model
    model_radiomics, # TransTab radomics model 
    device="cuda:0",
    temperature: float = 0.07,    
    ): 
    # 원본 training loop에서 
    # 'IMG emb <-> Rad emb contrastive' 부분만 남긴 공부용 함수. 
    model.eval()
    model_radiomics.eval()

    # batch에서 필요한 것만 꺼낸다. 
    images = batch["images"].to(device).float()
    apes = batch["apes"].to(device).float()
    point_coords = batch["point_coords"].to(device)
    point_labels = batch["point_labels"].to(device)

    radiomics_features = batch["radiomics_features"] # pd.DataFrame 
    ape_df = batch["ape_df"] # pd.DataFrame
    idxes = batch["idxes"] # same patient id for positives 

    # radiomics embedding 추출
    radiomics_embeddings, _ = model_radiomics(radiomics_features, ape_df) # logits 제외
    # (B, n_rad, D)

    # image side embedding 추출 
    with torch.no_grad():
        image_embeddings = model.image_encoder(images)
    
    sparse_embeddings, dense_embeddings, ape_down = model.prompt_encoder(
        points=[point_coords, point_labels],
        ape_map=apes,
        masks=None,
    )

    image_pe = model.prompt_encoder.get_dense_pe().expand_as(image_embeddings)
    if ape_df is not None:
        pos_src = image_pe + ape_down
    else: 
        pos_src = image_pe 

    # 원본에서는 mask_decoder가 여러 값을 반환함 
    (
        low_res_masks,
        iou,
        mask_tokens_out,
        iou_token_out,
        radiomics_token_out,
        cls_token_out,
    ) = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=pos_src,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    # radiomics_token_out: image branch에서 나온 contrastive용 token 
    image_patch_embeddings = model.projection_head(radiomics_token_out)
    # image_patch_embeddings: (B, D)

    # multimodal contrastive loss 
    contrastive_loss = compute_multimodal_contrastive_loss_singleSimCLR(
        image_token_embedding=image_patch_embeddings,
        radiomics_token_embedding=radiomics_embeddings, 
        idxes=idxes,
        temperature=temperature,
    )

    # shape 확인용 출력 
    print("radiomics_embeddings.shape :", tuple(radiomics_embeddings.shape))
    print("image_patch_embeddings.shape:", tuple(image_patch_embeddings.shape))
    print("contrastive_loss           :", float(contrastive_loss.detach().cpu()))


    return {
        "radiomics_embeddings": radiomics_embeddings,
        "image_patch_embeddings": image_patch_embeddings,
        "contrastive_loss": contrastive_loss,
    }

