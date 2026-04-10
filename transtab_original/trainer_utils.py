# TransTab에서 사용하는 핵심 유틸 모듈 (dataset / collator / scheduler 등)
# Trainer랑 연결되는 “데이터 처리 + 스케줄러 + 파라미터 유틸”을 담당

import pdb
import os
import random
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup
)

from transtab.modeling_transtab import TransTabFeatureExtractor


# schduler type별 함수 
TYPE_TO_SCHEDULER_FUNCTION = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule,
    'constant_with_warmup': get_constant_schedule_with_warmup,
}


######################################################

# TrainDataset
# pandas DataFrame --> PyTorch Dataset 변환 
class TrainDataset(Dataset):
    def __init__(self, trainset):
        self.x, self.y = trainset

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        # 한 번에 row 한개씩만 반환한다. 
        # (batch 아님에 주의, single sample)
        # batch는 collator에서 만든다. 
        x = self.x.iloc[index-1:index]
        if self.y is not None:
            y = self.y.iloc[index-1:index]
        else:
            y = None
        return x, y

# TrainCollator
# raw DataFrame --> 모델 입력 tensor 변환 
class TrainCollator:
    '''A base class for all collate function used for TransTab training.
    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        ignore_duplicate_cols=False,
        **kwargs,
        ):

        # tabular 데이터를 transformer 입력 형태로 바꿔주는 엔진 
        self.feature_extractor=TransTabFeatureExtractor(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            disable_tokenizer_parallel=True,
            ignore_duplicate_cols=ignore_duplicate_cols,
        )
    
    def save(self, path):
        # 저장 기능: 전처리도 같이 저장됨. 
        self.feature_extractor.save(path)
    
    def __call__(self, data):
        raise NotImplementedError
        # 하위 class에서 정의 (실제 사용은 SupervisedTrainCollator, TransTabCollatorForCL)

# SupervisedTrainCollator 
# 일반 supervised 학습용 
class SupervisedTrainCollator(TrainCollator):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        ignore_duplicate_cols=False,
        **kwargs,
        ):
        super().__init__(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        ignore_duplicate_cols=ignore_duplicate_cols,
        )
    
    def __call__(self, data):
        # Dataset에서 받은 row들을 --> batch로 합치고 --> feature_extractor로 반환. 
        x = pd.concat([row[0] for row in data])
        y = pd.concat([row[1] for row in data])
        inputs = self.feature_extractor(x)
        return inputs, y # --> Trainer에서 바로 사용 가능. 

# TransTabCollatorForCL
# Contrastive Learning용 
# : 하나의 데이터 --> 여러 "view" 생성 (**)
"""
입력 배치에서 여러 view 생성
같은 sample의 여러 view를 positive pair로 사용.
"""
"""
원본 테이블 한 배치
→ 같은 샘플의 서로 다른 column view 여러 개 생성
→ 각 view를 feature extractor로 인코딩
→ contrastive learning용 입력으로 반환
"""
"""
예시 
    원본: [A B C D E F]
    → view1: [A B C]
    → view2: [D E F]
    → 일부 overlap 추가
"""
class TransTabCollatorForCL(TrainCollator):
    '''support positive pair sampling for contrastive learning of transtab model.
    '''
    def __init__(self, 
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        overlap_ratio=0.5, 
        num_partition=3,
        ignore_duplicate_cols=False,
        **kwargs) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            ignore_duplicate_cols=ignore_duplicate_cols,
        )
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition,int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        # overlap_ratio: 각 view가 서로 얼마나 겹칠지 정한다. 
        self.overlap_ratio=overlap_ratio
        # num_partition: 원본 컬럼을 몇 개의 view로 나눌지 결정한다. 
        self.num_partition=num_partition

    def __call__(self, data):
        # DataLoader에서 batch 하나가 들어오면, 이 안에는 여러 row가 들어 있다.

        '''
        Take a list of subsets (views) from the original tests.
        '''
        # 1. build positive pairs
        # 2. encode each pair using feature extractor

        # 배치 합치기
        # 각 sample마다 (x_row, y_row) 형태로 들어왔던 걸 하나의 batch DataFrame으로 합친다.
        # 즉 supervised collator와 비슷하게 먼저 batch를 구성하는 단계. 
        df_x = pd.concat([row[0] for row in data])
        df_y = pd.concat([row[1] for row in data])

        # view 생성 
        # num_partition > 1이면 컬럼 분할 기반 multi-view 생성
        # num_partition == 1이면 special case로 single-view corruption 방식 사용
        if self.num_partition > 1:
            sub_x_list = self._build_positive_pairs(df_x, self.num_partition)
        else:
            sub_x_list = self._build_positive_pairs_single_view(df_x)

        # 각 view를 feature extractor로 변환 
        # view를 그냥 DataFrame으로 두지 않고, 모델이 읽을 수 있는 입력 형태로 바꾼다. 
        # 즉 각 sub-table마다 tokenizer/feature extraction이 따로 적용된다. 
        input_x_list = []
        for sub_x in sub_x_list:
            inputs = self.feature_extractor(sub_x)
            input_x_list.append(inputs)
        
        # 반환 값
        # res['input_sub_x']: 여러 view의 입력 리스트
        # df_y: 라벨
        res = {'input_sub_x':input_x_list}
        return res, df_y # return {'input_sub_x': input_x_list}, df_y 
        # 여러 view가 리스트로 전달됨. 

    def _build_positive_pairs(self, x, n):
        # 같은 샘플의 여러 view를 실제로 생성하는 핵심 로직. 
        # 같은 샘플의 다른 view = positive pair 
        '''build multi-view of each sample by spliting columns

        입력: 
            x: batch DataFrame
            n: partition 수 
        출력: 
            sub_x_list: 여러 개의 부분 DataFrame(view)
        ''' 
        # 컬럼 목록 추출: 현재 배치의 전체 컬럼 이름 리스트를 가져온다. 
        x_cols = x.columns.tolist()
        # 컬럼 분할: 전체 컬럼을 n개로 균등하게 나눈다. 
        sub_col_list = np.array_split(np.array(x_cols), n)

        # overlap 수 계산: 첫 번째 partition 길이를 기준으로 overlap 개수를 잡는다. 
        len_cols = len(sub_col_list[0])
        overlap = int(math.ceil(len_cols * (self.overlap_ratio)))
        # 각 partition에 이웃 partition 일부 붙이기 (overlapping)
        sub_x_list = []
        for i, sub_col in enumerate(sub_col_list):
            if overlap > 0 and i < n-1:
                # 마지막이 아닌 partition: 다음 partition의 앞부분을 일부 가져와서 겹치게 한다. 
                sub_col = np.concatenate([sub_col, sub_col_list[i+1][:overlap]])
            elif overlap >0 and i == n-1:
                # 마지막 partition: 이전 partition의 뒷부분을 일부 가져와서 겹치게 한다. 
                sub_col = np.concatenate([sub_col, sub_col_list[i-1][-overlap:]])
            # 실제 sub-table 생성: 원래 batch DataFrame에서 해당 컬럼들만 골라 새로운 view DataFrame을 만든다. 
            # np.random.shuffle(sub_col)
            sub_x = x.copy()[sub_col]
            sub_x_list.append(sub_x)
        return sub_x_list

    def _build_positive_pairs_single_view(self, x):
        # 이건 num_partition == 1일 때 쓰는 특별한 방식
        # 이 경우는 컬럼 분할이 아니라: 원본 view 하나, corruption된 view 하나 를 만든다. 
        x_cols = x.columns.tolist()  

        # 첫 번째 view는 원본 그대로   
        sub_x_list = [x]

        # 두 번째 view는 corruption된 것. 
        # : 절반 컬럼은 원래 값 유지, 절반 컬럼은 섞인 값으로 대체 
        # : 즉, 컬럼을 나눌 수 없으니까, 대신 같은 샘플의 약간 손상된 버전을 만들어 positive pair를 구성하는 것. (이미지 쪽에서의 augmentation을 주는 것과 비슷한 역할.)
        n_corrupt = int(len(x_cols)*0.5) # 전체 컬럼의 절반을 corruption 대상으로 잡는다. 
        corrupt_cols = x_cols[:n_corrupt]
        x_corrupt = x.copy()[corrupt_cols]
        np.random.shuffle(x_corrupt.values) # corruption 대상 컬럼들의 값을 섞는다. 즉, 컬럼 구조는 유지하되, 값의 row 대응을 흐트러뜨리는 방식. 
        sub_x_list.append(pd.concat([x.copy().drop(corrupt_cols,axis=1), x_corrupt], axis=1)) # 원본 일부 + 섞인 일부 합치기 
        
        return sub_x_list 

######################################################

# 특정 layer 제외하고 parameter 이름 가져오는 helper 
# weight decay 분리할 때 사용 (LayerNorm, bias 제외)
def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

# random seed 설정
# 역할: 재현성 (reproducibility)
def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# scheculder option에 따라 반환
# HuggingFace scheduler 그대로 사용 
# linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup 
def get_scheduler(
    name,
    optimizer,
    num_warmup_steps = None,
    num_training_steps = None,
    ):
    '''
    Unified API to get any scheduler from its name.

    Parameters
    ----------
    name: str
        The name of the scheduler to use.

    optimizer: torch.optim.Optimizer
        The optimizer that will be used during training.

    num_warmup_steps: int
        The number of warmup steps to do. This is not required by all schedulers (hence the argument being
        optional), the function will raise an error if it's unset and the scheduler type requires it.
    
    num_training_steps: int
        The number of training steps to do. This is not required by all schedulers (hence the argument being
        optional), the function will raise an error if it's unset and the scheduler type requires it.
    '''
    # 이름 --> 함수 매핑 
    name = name.lower()
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # constant의 경우 
    if name == 'constant':
        return schedule_func(optimizer)
    
    # warmup 필요한 경우 
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == 'constant_with_warmup':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
    
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    # 일반 경우 
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)