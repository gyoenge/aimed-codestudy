# DSBN: Domain-Specific Batch Normalization 
# DAR와 짝을 이루는 batch correction의 또 다른 핵심 축. 
# 도메인(batch)마다 서로 다른 BatchNorm을 사용하는 구조. 

from typing import Optional, Tuple

import torch 
from torch import nn 

# The code is modified from https://github.com/wgchang/DSBN/blob/master/model/dsbn.py


# 도메인(batch)마다 서로 다른 BatchNorm을 선택해서 적용하는 공통 베이스 클래스
# 입력 x + domain_label → 해당 domain의 BN 선택 → normalization  
class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(
        self, 
        num_features: int, 
        num_domains: int,  # 몇 개의 batch(domain)이 있는지 
        eps: float=1e-5, 
        momentum: bool=True, 
        affine: bool=True, 
        track_running_stats: bool=True, 
    ): 
        super(_DomainSpecificBatchNorm, self).__init__() 
        self._cur_domain = None 
        self.num_domains = num_domains
        # BN을 여러 개 만든다. 
        # BN_0, BN_1, BN_2, ..., BN_(num_domains-1) 
        # batch마다 하나씩 BN을 만들어 둠. 
        self.bns = nn.ModuleList([
            self.bn_handle(
                num_features, 
                eps, momentum, 
                affine, 
                track_running_stats, 
            )
            for _ in range(num_domains)
        ]) # domain(batch) 개수만큼 bn_handle 함수를 만든다. 

    @property
    def bn_handel(self) -> nn.Module:
        # 이를 상속하는 자식 클래스에서 구현하도록 함. (이 클래스는 추상 클래스)
        raise NotImplementedError
    
    # cur_domain ### 

    @property 
    def cur_domain(self) -> Optional[int]:
        # 현재(current) 어떤 domain BN을 쓰는지 기록하는 역할 
        return self._cur_domain
    
    @cur_domain.setter 
    def cur_domain(self, domain_label: int):
        # 현재(current) 어떤 domain BN을 쓰는지 기록하는 역할 
        self._cur_domain = domain_label

    # reset 함수들 ### 
    # 모든 domain BN에 대해 반복 수행 

    def reset_running_stats(self):
        # BN의 mean/var 초기화 
        for bn in self.bns:
            bn.reset_running_stats() 

    def reset_parameters(self):
        # BN의 weight, bias 초기화 
        for bn in self.bns:
            bn.reset_parameters() 

    def _check_input_dim(self, input: torch.Tensor):
        # 이것도 자식 클래스에서 구현. 입력 tensor shape를 체크하는 역할. 
        raise NotImplementedError 
    
    # forward (핵심) ### 

    def forward(self, x: torch.Tensor, domain_label: int) -> torch.Tensor:
        self._check_input_dim(x) # 입력 검증 
        if domain_label >= self.num_domains: # domain 체크, 범위 벗어나면 에러 
            raise ValueError(
                f"Domain label {domain_label} exceeds the number of domains {self.num_domains}"
            )
        # BN 선택: 이 샘플은 domain_label 번 BN을 사용해라 
        bn = self.bns[domain_label]
        # 현재 domain 기록 
        self.cur_domain = domain_label
        # normalization 적용 반환 
        return bn(x)


# 실제로 사용할 BatchNorm 종류를 결정하는 부분 ### 

# DSBN의 구체 구현 (1D)
class DomainSpecificBatchNorm1d(_DomainSpecificBatchNorm):
    @property 
    def bn_handle(self) -> nn.Module:
        return nn.BatchNorm1d
    
    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() > 3: 
            raise ValueError(
                "expected at most 3D input (got {}D input)".format(input.dim())
            )
        
# DSBN의 구체 구현 (2D)
class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    @property
    def bn_handle(self) -> nn.Module: 
        return nn.BatchNorm2d
    
    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() != 4: 
            raise ValueError(
                "expected at most 4D input (got {}D input)".format(input.dim())
            )

