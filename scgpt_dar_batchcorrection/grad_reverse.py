# DAR의 핵심인 Gradient Reversal Layer (GRL)
# forward에서는 그대로 통과시키고, backward에서는 gradient를 반대로 뒤집는 레이어

import torch
from torch.autograd import Function 


class GradReverse(Function):
    """
        PyToch의 `autograd.Function`을 상속. 
        forward/backward를 직접 정의할 수 있는 커스텀 연산. 
    """
    @staticmethod
    def forward(
        ctx,  # forward에서 저장한 값을 backward로 전달하기 위한 컨테이너. forward → backward로 정보 넘겨주는 “임시 저장 공간”.  
        x: torch.Tensor, 
        lambd: float, 
    ) -> torch.Tensor: 
        ctx.lambd = lambd  # ctx.lambd에 값을 저장
        return x.view_as(x) 
    
    """
        forward:
            ctx에 값 저장
        backward:
            ctx에서 값 꺼내서 gradient 계산
        --> backward에서 gradient를 조절해야 하기 때문 
    """
    """
        GradReverse.apply(x, lambd)

        이걸 호출하면:
        PyTorch가 내부적으로 ctx 객체 생성
        forward → backward까지 연결된 그래프에 포함
    """
    
    @staticmethod
    def backard(
        ctx, 
        grad_output: torch.Tensor, 
    ) -> torch.Tensor: 
        # gradient = -λ * gradient 
        """
            gradient 방향을 뒤집음 (neg)
            크기는 λ로 조절
        """
        return grad_output.neg() * ctx.lambd, None
    

def grad_reverse(
        x: torch.Tensor, 
        lambd: float=1.0,
    ) -> torch.Tensor: 
        return GradReverse.apply(x, lambd)


