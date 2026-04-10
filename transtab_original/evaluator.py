# TransTab의 평가/추론 유틸 모음. 
"""
predict : 모델로 예측하기
evaluate : metric 계산하기
get_eval_metric_fn 및 metric 함수들 : 어떤 지표 쓸지 결정
EarlyStopping : 성능이 안 좋아지면 학습 중단 + best model 저장
"""
"""
학습 자체 보다는, 학습 중간의 평가와 종료 판단을 담당하는 파일. 

이렇게 연결된다: 
모델 학습 중
→ validation set 예측
→ metric 계산
→ 성능이 좋아지면 checkpoint 저장
→ 성능 개선 없으면 early stopping
"""

from collections import defaultdict
import os
import pdb

import torch
import numpy as np
# metric 종류 
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

from transtab import constants

#############################
# 모델로 예측하기
# : 테스트 데이터를 batch 단위로 잘라서 모델 예측을 수행 
def predict(
    clf, # 학습된 모델 
    x_test, # 테스트용 feature DataFrame 
    y_test=None, # 정답 label (optional)
    return_loss=False, # loss도 반환지 여부
    eval_batch_size=256, # 추론 batch 크기 
    ):
    '''Make predictions by TransTabClassifier.

    Parameters
    ----------
    clf: TransTabClassifier
        the classifier model to make predictions.

    x_test: pd.DataFrame
            input tabular data.

    y_test: pd.Series
        target labels for input x_test. will be ignored if ``return_loss=False``.
    
    return_loss: bool
        set True will return the loss if y_test is given.
    
    eval_batch_size: int
        the batch size for inference.

    Returns
    -------
    pred_all: np.array
        if ``return_loss=False``, return the predictions made by TransTabClassifier.

    avg_loss: float
        if ``return_loss=True``, return the mean loss of the predictions made by TransTabClassifier.

    '''
    clf.eval() # 평가 모드 
    pred_list, loss_list = [], []

    # batch 단위로 반복 
    for i in range(0, len(x_test), eval_batch_size):
        bs_x_test = x_test.iloc[i:i+eval_batch_size]
        bs_y_test = y_test.iloc[i:i+eval_batch_size] if y_test is not None else None
        
        # forward (gradient 없이 -> 속도/메모리효율 좋음)
        with torch.no_grad():
            logits, loss = clf(bs_x_test, bs_y_test)
        
        # loss
        if loss is not None:
            loss_list.append(loss.item())
        # 출력 확률로 반환 
        if logits.shape[-1] == 1: # binary classification
            pred_list.append(logits.sigmoid().detach().cpu().numpy())
        else: # multi-class classification
            pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())
    
    # 모든 batch 결과 합치기 
    pred_all = np.concatenate(pred_list, 0)
        
    if logits.shape[-1] == 1:
        pred_all = pred_all.flatten()

    if return_loss:
        # 평균 loss 반환 
        avg_loss = np.mean(loss_list)
        return avg_loss
    else:
        # 예측값 전체 반환 
        return pred_all
"""
주의: 
이 함수는 현재 분류 문제 기준으로 작성돼 있다. 
왜냐하면 출력 처리에서 무조건 sigmoid 또는 softmax를 쓰기 때문.
그래서 regression이면 그대로 쓰기보다 수정이 필요. 
"""    


############################# 
# metric 계산하기
# : 예측값과 정답을 받아 metric을 계산한다. 
def evaluate(ypred, y_test, metric='auc', seed=123, bootstrap=False):
    np.random.seed(seed)
    eval_fn = get_eval_metric_fn(metric) # metric 함수 선택 
    res_list = []
    stats_dict = defaultdict(list)

    # bootstrap 여부에 따라 
    if bootstrap:
        # 복원 추출로 샘플을 다시 뽑아 여러 번 metric을 계산하고, 
        # 그 분포로 평균/구간을 추정한다. 
        """
        bootstrap의 의미: 
            metric 하나만 딱 보는 게 아니라,
            “이 성능이 어느 정도 안정적인가?”를 보고 싶을 때 쓰는 방식
        지금 코드는 10번 반복해서 95% 구간 비슷한 걸 출력한다. 
        """
        for i in range(10): # bootstrap resampling을 10번 (실제로는 10번은 조금 적은 편이고, 통계적으로는 보통 100번, 1000번 이상도 자주 쓴다.) 
            # 복원 추출 인덱스 생성 (어떤 샘플은 여러 번 뽑히고, 어떤 샘플은 이번 bootstrap 샘플에서는 아예 빠질 수도 있다.)
            sub_idx = np.random.choice(np.arange(len(ypred)), len(ypred), replace=True)
            # bootstrap 샘플 만들기 (원본에서 복원추출된 새 평가셋)
            sub_ypred = ypred[sub_idx]
            sub_ytest = y_test.iloc[sub_idx]
            # metric 계산 (예외 처리 포함: 예를 들어 AUC의 경우 샘플 안에 pos만/neg만 있으면 AUC 계산이 불가능해서 ValueError가 날 수 있다.)
            try:
                sub_res = eval_fn(sub_ytest, sub_ypred)
            except ValueError:
                print('evaluation went wrong!')
            # metric 값 저장. 실제로는 10번은 조금 적은 편이고, 통계적으로는 보통 100번, 1000번 이상도 자주 쓴다.
            stats_dict[metric].append(sub_res)
        # 저장된 metric 분포의 평균/구간 추정
        for key in stats_dict.keys():
            # 저장된 metric 분포 꺼내기 
            stats = stats_dict[key]
            alpha = 0.95 # 95% 구간을 보고 싶다는 뜻. 
            # 하한 percentile 계산 
            p = ((1-alpha)/2) * 100
            lower = max(0, np.percentile(stats, p)) # max(0, ...)를 쓰는 이유는, AUC 같은 metric은 음수가 나오면 이상하니까 0 아래로 안 내려가게 막으려는 의도
            # 상한 percentile 계산 
            p = (alpha+((1.0-alpha)/2.0)) * 100
            upper = min(1.0, np.percentile(stats, p))
            # 가운데값과 반폭 출력 (중심값 ± 반폭 형태로 보여준다.)
            print('{} {:.2f} mean/interval {:.4f}({:.2f})'.format(key, alpha, (upper+lower)/2, (upper-lower)/2))
            if key == metric: res_list.append((upper+lower)/2) # lower와 upper의 중간값을 대표값으로 쓰고 있다. 
    else:
        # 그냥 전체 데이터에 대해 metric 한 번 계산.
        # 보통은 이 모드로 많이 쓴다. 
        res = eval_fn(y_test, ypred)
        res_list.append(res)
    return res_list

############################# 
# 어떤 지표 쓸지 결정

def get_eval_metric_fn(eval_metric):
    fn_dict = {
        'acc': acc_fn,
        'auc': auc_fn,
        'mse': mse_fn,
        'val_loss': None,
    }
    return fn_dict[eval_metric]

# acc 
def acc_fn(y, p):
    y_p = np.argmax(p, -1)
    return accuracy_score(y, y_p)
# auc
def auc_fn(y, p):
    return roc_auc_score(y, p)
# mse
def mse_fn(y, p):
    return mean_squared_error(y, p)

#############################
# 성능이 안 좋아지면 학습 중단 + best model 저장
#: validation 성능이 좋아지지 않으면 학습을 멈추는 역할. 과적합 방지. 
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, output_dir='ckpt', trace_func=print, less_is_better=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print     
            less_is_better (bool): If True (e.g., val loss), the metric is less the better.       
        """
        self.patience = patience # 몇 번까지 성능 개선이 없어도 기다릴지 
        self.verbose = verbose 
        self.counter = 0 # 연속으로 개선되지 않은 횟수
        self.best_score = None # 지금까지 최고 성능 
        self.early_stop = False # 이 값이 True가 되면 학습 중단 
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = output_dir # best model을 저장할 위치 
        self.trace_func = trace_func
        self.less_is_better = less_is_better # loss처럼 작은 게 좋은 metric인지 여부 

    def __call__(self, val_loss, model):
        # 매 epoch마다 validation 결과를 넣어주면: 

        if self.patience < 0: # no early stop
            self.early_stop = False
            return
        
        # (i) 현재 점수가 이전 best보다 좋은지 확인 
        if self.less_is_better:
            score = val_loss
        else:    
            score = -val_loss

        # (ii) 좋으면 checkpoint 저장 
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # (iii) 안좋으면 counter 증가 
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # (iv) counter가 patience 넘으면 early_stop=True 
            if self.counter >= self.patience:
                self.early_stop = True
        # base case 
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    # checkpoint 저장 
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, constants.WEIGHTS_NAME))
        self.val_loss_min = val_loss
