import os
import pdb
import math
import time
import json

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm.autonotebook import trange

from transtab import constants
from transtab.evaluator import predict, get_eval_metric_fn, EarlyStopping
from transtab.modeling_transtab import TransTabFeatureExtractor
from transtab.trainer_utils import SupervisedTrainCollator, TrainDataset
from transtab.trainer_utils import get_parameter_names
from transtab.trainer_utils import get_scheduler

import logging
logger = logging.getLogger(__name__)


# TransTab 모델 학습을 위한 Trainer 클래스 전체
# : tabular 데이터를 여러 dataset으로 받아서 학습 + 평가 + early stopping까지 관리하는 엔진 
class Trainer:
    def __init__(self,
        # 모델 (TransTab) 
        # logits, loss = model(x, y) 형태. ㄴ
        model, 
        # 데이터 
        # [(x1, y1), (x2, y2), ...] 형태. 
        train_set_list,
        test_set_list=None,
        collate_fn=None,
        output_dir='./ckpt',
        # 학습 하이퍼파라미터 
        num_epoch=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=0,
        # 평가 / early stopping / 학습 전략 설정 
        patience=5, # early stopping 기준 
        eval_batch_size=256,
        warmup_ratio=None, # lr warmup 
        warmup_steps=None, # lr warmup 
        balance_sample=False, # class imbalance 해결 옵션 (True면: batch를 pos/neg 반반 샘플링)
        load_best_at_last=True, # 학습 끝나면 best ckpt 다시 로드 
        ignore_duplicate_cols=False, # tabular column 중복 무시 여부 
        eval_metric='auc', 
        eval_less_is_better=False, 
        num_workers=0, # dataloader 병렬 처리 수 
        **kwargs,
        ):
        '''args:
        train_set_list: a list of training sets [(x_1,y_1),(x_2,y_2),...]
        test_set_list: a list of tuples of test set (x, y), same as train_set_list. if set None, do not do evaluation and early stopping
        patience: the max number of early stop patience
        num_workers: how many workers used to process dataloader. recommend to be 0 if training data smaller than 10000.
        eval_less_is_better: if the set eval_metric is the less the better. For val_loss, it should be set True.
        ''' 
        # 모델 세팅 
        self.model = model 

        # 데이터 세팅 
        if isinstance(train_set_list, tuple): train_set_list = [train_set_list]
        if isinstance(test_set_list, tuple): test_set_list = [test_set_list]
        self.train_set_list = train_set_list
        self.test_set_list = test_set_list
        self.collate_fn = collate_fn
        if collate_fn is None:
            self.collate_fn = SupervisedTrainCollator(
                categorical_columns=model.categorical_columns,
                numerical_columns=model.numerical_columns,
                binary_columns=model.binary_columns,
                ignore_duplicate_cols=ignore_duplicate_cols,
            )
        self.trainloader_list = [
            self._build_dataloader(trainset, batch_size, collator=self.collate_fn, num_workers=num_workers) for trainset in train_set_list
        ]
        if test_set_list is not None:
            self.testloader_list = [
                self._build_dataloader(testset, eval_batch_size, collator=self.collate_fn, num_workers=num_workers, shuffle=False) for testset in test_set_list
            ]
        else:
            self.testloader_list = None
        
        self.test_set_list = test_set_list
        self.output_dir = output_dir

        # 학습 세팅 
        self.early_stopping = EarlyStopping(output_dir=output_dir, patience=patience, verbose=False, less_is_better=eval_less_is_better)
        self.args = {
            'lr':lr,
            'weight_decay':weight_decay,
            'batch_size':batch_size,
            'num_epoch':num_epoch,
            'eval_batch_size':eval_batch_size,
            'warmup_ratio': warmup_ratio,
            'warmup_steps': warmup_steps,
            'num_training_steps': self.get_num_train_steps(train_set_list, num_epoch, batch_size),
            'eval_metric': get_eval_metric_fn(eval_metric),
            'eval_metric_name': eval_metric,
            }
        self.args['steps_per_epoch'] = int(self.args['num_training_steps'] / (num_epoch*len(self.train_set_list)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.optimizer = None
        self.lr_scheduler = None
        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last

    # 학습 루프 실행 
    def train(self):
        # 학습 파라미터, optimizer, scheduler 초기화 
        args = self.args
        self.create_optimizer()
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:
            num_train_steps = args['num_training_steps']
            logger.info(f'set warmup training in initial {num_train_steps} steps')
            self.create_scheduler(num_train_steps, self.optimizer)

        # run epoches with time measurement 
        # 전체 데이터셋을 여러 번 반복 학습 
        start_time = time.time()
        for epoch in trange(args['num_epoch'], desc='Epoch'): # trange: tqdm progress bar (진행률 표시)
            """
            epoch 반복 →
                모든 dataset →
                    모든 batch →
                        forward → loss → backward → update
                evaluation →
                early stopping 체크
            """

            ite = 0 # iter count
            train_loss_all = 0 # epoch loss 
            for dataindex in range(len(self.trainloader_list)): # 여러 dataset 순회
                for data in self.trainloader_list[dataindex]: # dataset 마다 batch 반복 
                    self.optimizer.zero_grad() # 이전 gradient 제거 (안하면 gradient 누적됨)
                    # forward
                    logits, loss = self.model(data[0], data[1])
                    # backward 
                    loss.backward() # gradient 계산 
                    self.optimizer.step() # weight 파라미터 업데이트 
                    # epoch 로그용 
                    train_loss_all += loss.item() # loss 누적: epoch 평균 loss 계산용 
                    ite += 1 # iteration. 증가
                    # scheduler 적용 
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            # 평가 (validation set(test_set_list) 있을 때만 평가)
            if self.test_set_list is not None: 
                # evaluate: 여러 dataset 평균 성능 
                eval_res_list = self.evaluate()
                eval_res = np.mean(eval_res_list)
                print('epoch: {}, test {}: {:.6f}'.format(epoch, self.args['eval_metric_name'], eval_res))
                # early stop: 성능 개선 없으면 학습 중단 
                self.early_stopping(-eval_res, self.model) # -붙이는 이유는 metric은 클수록 좋지만 early stopping은 작을수록 좋다고 가정하는 것이기 때문. 
                if self.early_stopping.early_stop:
                    print('early stopped')
                    break
                
            # epoch log 
            print('epoch: {}, train loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch, train_loss_all, self.optimizer.param_groups[0]['lr'], time.time()-start_time))

        # 학습 완료 후. checkpoint 저장 (저장 경로(output_dir)있어야만 진행) 
        # : 학습 끝 → best 모델 불러오기 → 최종 저장 
        if os.path.exists(self.output_dir):
            if self.test_set_list is not None:
                # load checkpoints: best를 불러온다. 
                logger.info(f'load best at last from {self.output_dir}')
                state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

        # 학습 완료 log 
        logger.info('training complete, cost {:.1f} secs.'.format(time.time()-start_time))

    # 평가 (validation metric 계산)
    # : 모델 eval 모드 → 모든 데이터 예측 → pred/label 모으기 → metric 계산
    def evaluate(self):
        """
        for dataset:
            for batch:
                예측 (no_grad)
                pred 저장
                label 저장
            metric 계산
        return [dataset별 metric]
        """ 

        # evaluate in each epoch
        self.model.eval() # eval 모드 전환 
        eval_res_list = []
        for dataindex in range(len(self.testloader_list)): # dataset 반복 
            y_test, pred_list, loss_list = [], [], [] 
            for data in self.testloader_list[dataindex]: # batch 반복 
                # label 수집 
                if data[1] is not None:
                    label = data[1]
                    if isinstance(label, pd.Series):
                        label = label.values
                    y_test.append(label)
                # forward 
                with torch.no_grad():
                    logits, loss = self.model(data[0], data[1])
                # outputs 
                if loss is not None:
                    loss_list.append(loss.item())
                if logits is not None:
                    # prediction 처리 
                    if logits.shape[-1] == 1: # binary classification: 0~1 확률. 
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else: # multi-class classification: 클래스별 확률. 
                        pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())

            # 모든 batch에 대한 결과 전체 합치기 
            if len(pred_list)>0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten() # binary flatten 

            # metric 계산 
            if self.args['eval_metric_name'] == 'val_loss':
                # loss 기준일 때: 단순 평균 loss 
                eval_res = np.mean(loss_list)
            else:
                # 일반 metric (AUC/accuracy/..)
                y_test = np.concatenate(y_test, 0)
                eval_res = self.args['eval_metric'](y_test, pred_all)
            
            # dataset별 metric 저장 
            eval_res_list.append(eval_res)

        return eval_res_list # [dataset1_metric, dataset2_metric, ...]
    
    # PyTorch DataLoader를 쓰지 않고 직접 pandas DataFrame을 잘라서 학습하는 버전
    # : 앞 train()의 단순화된 옛 방식/직접 배치 처리 방식. 
    """
        DataLoader 없이
        x_train, y_train에서 batch를 직접 잘라
        forward → backward → update → evaluation
        하는 학습 루프

        왜 따로 이런 함수가 있나? --> train()은 TrainDataset, DataLoader, collate_fn을 이용해 batch를 자동으로 만들었는데, 
        그래서 구현은 단순하지만, tabular 전처리나 병렬 로딩 측면에서는 train()보다 덜 유연하다. 
    """
    def train_no_dataloader(self,
        resume_from_checkpoint = None,
        ):
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint
        args = self.args
        self.create_optimizer()
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:
            print('set warmup training.')
            self.create_scheduler(args['num_training_steps'], self.optimizer)

        for epoch in range(args['num_epoch']):
            ite = 0
            # go through all train sets
            for train_set in self.train_set_list:
                x_train, y_train = train_set
                train_loss_all = 0

                ### batch_size씩 직접 잘라 학습
                for i in range(0, len(x_train), args['batch_size']):
                    self.model.train()
                    if self.balance_sample:
                        bs_x_train_pos = x_train.loc[y_train==1].sample(int(args['batch_size']/2))
                        bs_y_train_pos = y_train.loc[bs_x_train_pos.index]
                        bs_x_train_neg = x_train.loc[y_train==0].sample(int(args['batch_size']/2))
                        bs_y_train_neg = y_train.loc[bs_x_train_neg.index]
                        bs_x_train = pd.concat([bs_x_train_pos, bs_x_train_neg], axis=0)
                        bs_y_train = pd.concat([bs_y_train_pos, bs_y_train_neg], axis=0)
                    else:
                        bs_x_train = x_train.iloc[i:i+args['batch_size']]
                        bs_y_train = y_train.loc[bs_x_train.index]
                    ### 

                    self.optimizer.zero_grad()
                    logits, loss = self.model(bs_x_train, bs_y_train)
                    loss.backward()

                    self.optimizer.step()
                    train_loss_all += loss.item()
                    ite += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            if self.test_set is not None:
                # evaluate in each epoch
                self.model.eval()
                x_test, y_test = self.test_set
                pred_all = predict(self.model, x_test, self.args['eval_batch_size'])
                eval_res = self.args['eval_metric'](y_test, pred_all)
                print('epoch: {}, test {}: {}'.format(epoch, self.args['eval_metric_name'], eval_res))
                self.early_stopping(-eval_res, self.model)
                if self.early_stopping.early_stop:
                    print('early stopped')
                    break

            print('epoch: {}, train loss: {}, lr: {:.6f}'.format(epoch, train_loss_all, self.optimizer.param_groups[0]['lr']))

        if os.path.exists(self.output_dir):
            if self.test_set is not None:
                # load checkpoints
                print('load best at last from', self.output_dir)
                state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

    # 모델 저장 
    # : 학습 결과를 “완전히 재현 가능하게” 저장하는 함수
    # : 단순히 weight만 저장하는 게 아니라, 학습에 필요한 모든 상태를 같이 저장.
    # : 모델 + 전처리 + optimizer + scheduler + 학습 설정까지 전부 저장. 
    def save_model(self, output_dir=None):
        # 경로 설정 
        if output_dir is None:
            print('no path assigned for save mode, default saved to ./ckpt/model.pt !')
            output_dir = self.output_dir

        # 모델, collate_fn 저장
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        logger.info(f'saving model checkpoint to {output_dir}')
        self.model.save(output_dir)
        self.collate_fn.save(output_dir)

        # optimizer, scheduler 저장 
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, constants.OPTIMIZER_NAME))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, constants.SCHEDULER_NAME))
        
        # 학습 설정(json) 저장 
        if self.args is not None:
            train_args = {}
            for k,v in self.args.items():
                if isinstance(v, int) or isinstance(v, str) or isinstance(v, float):
                    train_args[k] = v
            with open(os.path.join(output_dir, constants.TRAINING_ARGS_NAME), 'w', encoding='utf-8') as f:
                f.write(json.dumps(train_args))

    def create_optimizer(self):
        # 기본 optimizer : Adam 
        if self.optimizer is None:
            # weight decay 선택적으로 적용
            # decay 대상: 모델의 모든 파라미터 이름 중에서 LayerNorm 제외한 것들 가져옴. (LayerNorm은 weight decay하면 성능 떨어짐)
            # decay 제외: bias는 weight decay 적용 안함. (bias는 regularization이 필요 없고, 오히려 성능이 나빠질 수 있다.)
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            # Adam optimizer 
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args['lr'])

    def create_scheduler(self, num_training_steps, optimizer):
        # 기본 scheduler: cosine 
        # : 초반엔 천천히 lr 올리고(warmup), 이후엔 cosine 형태로 lr을 줄이는 스케쥴러. 
        self.lr_scheduler = get_scheduler(
            'cosine',
            optimizer = optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_num_train_steps(self, train_set_list, num_epoch, batch_size):
        # num_training_steps 계산: 
        total_step = 0
        for trainset in train_set_list:
            x_train, _ = trainset
            total_step += np.ceil(len(x_train) / batch_size)
        total_step *= num_epoch
        return total_step

    def get_warmup_steps(self, num_training_steps):
        # num_training_steps 중에 warmup_ratio를 반영한 step 수 계산 
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.args['warmup_steps'] if self.args['warmup_steps'] is not None else math.ceil(num_training_steps * self.args['warmup_ratio'])
        )
        return warmup_steps

    def _build_dataloader(self, trainset, batch_size, collator, num_workers=8, shuffle=True):
        # dataloader 세팅 
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            )
        return trainloader