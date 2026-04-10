# 모델 학습 결과를 저장할 때 사용하는 “파일 이름 규칙(constants)” 정의
# 즉, save_model()이나 checkpoint 저장할 때 어떤 파일을 어떤 이름으로 저장할지 정해둔 것.
"""
ckpt/
├── pytorch_model.bin       ← 모델
├── optimizer.pt            ← optimizer
├── scheduler.pt            ← scheduler
├── training_args.json      ← 설정
├── extractor/
│   └── extractor.json      ← feature 처리
├── input_encoder.bin       ← encoder
"""

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.json" # 학습 설정 (lr, batch_size 등). 목적 - 실험 재현 (reproducibility). 
TRAINER_STATE_NAME = "trainer_state.json" # Trainer 상태 (epoch, step 등). 보통 resume training 할 때 사용. 
OPTIMIZER_NAME = "optimizer.pt" # optimizer 내부 상태 (momentum, weight 등). 학습 이어서 할 때 필요. 
SCHEDULER_NAME = "scheduler.pt" # learning rate 상태. 없으면 lr 초기화 --> 학습 꼬임. 
WEIGHTS_NAME = "pytorch_model.bin" # (**) 모델 weight 파일. 
TOKENIZER_DIR = 'tokenizer' # 디렉토리 내부에 categorical encoding 관련 정보.
EXTRACTOR_STATE_DIR = 'extractor' # feature extractor 저장 폴더. 
EXTRACTOR_STATE_NAME = 'extractor.json' # 내용: feature 처리 방식 (column type, normalization 정보 등).
INPUT_ENCODER_NAME = 'input_encoder.bin' # 실제 입력 인코더. transformer 내부 encoder weight 파일. 

"""
참고: 
딥러닝 모델 저장은 보통 2가지 방식이 있음:
(i) 단순 방식: torch.save(model.state_dict())
        - 문제: 전처리 없음, 재현 불가능. 
(ii) TransTab 방식: 모델 + 전처리 + 학습 상태 전부 저장. 
        - 장점: inference 가능, resume training 가능, 실험 재현 가능. 
"""
