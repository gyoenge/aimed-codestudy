# source : https://github.com/bowang-lab/BLEEP 

import torch 
from torch import nn
import torch.nn.functional as F

import config as CFG 
from modules import (
    # 기본 이미지 인코더: backbone CNN/ViT 변경 가능 
    ImageEncoder,   

    # 인코더가 뽑은 원본 embedding을 바로 쓰지 않고, 학습하기 좋은 projection space로 한 번 더 변환하는 역할
    ProjectionHead, 
    # -- 구조: Linear -> GELU -> Linear -> Dropout -> residual conn -> LayerNorm
    # -- 비선형성 + residual + normalization 을 넣은 작은 MLP head

    # ViT 계열 인코더: Transformer 기반
    ImageEncoder_ViT,   # backbone을 ViT로 고정 
    ImageEncoder_ViT_L, # backbone을 ViT-L로 고정 
    # --> vit_large_patch32_224_in21k를 쓰므로, 기본 ViT보다 더 큰 모델. 
    # --> 더 강한 표현력을 기대할 수 있지만 메모리와 연산량 부담이 크다. 

    # CLIP 계열 인코더: image-text pretraining 기반
    ImageEncoder_CLIP,  # CLIP 계열 사전학습 ViT모델을 backbone으로 사용
    # --> vit_base_patch32_224_clip_laion2b를 사용하므로, 일반 ImageNet 분류 pretrained와 달리 대규모 image-text 정렬 데이터로 학습된 표현을 사용.  
    # --> 그래서 보통 zero-shot 성질이나 semantic feature가 더 강할 수 있다. 

    # ResNet 계열 인코더: CNN 기반
    ImageEncoder_resnet101, # backbone을 resnet101로 고정
    ImageEncoder_resnet152, # backbone을 resnet152로 고정
)

"""
CLIP-style contrastive learning을 이미지 ↔ 유전자(spot) 간에 적용한 모델
: 이미지 임베딩 ↔ gene(spot) 임베딩을 같은 공간에 매핑해서, 서로 맞는 쌍은 가깝게 / 아닌 쌍은 멀게 학습
--modification--> 
BLEEP-style Contrastive Learning **  
: Rather than one-hot target, BLEEP utilizes similarity awart soft target. 
""" 

# 1. 이미지 인코더로 ImageEncoder 사용. 
class CLIPModel(nn.Module):
    def __init__(
        self, 
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding, 
    ):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Image & Spot Features 
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        """
        image: CNN/ViT로 feature 추출 
        spot: 이미 reduced된 gene vector (예: HVG 3467개)

        image는 "representation learning"
        spot은 이미 vector라서 encoder 없이 바로 사용
        """ 

        # Image & Spot Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)
        """
        projection (같은 공간으로 정렬)
        : 서로 다른 modality (image vs gene)를 같은 차원의 embedding space로 맞춤 
        """

        # Loss 
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        """
        (i) Cross-modal similarity 
            - 각 spot vs 각 image의 similarity matrix
            - shape: (batch_size, batch_size)

            - temperature의 역할: 
                - temperature ↓ --> 분포 sharper, hard negative 강조.
                - temperature ↑ --> 분포 smoother, 학습 안정적. 
        """

        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T 
        """
        (ii) intra-modal similarity 
            - 각각: image끼리 similarity, spot끼리 similarity 
        """

        targets = F.softmax(
            ((images_similarity + spots_similarity) / 2) / self.temperature, 
            dim=-1,
        )
        """ 
        (iii) targets 생성
            - 일반 CLIP과 다른 점. 
            - 보통 CLIP은 targets = identity matrix.
            - 근데 여기서는: "유사한 것끼리는 soft하게 정답 확률을 공유".  

            - image끼리 비슷하면 → 대응 spot도 비슷해야 함 
            - spot끼리 비슷하면 → 대응 image도 비슷해야 함 

            - 그래서 (image similarity + spot similarity) / 2
            --> soft target distribution 생성. 
            --> == "cross modality 정렬을 할 때 비슷한 "정도"의 정답값을 부여한다"는 개념. 

            - temperature: target 분포의 sharpness을 조절. (softmax 전에 값의 스케일을 조절)
        """ 

        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (spots_loss + images_loss) / 2.0  
        """ 
        (iv) 최종 Loss 계산 
            - Cross Entropy는 
            - 양방향이라는 점. 
                - spot --> image : spot 기준으로 맞는 image 찾기.
                - image --> spot : image 기준으로 맞는 spot 찾기. 
            - 즉, 
                - "Gene-conditioned Image Embedding 학습"
                - "Image-conditioned Gene Embedding 학습"

            - 최종 shape: (batch_size)  
        """ 
        return loss.mean() 


# 2. 이미지 인코더로 ViT 계열 사용. 
class CLIPModel_ViT(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=768, ### 
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_ViT() ###
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) 
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) 
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]

        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 
        return loss.mean()


# 3. 이미지 인코더로 CLIP 인코더 사용. 
class CLIPModel_CLIP(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=768, ### 
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_CLIP() ### 
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) 
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) 
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]

        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 
        return loss.mean()


# 4. 이미지 인코더로 ViT-L 사용. 
class CLIPModel_ViT_L(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=1024, ### 
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_ViT_L() ### 
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) 
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) 
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]

        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 
        return loss.mean()


# 5. 이미지 인코더로 ResNet-101 사용. 
class CLIPModel_resnet101(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=2048,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet101()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) 
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) 
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]

        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 
        return loss.mean()


# 6. 이미지 인코더로 ResNet-152 사용. 
class CLIPModel_resnet152(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=2048,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet152()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) 
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 
        return loss.mean()


### 

# define cross entropy
# : 예측 logits vs 확률 분포(targets)를 비교하는 cross entropy

def cross_entropy(preds, targets, reduction='none'):
    """
    preds: 모델 출력 (logits) 
    targets: 정답 분포 (확률 형태, one-hot일 수도 있고 soft label일 수도 있음)
                - 일반적인 경우 (one-hot): targets = [0, 0, 1, 0]. 
                                        loss는 결국: - log(p_correct_class).
                - soft label 형태: targets = [0.1, 0.2, 0.7].
                                loss: - (0.1 log p1 + 0.2 log p2 + 0.7 log p3). 
                                즉, "정답이 하나가 아니라 분포". 정답이 “유사도 기반 soft distribution”. 

    nn.CrossEntropyLoss X --> (one-hot only)
    custom cross_entropy O -> (soft label 가능)                            
    
    reduction 옵션: 각 샘플별 loss 반환 or 평균 loss 반환 
    """
    log_softmax = nn.LogSoftmax(dim=-1)  # softmax를 적용한 뒤 log를 취한 함수. 마지막 차원 기준으로 계산. 
    loss = (-targets * log_softmax(preds)).sum(1)
    """
    $$ loss = - \Sigma_{i} target_i \cdot \log p_i $$  
        target_i: 정답분포, p_i: 모델이 예측한 확률 
    """
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean() 

# CLIP에서는 보통: none으로 계산 후, 양방향 loss 합치고, 마지막에 mean


# Usage: 
# 최소 실행 예제 (dummy test)
if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25)) # 의미 - input_ids: 토큰 ID (랜덤)
    attention_mask = torch.ones(8, 25) # 의미 - attention_mask: padding 없는 상태
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")

