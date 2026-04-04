import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        # Bottleneck 구조: 1x1 Conv로 채널 수를 먼저 줄인 후 3x3 Conv 적용
        # BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)
        # DenseNet은 층이 깊어질수록 입력 채널(num_input_features)이 계속 누적되어 엄청나게 커지는데,
        # bn_size * growth_rate 만큼 채널을 확 줄여버린다. 이를 통해 3x3 연산 시 발생하는 막대한 계산량을 방지.
        # - bn_size는 Bottleneck Size의 약자로, DenseNet 논문에서 제안한 하이퍼파라미터. 주로 4가 기본값.

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)
        # 압축된 데이터로부터 실제 공간적 특징(Spatial Feature)을 추출
        # 최종적으로 딱 growth_rate 개수만큼의 특징 맵만 새로 만들어낸다.

        self.drop_rate = float(drop_rate)

    def forward(self, *prev_features):
        # *prev_features: 가변 인자로, 지금까지 쌓여온 모든 레이어의 출력 리스트를 받는다.
        # 가변 인자: 개수가 정해지지 않은 여러 개의 인자를 하나의 꾸러미(튜플)로 받겠다는 것
        # 여기서 prev_features는 (tensor1, tensor2, tensor3, ...) 형태의 튜플이 된다.

        # Forward 연산의 핵심: Concatenation
        # 이전 계층들의 특성 맵을 모두 채널 차원(1: B/C/H/W 중)으로 연결(concatenate)
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        # dropout: 학습 과정에서 무작위로 뉴런의 일부를 꺼버리는(0으로 만드는) 과적합 방지 방식
        # 가중치(Weight)가 아니라 출력(Feature Map)에 적용: 어떤 숫자가 0이 되었다면, 역전파 때 그 통로로 흐르는 기울기도 0이 되므로,
        #                               그 0을 만드는 데 참여했던 앞선 레이어들의 가중치들은 이번 학습 단계에서 업데이트되지 않는다.
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features

class _DenseBlock(nn.ModuleDict):
    # nn.ModuleDict와 동적 레이어 생성
    # : 반복문(for i in range(num_layers))을 통해 여러 개의 레이어를 만들 때,
    #   각 레이어에 denselayer1, denselayer2와 같은 고유한 이름을 부여하여 딕셔너리 형태로 저장하기 위해서

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()

        # Dense Block 마다 dense layer 수가 달라진다.
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, # 입력 채널 수가 누적되어 늘어남
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer) # 레이어 이름 부여

    def forward(self, init_features):
        # forward 함수의 특징: 리스트를 활용한 누적
        features = [init_features]          # 1) 리스트에 초기 입력 저장
        for name, layer in self.items():
            new_features = layer(*features) # 2) 리스트 전체를 언패킹(*)하여 입력으로 전달
            features.append(new_features)   # 3) 새로 만든 특징을 리스트에 추가
        return torch.cat(features, 1)       # 4) 모든 특징을 하나로 합쳐서 다음 블록으로 전달

class _Transition(nn.Sequential):
    # Dense Block 사이를 연결(전환)해주는 역할
    # : Dense Block을 거치면서 생기는 두 가지 문제(채널 비대화, 해상도 유지)를 해결한다.
    def __init__(self, num_input_features, num_output_features):
        super().__init__()

        # (i) 채널 수의 제어 (Compression)
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False)

        # (ii) 공간 해상도 축소 (Downsampling)
        # : Dense Block 내부의 모든 _DenseLayer는 입력과 출력의 크기가 동일하다. (그래야 서로 이어 붙일 수 있음.)
        #   하지만 CNN 모델은 뒤로 갈수록 이미지의 크기를 줄여가며 더 추상적인 정보를 뽑아야 한다.
        #   해결책: 블록과 블록 사이에 AvgPool을 두어 이미지 크기를 줄여줌으로써 일반적인 CNN의 피라미드 구조를 완성
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):
        super().__init__()

        # 첫 번째 합성곱 계층 (이미지 크기를 줄이고 초기 채널(3 to 64) 생성)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # DenseBlock 및 Transition 계층 추가
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            # 마지막 블록이 아니면 Transition 계층을 추가하여 채널 수를 절반으로 줄임
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # 최종 배치 정규화
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # 주의: ST-Net에서는 이 DenseNet의 self.features 까지만 사용하므로
        # 원본 DenseNet의 최종 분류기(classifier) 부분은 의도적으로 제외.

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        return out

# DenseNet-121을 반환하는 함수 정의
def _densenet121():
    """
    DenseNet-121 구조를 생성
    block_config=(6, 12, 24, 16)이 121개 층(120개의 합성곱 층 (+ 분류기))을 만든다
    """
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64)