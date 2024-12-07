# RNN을 이용한 사인 파형 예측

이 프로젝트는 RNN(Recurrent Neural Network)을 사용하여 사인 파형 데이터를 예측하는 모델을 구현한 예제입니다. 주어진 시퀀스 데이터를 바탕으로 RNN 모델을 학습하고, 예측 결과를 시각화하여 모델의 성능을 평가합니다.

## 프로젝트 개요

이 코드는 사인 함수 데이터를 생성하여 모델을 학습시키고, 해당 데이터에 대한 예측값을 출력하는 간단한 예제입니다. RNN 모델을 사용하여 시계열 데이터 예측 문제를 해결하는 방법을 설명합니다.

## 주요 기능

- **사인 파형 데이터 생성**: 랜덤하게 시작하는 사인 파형 데이터를 생성합니다.
- **RNN 모델 학습**: `nn.RNN`을 사용하여 시계열 데이터를 학습하고 예측합니다.
- **모델 평가 및 시각화**: 예측 결과를 실제 데이터와 비교하여 모델 성능을 시각적으로 평가합니다.

## 요구사항

- Python 3.x
- PyTorch
- Matplotlib
- Numpy

## 설치 방법

이 프로젝트를 실행하려면 먼저 필요한 라이브러리를 설치해야 합니다. 아래 명령어를 통해 필요한 패키지를 설치할 수 있습니다.

```bash
pip install torch matplotlib numpy
```
코드 설명
1. 데이터 생성
사인 파형 데이터를 생성하는 함수입니다. 이 함수는 주어진 시퀀스 길이와 샘플 수에 대해 랜덤한 시작값을 사용하여 데이터를 생성합니다.

```python
import numpy as np

def create_sine_wave_data(seq_length, num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand()
        x = np.linspace(start, start + 2 * np.pi, seq_length)
        X.append(np.sin(x))
        y.append(np.sin(x + 0.1))  # 목표값은 약간 shift된 사인 파형
    return np.array(X), np.array(y)
```
2. 데이터 준비
생성된 데이터를 PyTorch 텐서로 변환하여 모델 학습에 사용할 수 있도록 준비합니다.

```python
import torch

seq_length = 50
num_samples = 1000
X, y = create_sine_wave_data(seq_length, num_samples)

# 데이터를 PyTorch 텐서로 변환
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (batch_size, seq_length, input_size)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (batch_size, seq_length, output_size)
```
3. RNN 모델 정의
RNN 모델을 정의하는 코드입니다. nn.RNN을 사용하여 입력 시퀀스를 처리하고, 마지막 시간 단계의 출력을 선형 계층을 통해 예측합니다.

```python
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)  # 초기 은닉 상태
        out, _ = self.rnn(x, h0)
        out = self.fc(out)  # 마지막 시간 단계의 출력
        return out
```
4. 모델 학습
모델을 학습하는 코드입니다. 주어진 에폭 수 만큼 학습을 진행하고, 매 10 에폭마다 손실을 출력합니다.
```python
import torch.optim as optim

input_size = 1
hidden_size = 32
output_size = 1
model = SimpleRNN(input_size, hidden_size, output_size)

# 손실 함수와 최적화 알고리즘 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 모델 학습
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X)
    optimizer.zero_grad()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Finished Training')
```
5. 모델 평가 및 시각화
모델을 평가한 후 예측값을 실제 값과 비교하여 시각화합니다.

```python
import matplotlib.pyplot as plt

# 모델 평가
model.eval()
with torch.no_grad():
    predicted = model(X).detach().numpy()

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(y.numpy().flatten()[:100], label='True')
plt.plot(predicted.flatten()[:100], label='Predicted')
plt.legend()
plt.title("RNN Sine Wave Prediction")
plt.show()
```
사용법
위의 코드를 실행하여 RNN 모델을 학습시킵니다.
학습이 완료되면, 예측 결과를 시각화하여 모델 성능을 확인할 수 있습니다.

결과
질이 별로인 데이터를 넣은 결과: ![image](https://github.com/user-attachments/assets/e33a5b15-f357-4988-a1ad-c18a84a79219)
질 좋은 데이터를 넣은 결과: ![download](https://github.com/user-attachments/assets/54c14e72-612f-4b8e-8eeb-367459c881e4)
