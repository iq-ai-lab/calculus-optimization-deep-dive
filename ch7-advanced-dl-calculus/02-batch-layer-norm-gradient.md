# 02. Batch Normalization과 Layer Normalization의 기울기

## 🎯 핵심 질문

- Batch Normalization의 역전파 계산이 복잡한 이유는?
- 배치 간 상호의존성이 기울기 계산에 미치는 영향은?
- Layer Normalization은 왜 배치 크기에 독립적인가?
- BN과 LN이 최적화 성능을 어떻게 향상시키는가?

## 🔍 왜 이 개념이 AI에서 중요한가

배치 정규화(Batch Normalization)는 깊은 신경망 훈련의 핵심 도구입니다:

1. **Internal Covariate Shift 감소**: 각 레이어 입력의 분포 안정화
2. **효과적인 학습률**: 학습률에 대한 민감도 감소
3. **헤시안 조건수 개선**: 최적화 랜드스케이프 개선
4. **수렴 속도**: 일반적으로 10-20배 빠른 수렴

Layer Normalization은 트랜스포머의 핵심:
- 시퀀스 길이에 독립적
- 배치 크기에 무관한 성능

## 📐 수학적 선행 조건

- 편미분, 연쇄법칙, 곱의 미분법
- 배치 연산과 브로드캐스팅
- 행렬 미분
- 분산과 표준편차의 성질

## ✏️ 정의와 핵심 도구

### Batch Normalization Forward

주어진 배치 $\mathbf{x} = [x_1, \ldots, x_B]^\top \in \mathbb{R}^{B \times D}$ (B: 배치 크기, D: 특징 차원):

**배치 통계**:
$$\mu_B = \frac{1}{B}\sum_{i=1}^B x_i \in \mathbb{R}^D$$

$$\sigma_B^2 = \frac{1}{B}\sum_{i=1}^B (x_i - \mu_B)^2 \in \mathbb{R}^D$$

**정규화**:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**스케일 및 시프트**:
$$y_i = \gamma \odot \hat{x}_i + \beta$$

여기서 $\gamma, \beta \in \mathbb{R}^D$는 학습 가능한 파라미터, $\epsilon$ = 수치 안정성을 위한 작은 상수

### Layer Normalization Forward

배치에 관계없이 각 샘플마다 정규화:

$$\mu_i = \frac{1}{D}\sum_{d=1}^D x_{id}$$

$$\sigma_i^2 = \frac{1}{D}\sum_{d=1}^D (x_{id} - \mu_i)^2$$

$$y_{id} = \gamma \cdot \frac{x_{id} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta$$

## 🔬 정리와 증명

### 정리 1: Batch Normalization 역전파

**명제**: BN의 역전파는 다음과 같이 구성된다:

$$\frac{\partial L}{\partial x_i} = \frac{1}{\sqrt{\sigma_B^2+\epsilon}}\left(\frac{\partial L}{\partial \hat{x}_i} - \frac{1}{B}\sum_{j=1}^B\frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i\frac{1}{B}\sum_{j=1}^B\frac{\partial L}{\partial \hat{x}_j}\odot\hat{x}_j\right)$$

여기서 $\odot$은 원소별 곱(Hadamard product)입니다.

**증명** (스칼라 z에 대해, 나중에 배치로 확장):

표준화 단계를 역전파하기 위해 세 부분으로 분해:

**Step 1: $\hat{x}$에서 $\gamma, \beta$로의 기울기**

$$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^B \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_{i=1}^B \frac{\partial L}{\partial y_i}$$

**Step 2: $\hat{x}$의 그라디언트**

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

**Step 3: $\hat{x}$에서 $x$로 역전파 (핵심)**

표준화는:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$\mu_B = \frac{1}{B}\sum_j x_j$와 $\sigma_B^2 = \frac{1}{B}\sum_j (x_j - \mu_B)^2$는 모두 배치의 모든 샘플에 의존합니다.

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \sum_{j \neq i} \frac{\partial L}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial x_i}$$

직접 항:
$$\frac{\partial \hat{x}_i}{\partial x_i} = \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} - \frac{(x_i - \mu_B)}{(\sigma_B^2+\epsilon)^{3/2}} \cdot \frac{\partial (x_i - \mu_B)}{\partial x_i}$$

$$= \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} - \frac{(x_i - \mu_B)}{(\sigma_B^2+\epsilon)^{3/2}} \cdot 1$$

간접항 ($\mu_B$를 통해):
$$\frac{\partial \hat{x}_j}{\partial x_i} \big|_{\mu_B} = -\frac{1}{B\sqrt{\sigma_B^2+\epsilon}}$$

간접항 ($\sigma_B^2$를 통해):
$$\frac{\partial \sigma_B^2}{\partial x_i} = \frac{2}{B}(x_i - \mu_B) \cdot \left(1 - \frac{1}{B}\right)$$

전체 계산은 복잡하므로, 결과를 정리하면:

$$\frac{\partial L}{\partial x_i} = \frac{1}{\sqrt{\sigma_B^2+\epsilon}} \left[\frac{\partial L}{\partial \hat{x}_i} - \frac{1}{B}\sum_j \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \frac{1}{B}\sum_j \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j\right]$$

**해석**: 세 항
1. 직접 기울기: $\frac{\partial L}{\partial \hat{x}_i}$
2. 평균 기울기를 빼기: 배치 평균 기여도 정규화
3. 분산 기울기를 고려: $\hat{x}_i$와 분산의 상호작용

### 정리 2: Batch Normalization의 헤시안 조건수 개선

**명제**: BN은 헤시안 조건수 $\kappa = \lambda_{\max}/\lambda_{\min}$을 개선합니다.

**직관적 설명**:

정규화 전 헤시안:
$$H = \begin{pmatrix} 100 & 0 \\ 0 & 1 \end{pmatrix}, \quad \kappa = 100$$

정규화 후 (표준편차 1로 스케일):
$$H' \approx \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad \kappa' = 1$$

**수치 증명 (3층 네트워크)**:

```python
import numpy as np

# 정규화 전
sigma_1 = 10.0
sigma_2 = 1.0
H_before = np.diag([sigma_1**2, sigma_2**2])
cond_before = np.linalg.cond(H_before)

# 정규화 후 (모든 특징이 평균 0, 표준편차 1)
H_after = np.eye(2)
cond_after = np.linalg.cond(H_after)

print(f"BN 전 조건수: {cond_before:.2f}")
print(f"BN 후 조건수: {cond_after:.2f}")
print(f"개선 비율: {cond_before/cond_after:.2f}배")
```

### 정리 3: Layer Normalization과 배치 독립성

**명제**: Layer Normalization은 배치 내 다른 샘플과 무관하게 각 샘플을 정규화합니다.

$$\frac{\partial y_{id}}{\partial x_{i'j}} = 0 \quad \text{for } i \neq i'$$

**증명**:

각 샘플 $i$에 대해:
$$\mu_i = \frac{1}{D}\sum_{d=1}^D x_{id}$$

$$\sigma_i^2 = \frac{1}{D}\sum_{d=1}^D (x_{id} - \mu_i)^2$$

다른 샘플 $i'$의 임의 특징 $x_{i'j}$는 $\mu_i$와 $\sigma_i^2$에 포함되지 않습니다:

$$\frac{\partial \mu_i}{\partial x_{i'j}} = 0, \quad \frac{\partial \sigma_i^2}{\partial x_{i'j}} = 0 \quad \text{for } i \neq i'$$

따라서:
$$\frac{\partial y_{id}}{\partial x_{i'j}} = 0 \quad \checkmark$$

이는 배치 크기가 LN의 성능에 영향을 주지 않음을 의미합니다.

### 정리 4: RMSNorm (바이어스 없는 LayerNorm)

**명제**:
$$\text{RMSNorm}(x) = x \odot \frac{\gamma}{\text{RMS}(x)}$$

여기서 $\text{RMS}(x) = \sqrt{\frac{1}{D}\sum_{d=1}^D x_d^2 + \epsilon}$

**역전파**:

$$\frac{\partial L}{\partial x_d} = \frac{\gamma}{RMS} \frac{\partial L}{\partial y_d} - \frac{x_d}{RMS^3}\left(\frac{\gamma}{RMS} \sum_j x_j \frac{\partial L}{\partial y_j}\right)$$

**효과**: 
- LN보다 계산 효율적 (평균 제거 불필요)
- LLaMA, PaLM 등 대규모 모델에서 사용

## 💻 NumPy/PyTorch 구현으로 검증

### Batch Normalization 역전파 구현

```python
import numpy as np
import torch
import torch.nn as nn

class BatchNormManual:
    """손동작 BN 구현"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x, training=True):
        """x: (B, D)"""
        self.B, self.D = x.shape
        
        if training:
            # 배치 통계
            self.mu = np.mean(x, axis=0)  # (D,)
            self.var = np.var(x, axis=0)   # (D,)
            
            # 실행 중 평균/분산 업데이트
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var
        else:
            self.mu = self.running_mean
            self.var = self.running_var
        
        # 정규화
        self.x_normalized = (x - self.mu) / np.sqrt(self.var + self.eps)  # (B, D)
        
        # 스케일 및 시프트
        y = self.gamma * self.x_normalized + self.beta
        
        return y
    
    def backward(self, dy):
        """dy: (B, D)"""
        
        # gamma, beta 기울기
        dgamma = np.sum(dy * self.x_normalized, axis=0)
        dbeta = np.sum(dy, axis=0)
        
        # x_normalized에서의 기울기
        dx_normalized = dy * self.gamma  # (B, D)
        
        # x로의 기울기 (배치 간 의존성 처리)
        dvar = np.sum(dx_normalized * (self.x_normalized) * (-0.5) * 
                      (self.var + self.eps)**(-1.5), axis=0)
        dmu = np.sum(dx_normalized * (-1) / np.sqrt(self.var + self.eps), axis=0) + \
              dvar * np.sum(-2 * (self.x_normalized) / self.B, axis=0)
        
        dx = dx_normalized / np.sqrt(self.var + self.eps) + \
             dvar * 2 * (self.x_normalized) / self.B + \
             dmu / self.B
        
        return dx, dgamma, dbeta

# 검증
print("=" * 60)
print("Batch Normalization 역전파 검증")
print("=" * 60)

np.random.seed(42)
B, D = 4, 3
x = np.random.randn(B, D) * 2 + 5  # 평균 5, 표준편차 2

bn = BatchNormManual(D)
y = bn.forward(x, training=True)

print(f"\nInput shape: {x.shape}")
print(f"Input statistics:")
print(f"  Mean: {np.mean(x, axis=0)}")
print(f"  Std:  {np.std(x, axis=0)}")

print(f"\nNormalized output statistics:")
print(f"  Mean: {np.mean(y, axis=0)}")
print(f"  Std:  {np.std(y, axis=0)}")

# 역전파 (더미 기울기)
dy = np.random.randn(B, D) * 0.01
dx, dgamma, dbeta = bn.backward(dy)

print(f"\nGradient shapes: dx={dx.shape}, dgamma={dgamma.shape}, dbeta={dbeta.shape}")

# 수치 미분으로 검증
def numerical_gradient(x_in, eps=1e-5):
    """수치 미분"""
    grad = np.zeros_like(x_in)
    for i in range(x_in.shape[0]):
        for j in range(x_in.shape[1]):
            x_plus = x_in.copy()
            x_minus = x_in.copy()
            x_plus[i, j] += eps
            x_minus[i, j] -= eps
            
            y_plus = bn.forward(x_plus, training=True)
            y_minus = bn.forward(x_minus, training=True)
            
            # 손실 = sum(dy * y)
            loss_plus = np.sum(dy * y_plus)
            loss_minus = np.sum(dy * y_minus)
            
            grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    return grad

print("\nNumerical gradient check:")
dx_numerical = numerical_gradient(x)
print(f"Max difference: {np.max(np.abs(dx - dx_numerical)):.2e}")
print(f"Relative error: {np.linalg.norm(dx - dx_numerical) / np.linalg.norm(dx_numerical):.2e}")
```

### Layer Normalization vs Batch Normalization

```python
class LayerNormManual:
    """손동작 LN 구현"""
    
    def __init__(self, num_features, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
    
    def forward(self, x):
        """x: (B, D)"""
        # 각 샘플별로 특징 차원에서 정규화
        self.mu = np.mean(x, axis=1, keepdims=True)  # (B, 1)
        self.var = np.var(x, axis=1, keepdims=True)  # (B, 1)
        self.x_normalized = (x - self.mu) / np.sqrt(self.var + self.eps)
        
        y = self.gamma * self.x_normalized + self.beta
        return y
    
    def backward(self, dy):
        dgamma = np.sum(dy * self.x_normalized, axis=0)
        dbeta = np.sum(dy, axis=0)
        
        # x_normalized에서의 기울기
        dx_normalized = dy * self.gamma
        
        # x로의 기울기
        D = dy.shape[1]
        dvar = np.sum(dx_normalized * self.x_normalized * (-0.5) * 
                      (self.var + self.eps)**(-1.5), axis=1, keepdims=True)
        dmu = np.sum(dx_normalized * (-1) / np.sqrt(self.var + self.eps), axis=1, keepdims=True) + \
              dvar * np.sum(-2 * self.x_normalized / D, axis=1, keepdims=True)
        
        dx = dx_normalized / np.sqrt(self.var + self.eps) + \
             dvar * 2 * self.x_normalized / D + \
             dmu / D
        
        return dx, dgamma, dbeta

# BN vs LN 비교
print("\n" + "=" * 60)
print("BN vs LN: 배치 크기 의존성")
print("=" * 60)

np.random.seed(42)
D = 5
x_base = np.random.randn(D) * 2 + 5

# 다양한 배치 크기에서 테스트
batch_sizes = [2, 4, 8, 16]

print(f"\nFeature dimension: {D}\n")
print(f"{'Batch Size':<12} | {'BN Output Std':<16} | {'LN Output Std':<16}")
print("-" * 50)

for B in batch_sizes:
    x = np.tile(x_base, (B, 1))  # 동일한 데이터를 반복
    
    bn = BatchNormManual(D)
    y_bn = bn.forward(x, training=True)
    
    ln = LayerNormManual(D)
    y_ln = ln.forward(x)
    
    # 각 배치에서의 출력 표준편차
    std_bn = np.std(y_bn)
    std_ln = np.std(y_ln)
    
    print(f"{B:<12} | {std_bn:<16.6f} | {std_ln:<16.6f}")

print("\n해석: BN은 배치 크기에 따라 출력 변동이 다름 (배치 통계 사용)")
print("     LN은 배치 크기와 무관하게 일관적 (샘플별 정규화)")
```

### RMSNorm 구현 및 비교

```python
class RMSNorm:
    """RMSNorm 구현"""
    
    def __init__(self, num_features, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(num_features)
    
    def forward(self, x):
        """x: (B, D)"""
        # RMS 계산
        self.rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + self.eps)  # (B, 1)
        self.x_normalized = x / self.rms
        y = self.gamma * self.x_normalized
        return y
    
    def backward(self, dy):
        D = dy.shape[1]
        dgamma = np.sum(dy * self.x_normalized, axis=0)
        
        dx_normalized = dy * self.gamma
        
        # 기울기: d(x/rms)/dx
        d_rms = -self.x_normalized / (self.rms**2) * np.sum(self.x_normalized * dx_normalized, axis=1, keepdims=True) / D
        dx = dx_normalized / self.rms + self.x_normalized * d_rms
        
        return dx, dgamma

print("\n" + "=" * 60)
print("RMSNorm vs LayerNorm")
print("=" * 60)

np.random.seed(42)
B, D = 4, 3
x = np.random.randn(B, D) * 2 + 5

ln = LayerNormManual(D)
y_ln = ln.forward(x)

rms = RMSNorm(D)
y_rms = rms.forward(x)

print(f"\nLayerNorm output (first row): {y_ln[0]}")
print(f"RMSNorm output (first row):   {y_rms[0]}")

print(f"\nLayerNorm - 의존 파라미터: gamma, beta (2D)")
print(f"RMSNorm - 의존 파라미터: gamma (D)")
```

### PyTorch 비교

```python
def pytorch_comparison():
    print("\n" + "=" * 60)
    print("PyTorch와의 비교")
    print("=" * 60)
    
    torch.manual_seed(42)
    B, D = 4, 3
    x_torch = torch.randn(B, D, requires_grad=True) * 2 + 5
    x_np = x_torch.detach().numpy()
    
    # PyTorch BN
    bn_torch = nn.BatchNorm1d(D, momentum=0.1)
    y_torch = bn_torch(x_torch)
    
    # 손수 BN
    bn_np = BatchNormManual(D, momentum=0.1)
    y_np = bn_np.forward(x_np, training=True)
    
    print(f"\nPyTorch BN output (first row): {y_torch[0].detach().numpy()}")
    print(f"Manual BN output (first row):  {y_np[0]}")
    print(f"Difference: {np.max(np.abs(y_torch[0].detach().numpy() - y_np[0])):.2e}")
    
    # PyTorch LN
    ln_torch = nn.LayerNorm(D)
    y_ln_torch = ln_torch(x_torch)
    
    # 손수 LN
    ln_np = LayerNormManual(D)
    y_ln_np = ln_np.forward(x_np)
    
    print(f"\nPyTorch LN output (first row): {y_ln_torch[0].detach().numpy()}")
    print(f"Manual LN output (first row):  {y_ln_np[0]}")
    print(f"Difference: {np.max(np.abs(y_ln_torch[0].detach().numpy() - y_ln_np[0])):.2e}")

pytorch_comparison()
```

### 헤시안 조건수 개선 시각화

```python
import matplotlib.pyplot as plt

def compute_hessian_condition_number(x, normalize=False):
    """간단한 2층 네트워크의 헤시안 조건수"""
    B, D = x.shape
    
    if normalize:
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)
    
    # Covariance 근사로 헤시안의 조건수 추정
    cov = np.cov(x.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    
    return np.max(eigvals) / np.min(eigvals)

print("\n" + "=" * 60)
print("헤시안 조건수 개선")
print("=" * 60)

np.random.seed(42)
B = 100
D = 10

# 스케일이 큰 차원
scales = [1, 10, 100, 1000]
cond_numbers_before = []
cond_numbers_after = []

for scale in scales:
    # 다양한 스케일의 특징
    x = np.hstack([
        np.random.randn(B, D//2) * scale,
        np.random.randn(B, D//2) * 1
    ])
    
    cond_before = compute_hessian_condition_number(x, normalize=False)
    cond_after = compute_hessian_condition_number(x, normalize=True)
    
    cond_numbers_before.append(cond_before)
    cond_numbers_after.append(cond_after)

plt.figure(figsize=(10, 5))
plt.plot(scales, cond_numbers_before, marker='o', linewidth=2, label='Before BN', markersize=8)
plt.plot(scales, cond_numbers_after, marker='s', linewidth=2, label='After BN', markersize=8)
plt.xlabel('Feature Scale Ratio', fontsize=12)
plt.ylabel('Condition Number κ', fontsize=12)
plt.title('BN에 의한 헤시안 조건수 개선', fontsize=13)
plt.yscale('log')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/bn_condition_number.png', dpi=150, bbox_inches='tight')
print("그래프 저장: /tmp/bn_condition_number.png")

print("\nCondition numbers:")
print(f"{'Scale':<10} | {'Before BN':<15} | {'After BN':<15} | {'Improvement':<15}")
print("-" * 60)
for scale, cond_b, cond_a in zip(scales, cond_numbers_before, cond_numbers_after):
    improvement = cond_b / cond_a
    print(f"{scale:<10} | {cond_b:<15.2f} | {cond_a:<15.2f} | {improvement:<15.2f}배")
```

## 🔗 AI/ML 연결

### 1. 심층 신경망 훈련

- **Internal Covariate Shift**: 각 레이어의 입력 분포가 훈련 중 변함
- **BN의 역할**: 각 미니배치의 통계로 정규화 → 분포 안정화

### 2. 트랜스포머 아키텍처

- Layer Normalization이 표준 (배치 크기 독립성)
- 자기 주의(Self-Attention) 전후에 LN 적용

### 3. 최적화 이론

- **Effective Learning Rate**: BN이 학습률 스케일 불변성 제공
- **Gradient Flow**: 깊은 네트워크에서 기울기 소실 완화

### 4. 이미지 처리

- **Convolutional Layers**: 채널별 BN
- **ResNet**: BN이 skip connection과 함께 잔차 네트워크 가능하게 함

## 📌 핵심 정리

| 개념 | 공식 | 특성 |
|------|------|------|
| **BN Forward** | $y = \gamma \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}} + \beta$ | 배치 통계 사용 |
| **LN Forward** | $y = \gamma \frac{x-\mu_i}{\sqrt{\sigma_i^2+\epsilon}} + \beta$ | 샘플별 정규화 |
| **BN Backward** | 복잡 (배치 간 의존성) | O(BD) 계산 |
| **LN Backward** | 간단 (샘플 독립) | 병렬화 용이 |
| **RMSNorm** | $y = \gamma \frac{x}{\text{RMS}(x)}$ | 평균 제거 없음 |
| **헤시안 개선** | $\kappa$ 감소 | 빠른 수렴 |

## 🤔 생각해볼 문제

1. **배치 의존성**: BN 역전파가 복잡한 이유를 $\frac{\partial \hat{x}_i}{\partial x_j}$의 항들로 설명하세요.

2. **훈련 vs 추론**: BN에서 running_mean/running_var는 왜 필요한가? LN에서는 왜 필요 없는가?

3. **헤시안 조건수**: BN이 왜 정확히 조건수를 개선하는지 고유값 관점에서 설명하세요.

4. **Group Normalization**: 
$$\mu_g = \frac{1}{|G|}\sum_{i \in G} x_i, \quad y_g = \gamma \frac{x_g - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} + \beta$$
   그룹 정규화와 BN, LN의 관계는?

5. **Instance Normalization**: 
$$y = \gamma \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} + \beta$$
   (각 샘플의 각 채널에서 정규화) 이것이 스타일 전이에 효과적인 이유는?

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Softmax 야코비안](./01-softmax-jacobian.md) | [📚 README](../README.md) | [03. NTK와 무한 폭 네트워크 ▶](./03-ntk-infinite-width.md) |

</div>
