# 01. Softmax의 야코비안

## 🎯 핵심 질문

- Softmax 함수의 야코비안은 정확히 어떤 구조를 가지는가?
- Cross-Entropy 손실과 Softmax를 결합하면 왜 역전파가 깔끔해지는가?
- 큰 입력값에서 Softmax 계산이 불안정해지는 이유는?
- Temperature와 Gumbel-Softmax는 어떤 역할을 하는가?

## 🔍 왜 이 개념이 AI에서 중요한가

Softmax는 신경망의 거의 모든 분류 문제의 마지막 활성화 함수입니다. 야코비안을 정확히 이해하면:

1. **역전파 효율성**: Cross-entropy와의 결합에서 gradient가 $\sigma_j - y_j$로 단순화됨
2. **수치 안정성**: Overflow/underflow 방지로 안정적인 훈련
3. **고급 기법**: Temperature scaling (증류학습), Gumbel-Softmax (이산 최적화) 등의 수학적 토대

## 📐 수학적 선행 조건

- 편미분, 연쇄법칙
- 지수함수와 로그함수의 미분
- 행렬 미분 (Jacobian, Hessian)
- 로그합지수(LogSumExp) 함수의 성질

## ✏️ 정의와 핵심 도구

### Softmax 정의

주어진 벡터 $\mathbf{z} = [z_1, \ldots, z_n]^\top$에 대해:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$$

성질:
- $\sum_{i=1}^n \sigma_i = 1$ (확률 분포)
- $\sigma_i > 0$ for all $i$

### 야코비안 (Jacobian) 정의

Softmax의 야코비안은 $\mathbf{J}_\sigma \in \mathbb{R}^{n \times n}$로:

$$J_\sigma(i, j) = \frac{\partial \sigma_i}{\partial z_j}$$

### Cross-Entropy 손실

$$\mathcal{L} = -\sum_{i=1}^n y_i \log \sigma_i$$

여기서 $\mathbf{y}$는 원-핫 인코딩된 정답 레이블입니다.

## 🔬 정리와 증명

### 정리 1: Softmax 야코비안의 명시적 형태

**명제**: 
$$\frac{\partial \sigma_i}{\partial z_j} = \begin{cases} 
\sigma_i(1 - \sigma_i) & \text{if } i = j \\
-\sigma_i \sigma_j & \text{if } i \neq j
\end{cases}$$

**증명**:

$i = j$인 경우:
$$\frac{\partial \sigma_i}{\partial z_i} = \frac{\partial}{\partial z_i}\left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right)$$

분자를 $u = e^{z_i}$, 분모를 $v = \sum_k e^{z_k}$라 하면, 몫의 미분법:

$$\frac{\partial \sigma_i}{\partial z_i} = \frac{e^{z_i} \cdot v - e^{z_i} \cdot e^{z_i}}{v^2}$$

$$= \frac{e^{z_i}(v - e^{z_i})}{v^2} = \frac{e^{z_i}}{v}\left(1 - \frac{e^{z_i}}{v}\right)$$

$$= \sigma_i(1 - \sigma_i) \quad \checkmark$$

$i \neq j$인 경우:
$$\frac{\partial \sigma_i}{\partial z_j} = \frac{\partial}{\partial z_j}\left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right)$$

분자는 $z_j$에 독립이고, 분모만 미분되므로:

$$\frac{\partial \sigma_i}{\partial z_j} = e^{z_i} \cdot \frac{\partial}{\partial z_j}\left(\sum_k e^{z_k}\right)^{-1}$$

$$= e^{z_i} \cdot (-1) \cdot \left(\sum_k e^{z_k}\right)^{-2} \cdot e^{z_j}$$

$$= -\frac{e^{z_i} \cdot e^{z_j}}{(\sum_k e^{z_k})^2} = -\sigma_i \sigma_j \quad \checkmark$$

### 정리 2: 야코비안의 행렬 형태

**명제**:
$$\mathbf{J}_\sigma = \text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^\top$$

여기서 $\text{diag}(\boldsymbol{\sigma})$는 대각원소가 $\sigma_1, \ldots, \sigma_n$인 대각행렬입니다.

**증명**:

우변을 전개하면:
$$[\text{diag}(\boldsymbol{\sigma})]_{ij} = \sigma_i \cdot \delta_{ij}$$
$$[\boldsymbol{\sigma}\boldsymbol{\sigma}^\top]_{ij} = \sigma_i \sigma_j$$

따라서:
$$[\text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^\top]_{ij} = \sigma_i\delta_{ij} - \sigma_i\sigma_j$$

$i = j$일 때: $\sigma_i - \sigma_i^2 = \sigma_i(1-\sigma_i)$ ✓

$i \neq j$일 때: $0 - \sigma_i\sigma_j = -\sigma_i\sigma_j$ ✓

### 정리 3: Cross-Entropy + Softmax의 역전파

**명제**: $\mathcal{L} = -\sum_i y_i \log \sigma_i$일 때,
$$\frac{\partial \mathcal{L}}{\partial z_j} = \sigma_j - y_j$$

**증명**:

연쇄법칙으로:
$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{i=1}^n \frac{\partial \mathcal{L}}{\partial \sigma_i} \cdot \frac{\partial \sigma_i}{\partial z_j}$$

먼저 $\frac{\partial \mathcal{L}}{\partial \sigma_i}$:
$$\frac{\partial \mathcal{L}}{\partial \sigma_i} = \frac{\partial}{\partial \sigma_i}\left(-\sum_k y_k \log \sigma_k\right) = -\frac{y_i}{\sigma_i}$$

따라서:
$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{i=1}^n \left(-\frac{y_i}{\sigma_i}\right) \cdot \frac{\partial \sigma_i}{\partial z_j}$$

정리 1을 사용하면:
$$= -y_j \cdot \frac{1}{\sigma_j} \cdot \sigma_j(1-\sigma_j) + \sum_{i \neq j} \left(-\frac{y_i}{\sigma_i}\right) \cdot (-\sigma_i \sigma_j)$$

$$= -y_j(1-\sigma_j) + \sigma_j \sum_{i \neq j} y_i$$

$$= -y_j + y_j\sigma_j + \sigma_j\sum_{i \neq j} y_i$$

$$= -y_j + \sigma_j\left(y_j + \sum_{i \neq j} y_i\right)$$

$\sum_i y_i = 1$ (원-핫 인코딩)이므로 $y_j + \sum_{i \neq j} y_i = 1$:

$$= -y_j + \sigma_j = \sigma_j - y_j \quad \checkmark$$

### 정리 4: Softmax 수치 안정성

**명제**: 임의의 상수 $c$에 대해 $\sigma(\mathbf{z}) = \sigma(\mathbf{z} - c)$

**증명**:
$$\sigma(\mathbf{z} - c)_i = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c} \cdot e^{z_i}}{e^{-c} \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}} = \sigma(\mathbf{z})_i \quad \checkmark$$

**실무**: $c = \max(\mathbf{z})$로 설정하여 $e^{z_i - \max z}$ 계산으로 overflow 방지

### 정리 5: Log-Sum-Exp 트릭

**명제**: 
$$\log\sum_{i=1}^n e^{z_i} = \max_i z_i + \log\sum_{i=1}^n e^{z_i - \max_i z_i}$$

**증명**:
$$\log\sum_{i=1}^n e^{z_i} = \log\left(e^{\max z} \sum_{i=1}^n e^{z_i - \max z}\right)$$

$$= \log e^{\max z} + \log\sum_{i=1}^n e^{z_i - \max z}$$

$$= \max z + \log\sum_{i=1}^n e^{z_i - \max z} \quad \checkmark$$

$e^{z_i - \max z} \le 1$이므로 exponential underflow 방지

### 정리 6: Temperature Softmax

**명제**: Temperature $T > 0$일 때,
$$\sigma_T(\mathbf{z})_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

극한:
- $T \to 0^+$: $\sigma_T \to \text{one-hot}(\arg\max z_i)$ (hard max)
- $T \to \infty$: $\sigma_T \to \text{uniform}$ (완전히 부드러워짐)

**증명**: $T \to 0$인 경우

$z_1 = \max(\mathbf{z})$이라 하면:
$$\sigma_T(\mathbf{z})_1 = \frac{e^{z_1/T}}{\sum_j e^{z_j/T}} = \frac{1}{1 + \sum_{j \neq 1} e^{(z_j - z_1)/T}}$$

$T \to 0$일 때, $z_j - z_1 < 0$ ($j \neq 1$)이므로 $e^{(z_j-z_1)/T} \to 0$:

$$\sigma_T(\mathbf{z})_1 \to \frac{1}{1+0} = 1 \quad \checkmark$$

다른 $i$에 대해서도 $\sigma_T(\mathbf{z})_i \to 0$

### 정리 7: Gumbel-Softmax

**명제**: $g_i = -\log(-\log U_i)$ (Gumbel 분포, $U_i \sim \text{Uniform}(0,1)$)라 할 때,
$$\tilde{\sigma}_i = \frac{e^{(z_i + g_i)/T}}{\sum_j e^{(z_j + g_j)/T}}$$

는 이산 분포의 연속 완화로, $T \to 0$일 때 카테고리 분포의 샘플로 수렴합니다.

**증명 개요**:

Gumbel-Max Trick: 이산 분포에서 샘플링하려면
$$i^* = \arg\max_i (z_i + g_i)$$

이 정확히 다항 분포를 따릅니다. Temperature를 도입한 Gumbel-Softmax는 이를 미분 가능하게 만듭니다:

$$\log p(i^*) = z_i - \log\sum_j e^{z_j}$$

Gumbel-Softmax의 기댓값은 $T \to 0$일 때 이 확률로 수렴합니다.

## 💻 NumPy/PyTorch 구현으로 검증

### Softmax 야코비안 수치 계산

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z, T=1.0):
    """수치 안정적인 softmax"""
    z = z / T
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def softmax_jacobian_numerical(z, epsilon=1e-5):
    """수치 미분으로 야코비안 계산"""
    n = len(z)
    J = np.zeros((n, n))
    
    for j in range(n):
        z_plus = z.copy()
        z_minus = z.copy()
        z_plus[j] += epsilon
        z_minus[j] -= epsilon
        
        sigma_plus = softmax(z_plus)
        sigma_minus = softmax(z_minus)
        
        J[:, j] = (sigma_plus - sigma_minus) / (2 * epsilon)
    
    return J

def softmax_jacobian_analytical(sigma):
    """해석적 야코비안: J = diag(sigma) - sigma @ sigma.T"""
    n = len(sigma)
    return np.diag(sigma) - np.outer(sigma, sigma)

# 검증
z = np.array([1.0, 2.0, 0.5])
sigma = softmax(z)

J_numerical = softmax_jacobian_numerical(z)
J_analytical = softmax_jacobian_analytical(sigma)

print("=" * 60)
print("Softmax 야코비안 검증")
print("=" * 60)
print(f"\nInput z: {z}")
print(f"Softmax σ(z): {sigma}")
print(f"Sum of σ: {np.sum(sigma)}")

print("\nNumerical Jacobian:")
print(J_numerical)

print("\nAnalytical Jacobian:")
print(J_analytical)

print("\nDifference (L2 norm):")
diff = np.linalg.norm(J_numerical - J_analytical)
print(f"{diff:.2e}")

# 성질 검증: 각 열의 합이 0 (확률 제약)
print("\nJacobian 각 열의 합 (0이어야 함):")
print(np.sum(J_analytical, axis=0))
```

### Cross-Entropy + Softmax 역전파

```python
def cross_entropy_loss(sigma, y_true):
    """Cross-entropy 손실"""
    epsilon = 1e-10
    return -np.sum(y_true * np.log(np.clip(sigma, epsilon, 1.0)))

def cross_entropy_softmax_gradient_numerical(z, y_true, epsilon=1e-5):
    """수치 미분"""
    n = len(z)
    grad = np.zeros(n)
    
    for j in range(n):
        z_plus = z.copy()
        z_minus = z.copy()
        z_plus[j] += epsilon
        z_minus[j] -= epsilon
        
        loss_plus = cross_entropy_loss(softmax(z_plus), y_true)
        loss_minus = cross_entropy_loss(softmax(z_minus), y_true)
        
        grad[j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    return grad

def cross_entropy_softmax_gradient_analytical(z, y_true):
    """해석적: dL/dz = σ - y"""
    sigma = softmax(z)
    return sigma - y_true

# 검증
z = np.array([1.0, 2.0, 0.5])
y_true = np.array([0.0, 1.0, 0.0])  # 정답은 클래스 1

grad_numerical = cross_entropy_softmax_gradient_numerical(z, y_true)
grad_analytical = cross_entropy_softmax_gradient_analytical(z, y_true)

print("\n" + "=" * 60)
print("Cross-Entropy + Softmax Gradient 검증")
print("=" * 60)
print(f"\nInput z: {z}")
print(f"True label y: {y_true}")
print(f"Softmax σ(z): {softmax(z)}")

print("\nNumerical Gradient:")
print(grad_numerical)

print("\nAnalytical Gradient (σ - y):")
print(grad_analytical)

print("\nDifference (L2 norm):")
diff = np.linalg.norm(grad_numerical - grad_analytical)
print(f"{diff:.2e}")
```

### Temperature Softmax 시각화

```python
def softmax_with_temperature(z, T):
    """Temperature를 포함한 softmax"""
    z_scaled = z / T
    z_shifted = z_scaled - np.max(z_scaled)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

z = np.array([1.0, 2.0, 0.5])
temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

print("\n" + "=" * 60)
print("Temperature Softmax 효과")
print("=" * 60)
print(f"\nInput z: {z}\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 각 클래스의 확률
for i in range(len(z)):
    probs = [softmax_with_temperature(z, T)[i] for T in temperatures]
    ax1.plot(temperatures, probs, marker='o', label=f'Class {i}')

ax1.set_xlabel('Temperature T', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title('Temperature에 따른 확률 변화', fontsize=13)
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 엔트로피 변화
entropy = []
for T in temperatures:
    sigma = softmax_with_temperature(z, T)
    H = -np.sum(sigma * np.log(sigma + 1e-10))
    entropy.append(H)

ax2.plot(temperatures, entropy, marker='s', color='red', linewidth=2)
ax2.set_xlabel('Temperature T', fontsize=12)
ax2.set_ylabel('Entropy', fontsize=12)
ax2.set_title('Temperature에 따른 엔트로피', fontsize=13)
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/temperature_softmax.png', dpi=150, bbox_inches='tight')
print("그래프 저장: /tmp/temperature_softmax.png")

# 정량적 표
print("Temperature에 따른 확률분포:\n")
print(f"{'T':<8} | σ₁ (class 0)  | σ₂ (class 1)  | σ₃ (class 2)  | Entropy")
print("-" * 65)
for T in temperatures:
    sigma = softmax_with_temperature(z, T)
    H = -np.sum(sigma * np.log(sigma + 1e-10))
    print(f"{T:<8.2f} | {sigma[0]:12.6f} | {sigma[1]:12.6f} | {sigma[2]:12.6f} | {H:.6f}")
```

### Gumbel-Softmax 시뮬레이션

```python
def gumbel_noise(shape, seed=None):
    """Gumbel 분포 샘플"""
    if seed is not None:
        np.random.seed(seed)
    U = np.random.uniform(0, 1, shape)
    return -np.log(-np.log(U))

def gumbel_softmax(z, T, seed=None):
    """Gumbel-Softmax"""
    g = gumbel_noise(z.shape, seed)
    return softmax((z + g) / T)

def gumbel_softmax_hard(z, T, seed=None):
    """Gumbel-Softmax with straight-through estimator"""
    g = gumbel_noise(z.shape, seed)
    y = softmax((z + g) / T)
    y_hard = np.eye(len(z))[np.argmax(z + g)]
    return y + (y_hard - y)  # 역전파는 y 사용, forward는 y_hard

print("\n" + "=" * 60)
print("Gumbel-Softmax 시뮬레이션")
print("=" * 60)

z = np.array([2.0, 0.5, 1.0])
T = 0.5

print(f"\nInput z: {z}")
print(f"Temperature T: {T}\n")

# 여러 샘플 생성
n_samples = 10000
samples = np.array([gumbel_softmax(z, T) for _ in range(n_samples)])

print("Gumbel-Softmax 샘플의 클래스별 선택 확률:")
class_probs = np.mean(samples == np.max(samples, axis=1, keepdims=True), axis=0)
for i, p in enumerate(class_probs):
    print(f"  Class {i}: {p:.4f}")

print("\n수학적 확률 (softmax):")
sigma = softmax(z)
for i, p in enumerate(sigma):
    print(f"  Class {i}: {p:.4f}")

# T → 0일 때 one-hot 수렴
print(f"\nT → 0 극한에서 Gumbel-Softmax:")
for T_test in [0.1, 0.01, 0.001]:
    samples_test = np.array([gumbel_softmax(z, T_test) for _ in range(1000)])
    class_probs_test = np.mean(samples_test == np.max(samples_test, axis=1, keepdims=True), axis=0)
    print(f"  T={T_test}: {class_probs_test}")
```

### PyTorch 검증

```python
import torch
import torch.nn.functional as F

def verify_with_pytorch():
    print("\n" + "=" * 60)
    print("PyTorch 검증")
    print("=" * 60)
    
    z = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
    y_true = torch.tensor([0.0, 1.0, 0.0])
    
    # Forward pass
    sigma = F.softmax(z, dim=0)
    loss = F.cross_entropy(z.unsqueeze(0), torch.argmax(y_true).unsqueeze(0))
    
    # Backward pass
    loss.backward()
    
    print(f"\nInput z: {z.detach().numpy()}")
    print(f"Softmax σ(z): {sigma.detach().numpy()}")
    print(f"Cross-Entropy Loss: {loss.item():.6f}")
    print(f"\nGradient dL/dz (PyTorch):")
    print(sigma.detach().numpy() - y_true.numpy())
    print(f"\nGradient from backprop (z.grad):")
    print(z.grad.numpy())

verify_with_pytorch()
```

## 🔗 AI/ML 연결

### 1. 분류 모델의 기본

- 모든 신경망 분류기는 마지막 레이어에서 Softmax를 사용
- 역전파의 가장 깔끔한 형태: $\nabla_z \mathcal{L} = \sigma - y$

### 2. 증류학습 (Knowledge Distillation)

- Teacher 네트워크: high temperature로 부드러운 확률 출력
- Student 네트워크: 이를 모방하여 일반화 성능 향상
- Temperature scaling이 KL divergence 기울기 scale 조정

### 3. 강화학습의 정책 경사

- 정책 $\pi_\theta(a|s) = \text{softmax}(\phi(s, a)^\top \theta)$
- Softmax 야코비안이 정책 경사의 variance 조절

### 4. Mixture of Experts (MoE)

- Gating network: Softmax로 expert 선택
- 이산 선택을 미분 가능하게 → Gumbel-Softmax

## 📌 핵심 정리

| 개념 | 공식 | 역할 |
|------|------|------|
| **Softmax** | $\sigma_i = e^{z_i}/\sum_j e^{z_j}$ | 확률분포로 변환 |
| **야코비안** | $J = \text{diag}(\sigma) - \sigma\sigma^\top$ | 역전파 계산 |
| **CE + Softmax** | $\nabla_z L = \sigma - y$ | 간결한 기울기 |
| **수치 안정성** | $\sigma(z) = \sigma(z - \max z)$ | Overflow 방지 |
| **Temperature** | $\sigma(z/T)$ | 확률분포 경도 조절 |
| **Gumbel-Softmax** | $\sigma((z+g)/T)$ | 이산 선택의 미분 가능화 |

## 🤔 생각해볼 문제

1. **야코비안의 특성**: $\text{diag}(\sigma) - \sigma\sigma^\top$가 항상 singular matrix인 이유는? (힌트: 고유값)

2. **Log-Sum-Exp**: 왜 $\log \sum e^{z_i}$를 직접 계산하면 위험한가? 구체적인 수치 예시를 들어보세요.

3. **Temperature 극한**: $T \to \infty$일 때 Softmax 야코비안의 고유값은?

4. **Gumbel-Softmax의 미분**: 
$$\frac{\partial \text{GumbelSoftmax}_i}{\partial z_j} = ?$$
이것이 수치적으로 항상 계산 가능한가?

5. **배치 정규화와의 상호작용**: Softmax 앞에 배치 정규화를 넣으면 야코비안이 어떻게 변하는가?

<div align="center">

| | | |
|---|---|---|
| [◀ Ch6-05. AI 제약 최적화 응용](../ch6-constrained-optimization/05-constrained-ai-applications.md) | [📚 README](../README.md) | [02. BatchNorm·LayerNorm 기울기 ▶](./02-batch-layer-norm-gradient.md) |

</div>
